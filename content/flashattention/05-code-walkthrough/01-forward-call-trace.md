---
title: "前向调用链追踪"
weight: 1
---

以 FA2 `v2.8.3.post2` 的 NVIDIA CUDA 稠密前向为例。输入 `[2, 1024, 32, 128]` 表示两条等长序列，每个位置 32 个头，每头 128 维。调用者给出 `causal=True`，函数按这些条件分配输出并派发内核。[FA2 Python 入口](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/flash_attn/flash_attn_interface.py)

```mermaid
flowchart LR
    A["flash_attn_func"] --> B["FlashAttnFunc.apply"]
    B --> C["_wrapped_flash_attn_forward"]
    C --> D["flash_attn_gpu.fwd"]
    D --> E["mha_fwd"]
    E --> F["run_mha_fwd"]
    F --> G["FA2 CUDA 前向内核"]
```

| 边界 | 本例要追的状态 |
|---|---|
| Python → Autograd | `flash_attn_func` 调用 `FlashAttnFunc.apply`。前向处理最后一维的布局和 padding。存在梯度需求时，保存反向所需张量。 |
| Custom op → 扩展 | `_flash_attn_forward` 调用 `flash_attn_gpu.fwd`。NVIDIA 构建中的模块为 `flash_attn_2_cuda`。 |
| C++ → CUDA | `mha_fwd` 校验设备、形状、dtype，构造 `Flash_fwd_params`；`run_mha_fwd` 按 dtype、head dimension、causal 等条件选择已实例化的内核。 |
| 内核 → 调用者 | 每个 Q 行块遍历可见的 K/V 块，更新行统计量，写回 O 和反向所需的 log-sum-exp。Python 返回去掉额外 padding 的输出。 |

[扩展绑定](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/csrc/flash_attn/flash_api.cpp) · [FA2 前向内核](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/csrc/flash_attn/src/flash_fwd_kernel.h)

## FA2 的 Python 包装与 C++ 参数

以下节选省略错误分支与部分参数，便于按函数名对照固定 tag。

### 用户 API 入口

```python
# flash_attn/flash_attn_interface.py
def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False,
                    window_size=(-1, -1), softcap=0.0, alibi_slopes=None,
                    deterministic=False, return_attn_probs=False):
    return FlashAttnFunc.apply(
        q, k, v, dropout_p, softmax_scale, causal,
        window_size, softcap, alibi_slopes, deterministic,
        return_attn_probs, torch.is_grad_enabled(),  # 传入梯度开关状态
    )
```

**操作**：将用户参数直接转发给 `FlashAttnFunc.apply()`，额外传入 `torch.is_grad_enabled()` 的结果。

### Autograd Function

```python
# flash_attn/flash_attn_interface.py
class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, dropout_p, softmax_scale, causal,
                window_size, softcap, alibi_slopes, deterministic,
                return_softmax, is_grad_enabled):
        # 1. 判断是否需要梯度
        is_grad = is_grad_enabled and any(x.requires_grad for x in [q, k, v])

        # 2. 默认 softmax_scale
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        # 3. Head dimension padding 到 8 的倍数
        head_size_og = q.size(3)
        if head_size_og % 8 != 0:
            q = F.pad(q, [0, 8 - head_size_og % 8])
            k = F.pad(k, [0, 8 - head_size_og % 8])
            v = F.pad(v, [0, 8 - head_size_og % 8])

        # 4. 调用 CUDA 前向
        out_padded, softmax_lse, S_dmask, rng_state = \
            _wrapped_flash_attn_forward(q, k, v, dropout_p, softmax_scale, ...)

        # 5. 保存反向所需张量
        if is_grad:
            ctx.save_for_backward(q, k, v, out_padded, softmax_lse, rng_state)
            ctx.dropout_p = dropout_p
            # ... 保存其他标量

        # 6. 裁剪 padding
        out = out_padded[..., :head_size_og]
        return out
```

### Custom Op 层

```python
# flash_attn/flash_attn_interface.py
@_torch_custom_op_wrapper("flash_attn::_flash_attn_forward", mutates_args=(), device_types="cuda")
def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal,
                         window_size_left, window_size_right, softcap,
                         alibi_slopes, return_softmax):
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    out, softmax_lse, S_dmask, rng_state = flash_attn_gpu.fwd(
        q, k, v, None,        # out=None（由 C++ 层分配）
        alibi_slopes,
        dropout_p, softmax_scale, causal,
        window_size_left, window_size_right, softcap,
        return_softmax, None,  # generator=None
    )
    return out, softmax_lse, S_dmask, rng_state
```

**关键操作**：
- `maybe_contiguous()`：确保张量最后一个维度连续（stride[-1] == 1）
- `flash_attn_gpu` 是编译后的 C++ 扩展模块（`flash_attn_2_cuda`）



### 参数验证

```cpp
// csrc/flash_attn/flash_api.cpp
// 设备检查
TORCH_CHECK(q.is_cuda(), "Input tensor must be on CUDA device");
// 数据类型检查
TORCH_CHECK(q.dtype() == torch::kFloat16 || q.dtype() == torch::kBFloat16);
// 形状提取
const int batch_size = sizes[0];
const int seqlen_q = sizes[1];
const int num_heads = sizes[2];
const int head_size = sizes[3];
// 维度约束
TORCH_CHECK(head_size <= 256);
TORCH_CHECK(head_size % 8 == 0, "head_size should be a multiple of 8");
```

### 维度对齐与输出分配

```cpp
// csrc/flash_attn/flash_api.cpp
// 向上取整到 32/64 的倍数
const int head_size_rounded = round_multiple(head_size, head_size <= 128 ? 32 : 64);
const int seqlen_q_rounded = round_multiple(seqlen_q, 128);
const int seqlen_k_rounded = round_multiple(seqlen_k, 128);

// 分配输出张量
at::Tensor out;
if (out_.has_value()) { out = out_.value(); }
else { out = torch::empty_like(q); }

// 分配 softmax_lse
auto softmax_lse = torch::empty({batch_size, num_heads, seqlen_q},
                                opts.dtype(at::kFloat));
```

### Flash_fwd_params 结构体填充

`set_params_fprop()` 将所有参数打包到 `Flash_fwd_params` 结构体：

```cpp
// csrc/flash_attn/flash_api.cpp
void set_params_fprop(Flash_fwd_params &params, ...) {
    // 张量指针
    params.q_ptr = q.data_ptr();
    params.k_ptr = k.data_ptr();
    params.v_ptr = v.data_ptr();
    params.o_ptr = out.data_ptr();

    // 步长（以元素为单位）
    params.q_row_stride = q.stride(-3);
    params.q_head_stride = q.stride(-2);

    // 维度信息
    params.d = head_size;
    params.d_rounded = head_size_rounded;

    // Softmax 缩放与 Softcap
    params.scale_softmax = softmax_scale;
    params.scale_softmax_log2 = softmax_scale * M_LOG2E;  // log2(e) 优化

    // Dropout
    params.p_dropout = 1.f - p_dropout;  // 注意：存储的是保留概率
    params.rp_dropout = 1.f / (1.f - p_dropout);
    params.scale_softmax_rp_dropout = params.rp_dropout * params.scale_softmax;

    // 窗口与因果
    if (is_causal) { window_size_left = -1; window_size_right = 0; }
    params.window_size_left = window_size_left;
    params.window_size_right = window_size_right;
    params.is_causal = window_size_left < 0 && window_size_right == 0;
}
```

**关键参数变换**：

| Python 参数 | C++ params 字段 | 变换 |
|------------|----------------|------|
| `softmax_scale` | `scale_softmax_log2` | `× log2(e)`（用于 `exp2f` 优化） |
| `dropout_p` | `p_dropout` | `1 - dropout_p`（保留概率） |
| `dropout_p` | `rp_dropout` | `1 / (1 - dropout_p)`（缩放因子） |
| `head_size` | `d_rounded` | 向上取整到 32/64 的倍数 |
| `causal=True` | `window_size_right=0` | 因果转化为窗口参数 |

## 从调用方反查 FA3 和 FA4

FA3 的 `flash_attn_3.flash_attn_interface` 进入 Hopper 扩展。在这个目录内才会看到 TMA、warp-group 矩阵乘和 `FlashAttnFwdSm90`。FA4 的 `flash_attn.cute.flash_attn_func` 则在 CuTeDSL 的 `interface.py` 进入 `FlashAttnFunc.apply` 和 `_flash_attn_fwd`；在 beta39，Python 返回 `(out, lse)`。单看同名的 `flash_attn_func` 或卡型不足以选择这三条调用链。[FA3 入口](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/hopper/flash_attn_interface.py) · [FA4 入口](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/flash_attn/cute/interface.py)

### FA3 的 Hopper 时序

从 FA3 的 Python 接口进入，`torch.ops.flash_attn_3` 的 CUDA 实现绑定在 `hopper/flash_api.cpp`。`run_mha_fwd` 为 dtype、head dimension 等运行时条件选择提前实例化的模板；随后 `hopper/flash_fwd_launch_template.h` 选择 tile、scheduler、mainloop 和 epilogue。下图仅示意可重叠的一次流水，实际 stage 与 warp 分工由实例化参数决定。[Hopper 启动模板](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/hopper/flash_fwd_launch_template.h)

{{< trace >}}
title: "调度器和 warp group 交接一个 tile"
actors: ["调度器", "搬运组", "计算组", "Epilogue"]
steps:
  - from: 0
    to: 1
    label: "分配 Q 行块"
    detail: "工作项含 batch、head 和 Q block，决定本次要写哪个 O 范围。"
  - from: 1
    to: 2
    label: "交付一个 KV 块"
    detail: "搬运组把需要的 K/V 数据送到可用 stage，并在就绪点通知计算方。"
  - from: 2
    to: 2
    label: "累计结果"
    detail: "矩阵乘与行统计共同更新当前输出；之后可继续消费新的 K/V 块。"
  - from: 2
    to: 3
    label: "最后一块结束"
    detail: "归一化参数就绪，epilogue 把当前行块的 O 和相应统计写回。"
{{< /trace >}}

[反向调用链](/flashattention/05-code-walkthrough/02-backward-call-trace/) · [数值验证与调试](/flashattention/05-code-walkthrough/03-debug-guide/)
