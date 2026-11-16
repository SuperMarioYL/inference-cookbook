---
title: "常见问题"
weight: 3
---

## 目录

- [安装与环境](#安装与环境)
- [API 使用](#api-使用)
- [数值与正确性](#数值与正确性)
- [性能问题](#性能问题)
- [推理与部署](#推理与部署)
- [高级功能](#高级功能)

---

## 安装与环境

### 如何判断安装问题发生在哪一层？

先运行 `python -m pip show flash-attn flash-attn-3 flash-attn-4`。包名、导入路径和设备能力是三个不同的问题。FA2 的编译进程被系统终止时，调低 `MAX_JOBS` 并检查机器剩余内存。导入成功后用[已知输入的最小例子](/flashattention/00-overview/03-quick-start/)执行 GPU 内核并比较数值。

### H100 上应该导入哪个包？

FA2 稳定包、FA3 Hopper 实现和 FA4 CuTeDSL 预览包都提供稠密 attention 路径，但函数签名、返回值和支持的 dtype 有差异。对已有模型，先追踪它实际导入的函数；自行选择时，按需要的输入形状、特性和工具链筛选，然后在同一硬件比较结果和延迟。[分代架构](/flashattention/00-overview/02-architecture-overview/)

## API 使用

### Q5: Flash Attention 的张量布局与 PyTorch 不同吗？

是的。Flash Attention 使用 `(batch, seqlen, num_heads, head_dim)` 布局，而 PyTorch 的 `nn.MultiheadAttention` 默认使用 `(seqlen, batch, embed_dim)`。

```python
# PyTorch 风格 → Flash Attention 风格
# 假设 x: (seqlen, batch, embed_dim)
x = x.transpose(0, 1)  # → (batch, seqlen, embed_dim)
x = x.view(batch, seqlen, num_heads, head_dim)  # → Flash Attention 格式
```

### Q6: 因果遮蔽在 seqlen_q != seqlen_k 时如何对齐？

FA2 的因果遮蔽向注意力矩阵的右下角对齐。Q、K 长度不同时，偏移可直接从两者的长度差算出。

- 当 `seqlen_q < seqlen_k` 时，前缀部分的 K 对所有 Q 可见（适用于 prefix-filling）
- 当 `seqlen_q > seqlen_k` 时，前面的 Q 位置没有可注意的 K

这是有意的设计决策，支持 KV Cache 场景中 seqlen_q=1 而 seqlen_k 很大的情况。

### Q7: 如何实现交叉注意力？

```python
from flash_attn import flash_attn_func

# Q 来自解码器，K/V 来自编码器
q = decoder_output  # (batch, seqlen_q, num_heads, head_dim)
k = encoder_output  # (batch, seqlen_k, num_heads, head_dim)
v = encoder_output  # (batch, seqlen_k, num_heads, head_dim)

out = flash_attn_func(q, k, v, causal=False)  # 交叉注意力不用因果遮蔽
```

也可以使用 KV 打包格式以获得更快的反向传播：

```python
from flash_attn import flash_attn_kvpacked_func

kv = torch.stack([k, v], dim=2)  # (batch, seqlen_k, 2, num_heads, head_dim)
out = flash_attn_kvpacked_func(q, kv)
```

### Q8: 如何处理变长序列的 padding？

```python
from flash_attn import flash_attn_varlen_func
from flash_attn.bert_padding import unpad_input, pad_input

# 方法一：手动构造 cu_seqlens
seqlens = [100, 200, 150]  # 各序列长度
cu_seqlens = torch.tensor([0] + list(torch.cumsum(torch.tensor(seqlens), 0).numpy()),
                          dtype=torch.int32, device='cuda')

# 方法二：使用 unpad_input 工具
x_unpad, indices, cu_seqlens, max_seqlen, _ = unpad_input(x_padded, attention_mask)
# ... 执行变长注意力 ...
out_padded = pad_input(out_unpad, indices, batch_size, max_seqlen_padded)
```

### Q9: `return_attn_probs=True` 是否影响性能？

FA2 打开这个测试接口后会额外返回 log-sum-exp。启用 dropout 时，内核还会写出 `S_dmask`，其空间随查询长度和键长度的乘积增长。测量日常推理的耗时与显存时，使用推理配置 `dropout_p=0` 和默认的 `return_attn_probs=False`。[查看测试输出示例](../07-usage-tutorial/01-basic-usage.md)

---

## 数值与正确性

### Q10: Flash Attention 的结果与标准注意力完全一致吗？

两者计算同一个注意力公式。FlashAttention 按块归约，浮点加法的顺序发生变化，最后几位可能不同。可以用相同的 Q、K、V 和 mask 做一次 FP32 参考计算，再比较输出误差。

参考容差：
| 数据类型 | 绝对误差 (atol) | 相对误差 (rtol) |
|---------|----------------|----------------|
| FP16 | 1e-3 | 1e-3 |
| BF16 | 5e-3 | 5e-3 |
| FP8 | 1e-2 | 1e-2 |

### Q11: 反向传播的结果是否确定性的？

默认不是。Flash Attention 使用原子操作（`atomicAdd`）累加 dQ 梯度，不同运行的结果可能有微小差异。

启用确定性模式：
```python
out = flash_attn_func(q, k, v, deterministic=True)
```

FA2 的 `deterministic=True` 选择确定性反向路径，所需工作区和延迟按当前 GPU、输入形状与精度测量。

### Q12: 训练中出现 NaN/Inf 怎么办？

排查步骤：
1. 检查输入是否包含 NaN：`torch.isnan(q).any()`
2. 检查 `softmax_scale` 是否过大（默认 $1/\sqrt{d}$）
3. 启用 `residual_in_fp32=True` 保持残差路径精度
4. 使用 BF16 替代 FP16（更大的动态范围）
5. 降低学习率

---

## 性能问题

### Q13: Flash Attention 比标准实现慢？

可能的原因：
1. **序列太短**（< 128）：内核启动开销占比大
2. **Head dimension 不是 8 的倍数**：需要 padding
3. **输入不连续**：需要先 `.contiguous()`
4. **GPU 利用率不足**：batch × heads 太小
5. **首次调用开销**：CUDA 内核编译/加载，后续调用会快很多

要确认实际运行的路径，先记录 Python 导入的包名和版本，再在 GPU profiler 中检查对应的 kernel。对于 PyTorch SDPA，用 `torch.nn.attention.sdpa_kernel` 临时限制候选后端，可得到不满足输入条件的原因。

### Q14: 如何选择最优的 num_splits？

推荐使用 `num_splits=0`（自动决策）。自动策略考虑了 SM 数量、L2 Cache 大小和序列长度。手动调优：

- `batch × heads >= SM_count`：`num_splits=1`
- `batch × heads < SM_count / 2`：增加 splits 直到接近 SM_count
- 超长序列（KV > L2）：增加 splits 使每个 split 的 KV 能放入 L2

### Q15: GQA 比 MHA 慢？

当 `seqlen_q` 足够长且 `num_heads_q` 已经接近 `kBlockM` 的倍数时，PackGQA 可能比非打包模式略慢（额外的 divmod 和 `__shfl_sync` 开销）。此时可以手动禁用：

```python
from flash_attn_3.flash_attn_interface import flash_attn_func
out = flash_attn_func(q, k, v, pack_gqa=False)
```

### Q16: 如何测量 Flash Attention 的实际吞吐量？

```python
from triton.testing import do_bench

# 正确的测量方式（包含 warmup）
time_ms = do_bench(lambda: flash_attn_func(q, k, v, causal=True),
                   warmup=5, rep=30)

# 计算 TFLOPS
total_flops = batch * nheads * 2 * seqlen_q * (seqlen_k / 2) * (headdim * 2)  # causal
tflops = total_flops / (time_ms * 1e-3 * 1e12)
```

> 详见 [性能测试](/flashattention/07-usage-tutorial/04-benchmarking/)

---

## 推理与部署

### Q17: KV Cache 需要手动管理 cache_seqlens 吗？

是的。`flash_attn_with_kvcache` 不会自动更新 `cache_seqlens`。每次追加新 token 后需要手动递增：

```python
out = flash_attn_with_kvcache(q, k_cache, v_cache, k=k_new, v=v_new,
                              cache_seqlens=cache_seqlens, causal=True)
cache_seqlens += k_new.shape[1]  # 手动更新
```

### Q18: Paged KV Cache 的 page_size 如何选择？

FA2 稳定包的 CUDA `flash_attn_with_kvcache` 在入口检查页大小能否被 256 整除。页表用物理块编号，页内的 token 数、张量布局与 dtype 都必须与调用者持有的缓存一致。[写入示例](/flashattention/07-usage-tutorial/03-inference-optimization/)

### Q19: 可以使用 CUDA Graph 加速推理吗？

可以。Flash Attention 兼容 CUDA Graph：

```python
# 预热
for _ in range(3):
    out = flash_attn_with_kvcache(q, k_cache, v_cache, ...)
torch.cuda.synchronize()

# 捕获 Graph
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    out = flash_attn_with_kvcache(q, k_cache, v_cache, ...)

# 重放（极低延迟）
g.replay()
```

注意：使用 CUDA Graph 时，输入张量的地址不能改变（只能改变内容）。

### Q20: FP8 推理的输出为什么是 BF16？

本例指 FA3 的 Hopper FP8 前向。该 API 以 BF16 写出结果。这在 softmax 归一化和 V 的 descale 应用后已经需要更高精度的中间结果。

---

## 高级功能

### Q21: 如何同时使用因果遮蔽和滑动窗口？

```python
# 因果 + 向左 256 token 的滑动窗口
out = flash_attn_func(q, k, v, window_size=(256, 0))

# 等价于 causal=True 并限制左侧窗口
# 位置 i 只能看到 [max(0, i-256), i] 范围的 K
```

`window_size=(left, right)` 中 `right=0` 即因果，`left` 控制滑动窗口大小。

### Q22: Softcap 和因果遮蔽可以同时使用吗？

可以。Softcap 在 masking 之前应用：

```python
out = flash_attn_func(q, k, v, causal=True, softcap=50.0)
# 计算顺序: QK^T → softcap(tanh) → causal mask → softmax → PV
```

### Q23: 用 PyTorch 核对 FA2 的结果时，怎样处理不等长因果遮蔽？

对 `Sq != Sk`，FA2 的因果 mask 向右下角对齐。用 SDPA 作参考时显式构造相同的布尔 mask。下面覆盖解码常见的 `Sq <= Sk` 情形，且 Q/K/V 头数相同。

```python
import torch
from torch.nn.functional import scaled_dot_product_attention

def causal_reference(q, k, v):
    sq, sk = q.size(1), k.size(1)
    assert 0 < sq <= sk
    row = torch.arange(sq, device=q.device)[:, None]
    col = torch.arange(sk, device=q.device)[None, :]
    allowed = col <= row + sk - sq
    qh, kh, vh = (x.transpose(1, 2) for x in (q, k, v))
    return scaled_dot_product_attention(qh, kh, vh, attn_mask=allowed).transpose(1, 2)
```

### Q24: torch.compile 与 Flash Attention 兼容吗？

兼容。Flash Attention 使用 `torch.library.custom_op` 注册，支持 `torch.compile` 的 tracing。

```python
model = MyModel()  # 内含 flash_attn_func 调用
compiled = torch.compile(model, mode='reduce-overhead')
out = compiled(x)  # 正常工作
```

限制：`return_attn_probs=True` 不兼容 compile。

---

## 导航

- 上一篇：[论文阅读指南](/flashattention/08-appendix/02-paper-reading-guide/)
- [返回目录](/flashattention/00-overview/overview/)
