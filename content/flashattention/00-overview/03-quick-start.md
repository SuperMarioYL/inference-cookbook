---
title: "Flash Attention 快速上手指南"
weight: 3
---

## 先确定设备与入口

在已安装适配设备的 PyTorch 环境中运行：

```python
import torch

print(torch.__version__, torch.version.cuda, torch.version.hip)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))
```

下表给出独立安装路径；运行一个例子只需选择一条。FA2 的 NVIDIA 示例面向 Ampere、Ada 或 Hopper，要求 PyTorch ≥ 2.2、CUDA toolkit ≥ 12.0；FA3 上游列出 H100/H800 与 CUDA ≥ 12.3。FA4 预览包主要面向 Hopper 和 Blackwell，各功能支持范围还取决于架构。[FA2 安装说明](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/README.md) · [FA3、FA4 安装说明](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/README.md)

| 路线 | 安装命令 | 对应的 Python 导入 |
|---|---|---|
| FA2 稳定版 | `python -m pip install packaging psutil ninja`，随后 `python -m pip install 'flash-attn==2.8.3.post2' --no-build-isolation` | `from flash_attn import flash_attn_func` |
| FA3 Hopper | 检出下列固定源码，按上游说明在 `hopper/` 构建 | `from flash_attn_3.flash_attn_interface import flash_attn_func` |
| FA4 预览版 | `python -m pip install 'flash-attn-4==4.0.0b39'`；使用 CUDA 13 的环境可安装 `'flash-attn-4[cu13]==4.0.0b39'` | `from flash_attn.cute import flash_attn_func` |

```bash
git clone --branch fa4-v4.0.0.beta39 --recursive https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
python setup.py install
```

## 验证一次计算

下面选 FA2 的导入。Q/K/V 的布局为 `[B,S,H,D]`。使用长度相同的 Q/K、无 dropout 和因果 mask，将结果与 FP32 显式实现比较。FA3 可以按表替换导入。FA4 beta39 的函数返回一对值，调用行要写成 `actual, _ = flash_attn_func(q, k, v, causal=True)`。参考计算保持不变。[FA4 返回值](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/flash_attn/cute/interface.py)

```python
import torch
from flash_attn import flash_attn_func

torch.manual_seed(0)
shape = (1, 32, 2, 64)
q, k, v = [torch.randn(shape, device="cuda", dtype=torch.float16) for _ in range(3)]
actual = flash_attn_func(q, k, v, causal=True)
qh, kh, vh = [tensor.float().transpose(1, 2) for tensor in (q, k, v)]
scores = qh @ kh.transpose(-2, -1) / shape[-1]**0.5
allowed = torch.ones(shape[1], shape[1], device=q.device, dtype=torch.bool).tril()
expected = (scores.masked_fill(~allowed, -torch.inf).softmax(-1) @ vh).transpose(1, 2)
torch.testing.assert_close(actual.float(), expected, atol=0.005, rtol=0.005)
print(actual.shape)
```

这里的容差用于这组小规模 FP16 输入。将 Q/K 长度改为不相等时，要分别确认两种实现的 mask 对齐；直接用 PyTorch SDPA 的 `is_causal=True` 当参考可能会比较两个不同的问题。[FA2 不等长 mask 语义](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/README.md#21-change-behavior-of-causal-flag)

## FA2 的两种常见输入

### 打包 QKV

当 Q, K, V 已经拼接在一起时（如自注意力场景），使用此函数可以避免反向传播中显式拼接梯度。是否缩短 step 时间可以用同一训练 batch 测量。

```python
import torch
from flash_attn import flash_attn_qkvpacked_func

batch_size = 2
seqlen = 1024
nheads = 32
headdim = 128

# QKV 打包为一个张量, shape: (batch_size, seqlen, 3, nheads, headdim)
qkv = torch.randn(batch_size, seqlen, 3, nheads, headdim, device='cuda', dtype=torch.float16)

# 因果注意力
output = flash_attn_qkvpacked_func(qkv, causal=True)
print(f"输出形状: {output.shape}")  # torch.Size([2, 1024, 32, 128])
```

### 批中各序列长度不同

处理一个 batch 中不同长度的序列时使用，通过 cumulative sequence lengths 来索引:

```python
import torch
from flash_attn import flash_attn_varlen_func

# 假设 batch 中有 3 个序列，长度分别为 100, 200, 150
seqlens = [100, 200, 150]
total_tokens = sum(seqlens)  # 450
nheads = 32
nheads_k = 8
headdim = 128

# 所有序列拼接成一个张量, shape: (total_tokens, nheads, headdim)
q = torch.randn(total_tokens, nheads, headdim, device='cuda', dtype=torch.float16)
k = torch.randn(total_tokens, nheads_k, headdim, device='cuda', dtype=torch.float16)
v = torch.randn(total_tokens, nheads_k, headdim, device='cuda', dtype=torch.float16)

# 累积序列长度 (batch_size + 1,), dtype 必须是 int32
cu_seqlens_q = torch.tensor([0, 100, 300, 450], device='cuda', dtype=torch.int32)
cu_seqlens_k = torch.tensor([0, 100, 300, 450], device='cuda', dtype=torch.int32)

max_seqlen_q = max(seqlens)
max_seqlen_k = max(seqlens)

output = flash_attn_varlen_func(
    q, k, v,
    cu_seqlens_q=cu_seqlens_q,
    cu_seqlens_k=cu_seqlens_k,
    max_seqlen_q=max_seqlen_q,
    max_seqlen_k=max_seqlen_k,
    causal=True,
)
print(f"输出形状: {output.shape}")  # torch.Size([450, 32, 128])
```

## 定位安装问题

用 `python -m pip show flash-attn flash-attn-3 flash-attn-4` 检查当前解释器能看到的包，再核对表中的导入名。FA2 在本机源码编译遇到内存不足时，可通过 `MAX_JOBS=2` 减少并发。若导入成功但调用失败，先恢复上面的小输入，区分扩展加载、架构支持和特定形状限制。没有受支持 GPU 时可使用[在线 softmax 手算](/flashattention/02-core-algorithm/01-online-softmax/)检查数学过程。
