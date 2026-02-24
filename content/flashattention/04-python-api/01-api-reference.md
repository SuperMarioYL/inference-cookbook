---
title: "Python API 参考"
weight: 1
---

## 目录

- [1. 核心函数概览](#1-核心函数概览)
- [2. flash_attn_func — 标准 Attention](#2-flash_attn_func--标准-attention)
- [3. 打包变体 — QKVPacked 与 KVPacked](#3-打包变体--qkvpacked-与-kvpacked)
- [4. Varlen 变体 — 变长序列](#4-varlen-变体--变长序列)
- [5. flash_attn_with_kvcache — 推理专用](#5-flash_attn_with_kvcache--推理专用)
- [6. 后端选择逻辑](#6-后端选择逻辑)
- [7. 张量形状速查](#7-张量形状速查)

---

## 1. 核心函数概览

Flash Attention 提供 7 个公开 API 函数：

| 函数 | 用途 | 训练/推理 |
|------|------|----------|
| `flash_attn_func` | 标准 Q/K/V 分离输入 | 训练 + 推理 |
| `flash_attn_qkvpacked_func` | QKV 打包输入 | 训练 + 推理 |
| `flash_attn_kvpacked_func` | Q 分离, KV 打包 | 训练 + 推理 |
| `flash_attn_varlen_func` | 变长序列（无 padding） | 训练 + 推理 |
| `flash_attn_varlen_qkvpacked_func` | 变长 + QKV 打包 | 训练 + 推理 |
| `flash_attn_varlen_kvpacked_func` | 变长 + KV 打包 | 训练 + 推理 |
| `flash_attn_with_kvcache` | KV Cache 推理 | 仅推理 |

所有函数都支持：
- **MQA / GQA**：`nheads_k` 可以小于 `nheads`，自动广播
- **Causal Masking**：通过 `causal=True` 启用
- **Sliding Window**：通过 `window_size=(left, right)` 设置
- **Softcap**：通过 `softcap > 0` 启用分数截断

```python
# 导入
from flash_attn import (
    flash_attn_func,
    flash_attn_qkvpacked_func,
    flash_attn_kvpacked_func,
    flash_attn_varlen_func,
    flash_attn_varlen_qkvpacked_func,
    flash_attn_varlen_kvpacked_func,
    flash_attn_with_kvcache,
)
```

---

## 2. flash_attn_func — 标准 Attention

### 2.1 函数签名

```python
# flash_attn/flash_attn_interface.py:1145-1219
def flash_attn_func(
    q,                          # (batch_size, seqlen, nheads, headdim)
    k,                          # (batch_size, seqlen, nheads_k, headdim)
    v,                          # (batch_size, seqlen, nheads_k, headdim)
    dropout_p=0.0,              # Attention dropout 概率
    softmax_scale=None,         # 缩放因子，默认 1/sqrt(headdim)
    causal=False,               # 是否启用因果遮蔽
    window_size=(-1, -1),       # 滑动窗口 (left, right)，-1 表示无限
    softcap=0.0,                # 分数截断上限，0 表示不启用
    alibi_slopes=None,          # ALiBi 位置偏置斜率
    deterministic=False,        # 确定性反向传播（更慢但可复现）
    return_attn_probs=False,    # 返回 attention 概率（仅测试用）
):
```

### 2.2 参数详解

**输入张量**：

| 参数 | 形状 | 类型 | 说明 |
|------|------|------|------|
| `q` | `(B, S_q, H, D)` | FP16/BF16 | Query 张量 |
| `k` | `(B, S_k, H_k, D)` | FP16/BF16 | Key 张量 |
| `v` | `(B, S_k, H_k, D)` | FP16/BF16 | Value 张量 |

- `H_k` 可以小于 `H`，此时自动启用 GQA（`H` 必须能被 `H_k` 整除）
- `S_q` 和 `S_k` 可以不同（交叉注意力场景）
- `D`（headdim）必须是 8 的倍数；不满足时自动 padding

**关键参数**：

| 参数 | 默认值 | 说明 |
|------|-------|------|
| `softmax_scale` | `None` → `D**(-0.5)` | Attention 分数缩放因子 |
| `causal` | `False` | Causal mask 对齐到矩阵右下角 |
| `window_size` | `(-1, -1)` | 左右窗口大小，`(256, 0)` = 因果 + 窗口 256 |
| `softcap` | `0.0` | 启用时 $S = \text{softcap} \cdot \tanh(S/\text{softcap})$ |
| `deterministic` | `False` | True 时反向传播可复现但约慢 10-20% |

**ALiBi 支持**：

```python
# ALiBi 位置偏置
alibi_slopes = torch.tensor([1/2, 1/4, 1/8, ...], dtype=torch.float32)  # (nheads,)
# 或按 batch 不同：
alibi_slopes = torch.tensor(..., shape=(batch_size, nheads))
```

### 2.3 返回值

```python
out = flash_attn_func(q, k, v, causal=True)
# out: (batch_size, seqlen, nheads, headdim)

# 如果 return_attn_probs=True:
out, softmax_lse, S_dmask = flash_attn_func(q, k, v, return_attn_probs=True)
# softmax_lse: (batch_size, nheads, seqlen) — Log-Sum-Exp
# S_dmask: (batch_size, nheads, seqlen_q, seqlen_k) — Dropout 后的 attention 矩阵
```

### 2.4 使用示例

```python
import torch
from flash_attn import flash_attn_func

# 基础用法
B, S, H, D = 2, 1024, 32, 128
q = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(B, S, H, D, dtype=torch.float16, device="cuda")

out = flash_attn_func(q, k, v)  # (2, 1024, 32, 128)

# 因果 + GQA
H_kv = 8  # 4 个 query head 共享 1 个 KV head
k_gqa = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")
v_gqa = torch.randn(B, S, H_kv, D, dtype=torch.float16, device="cuda")
out = flash_attn_func(q, k_gqa, v_gqa, causal=True)

# 滑动窗口 + Softcap
out = flash_attn_func(q, k, v, window_size=(256, 256), softcap=50.0)

# 训练时的 Dropout
out = flash_attn_func(q, k, v, dropout_p=0.1, causal=True)
```

---

## 3. 打包变体 — QKVPacked 与 KVPacked

### 3.1 flash_attn_qkvpacked_func

```python
# flash_attn/flash_attn_interface.py:1008-1064
def flash_attn_qkvpacked_func(
    qkv,                        # (batch_size, seqlen, 3, nheads, headdim)
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
```

**适用场景**：自注意力（Self-Attention），Q/K/V 来自同一输入的线性投影。

**性能优势**：反向传播时避免 `dQ`, `dK`, `dV` 的拼接操作（因为梯度直接写入 `dQKV` 张量的对应位置）。

```python
# 使用示例
qkv = torch.randn(B, S, 3, H, D, dtype=torch.float16, device="cuda")
out = flash_attn_qkvpacked_func(qkv, causal=True)
# out: (B, S, H, D)
```

### 3.2 flash_attn_kvpacked_func

```python
# flash_attn/flash_attn_interface.py:1067-1142
def flash_attn_kvpacked_func(
    q,                          # (batch_size, seqlen_q, nheads, headdim)
    kv,                         # (batch_size, seqlen_k, 2, nheads_k, headdim)
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
):
```

**适用场景**：交叉注意力（Cross-Attention），K 和 V 来自同一编码器输出。

---

## 4. Varlen 变体 — 变长序列

### 4.1 核心概念

Varlen（Variable Length）变体处理 **无 padding 的批量序列**。序列按照 batch 维度拼接成一个长张量，通过累计长度数组 `cu_seqlens` 标记每个序列的边界：

```
标准格式 (有 padding):
  Batch 0: [tok0, tok1, tok2, PAD, PAD]    shape: (B=2, S=5, H, D)
  Batch 1: [tok0, tok1, tok2, tok3, tok4]

Varlen 格式 (无 padding):
  [tok0_0, tok1_0, tok2_0, tok0_1, tok1_1, tok2_1, tok3_1, tok4_1]
  cu_seqlens = [0, 3, 8]                   shape: (total=8, H, D)
  max_seqlen = 5
```

### 4.2 flash_attn_varlen_func

```python
# flash_attn/flash_attn_interface.py:1380-1471
def flash_attn_varlen_func(
    q,                          # (total_q, nheads, headdim)
    k,                          # (total_k, nheads_k, headdim)
    v,                          # (total_k, nheads_k, headdim)
    cu_seqlens_q,               # (batch_size + 1,) int32
    cu_seqlens_k,               # (batch_size + 1,) int32
    max_seqlen_q,               # int
    max_seqlen_k,               # int
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,           # 分页 KV Cache 页表
):
```

**关键参数**：

| 参数 | 说明 |
|------|------|
| `cu_seqlens_q` | 累计 Q 序列长度，如 `[0, 128, 384, 512]` 表示 3 个序列 |
| `cu_seqlens_k` | 累计 K 序列长度（可与 Q 不同） |
| `max_seqlen_q` | 最大 Q 序列长度（用于内核分块优化） |
| `max_seqlen_k` | 最大 K 序列长度 |
| `block_table` | 可选的分页 KV Cache 页表 |

```python
# 使用示例
total_q = 512
total_k = 512
cu_seqlens = torch.tensor([0, 128, 384, 512], dtype=torch.int32, device="cuda")
q = torch.randn(total_q, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(total_k, H, D, dtype=torch.float16, device="cuda")
v = torch.randn(total_k, H, D, dtype=torch.float16, device="cuda")

out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, 256, 256, causal=True)
# out: (512, H, D)
```

---

## 5. flash_attn_with_kvcache — 推理专用

### 5.1 函数签名

```python
# flash_attn/flash_attn_interface.py:1474-1616
def flash_attn_with_kvcache(
    q,                          # (batch_size, seqlen, nheads, headdim)
    k_cache,                    # (batch_size_cache, seqlen_cache, nheads_k, headdim)
    v_cache,                    # (batch_size_cache, seqlen_cache, nheads_k, headdim)
    k=None,                     # 新的 K 数据，追加到 cache
    v=None,                     # 新的 V 数据，追加到 cache
    rotary_cos=None,            # Rotary 余弦部分
    rotary_sin=None,            # Rotary 正弦部分
    cache_seqlens=None,         # 当前 cache 已有长度
    cache_batch_idx=None,       # Cache 行索引映射
    cache_leftpad=None,         # Cache 左填充偏移
    block_table=None,           # 分页 KV Cache 页表
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    rotary_interleaved=True,    # GPT-J 风格旋转
    alibi_slopes=None,
    num_splits=0,               # Flash Decoding 分块数
    return_softmax_lse=False,
):
```

### 5.2 核心特性

**In-place KV Cache 更新**：

```python
# 自动将 k, v 追加到 k_cache, v_cache
out = flash_attn_with_kvcache(
    q,                    # 新的 query token(s)
    k_cache, v_cache,     # 已有的 KV cache
    k=k_new, v=v_new,     # 新的 KV 数据
    cache_seqlens=cache_seqlens,  # 当前 cache 长度
    causal=True,
)
# k_cache 和 v_cache 已被 in-place 更新
```

**分页 KV Cache（Paged Attention）**：

```python
# 分页模式
block_table = torch.tensor([[0, 3, 5, 7], [1, 2, 4, 6]], dtype=torch.int32, device="cuda")
# block_table[i, j] = 第 i 个序列的第 j 个物理块编号
k_cache = torch.empty(num_blocks, page_size, H_kv, D, dtype=torch.float16, device="cuda")
v_cache = torch.empty(num_blocks, page_size, H_kv, D, dtype=torch.float16, device="cuda")

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    block_table=block_table,
    cache_seqlens=cache_seqlens,
    causal=True,
)
```

**融合 Rotary Embedding**：

```python
# Rotary 在内核内部应用，避免额外的内存读写
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    k=k_new, v=v_new,
    rotary_cos=cos, rotary_sin=sin,
    cache_seqlens=cache_seqlens,
    rotary_interleaved=True,   # GPT-J 风格
    causal=True,
)
```

### 5.3 Flash Decoding (num_splits)

```python
# 长序列推理时自动拆分 K/V 维度
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    num_splits=0,  # 0 = 自动选择最优分块数
    causal=True,
)
```

`num_splits > 1` 时启用 Split-K 策略，将 K/V 序列拆分到多个 SM 上并行计算，通过 combine kernel 合并结果。这对长序列推理（seqlen_k >> seqlen_q）尤其有效。

---

## 6. 后端选择逻辑

### 6.1 CUDA vs Triton

```python
# flash_attn/flash_attn_interface.py:11-15
USE_TRITON_ROCM = os.getenv("FLASH_ATTENTION_TRITON_AMD_ENABLE", "FALSE") == "TRUE"
if USE_TRITON_ROCM:
    from .flash_attn_triton_amd import flash_attn_2 as flash_attn_gpu
else:
    import flash_attn_2_cuda as flash_attn_gpu
```

- **NVIDIA GPU**：默认使用 CUDA 后端（`flash_attn_2_cuda`）
- **AMD GPU**：可通过环境变量启用 Triton 后端

### 6.2 SM80 vs SM90 选择

架构选择在 C++ 层面通过 `params.arch` 完成：

```python
# flash_attn/flash_attn_interface.py:23-46
def _get_block_size_n(device, head_dim, is_dropout, is_causal):
    major, minor = torch.cuda.get_device_capability(device)
    is_sm80 = major == 8 and minor == 0   # A100
    is_sm8x = major == 8 and minor > 0    # A6000, L40
    is_sm90 = major == 9 and minor == 0   # H100
    # 根据架构选择不同的 block size
```

### 6.3 torch.compile 集成

Flash Attention 通过 `torch.library.custom_op` 注册自定义算子，支持 `torch.compile`：

```python
# flash_attn/flash_attn_interface.py:53-73
# PyTorch >= 2.4.0
@torch.library.custom_op("flash_attn::_flash_attn_forward", mutates_args=())
def _flash_attn_forward(q, k, v, ...):
    return flash_attn_gpu.fwd(q, k, v, ...)

@_flash_attn_forward.register_fake
def _flash_attn_forward_fake(q, k, v, ...):
    # 返回正确形状的空张量（符号推断）
    return torch.empty_like(q), torch.empty(...), ...
```

这允许 `torch.compile` 在不执行实际内核的情况下推断输出形状和类型。

---

## 7. 张量形状速查

### 7.1 标准函数

| 函数 | Q | K | V | Output |
|------|---|---|---|--------|
| `flash_attn_func` | `(B, S_q, H, D)` | `(B, S_k, H_k, D)` | `(B, S_k, H_k, D)` | `(B, S_q, H, D)` |
| `flash_attn_qkvpacked_func` | `(B, S, 3, H, D)` | — | — | `(B, S, H, D)` |
| `flash_attn_kvpacked_func` | `(B, S_q, H, D)` | `(B, S_k, 2, H_k, D)` | — | `(B, S_q, H, D)` |

### 7.2 Varlen 函数

| 函数 | Q | K | V | Output |
|------|---|---|---|--------|
| `flash_attn_varlen_func` | `(T_q, H, D)` | `(T_k, H_k, D)` | `(T_k, H_k, D)` | `(T_q, H, D)` |
| `flash_attn_varlen_qkvpacked_func` | `(T, 3, H, D)` | — | — | `(T, H, D)` |
| `flash_attn_varlen_kvpacked_func` | `(T_q, H, D)` | `(T_k, 2, H_k, D)` | — | `(T_q, H, D)` |

### 7.3 KV Cache 函数

| 参数 | 形状 | 说明 |
|------|------|------|
| q | `(B, S_q, H, D)` | 通常 `S_q = 1`（逐 token） |
| k_cache | `(B_c, S_c, H_k, D)` 或分页格式 | 已有 cache |
| v_cache | `(B_c, S_c, H_k, D)` 或分页格式 | 已有 cache |
| k (新) | `(B, S_q, H_k, D)` | 追加到 cache |
| v (新) | `(B, S_q, H_k, D)` | 追加到 cache |
| block_table | `(B, max_blocks)` | 分页页表 |

### 7.4 HeadDim 对齐

所有 API 要求 `headdim` 为 8 的倍数。不满足时自动 padding：

```python
# flash_attn/flash_attn_interface.py:470-473
head_size_og = q.size(3)
if head_size_og % 8 != 0:
    q = torch.nn.functional.pad(q, [0, 8 - head_size_og % 8])
    k = torch.nn.functional.pad(k, [0, 8 - head_size_og % 8])
    v = torch.nn.functional.pad(v, [0, 8 - head_size_og % 8])
# 输出时 unpad: out = out[..., :head_size_og]
```

---

## 导航

- 上一篇：[SM80 内核实现对比](../03-cuda-kernel/05-kernel-sm80.md)
- 下一篇：[Autograd 集成](02-autograd-integration.md)
- [返回目录](../README.md)
