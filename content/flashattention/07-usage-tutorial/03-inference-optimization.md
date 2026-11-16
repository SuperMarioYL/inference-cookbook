---
title: "推理优化"
weight: 3
---

## 目录

- [1. 概述](#1-概述)
- [2. KV Cache 基础](#2-kv-cache-基础)
- [3. flash_attn_with_kvcache 详解](#3-flash_attn_with_kvcache-详解)
- [4. Paged KV Cache](#4-paged-kv-cache)
- [5. 自回归生成实现](#5-自回归生成实现)
- [6. Flash Decoding（Split-KV）](#6-flash-decodingsplit-kv)
- [7. FP8 推理](#7-fp8-推理)
- [8. 性能优化策略](#8-性能优化策略)

---

## 1. 概述

推理阶段的注意力计算与训练有显著不同。自回归生成分为两个阶段：

```mermaid
graph LR
    subgraph prefill["Prefill 阶段"]
        P1["输入完整 prompt"]
        P2["一次性计算所有 token"]
        P3["填充 KV Cache"]
        P1 --> P2 --> P3
    end

    subgraph decode["Decode 阶段"]
        D1["每次生成 1 个 token"]
        D2["seqlen_q = 1"]
        D3["从 KV Cache 读取历史"]
        D1 --> D2 --> D3
        D3 -->|"循环"| D1
    end

    prefill -->|"进入逐 token 生成"| decode
```

| 阶段 | seqlen_q | seqlen_k | 特点 | 瓶颈 |
|------|----------|----------|------|------|
| Prefill | 长（数百~数千） | 等于 seqlen_q | 计算密集 | Compute Bound |
| Decode | 1 | 递增（cache 长度） | 带宽密集 | Memory Bound |

Flash Attention 针对两个阶段提供不同的优化策略。

---

## 2. KV Cache 基础

### 2.1 什么是 KV Cache

自回归模型每生成一个 token，需要与之前所有 token 计算注意力。KV Cache 将之前计算的 K、V 缓存在 GPU 内存中，避免重复计算：

```python
# 不使用 KV Cache：每步重新计算所有 K、V
# 步骤 1: Q=[t1], K=[t1], V=[t1]           → 1×1 注意力
# 步骤 2: Q=[t1,t2], K=[t1,t2], V=[t1,t2]  → 2×2 注意力
# 步骤 3: Q=[t1,t2,t3], K=[...], V=[...]   → 3×3 注意力
# 总计算: 1+4+9 = O(N³)

# 使用 KV Cache：只计算新 token，复用历史
# 步骤 1: Q=[t1], K=cache[t1], V=cache[t1]          → 1×1
# 步骤 2: Q=[t2], K=cache[t1,t2], V=cache[t1,t2]    → 1×2
# 步骤 3: Q=[t3], K=cache[t1,t2,t3], V=cache[...]   → 1×3
# 总计算: 1+2+3 = O(N²)
```

### 2.2 预分配 Cache

使用 MHA 模块时，可以预分配 KV Cache：

```python
from flash_attn.modules.mha import MHA

mha = MHA(
    embed_dim=1024,
    num_heads=16,
    num_heads_kv=4,        # GQA
    layer_idx=0,           # 必须指定 layer_idx
    causal=True,
    use_flash_attn=True,
)

# 预分配 KV Cache
kv_cache = mha.allocate_inference_cache(
    batch_size=32,
    max_seqlen=2048,
    dtype=torch.float16,
)
# kv_cache 形状: (batch_size, max_seqlen, 2, num_heads_kv, head_dim)
```

---

## 3. flash_attn_with_kvcache 详解

### 3.1 基本用法

`flash_attn_with_kvcache` 是推理的核心 API，支持 KV Cache 的原地更新和注意力计算：

```python
from flash_attn import flash_attn_with_kvcache

batch_size = 4
max_seqlen_k = 2048
num_heads = 32
num_heads_kv = 8
head_dim = 128

# 预分配 KV Cache
k_cache = torch.zeros(batch_size, max_seqlen_k, num_heads_kv, head_dim,
                      device='cuda', dtype=torch.float16)
v_cache = torch.zeros(batch_size, max_seqlen_k, num_heads_kv, head_dim,
                      device='cuda', dtype=torch.float16)

# 当前序列长度（每个 batch 元素的已缓存长度）
cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device='cuda')
```

### 3.2 Prefill 阶段

```python
# Prefill: 处理完整 prompt
prompt_len = 512
q = torch.randn(batch_size, prompt_len, num_heads, head_dim,
               device='cuda', dtype=torch.float16)
k = torch.randn(batch_size, prompt_len, num_heads_kv, head_dim,
               device='cuda', dtype=torch.float16)
v = torch.randn(batch_size, prompt_len, num_heads_kv, head_dim,
               device='cuda', dtype=torch.float16)

out = flash_attn_with_kvcache(
    q,
    k_cache, v_cache,
    k=k, v=v,               # 新的 K/V 会被写入 cache
    cache_seqlens=cache_seqlens,  # 告诉内核当前 cache 长度
    causal=True,
)
# cache_seqlens 不会自动更新，需要手动更新
cache_seqlens += prompt_len
```

### 3.3 Decode 阶段

```python
# Decode: 每次生成一个 token
q_new = torch.randn(batch_size, 1, num_heads, head_dim,
                    device='cuda', dtype=torch.float16)
k_new = torch.randn(batch_size, 1, num_heads_kv, head_dim,
                    device='cuda', dtype=torch.float16)
v_new = torch.randn(batch_size, 1, num_heads_kv, head_dim,
                    device='cuda', dtype=torch.float16)

out = flash_attn_with_kvcache(
    q_new,
    k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    causal=True,
)
cache_seqlens += 1
```

### 3.4 融合 Rotary Embedding

`flash_attn_with_kvcache` 支持在内核中直接应用 Rotary Embedding，避免额外的内存读写：

```python
from flash_attn.layers.rotary import RotaryEmbedding

rotary = RotaryEmbedding(dim=head_dim // 2)

# 获取 cos/sin 缓存
cos = rotary._cos_cached
sin = rotary._sin_cached

out = flash_attn_with_kvcache(
    q_new,
    k_cache, v_cache,
    k=k_new, v=v_new,
    rotary_cos=cos,
    rotary_sin=sin,
    cache_seqlens=cache_seqlens,
    rotary_interleaved=False,   # GPT-NeoX 风格
    causal=True,
)
```

融合 Rotary 的好处：
- 减少一次 Q 和 K 的全局内存读写
- 在写入 Cache 前完成旋转，Cache 中存储的是旋转后的 K/V

### 3.5 获取 Softmax LSE

```python
out, softmax_lse = flash_attn_with_kvcache(
    q_new, k_cache, v_cache,
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    return_softmax_lse=True,
)
# softmax_lse: (batch_size, num_heads, seqlen_q)
```

---

## 4. Paged KV Cache

### 4.1 为什么需要 Paged KV Cache

连续 KV Cache 要求为每个序列预分配最大长度的内存。当 batch 中序列长度差异大时，造成大量浪费。Paged KV Cache（灵感来自操作系统的虚拟内存）将 Cache 分成固定大小的页，按需分配：

为了看清页内空槽，这里把页长缩小到四个 token。下节的 FA2 CUDA 接口使用实际支持的页长。

{{< stateboard >}}
title: "两条不等长序列怎样占用页池"
rows: ["页号", "已用位置", "空槽"]
steps:
  - title: "A 写入四个 token"
    detail: "用每页四槽的教学模型，A0 到 A3 填满 P0。"
    cells: [["P0", "P1", "P2"], ["A×4", "·", "·"], ["0", "4", "4"]]
    focus: [[1, 0]]
  - title: "B 写入两个 token"
    detail: "为 B 分配 P1，B0、B1 占前两槽；P1 内余下两槽仍归 B 的末页。"
    cells: [["P0", "P1", "P2"], ["A×4", "B×2", "·"], ["0", "2", "4"]]
    focus: [[1, 1], [2, 1]]
  - title: "A 继续生成"
    detail: "A4 落到新分配的 P2。A 的页表为 [P0,P2]；B 的 P1 不必移动。"
    cells: [["P0", "P1", "P2"], ["A×4", "B×2", "A×1"], ["0", "2", "3"]]
    focus: [[1, 2], [2, 2]]
{{< /stateboard >}}

### 4.2 初始化页表与追加 token

FA2 的 `flash_attn_with_kvcache` 接受 `int32` 的逻辑到物理块编号。这个例子从空 cache 开始，两条序列各占一页；故意让第一条使用物理页 1，第二条使用物理页 0。内核写入 K/V，写入后的长度由调用方维护。

```python
import torch
from flash_attn import flash_attn_with_kvcache

device = "cuda"
page_size, heads, dim = 256, 4, 64
k_cache = torch.zeros(2, page_size, heads, dim, device=device, dtype=torch.float16)
v_cache = torch.zeros_like(k_cache)
block_table = torch.tensor([[1], [0]], dtype=torch.int32, device=device)
lengths = torch.zeros(2, dtype=torch.int32, device=device)
q, k, v = [torch.randn(2, 1, heads, dim, device=device, dtype=k_cache.dtype) for _ in range(3)]
out = flash_attn_with_kvcache(
    q, k_cache, v_cache, k=k, v=v, cache_seqlens=lengths,
    block_table=block_table, causal=True,
)
lengths += k.size(1)
torch.testing.assert_close(k_cache[1, 0], k[0, 0])
torch.testing.assert_close(k_cache[0, 0], k[1, 0])
torch.testing.assert_close(out, v)
```

两条序列此时都只看得到各自的一个新 token，因此输出等于 V。继续生成前先判断将写入的位置是否已有物理页。FA2 此接口要求每页 token 数是 256 的倍数；接入服务框架还要核对页布局、dtype、KV 头数与设备，页表概念相同并不保证缓存可直接传递。[FA2 参数检查](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/csrc/flash_attn/flash_api.cpp)

## 5. 自回归生成实现

### 5.1 完整生成循环

```python
import torch
from flash_attn import flash_attn_with_kvcache

def generate(model, prompt_ids, max_new_tokens, temperature=1.0):
    """简化的自回归生成循环"""
    batch_size = prompt_ids.shape[0]
    device = prompt_ids.device

    # 预分配 KV Cache（每一层）
    num_layers = len(model.layers)
    max_seqlen = prompt_ids.shape[1] + max_new_tokens
    kv_caches = [
        (
            torch.zeros(batch_size, max_seqlen, num_heads_kv, head_dim,
                       device=device, dtype=torch.float16),
            torch.zeros(batch_size, max_seqlen, num_heads_kv, head_dim,
                       device=device, dtype=torch.float16),
        )
        for _ in range(num_layers)
    ]
    cache_seqlens = torch.zeros(batch_size, dtype=torch.int32, device=device)

    # === Prefill 阶段 ===
    prompt_embeds = model.embed(prompt_ids)
    hidden = prompt_embeds
    for layer_idx, layer in enumerate(model.layers):
        q, k, v = layer.qkv_proj(hidden)
        k_cache, v_cache = kv_caches[layer_idx]
        attn_out = flash_attn_with_kvcache(
            q, k_cache, v_cache,
            k=k, v=v,
            cache_seqlens=cache_seqlens,
            causal=True,
        )
        hidden = layer.ffn(layer.norm(attn_out + hidden))

    cache_seqlens += prompt_ids.shape[1]
    logits = model.lm_head(hidden[:, -1:, :])
    next_token = torch.argmax(logits / temperature, dim=-1)
    generated = [next_token]

    # === Decode 阶段 ===
    for step in range(max_new_tokens - 1):
        hidden = model.embed(next_token)
        for layer_idx, layer in enumerate(model.layers):
            q, k, v = layer.qkv_proj(hidden)
            k_cache, v_cache = kv_caches[layer_idx]
            attn_out = flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=cache_seqlens,
                causal=True,
            )
            hidden = layer.ffn(layer.norm(attn_out + hidden))

        cache_seqlens += 1
        logits = model.lm_head(hidden)
        next_token = torch.argmax(logits / temperature, dim=-1)
        generated.append(next_token)

    return torch.cat(generated, dim=1)
```

### 5.2 使用 cache_batch_idx

当 batch 中某些序列已完成生成时，可以使用 `cache_batch_idx` 跳过它们：

```python
# 只处理 batch 中的部分序列
active_indices = torch.tensor([0, 2, 5], dtype=torch.int32, device='cuda')

out = flash_attn_with_kvcache(
    q_active,              # 只包含活跃序列的 Q
    k_cache, v_cache,      # 完整 cache
    k=k_new, v=v_new,
    cache_seqlens=cache_seqlens,
    cache_batch_idx=active_indices,  # 映射到 cache 中的位置
    causal=True,
)
```

### 5.3 Left Padding 支持

```python
# 当 cache 的有效数据不从位置 0 开始时
cache_leftpad = torch.tensor([0, 100, 50, 0], dtype=torch.int32, device='cuda')

out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    cache_leftpad=cache_leftpad,  # 跳过每个序列开头的空白
    causal=True,
)
```

---

## 6. Flash Decoding（Split-KV）

### 6.1 原理

Decode 阶段（seqlen_q=1）的计算量小，单个 Thread Block 无法充分利用 GPU。Flash Decoding 将 KV 序列分成多个 split，每个 split 由独立的 Thread Block 并行处理，然后合并结果：

{{< stateboard >}}
title: "两个 split 的输出怎样合并"
rows: ["split", "归一化权重和", "局部输出"]
steps:
  - title: "第一个 split"
    detail: "为便于手算，两段使用同一分数基准。第一段的未归一化权重之和为 1，局部输出为 2。"
    cells: [["A", "B"], ["1", "·"], ["2", "·"]]
    focus: [[1, 0], [2, 0]]
  - title: "第二个 split"
    detail: "第二段权重和为 3，局部输出为 6。两段局部输出不能按相同权重平均。"
    cells: [["A", "B"], ["1", "3"], ["2", "6"]]
    focus: [[1, 1], [2, 1]]
  - title: "归并两个结果"
    detail: "归一化系数为 1/4 与 3/4，全局 O=2×1/4+6×3/4=5。实际实现可用局部 LSE 对齐尺度。"
    cells: [["A", "B"], ["0.25", "0.75"], ["0.5", "4.5"]]
    focus: [[2, 0], [2, 1]]
{{< /stateboard >}}

### 6.2 使用方式

```python
# 自动决策 split 数（推荐）
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    num_splits=0,       # 0 = 自动，基于启发式决策
    causal=True,
)

# 手动指定 split 数
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    num_splits=4,       # 4 个 split
    causal=True,
)

# 禁用 split（不推荐，除非确认单 block 已饱和）
out = flash_attn_with_kvcache(
    q, k_cache, v_cache,
    cache_seqlens=cache_seqlens,
    num_splits=1,
    causal=True,
)
```

### 6.3 FA2 怎样决定 split 数

`num_splits=0` 时，FA2 的 `num_splits_heuristic` 查看 batch、Q 头与行块构成的并行工作量，以及 K 的块数和可用 SM 数。工作量已经达到判定阈值时会选择 1；否则在有效候选中选达到最佳估计效率 85% 的最小 split 数，以减少额外读写。Hopper 的 `hopper/heuristics.h` 属于另一套实现，其启发式条件需要独立阅读。[FA2 分发与启发式](https://github.com/Dao-AILab/flash-attention/blob/v2.8.3.post2/csrc/flash_attn/flash_api.cpp)

## 7. FP8 推理

### 7.1 基本用法

```python
from flash_attn_3.flash_attn_interface import flash_attn_func

# FP8 推理（仅限 H100/H200）
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

# Per-head 缩放因子
q_descale = torch.ones(batch_size, num_heads_kv, device='cuda', dtype=torch.float32)
k_descale = torch.ones(batch_size, num_heads_kv, device='cuda', dtype=torch.float32)
v_descale = torch.ones(batch_size, num_heads_kv, device='cuda', dtype=torch.float32)

out = flash_attn_func(
    q_fp8, k_fp8, v_fp8,
    causal=True,
    q_descale=q_descale,
    k_descale=k_descale,
    v_descale=v_descale,
)
# out.dtype == torch.bfloat16（FP8 输出固定为 BF16）
```

### 7.2 FP8 KV Cache

FP8 KV Cache 可以将缓存大小减半，适合长序列推理：

```python
k_cache_fp8 = torch.zeros(batch_size, max_seqlen, num_heads_kv, head_dim,
                          device='cuda', dtype=torch.float8_e4m3fn)
v_cache_fp8 = torch.zeros(batch_size, max_seqlen, num_heads_kv, head_dim,
                          device='cuda', dtype=torch.float8_e4m3fn)
```

> 详见 [FP8 支持](/flashattention/06-advanced-features/04-fp8-support/)

---

## 8. 性能优化策略

### 8.1 Prefill vs Decode 优化

| 策略 | Prefill | Decode |
|------|---------|--------|
| Split-KV | 通常不需要 | 推荐 |
| PackGQA | 取决于序列长度 | 推荐 |
| FP8 | 可选（减少带宽） | 推荐（带宽瓶颈） |
| Paged KV | 可选 | 推荐（长序列） |

### 8.2 批量大小与 Split 平衡

Decode 阶段的关键是平衡 batch 大小和 split 数：

```
总 Thread Blocks = batch_size × num_heads_kv × num_splits

目标：总 Blocks ≈ GPU SM 数量 × 2~4

例如 H100 (132 SMs):
  batch=1, heads=8:  8 blocks → 需要 ~16 splits
  batch=32, heads=8: 256 blocks → num_splits=1 即可
  batch=4, heads=32: 128 blocks → num_splits=1~2
```

### 8.3 内存带宽优化

Decode 阶段是 memory-bound，优化重点在减少内存访问：

1. **GQA/MQA**：减少 KV Cache 的读取量
2. **FP8 KV Cache**：数据量减半
3. **Paged KV Cache**：避免无效内存分配
4. **融合 Rotary**：减少一次 Q/K 的读写
5. **PackGQA**：减少内核启动数

### 8.4 延迟优化清单

```python
# 1. 使用 flash_attn_with_kvcache（避免 Python 开销）
out = flash_attn_with_kvcache(q, k_cache, v_cache, ...)

# 2. 使用 CUDA Graph 消除 CPU 开销
with torch.cuda.graph(graph):
    out = flash_attn_with_kvcache(q, k_cache, v_cache, ...)

# 3. 预热 CUDA 内核（避免首次调用延迟）
for _ in range(3):
    _ = flash_attn_with_kvcache(q_dummy, k_cache, v_cache, ...)
torch.cuda.synchronize()
```

---

## 导航

- 上一篇：[训练集成](/flashattention/07-usage-tutorial/02-training-integration/)
- 下一篇：[性能测试](/flashattention/07-usage-tutorial/04-benchmarking/)
- [返回目录](/flashattention/00-overview/overview/)
