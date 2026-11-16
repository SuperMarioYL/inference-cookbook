---
title: "KV Cache 核心概念"
weight: 2
---


> **前置要求**: [Transformer 与注意力机制基础](./01-transformer-attention.md)

---

## 概述

每追加一个 token，全注意力层就要为它保留一份 K 和 V。把每层的 KV 头数、头维度和元素字节数相乘，再加总各层，就能算出序列增长一格的代价。

---

## 1. 为什么需要 KV Cache

### 1.1 自回归生成的重复计算问题

在没有 KV Cache 的情况下，生成每个新 token 都需要重新计算所有历史 token 的 Key 和 Value：

```mermaid
sequenceDiagram
    participant Gen as 生成过程

    Note over Gen: 生成 Token 1
    Gen->>Gen: 计算 K1, V1
    Note over Gen: 生成 Token 2
    Gen->>Gen: 重新计算 K1, V1 (重复!)
    Gen->>Gen: 计算 K2, V2
    Note over Gen: 生成 Token 3
    Gen->>Gen: 重新计算 K1, V1 (重复!)
    Gen->>Gen: 重新计算 K2, V2 (重复!)
    Gen->>Gen: 计算 K3, V3

    Note over Gen: 生成 Token n
    Gen->>Gen: 重复计算 K1...K(n-1), V1...V(n-1)
    Gen->>Gen: 计算 Kn, Vn
```
**问题**: 生成第 n 个 token 需要 $O(n)$ 次 K、V 计算，生成 n 个 token 总计 $O(n^2)$ 次计算。
### 1.2 KV Cache 的解决方案
KV Cache 缓存已计算的 Key 和 Value，避免重复计算：
```mermaid
sequenceDiagram
    participant Gen as 生成过程
    participant Cache as KV Cache

    Note over Gen,Cache: 生成 Token 1
    Gen->>Gen: 计算 K1, V1
    Gen->>Cache: 存储 K1, V1

    Note over Gen,Cache: 生成 Token 2
    Cache-->>Gen: 读取 K1, V1
    Gen->>Gen: 只计算 K2, V2
    Gen->>Cache: 追加 K2, V2
    Note over Gen,Cache: 生成 Token 3
    Cache-->>Gen: 读取 K1, V1, K2, V2
    Gen->>Gen: 只计算 K3, V3
    Gen->>Cache: 追加 K3, V3
```
**优化效果**: 生成第 n 个 token 只需 $O(1)$ 次 K、V 计算，总计 $O(n)$ 次。
---
## 2. KV Cache 的数据结构

### 2.1 张量形状

对于一个 Transformer 模型，KV Cache 的形状为：

```
KV Cache 形状:
┌─────────────────────────────────────────────────────────────┐
│ K: [batch_size, num_layers, num_heads, seq_len, head_dim]  │
│ V: [batch_size, num_layers, num_heads, seq_len, head_dim]  │
└─────────────────────────────────────────────────────────────┘

示例 (Llama-7B):
- batch_size = 1
- num_layers = 32
- num_heads = 32
- seq_len = 2048 (动态增长)
- head_dim = 128

K 和 V 各: 1 x 32 x 32 x 2048 x 128 = 268,435,456 个元素
```

### 2.2 内存布局

```mermaid
graph TB
    subgraph layer["单层的 KV Cache"]
        subgraph heads["32 个注意力头"]
            H1["Head 1<br/>K: (seq, 128)<br/>V: (seq, 128)"]
            H2["Head 2<br/>K: (seq, 128)<br/>V: (seq, 128)"]
            Hn["Head 32<br/>K: (seq, 128)<br/>V: (seq, 128)"]
        end
    end

    subgraph model["完整模型"]
        L1["Layer 1"]
        L2["Layer 2"]
        Ln["Layer 32"]
    end

    layer --> model
```

---
## 3. KV Cache 内存占用计算
### 3.1 计算公式
单个请求的 KV Cache 内存占用：
$$
\text{KV Cache Size} = 2 \times L \times H \times S \times D \times \text{dtype\_size}
$$
其中：
- $2$ = Key 和 Value 两部分
- $L$ = 层数 (num_layers)
- $H$ = 注意力头数 (num_heads)
- $S$ = 序列长度 (seq_len)
- $D$ = 每个头的维度 (head_dim)
- dtype_size = 数据类型大小（FP16 = 2 bytes，FP32 = 4 bytes）
### 3.2 典型模型的 KV Cache 大小
| 模型 | 参数量 | 层数 | 头数 | head_dim | 每 token KV (FP16) | 2K 序列 |
|------|--------|------|------|----------|-------------------|---------|
| Llama-7B | 7B | 32 | 32 | 128 | 512 KB | 1 GB |
| Llama-13B | 13B | 40 | 40 | 128 | 800 KB | 1.6 GB |
| Llama-70B | 70B | 80 | 64 | 128 | 2.56 MB | 5.12 GB |
| GPT-4 (估计) | 1.8T | 120 | 128 | 128 | 15.36 MB | 30.72 GB |

### 3.3 线性增长的挑战

{{< stateboard >}}
title: "长度翻倍时，KV 占用如何变化"
rows: ["每层字节", "层数", "总计"]
steps:
  - title: "一个 token"
    detail: "示例用 2 个 KV 头、每头 4 维、FP16。K 和 V 合计每层 32 字节。"
    cells:
      - ["32 B"]
      - ["2"]
      - ["64 B"]
    focus: [[0, 0], [2, 0]]
  - title: "两个 token"
    detail: "模型有两层，保存两个位置需 128 字节。"
    cells:
      - ["64 B"]
      - ["2"]
      - ["128 B"]
    focus: [[0, 0], [2, 0]]
  - title: "四个 token"
    detail: "保持头数、维度和精度不变，总字节数继续按位置数线性增长。"
    cells:
      - ["128 B"]
      - ["2"]
      - ["256 B"]
    focus: [[0, 0], [2, 0]]
{{< /stateboard >}}

这里计算 K/V 张量的有效数据量。引擎实际分配还要考虑块对齐、保留容量和每个 rank 的布局。
**关键问题**:
- 单请求长序列会占用大量显存
- 并发请求会快速耗尽显存
- 显存成为 LLM 服务的主要瓶颈

---

## 4. KV Cache 的访问模式

### 4.1 时间特性

KV Cache 的访问分为两个阶段：

```mermaid
graph TB
    subgraph prefill["Prefill 阶段 - 初始化"]
        P1["输入: 完整的 prompt"]
        P2["计算: 所有 token 的 K, V"]
        P3["写入: 一次性写入 Cache"]
        P4["特点: 计算密集型"]
        P1 --> P2 --> P3 --> P4
    end

    subgraph decode["Decode 阶段 - 生成"]
        D1["输入: 新生成的 1 个 token"]
        D2["读取: 所有历史 K, V"]
        D3["计算: 1 个新的 K, V"]
        D4["追加: 新 K, V 到 Cache"]
        D5["特点: 内存带宽密集型"]
        D1 --> D2 --> D3 --> D4 --> D5
    end
    prefill --> decode
```

### 4.2 空间特性

```mermaid
graph TB
    subgraph spatial["KV Cache 空间访问模式"]
        direction TB
        subgraph read["读取模式"]
            R1["全量读取<br/>每次 Decode 读取所有历史"]
            R2["顺序访问<br/>按层、按头顺序"]
        end

        subgraph write["写入模式"]
            W1["追加写入<br/>每次只追加新 token"]
            W2["位置确定<br/>写入位置已知"]
        end
    end
```
### 4.3 瓶颈分析
| 阶段 | 主要瓶颈 | 原因 |
|------|----------|------|
| Prefill | 计算 | 需要计算所有 token 的注意力 |
| Decode | 内存带宽 | 每个 token 都要读取全部历史 KV |

**Decode 阶段的带宽问题**:
- 生成 1 个 token 需要读取所有历史 KV
- 序列越长，读取量越大
- GPU 计算能力被浪费在等待数据传输
---
## 5. KV Cache 优化动机

### 5.1 问题汇总

```mermaid
mindmap
  root["KV Cache 挑战"]
    memory["内存压力"]
      m1["显存有限"]
      m2["线性增长"]
      m3["并发受限"]
    bandwidth["带宽瓶颈"]
      b1["每次全量读取"]
      b2["Decode 阶段慢"]
    redundancy["重复计算"]
      r1["相同前缀"]
      r2["多轮对话"]
```

### 5.2 优化方向

| 优化方向 | 技术方案 | UCM 对应模块 |
|----------|----------|-------------|
| 减少存储 | KV 压缩、量化 | `ucm/sparse/kvcomp/` |
| 减少读取 | 稀疏注意力 | `ucm/sparse/esa/`, `gsa/` |
| 复用 KV | 前缀缓存 | `ucm/store/` |
| 扩展容量 | 外部存储 | `ucm/store/posix/`, `nfs/` |
| 跨请求共享 | 内容寻址 | `ucm/integration/vllm/ucm_connector.py` |

### 5.3 UCM 的核心思路

```mermaid
graph TB
    subgraph problem["问题"]
        P1["KV Cache 占用大量显存"]
        P2["相同前缀重复计算"]
        P3["长序列带宽瓶颈"]
    end

    subgraph solution["UCM 解决方案"]
        S1["多级存储<br/>HBM → CPU → 外部"]
        S2["内容寻址<br/>哈希复用相同 KV"]
        S3["稀疏注意力<br/>只读取重要 KV"]
    end

    P1 --> S1
    P2 --> S2
    P3 --> S3
```

---
## 6. KV Cache 复用场景
### 6.1 前缀缓存（Prefix Caching）
当多个请求共享相同前缀时，可以复用已计算的 KV：
```
请求 1: "请帮我翻译以下内容：Hello World"
请求 2: "请帮我翻译以下内容：Good Morning"
请求 3: "请帮我翻译以下内容：How are you"

共同前缀: "请帮我翻译以下内容："
         ↓
    只计算一次 KV，三个请求共享
```
### 6.2 多轮对话
{{< stateboard >}}
title: "第二轮对话可以接着哪部分算"
rows: ["轮次一", "轮次二", "KV"]
steps:
  - title: "第一轮"
    detail: "提示 P1 和回答 A1 构成已经处理的历史。"
    cells:
      - ["P1", "A1", "·"]
      - ["·", "·", "·"]
      - ["P1", "A1", "·"]
    focus: [[2, 0], [2, 1]]
  - title: "追加问题"
    detail: "聊天模板与历史 token 完全一致时，前两格形成可复用前缀。"
    cells:
      - ["P1", "A1", "·"]
      - ["P1", "A1", "P2"]
      - ["P1", "A1", "·"]
    focus: [[1, 0], [1, 1]]
  - title: "处理 P2"
    detail: "引擎读取仍然可用的完整块，计算新问题以及块边界处的余量。"
    cells:
      - ["P1", "A1", "·"]
      - ["P1", "A1", "P2"]
      - ["P1", "A1", "P2"]
    focus: [[1, 2], [2, 2]]
{{< /stateboard >}}

### 6.3 RAG 检索增强

```
System Prompt (10K tokens): 固定不变，可以缓存
Retrieved Context (5K tokens): 部分可能重复
User Query (100 tokens): 每次不同
优化: 缓存 System Prompt 的 KV，只计算变化部分
```

---

## 7. 关键概念总结

| 概念 | 说明 | 重要性 |
|------|------|--------|
| KV Cache | 缓存历史 token 的 Key 和 Value | 避免重复计算 |
| 线性增长 | 内存随序列长度线性增长 | 显存瓶颈 |
| Prefill vs Decode | 两个阶段的不同特性 | 针对性优化 |
| 前缀复用 | 相同前缀共享 KV | 减少计算 |
| 内容寻址 | 基于内容哈希查找 KV | UCM 核心机制 |

---

## 延伸阅读

- [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180)
- [vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)
