---
title: "LLM 推理流程"
weight: 3
---

一条提示先做 prefill，得到第一个输出 token；后续 decode 每步读取已有 KV 并追加一个位置。

---


## 1. LLM 推理概述

### 1.1 推理流程总览

```mermaid
sequenceDiagram
    participant User as 用户
    participant Tokenizer as 分词器
    participant Model as LLM 模型
    participant Detokenizer as 解码器

    User->>Tokenizer: "What is AI?"
    Tokenizer->>Model: [1, 2, 3, 4, 5]

    rect rgb(200, 230, 200)
        Note over Model: Prefill 阶段
        Model->>Model: 并行处理所有 prompt tokens
        Model->>Model: 生成 KV Cache
    end

    rect rgb(230, 200, 200)
        Note over Model: Decode 阶段
        loop 自回归生成
            Model->>Model: 生成下一个 token
            Model->>Model: 更新 KV Cache
        end
    end

    Model->>Detokenizer: [6, 7, 8, ...]
    Detokenizer->>User: "Artificial Intelligence is..."
```

### 1.2 关键指标

| 指标 | 定义 | 重要性 |
|------|------|--------|
| **TTFT** | Time-To-First-Token，首 token 延迟 | 用户体验关键 |
| **TPS** | Tokens Per Second，生成速度 | 吞吐量指标 |
| **Latency** | 端到端延迟 | 总体响应时间 |
| **Throughput** | 单位时间处理的请求数 | 系统容量 |

---

## 2. Tokenization

### 2.1 什么是 Tokenization？

将文本转换为模型可以理解的数字序列。

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

text = "What is artificial intelligence?"
tokens = tokenizer.encode(text)
# [1, 1724, 338, 23116, 21082, 29973]

# 解码回文本
decoded = tokenizer.decode(tokens)
# "What is artificial intelligence?"
```

### 2.2 常见 Tokenization 方法

```mermaid
graph TB
    subgraph "Tokenization 方法演进"
        A["Word-level<br>词级别"] --> B["Character-level<br>字符级别"]
        B --> C["Subword<br>子词级别"]
        C --> D["BPE<br>(Byte Pair Encoding)"]
        C --> E["WordPiece"]
        C --> F["SentencePiece"]
    end
```

### 2.3 BPE 算法示例

```
原始文本: "low lower lowest"

迭代过程:
1. 初始词表: ['l', 'o', 'w', 'e', 'r', 's', 't', ' ']
2. 合并 'l' + 'o' -> 'lo'
3. 合并 'lo' + 'w' -> 'low'
4. 合并 'e' + 'r' -> 'er'
5. 合并 'e' + 's' -> 'es'
6. ...

最终: ['low', 'er', 'est']
```

---

## 3. Prefill 阶段

### 3.1 什么是 Prefill？

Prefill（预填充）是处理输入 prompt 的阶段，**并行**计算所有 prompt tokens 的 KV Cache。

```mermaid
graph TB
    subgraph "Prefill 阶段"
        Input["Prompt Tokens<br>[t1, t2, t3, ..., tn]"] --> Embed["Embedding Layer"]
        Embed --> Layer1["Transformer Layer 1"]
        Layer1 --> Layer2["Transformer Layer 2"]
        Layer2 --> Dots["..."]
        Dots --> LayerN["Transformer Layer N"]
        LayerN --> KV["KV Cache<br>K: [n, d], V: [n, d]"]
        LayerN --> Hidden["Final Hidden State"]
    end
```

### 3.2 Prefill 的特点

| 特点 | 说明 |
|------|------|
| **并行计算** | 所有 prompt tokens 同时处理 |
| **Compute-Bound** | 计算密集，GPU 利用率高 |
| **一次性** | 每个请求只执行一次 |
| **生成 KV Cache** | 为 Decode 阶段准备缓存 |

### 3.3 Prefill 的计算量

```python
def estimate_prefill_flops(prompt_len, hidden_size, num_layers, num_heads):
    """估算 Prefill 阶段的计算量"""

    # 注意力计算: O(n^2 * d)
    attn_flops = 2 * prompt_len * prompt_len * hidden_size * num_layers

    # FFN 计算: O(n * d * 4d)
    ffn_flops = 2 * prompt_len * hidden_size * 4 * hidden_size * num_layers

    total_flops = attn_flops + ffn_flops
    return total_flops

# 示例：Llama-7B, prompt_len=1000
flops = estimate_prefill_flops(1000, 4096, 32, 32)
print(f"Prefill FLOPs: {flops / 1e12:.2f} TFLOPs")
```

### 3.4 CacheBlend 对 Prefill 的优化

CacheBlend 的核心贡献是**减少 Prefill 阶段的计算量**：

```mermaid
graph LR
    subgraph "传统 Prefill"
        T1["计算所有 tokens 的 KV"]
        T1 --> T2["100% 计算量"]
    end

    subgraph "CacheBlend Prefill"
        C1["加载预计算的 KV Cache"]
        C1 --> C2["选择 HKVD Tokens"]
        C2 --> C3["仅重计算 ~15% tokens"]
        C3 --> C4["融合 KV Cache"]
    end
```

---

## 4. Decode 阶段

### 4.1 什么是 Decode？

Decode（解码）是自回归生成 token 的阶段，每次只生成**一个** token。

{{< trace >}}
title: "一轮 decode 读取什么、留下什么"
actors: ["当前 token", "模型层", "KV 缓存", "输出"]
steps:
  - from: 0
    to: 1
    label: "输入 t3"
    detail: "模型为当前位置 t3 计算 Q、K、V。"
  - from: 2
    to: 1
    label: "读取历史"
    detail: "该层把 t3 的 Q 与先前位置的 K 以及本位置的 K 一起做注意力。"
  - from: 1
    to: 2
    label: "追加 t3"
    detail: "本层留下 t3 的 K、V，供下一步使用。"
  - from: 1
    to: 3
    label: "采样下一个词"
    detail: "所有层完成后输出分布，选出的 t4 成为下一步输入。"
{{< /trace >}}

### 4.2 Decode 的特点

| 特点 | 说明 |
|------|------|
| **顺序生成** | 每次只生成一个 token |
| **Memory-Bound** | 内存带宽受限 |
| **使用 KV Cache** | 避免重复计算 |
| **延迟敏感** | 影响用户体验 |

### 4.3 Decode 的计算量

```python
def decode_step_flops(seq_len, hidden_size, num_layers):
    """单步 Decode 的计算量"""

    # 注意力: 只计算最后一个 token
    # Q 与 所有 K 的点积: O(seq_len * d)
    attn_flops = 2 * seq_len * hidden_size * num_layers

    # FFN: 只处理最后一个 token
    ffn_flops = 2 * hidden_size * 4 * hidden_size * num_layers

    return attn_flops + ffn_flops

# 对比
prefill_1000 = estimate_prefill_flops(1000, 4096, 32, 32)
decode_step = decode_step_flops(1000, 4096, 32)

print(f"Prefill (1000 tokens): {prefill_1000 / 1e9:.2f} GFLOPs")
print(f"Single Decode step:    {decode_step / 1e9:.2f} GFLOPs")
print(f"Ratio: {prefill_1000 / decode_step:.0f}x")
```

### 4.4 Decode 的瓶颈

```mermaid
graph TB
    subgraph "Decode 瓶颈分析"
        A["读取 KV Cache"] --> B["内存带宽瓶颈"]
        C["读取模型权重"] --> B
        D["计算 Attention"] --> E["计算很快完成"]

        B --> F["GPU 利用率低<br>(10-30%)"]
    end
```

---

## 5. Memory-Bound vs Compute-Bound

### 5.1 两种瓶颈

```mermaid
graph TB
    subgraph "Compute-Bound (Prefill)"
        C1["大矩阵乘法"]
        C2["GPU 算力充分利用"]
        C3["内存带宽有余"]
    end

    subgraph "Memory-Bound (Decode)"
        M1["小矩阵运算"]
        M2["频繁读取 KV Cache"]
        M3["内存带宽成为瓶颈"]
    end
```

### 5.2 Roofline 模型

```
          ↑ Performance (FLOPs/s)
          │
    Peak  │         _______________
  Compute │        /
          │       /  Compute-Bound
          │      /   (Prefill)
          │     /
          │    /  Memory-Bound
          │   /   (Decode)
          │  /
          │ /
          └─────────────────────────→
                   Arithmetic Intensity (FLOPs/Byte)
```

### 5.3 优化策略

| 阶段 | 瓶颈 | 优化策略 |
|------|------|---------|
| Prefill | Compute | 算子融合、Flash Attention |
| Decode | Memory | KV Cache 压缩、批处理 |
| 两者 | - | 量化、Speculative Decoding |

---

## 6. Continuous Batching

### 6.1 传统 Batching 的问题

```mermaid
graph TB
    subgraph "传统 Static Batching"
        R1["Request 1: 100 tokens"] --> Wait["等待最长请求"]
        R2["Request 2: 50 tokens"] --> Wait
        R3["Request 3: 200 tokens"] --> Wait
        Wait --> Done["全部完成才返回"]
    end
```

问题：短请求需要等待长请求完成。

### 6.2 Continuous Batching

{{< stateboard >}}
title: "每一步结束后，空槽可以接新请求"
rows: ["槽 0", "槽 1"]
steps:
  - title: "第一次解码"
    detail: "A 和 B 同时占用两个槽，B 在本步结束。"
    cells:
      - ["A"]
      - ["B"]
    focus: [[0, 0], [1, 0]]
  - title: "换入请求 C"
    detail: "调度器保留 A，把释放的槽分给正在等待的 C。"
    cells:
      - ["A"]
      - ["C"]
    focus: [[1, 0]]
  - title: "下一轮"
    detail: "A 和 C 各继续一步，B 的结束时间不会推迟 C 入批。"
    cells:
      - ["A→"]
      - ["C→"]
    focus: [[0, 0], [1, 0]]
{{< /stateboard >}}

优势：
- 请求完成后立即返回
- 新请求立即加入
- GPU 利用率更高

---

## 7. vLLM 与 PagedAttention

### 7.1 vLLM 简介

vLLM 是目前最流行的 LLM 推理引擎，核心创新是 **PagedAttention**。

### 7.2 传统 KV Cache 的问题

```mermaid
graph TB
    subgraph "传统 KV Cache 分配"
        R1["Request 1<br>需要 1000 tokens"]
        R2["Request 2<br>需要 500 tokens"]
        R3["Request 3<br>需要 800 tokens"]

        R1 --> M1["预分配 1000 slots"]
        R2 --> M2["预分配 500 slots"]
        R3 --> M3["预分配 800 slots"]

        M1 --> Waste["内存碎片<br>浪费严重"]
    end
```

问题：
- 不知道生成多少 tokens，必须预分配最大长度
- 内存碎片化严重
- 难以支持长序列

### 7.3 PagedAttention 解决方案

```mermaid
graph TB
    subgraph "PagedAttention"
        subgraph "物理内存块"
            B1["Block 1"]
            B2["Block 2"]
            B3["Block 3"]
            B4["Block 4"]
        end

        subgraph "逻辑视图"
            R1["Request 1"] --> B1
            R1 --> B3
            R2["Request 2"] --> B2
            R2 --> B4
        end
    end
```

优势：
- **按需分配**: 用多少分配多少
- **无碎片**: 块大小固定
- **高效共享**: 相同 prefix 可共享

### 7.4 CacheBlend 在 vLLM 中的实现

CacheBlend 基于 vLLM 实现，主要修改：

```mermaid
graph TB
    subgraph "CacheBlend 在 vLLM 中的位置"
        vLLM["vLLM Engine"]
        Scheduler["Scheduler"]
        Executor["Model Executor"]
        Model["LlamaModel"]
        Attn["XFormers Attention"]

        vLLM --> Scheduler
        Scheduler --> Executor
        Executor --> Model
        Model --> Attn

        Model --> |"修改"| CacheBlend1["cache_fuse_metadata"]
        Model --> |"修改"| CacheBlend2["old_kvs"]
        Attn --> |"修改"| CacheBlend3["HKVD 选择"]
        Attn --> |"修改"| CacheBlend4["KV 融合"]
    end
```

---

## 总结

1. **Tokenization**: 文本到数字的转换
2. **Prefill**: 并行处理 prompt，生成 KV Cache
3. **Decode**: 自回归生成，使用 KV Cache
4. **瓶颈分析**: Prefill 计算密集，Decode 内存密集
5. **Continuous Batching**: 提高 GPU 利用率
6. **PagedAttention**: 高效 KV Cache 管理

CacheBlend 的核心价值是**优化 Prefill 阶段**，通过选择性重计算减少延迟，同时保持生成质量。

---

## 下一步

- [KV Cache 基础原理](./04-kv-cache-fundamentals.md)
