---
title: "LLM 推理基础"
weight: 1
---

一条生成请求先利用提示建立各层状态，再根据已知 token 逐步预测后续输出。下文以普通自回归注意力为例；跨 GPU 时，模型的缓存还要在相应设备上可见。[Dynamo 1.6.2 对 P/D 交接的说明](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md)

---

## 1. Transformer 架构回顾

现代大语言模型（LLM）几乎都基于 Transformer 架构构建。理解 Transformer 的工作原理是理解 LLM 推理优化的基础。

### 1.1 Self-Attention 机制原理

Transformer 的核心是 Self-Attention（自注意力）机制。对于输入序列 X，Self-Attention 通过三个线性变换生成 Query（Q）、Key（K）、Value（V）：

```
Q = X · W_Q
K = X · W_K
V = X · W_V
```

然后计算注意力分数：

```
Attention(Q, K, V) = softmax(Q · K^T / √d_k) · V
```

其中 d_k 是 Key 向量的维度，用于缩放防止点积过大。

```mermaid
graph LR
    subgraph attention["Self-Attention 计算流程"]
        X["输入 X"] --> WQ["W_Q"]
        X --> WK["W_K"]
        X --> WV["W_V"]
        WQ --> Q["Query"]
        WK --> K["Key"]
        WV --> V["Value"]
        Q --> MM1["Q · K^T"]
        K --> MM1
        MM1 --> Scale["÷ √d_k"]
        Scale --> SM["Softmax"]
        SM --> MM2["× V"]
        V --> MM2
        MM2 --> Out["输出"]
    end
```

### 1.2 Multi-Head Attention

实际应用中，Transformer 使用 Multi-Head Attention，将注意力计算分成多个"头"并行执行，每个头关注不同的特征子空间：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) · W_O

其中 head_i = Attention(Q · W_Q^i, K · W_K^i, V · W_V^i)
```

```mermaid
graph TB
    subgraph mha["Multi-Head Attention"]
        Input["输入"] --> H1["Head 1"]
        Input --> H2["Head 2"]
        Input --> H3["Head 3"]
        Input --> Hn["Head N"]
        H1 --> Concat["Concat"]
        H2 --> Concat
        H3 --> Concat
        Hn --> Concat
        Concat --> WO["W_O"]
        WO --> Output["输出"]
    end
```

### 1.3 Transformer Decoder Layer

一个标准的 Transformer Decoder Layer 包含：

1. **Masked Self-Attention**：处理当前位置与之前所有位置的关系
2. **Feed-Forward Network (FFN)**：两层全连接网络，通常有较大的中间维度
3. **Layer Normalization**：稳定训练
4. **Residual Connection**：残差连接

```mermaid
graph TB
    subgraph decoder["Transformer Decoder Layer"]
        Input["输入"] --> LN1["Layer Norm"]
        LN1 --> MSA["Masked Self-Attention"]
        MSA --> Add1["+"]
        Input --> Add1
        Add1 --> LN2["Layer Norm"]
        LN2 --> FFN["Feed-Forward Network"]
        FFN --> Add2["+"]
        Add1 --> Add2
        Add2 --> Output["输出"]
    end
```

### 1.4 参数规模与计算量

对于一个 L 层、隐藏维度 d、注意力头数 h 的 Transformer 模型：

- **参数量**：约 12 × L × d²（主要来自 Q/K/V 投影和 FFN）
- **FLOPs**：对于序列长度 n，单次前向传播约 2 × n × 参数量

| 模型 | 参数量 | 层数 | 隐藏维度 | 注意力头数 |
|------|--------|------|----------|------------|
| Llama-7B | 7B | 32 | 4096 | 32 |
| Llama-70B | 70B | 80 | 8192 | 64 |
| GPT-4 (估计) | ~1.7T | ~120 | ~12288 | ~96 |

---

## 2. 自回归生成过程

LLM 生成文本采用**自回归（Autoregressive）**方式：每次只生成一个 token，然后将其作为下一次生成的输入。这个过程分为两个明显不同的阶段：

### 2.1 Prefill 阶段（Prompt Processing）

- **输入**：用户的完整 prompt（可能数千个 token）
- **计算**：一次性并行处理所有输入 token
- **输出**：生成第一个输出 token + KV Cache
- **特点**：计算密集型（Compute-bound）

```mermaid
sequenceDiagram
    participant User as 用户
    participant Model as 模型

    Note over User,Model: Prefill 阶段
    User->>Model: "What is the capital of France?"
    Note right of Model: 并行处理所有 token
    Note right of Model: 计算 Q, K, V
    Note right of Model: 存储 K, V 到 Cache
    Model->>User: "The" (第一个 token)
```

### 2.2 Decode 阶段（Token Generation）

- **输入**：单个 token（上一步生成的）
- **计算**：利用 KV Cache，只计算新 token 的 attention
- **输出**：下一个 token
- **特点**：内存带宽受限（Memory-bound）

```mermaid
sequenceDiagram
    participant User as 用户
    participant Model as 模型
    participant KVCache as KV Cache

    Note over User,Model: Decode 阶段（循环）

    loop 直到生成结束符
        Model->>KVCache: 读取历史 K, V
        Note right of Model: 为当前位置计算 Q、K、V
        Note right of Model: Attention 使用缓存的 K, V
        Model->>KVCache: 追加新的 K, V
        Model->>User: 输出 token
    end
```

---

## 3. 两阶段的计算特性差异

### 3.1 特性对比

| 特性 | Prefill | Decode |
|------|---------|--------|
| Token 数量 | 多（整个 prompt） | 1（每次） |
| 并行度 | 高 | 低 |
| 常见瓶颈 | 长输入下可能受算力或 attention 实现限制 | 低并发可能受权重与 KV 读取带宽限制 |
| 观察重点 | 提示长度、分块与每轮 token 数 | 活动请求数、读取的 KV 和每轮时间 |
| 批处理影响 | 改变本轮 token 和数据复用 | 多条请求可共同摊薄权重读取，收益受带宽及并行限制 |

### 3.2 计算特性图解

```mermaid
graph TB
    subgraph prefill["Prefill 特性"]
        P1["计算密集"] --> P2["高 GPU 利用率"]
        P1 --> P3["并行度高"]
        P3 --> P4["批处理收益线性"]
    end

    subgraph decode["Decode 特性"]
        D1["内存带宽受限"] --> D2["低 GPU 利用率"]
        D1 --> D3["需要批处理"]
        D3 --> D4["批处理收益显著"]
    end
```

### 3.3 KV Cache 的作用

KV Cache 是 LLM 推理优化的关键：

- **避免重复计算**：历史 token 的 K、V 只计算一次，后续直接复用
- **内存换时间**：用额外的内存存储换取计算时间
- **线性增长**：Cache 大小随序列长度线性增长

```mermaid
flowchart LR
    subgraph kvcache["KV Cache 工作原理"]
        T1["Token 1"] --> KV1["K1, V1"]
        T2["Token 2"] --> KV2["K2, V2"]
        T3["Token 3"] --> KV3["K3, V3"]

        KV1 --> Cache["KV Cache"]
        KV2 --> Cache
        KV3 --> Cache

        NewToken["新 Token"] --> Q["Query"]
        Q --> Attn["Attention"]
        Cache --> Attn
        Attn --> Output["输出"]
    end
```

---

## 4. 关键指标定义

理解 LLM 推理性能需要关注以下指标：

| 指标 | 全称 | 说明 |
|------|------|------|
| **TTFT** | Time To First Token | 从请求到第一个 token 的延迟 |
| **ITL** | Inter-Token Latency | 相邻 token 之间的延迟 |
| **Throughput** | 吞吐量 | tokens/s 或 requests/s |
| **Latency** | 端到端延迟 | 完整请求的响应时间 |

{{< trace >}}
title: 从客户端时钟读 TTFT 与 ITL
actors: ["客户端", "Frontend", "Worker"]
steps:
  - from: 0
    to: 1
    label: "0 ms · 发出"
    detail: "请求开始计时。"
  - from: 1
    to: 2
    label: "15 ms · 分发"
    detail: "Frontend 已选中模型 worker，服务端继续预填充。"
  - from: 2
    to: 0
    label: "90 ms · 首个 token"
    detail: "结果经前端发回客户端。TTFT 从 0 ms 算起，为 90 ms。"
  - from: 2
    to: 0
    label: "110 ms · 第二个"
    detail: "这一段 ITL 为 20 ms。"
  - from: 2
    to: 0
    label: "135 ms · 第三个"
    detail: "第二段 ITL 为 25 ms；三次输出只有两个相邻间隔。"
{{< /trace >}}

---

## 下一篇

继续阅读 [02-行业痛点与解决方案](02-industry-challenges.md)，了解传统推理框架的问题以及业界的解决思路。
