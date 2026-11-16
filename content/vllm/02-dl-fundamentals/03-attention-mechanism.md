---
title: "注意力机制原理"
weight: 3
---

处理一句话的最后一个 token 时，模型用它的查询向量 Q 与可见位置的 K 求相似度，再用得到的权重对 V 求和。下面把这个过程写成张量运算，并比较 MHA、GQA 和 MQA 中 KV 头的数量。

## 1. 注意力的直觉理解

### 1.1 人类注意力的类比

想象你在阅读一篇文章，当你看到"他"这个代词时，你会自动"关注"前文中提到的人名，以理解"他"指的是谁。

这就是注意力机制的核心思想：**让模型学会"关注"序列中最相关的部分**。

```mermaid
graph LR
    subgraph reading["阅读理解"]
        T1["张三"] --> T2["今天"] --> T3["去了"] --> T4["公园"]
        T4 --> T5["他"]
        T5 -.->|关注| T1
    end
```

### 1.2 从"全局视野"到"重点关注"

没有注意力机制时，模型只能看到固定窗口内的信息。有了注意力机制：

```mermaid
graph TB
    subgraph fixed_window["固定窗口"]
        FW["只能看到附近几个 token"]
    end

    subgraph attention_mechanism["注意力机制"]
        ATT["可以关注序列中任意位置<br/>并根据相关性分配权重"]
    end

    style ATT fill:#c8e6c9
```

---

## 2. 自注意力（Self-Attention）计算

### 2.1 Query、Key、Value 的含义

自注意力使用三个向量：

| 向量 | 类比 | 作用 |
|------|------|------|
| **Query (Q)** | "我要找什么" | 当前位置的查询向量 |
| **Key (K)** | "我是什么" | 每个位置的索引向量 |
| **Value (V)** | "我的内容" | 每个位置的值向量 |

**直觉理解**：
- Q 是"问题"
- K 是"索引/标签"
- V 是"内容"
- 计算 Q 和所有 K 的相似度，用相似度加权所有 V

### 2.2 计算公式

自注意力的核心公式：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

其中：
- $Q$：Query 矩阵，形状 $[seq\_len, d_k]$
- $K$：Key 矩阵，形状 $[seq\_len, d_k]$
- $V$：Value 矩阵，形状 $[seq\_len, d_v]$
- $d_k$：Key 的维度（用于缩放）

### 2.3 计算步骤详解

```mermaid
flowchart TD
    subgraph step1["步骤1 - 生成 Q, K, V"]
        X["输入 X<br/>seq_len × hidden_dim"]
        X --> WQ["W_Q 投影"]
        X --> WK["W_K 投影"]
        X --> WV["W_V 投影"]
        WQ --> Q["Query<br/>seq_len × d_k"]
        WK --> K["Key<br/>seq_len × d_k"]
        WV --> V["Value<br/>seq_len × d_v"]
    end

    subgraph step2["步骤2 - 计算注意力分数"]
        Q --> MM["Q × K^T"]
        K --> MM
        MM --> SC["÷ √d_k<br/>缩放"]
        SC --> MASK["+ Mask<br/>可选"]
        MASK --> SM[Softmax]
        SM --> ATT["注意力权重<br/>seq_len × seq_len"]
    end

    subgraph step3["步骤3 - 加权求和"]
        ATT --> OUT["× V"]
        V --> OUT
        OUT --> O["输出<br/>seq_len × d_v"]
    end

    style SC fill:#fff9c4
    style SM fill:#c8e6c9
```

### 2.4 逐步计算示例

假设我们有一个简单的序列，3 个 token，每个 token 的隐藏维度是 4：

```python
import torch
import torch.nn.functional as F

seq_len = 3
d_k = 4

Q = torch.tensor([
    [1.0, 0.0, 1.0, 0.0],  # token 0 的 query
    [0.0, 1.0, 0.0, 1.0],  # token 1 的 query
    [1.0, 1.0, 0.0, 0.0],  # token 2 的 query
])

K = torch.tensor([
    [1.0, 0.0, 0.0, 1.0],  # token 0 的 key
    [0.0, 1.0, 1.0, 0.0],  # token 1 的 key
    [1.0, 1.0, 1.0, 1.0],  # token 2 的 key
])

V = torch.tensor([
    [1.0, 2.0, 3.0, 4.0],  # token 0 的 value
    [5.0, 6.0, 7.0, 8.0],  # token 1 的 value
    [9.0, 10., 11., 12.],  # token 2 的 value
])

scores = Q @ K.T
print("注意力分数 (未缩放):")
print(scores)
#         [1., 1., 2.],

d_k = 4
scaled_scores = scores / (d_k ** 0.5)
print("\n缩放后的分数:")
print(scaled_scores)

attention_weights = F.softmax(scaled_scores, dim=-1)
print("\n注意力权重:")
print(attention_weights)

output = attention_weights @ V
print("\n输出:")
print(output)
```

### 2.5 注意力权重可视化

注意力权重形成一个 `[seq_len, seq_len]` 的矩阵：

```
         Token 0  Token 1  Token 2
Token 0 [  0.30    0.30     0.40  ]  # Token 0 关注谁
Token 1 [  0.30    0.30     0.40  ]  # Token 1 关注谁
Token 2 [  0.20    0.20     0.60  ]  # Token 2 关注谁
```

每一行表示一个 token 对所有 token 的注意力分布（和为 1）。

---

## 3. 缩放因子 √d 的作用

### 3.1 为什么需要缩放

当 $d_k$ 较大时，$QK^T$ 的点积结果会变得很大。这会导致：

1. **Softmax 饱和**：大值经过 softmax 后趋近于 1，小值趋近于 0
2. **梯度消失**：softmax 在饱和区域的梯度接近 0

```mermaid
graph LR
    subgraph no_scale["无缩放"]
        S1["大的点积值"] --> SM1["Softmax 饱和"]
        SM1 --> G1["梯度消失"]
    end

    subgraph with_scale["有缩放"]
        S2["缩放后的点积"] --> SM2["Softmax 正常"]
        SM2 --> G2["梯度正常"]
    end

    style G1 fill:#ffcdd2
    style G2 fill:#c8e6c9
```

### 3.2 数学解释

假设 Q 和 K 的元素服从均值 0、方差 1 的分布，那么：

- $Q \cdot K$ 的均值为 0
- $Q \cdot K$ 的方差为 $d_k$

除以 $\sqrt{d_k}$ 后，方差变为 1，分布更稳定。

---

## 4. 多头注意力（Multi-Head Attention）

### 4.1 为什么需要多头

单头注意力只能学习一种"关注模式"。多头注意力让模型同时学习多种不同的关系：

```mermaid
graph TB
    subgraph multi_head_advantage["多头注意力的优势"]
        H1["Head 1<br/>关注语法关系"]
        H2["Head 2<br/>关注语义关系"]
        H3["Head 3<br/>关注位置关系"]
        H4["Head 4<br/>关注其他模式"]
    end
```

### 4.2 多头计算过程

```mermaid
graph TD
    X["输入 X<br/>batch × seq × hidden"] --> SPLIT["分割成多个头"]

    subgraph parallel_compute["并行计算"]
        SPLIT --> H1["Head 1<br/>Attention"]
        SPLIT --> H2["Head 2<br/>Attention"]
        SPLIT --> H3["Head 3<br/>Attention"]
        SPLIT --> HN["Head N<br/>Attention"]
    end

    H1 --> CAT[Concat]
    H2 --> CAT
    H3 --> CAT
    HN --> CAT

    CAT --> WO["W_O 投影"]
    WO --> O["输出"]
```

### 4.3 代码实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        # Q, K, V 投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 输出投影
        self.o_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # 投影
        Q = self.q_proj(x)  # [batch, seq, hidden]
        K = self.k_proj(x)
        V = self.v_proj(x)

        # 重塑为多头: [batch, seq, num_heads, head_dim]
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim)

        # 转置: [batch, num_heads, seq, head_dim]
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # 注意力计算
        scores = Q @ K.transpose(-2, -1) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = attn_weights @ V  # [batch, num_heads, seq, head_dim]

        # 合并多头
        output = output.transpose(1, 2)  # [batch, seq, num_heads, head_dim]
        output = output.reshape(batch_size, seq_len, -1)  # [batch, seq, hidden]

        # 输出投影
        output = self.o_proj(output)

        return output
```

### 4.4 头数与维度的关系

```
hidden_dim = num_heads × head_dim
```

**常见配置**：

| 模型 | hidden_dim | num_heads | head_dim |
|------|-----------|-----------|----------|
| GPT-2 Small | 768 | 12 | 64 |
| GPT-2 Large | 1280 | 20 | 64 |
| LLaMA-7B | 4096 | 32 | 128 |
| LLaMA-70B | 8192 | 64 | 128 |

---

## 5. Masked Attention（因果掩码）

### 5.1 为什么需要掩码

预测下一个 token 时，位置 i 的查询可读取位置 0 到 i 的键值，包括 i 自身。右侧尚未出现的位置被屏蔽。

{{< mechanism "vllm-causal-mask" >}}

### 5.2 掩码矩阵

因果掩码是一个下三角矩阵：

```python
seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len))
print(mask)
#         [1., 1., 0., 0.],
#         [1., 1., 1., 1.]])
```

**可视化**：

```
         位置 0  位置 1  位置 2  位置 3
位置 0  [  1      0       0       0   ]  → 只能看自己
位置 1  [  1      1       0       0   ]  → 可看 0, 1
位置 2  [  1      1       1       0   ]  → 可看 0, 1, 2
位置 3  [  1      1       1       1   ]  → 可看全部
```

### 5.3 应用掩码

在 softmax 之前应用掩码，将不允许关注的位置设为负无穷：

```python
def masked_attention(Q, K, V, mask):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / (d_k ** 0.5)

    # 应用掩码：将 mask=0 的位置设为 -inf
    scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = F.softmax(scores, dim=-1)
    output = attn_weights @ V
    return output
```

**掩码后的注意力分数**：

```
before softmax:
[[ 0.5   -inf   -inf   -inf]
 [ 0.3    0.7   -inf   -inf]
 [ 0.2    0.4    0.6   -inf]
 [ 0.1    0.3    0.5    0.8]]

after softmax:
[[1.00   0.00   0.00   0.00]  # 只关注位置 0
 [0.40   0.60   0.00   0.00]  # 关注位置 0, 1
 [0.25   0.33   0.42   0.00]  # 关注位置 0, 1, 2
 [0.15   0.22   0.28   0.35]] # 关注全部
```

---

## 6. 注意力的计算复杂度

### 6.1 时间复杂度

核心计算 $QK^T$ 和 $(\text{softmax})V$：

- $QK^T$：$[n, d] \times [d, n] = O(n^2 d)$
- $\text{Attention} \times V$：$[n, n] \times [n, d] = O(n^2 d)$

**总时间复杂度**：$O(n^2 d)$

其中 $n$ 是序列长度，$d$ 是维度。

### 6.2 空间复杂度

需要存储注意力权重矩阵：

**空间复杂度**：$O(n^2)$

### 6.3 长序列的挑战

```mermaid
graph LR
    subgraph seq_len_impact["序列长度影响"]
        L1[n=512] --> C1["计算量 262K"]
        L2[n=2048] --> C2["计算量 4.2M"]
        L3[n=8192] --> C3["计算量 67M"]
        L4[n=32768] --> C4["计算量 1B"]
    end
```

在全注意力 prefill 中，序列长度变为 4 倍，分数矩阵的元素数变为 16 倍。

### 6.4 优化方法简介

| 方法 | 原理 | 复杂度 |
|------|------|--------|
| Flash Attention | IO 优化，减少内存访问 | O(n²) 但更快 |
| Sparse Attention | 稀疏注意力模式 | O(n√n) 或 O(n) |
| Linear Attention | 核方法近似 | O(n) |
| Sliding Window | 只关注局部窗口 | O(nw) |

vLLM 主要使用 **Flash Attention** 作为注意力后端。

---

## 7. Grouped-Query Attention (GQA)

### 7.1 传统 MHA vs GQA

为了减少 KV Cache 的内存占用，现代模型使用 GQA：

```mermaid
graph TB
    subgraph mha["MHA - Multi-Head Attention"]
        MQ1[Q Head 1] --> MK1[K Head 1]
        MQ2[Q Head 2] --> MK2[K Head 2]
        MQ3[Q Head 3] --> MK3[K Head 3]
        MQ4[Q Head 4] --> MK4[K Head 4]
    end

    subgraph gqa["GQA - Grouped-Query Attention"]
        GQ1[Q Head 1] --> GK1[K Group 1]
        GQ2[Q Head 2] --> GK1
        GQ3[Q Head 3] --> GK2[K Group 2]
        GQ4[Q Head 4] --> GK2
    end
```

### 7.2 GQA 的优势

| 特性 | MHA | GQA |
|------|-----|-----|
| Q heads | N | N |
| K/V heads | N | N/group_size |
| KV Cache 大小 | 100% | 减少到 1/group_size |
| 模型质量 | 基准 | 接近基准 |

**示例**（LLaMA-2-70B）：
- Q heads: 64
- KV heads: 8
- 同模型形状和精度下，KV 字节数是 64 个 KV 头时的 1/8。

---

## 8. 注意力与 KV Cache 的关系

### 8.1 为什么需要缓存 K 和 V

在自回归生成中，每生成一个新 token，都需要计算它与所有历史 token 的注意力。

**不使用 KV Cache**：每次都重新计算所有 token 的 K 和 V
**使用 KV Cache**：缓存历史 token 的 K 和 V，只计算新 token 的

这正是下一章的主题！

### 8.2 预览：KV Cache 的作用

{{< trace >}}
title: 下一个位置怎样读取历史 KV
actors: ["新位置", "注意力层", "KV 缓存"]
steps:
  - from: 0
    to: 1
    label: "输入位置 3"
    detail: "位置 3 的隐藏状态进入这一层。"
  - from: 1
    to: 1
    label: "投影 Q、K、V"
    detail: "产生当前位置的查询和键值，历史位置已计算过。"
  - from: 2
    to: 1
    label: "读 K₀ 至 K₂"
    detail: "缓存给出位置 0 到 2 的键与值。"
  - from: 1
    to: 1
    label: "求加权和"
    detail: "Q₃ 对 K₀…K₃ 求分数，归一化后汇总 V₀…V₃。"
  - from: 1
    to: 2
    label: "保留 K₃、V₃"
    detail: "完成本轮后，下个位置可读取四个位置。具体写入可与设备计算融合。"
{{< /trace >}}

---

## 注意力输出的计算式

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

分数矩阵增大时，还可以沿 K/V 序列分块读取。[FlashAttention 的两块手算例子](/flashattention/02-core-algorithm/01-online-softmax/) 展示怎样保留行最大值和加权和，最后得到同一个注意力输出。

## 思考题

1. 如果没有缩放因子 $\sqrt{d_k}$，会发生什么？
2. 为什么 GQA 可以在减少 KV heads 的同时保持模型质量？
3. 在因果掩码下，位置 0 的 token 只能关注自己，这会影响模型效果吗？

---

[计算一次生成需要保留多少 KV](04-kv-cache-concept.md)
