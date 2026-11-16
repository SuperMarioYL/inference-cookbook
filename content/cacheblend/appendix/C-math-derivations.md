---
title: "数学推导"
weight: 3
---

文档移位同时改变了位置编码和可见的上文。先固定 Q、K 推导位置差，再考察隐藏状态变化怎样进入 K、V 偏差。

---

## RoPE 位置编码的相对位置不变性证明

**定理**: RoPE（Rotary Position Embedding，旋转位置编码）保持相对位置不变性，即两个 token 之间的注意力分数中的位置因子只取决于相对位置差。

**证明**:

设 $q_m$ 和 $k_n$ 分别是位置 $m$ 和 $n$ 的 Query 和 Key 向量。RoPE 对它们进行旋转编码：

$$
\begin{aligned}
\tilde{q}_m &= R_m \cdot q_m \\
\tilde{k}_n &= R_n \cdot k_n
\end{aligned}
$$

其中 $R_\theta$ 是旋转矩阵：

$$
R_\theta = \begin{pmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{pmatrix}
$$

注意力分数为：

$$
\begin{aligned}
\tilde{q}_m^T \tilde{k}_n &= (R_m q_m)^T (R_n k_n) \\
&= q_m^T R_m^T R_n k_n \\
&= q_m^T R_{n-m} k_n
\end{aligned}
$$

最后一步利用了旋转矩阵的性质 $R_m^T R_n = R_{n-m}$。

这证明了注意力分数中的旋转因子只依赖两者的相对位置 $(n-m)$。

### CacheBlend 中的应用

若当前引擎保存的是已加入 RoPE 的 Key，它位于原始位置 $p_{old}$：

$$
k_{old} = R_{p_{old}} \cdot k_{base}
$$

在融合时，我们需要将其恢复到新位置 $p_{new}$。利用旋转矩阵的逆：

$$
\begin{aligned}
k_{new} &= R_{p_{new}} \cdot k_{base} \\
&= R_{p_{new}} \cdot R_{p_{old}}^{-1} \cdot k_{old} \\
&= R_{p_{new} - p_{old}} \cdot k_{old}
\end{aligned}
$$

作者原型在应用 RoPE 之前收集 `hack_kv`，因此复用时对这份原始 K 用组合后的位置直接旋转。若输入已经带旧旋转，才使用上面的差分关系。跟踪某个具体实现应从保存 K 的位置开始。[原型源码](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py)

---

## HKVD 选择的理论基础

### 问题定义

给定：
- 旧 KV Cache: $K_{old}, V_{old} \in \mathbb{R}^{n \times d}$
- 新 KV Cache: $K_{new}, V_{new} \in \mathbb{R}^{n \times d}$
- 重计算预算: $k$ 个 token（$k << n$）

目标：选择 $k$ 个 token 进行重计算，使注意力输出偏差最小化。

### 形式化

设选择的 token 索引集合为 $S$，$|S| = k$。定义融合后的 KV：

$$
\begin{aligned}
K_{fused}[i] &= \begin{cases}
K_{new}[i] & \text{if } i \in S \\
K_{old}[i] & \text{otherwise}
\end{cases}
\end{aligned}
$$

目标是最小化注意力输出偏差：

$$
\min_{|S|=k} \| \text{Attention}(Q, K_{fused}, V_{fused}) - \text{Attention}(Q, K_{new}, V_{new}) \|
$$

### 近似解

直接优化上述目标是 NP-hard 的。CacheBlend 使用了一个贪婪近似：选择 KV 偏差最大的 token。

$$
S = \text{TopK}_i \left( \| V_{new}[i] - V_{old}[i] \|_2^2 \right)
$$

### 为什么这个近似有效？

1. **KV 偏差与注意力偏差的相关性**: 实验表明，KV 偏差大的 token 更可能导致较大的注意力偏差
2. **注意力稀疏性**: 大多数 token 的注意力权重很小，因此它们的 KV 偏差对输出影响有限
3. **层间相关性**: 第一层选择的 HKVD tokens 在后续层仍然是高偏差的

---

## 注意力稀疏性的量化分析

### 定义

对于注意力权重矩阵 $A \in \mathbb{R}^{n \times n}$，稀疏度定义为：

$$
\text{Sparsity}(A, \epsilon) = \frac{|\{(i,j): A_{ij} < \epsilon\}|}{n^2}
$$

### 实验观察

在 Mistral-7B 上的测量结果：

| 层 | 稀疏度 ($\epsilon=0.01$) | 稀疏度 ($\epsilon=0.001$) |
|---|---|---|
| Layer 1 | 92.3% | 87.5% |
| Layer 8 | 94.1% | 89.2% |
| Layer 16 | 95.7% | 91.8% |
| Layer 24 | 96.3% | 93.1% |
| Layer 32 | 97.2% | 94.5% |

### 结论

深层的注意力更加稀疏，这解释了为什么选择少量 HKVD tokens 就能恢复大部分 Cross-Attention。

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
2. RoFormer: Enhanced Transformer with Rotary Position Embedding
