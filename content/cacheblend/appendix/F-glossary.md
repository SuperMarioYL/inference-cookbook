---
title: "术语表"
weight: 6
---

术语按注意力计算、缓存复用和论文评测分组。

---

## 核心术语

| 术语 | 中文 | 解释 |
|------|------|------|
| **Attention** | 注意力机制 | Transformer 的核心机制，计算 token 间的关联 |
| **Cross-Attention** | 交叉注意力 | 不同文本块 token 之间的注意力 |
| **Decode** | 解码阶段 | 自回归生成 token 的阶段 |
| **F1-Score** | F1 分数 | 精确率和召回率的调和平均 |
| **GPU HBM** | GPU 高带宽内存 | GPU 上的主内存 |
| **HKVD** | 高 KV 偏差 | High-KV-Deviation 的缩写 |
| **KV Cache** | 键值缓存 | 存储 Key 和 Value 张量的缓存 |
| **LLM** | 大语言模型 | Large Language Model |
| **MLP** | 多层感知机 | 前馈神经网络层 |
| **NVMe** | 非易失性存储器 | 高速 SSD 接口标准 |
| **Prefill** | 预填充阶段 | 处理输入 prompt 生成初始 KV Cache 的阶段 |
| **RAG** | 检索增强生成 | Retrieval-Augmented Generation |
| **RMSNorm** | 均方根归一化 | LLaMA 使用的归一化方法 |
| **RoPE** | 旋转位置编码 | Rotary Position Embedding |
| **Rouge-L** | Rouge-L 分数 | 基于最长公共子序列的评估指标 |
| **Self-Attention** | 自注意力 | 同一文本块 token 之间的注意力 |
| **TTFT** | 首 Token 时间 | Time-To-First-Token |
| **Tensor Parallel** | 张量并行 | 将模型张量分布到多个 GPU |
| **Token** | 令牌 | 文本的基本单位 |
| **Throughput** | 吞吐量 | 单位时间处理的请求数 |
| **xFormers** | xFormers 库 | Meta 开发的高效注意力计算库 |
| **PagedAttention** | 分页注意力 | vLLM 使用的内存高效注意力机制 |
| **GQA** | 分组查询注意力 | Grouped Query Attention，减少 KV heads 的技术 |
| **Chunk** | 文本块 | 将长文本分割成的较小片段 |
| **Embedding** | 嵌入 | 将离散 token 映射到连续向量空间 |
| **Residual** | 残差连接 | 跨层的跳跃连接，缓解梯度消失 |
| **LayerNorm** | 层归一化 | 神经网络归一化技术 |
| **Softmax** | Softmax 函数 | 将分数转换为概率分布 |
| **Batch** | 批次 | 同时处理的多个请求 |

---

## CacheBlend 特定术语

| 术语 | 解释 |
|------|------|
| **HKVD Token** | 高 KV 偏差令牌，指那些在预计算 KV Cache 和完整 Prefill KV Cache 之间差异最大的 token |
| **Selective KV Recompute** | 选择性 KV 重计算，CacheBlend 的核心技术，只重计算部分 token 的 KV |
| **KV Deviation** | KV 偏差，衡量预计算 KV 与完整计算 KV 之间的差异 |
| **Attention Deviation** | 注意力偏差，衡量前向注意力矩阵与完整计算的差异 |
| **Gradual Filtering** | 渐进过滤，逐层减少需要重计算的 token 数量 |
| **Loading Controller** | 加载控制器，负责协调 KV Cache 加载和重计算的调度 |
| **Fusor** | 融合器，负责执行 KV Cache 的融合操作 |
| **KV Cache Store** | KV 缓存存储，管理 KV Cache 的多级存储 |

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
2. Attention Is All You Need (Transformer 原始论文)
3. vLLM: Efficient Memory Management for Large Language Model Serving
