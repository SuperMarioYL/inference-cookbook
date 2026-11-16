---
title: "实验设置"
weight: 1
---

论文用 Mistral-7B、Yi-34B 和 Llama-70B 比较重算、前缀缓存与融合。硬件、量化、输入长度和命中条件决定了其中每组数字解释的场景。

---

## 6.1.1 模型配置

| 模型 | 参数量 | 量化 | GPU 数量 |
|------|--------|------|----------|
| Mistral-7B | 7B | 无 | 1 x A40 |
| Yi-34B | 34B | 8-bit | 1 x A40 |
| Llama-70B | 70B | 8-bit | 2 x A40 |

## 6.1.2 硬件环境

- **平台**: Runpod GPUs
- **内存**: 128 GB RAM
- **GPU**: 2 x Nvidia A40
- **存储**: 1TB NVME SSD（测量吞吐量 4.8 GB/s）

## 6.1.3 数据集描述

| 数据集 | 任务类型 | 样本数 | 说明 |
|--------|----------|--------|------|
| 2WikiMQA | 多跳问答 | 200 | 测试多段落推理能力 |
| Musique | 多跳问答 | 150 | 测试多跳推理能力 |
| SAMSum | 文本摘要 | 200 | 测试对话摘要能力 |
| MultiNews | 文本摘要 | 60 | 测试多文档摘要能力 |

**数据处理**：
- 使用 Langchain 将上下文分割成 512-token 的 chunk
- SAMSum 使用原始 200-400 token 的 chunk

## 6.1.4 评估指标

| 指标 | 应用数据集 | 说明 |
|------|------------|------|
| **F1-Score** | 2WikiMQA, Musique | 基于词重叠计算 |
| **Rouge-L** | SAMSum, MultiNews | 基于最长公共子序列 |
| **TTFT** | 所有 | Time-To-First-Token |
| **Throughput** | 所有 | 推理吞吐量 |

## 6.1.5 基线方法

1. **Full KV Recompute**: 不复用任何 KV Cache
2. **Prefix Caching**: 仅复用前缀的 KV Cache（SGLang 实现）
3. **Full KV Reuse**: 复用所有 KV Cache，忽略 Cross-Attention（PromptCache）
4. **MapReduce**: LangChain 的替代 RAG 方法，先摘要后合并
5. **MapRerank**: LangChain 的替代 RAG 方法，独立生成答案后排序

---

## 下一步

- [性能结果](./02-results.md) - 查看 CacheBlend 的性能评估结果

硬件、量化、测试样本和指标来自论文第 7.1 节；吞吐实验另外使用扩展 RAG 负载，和单题质量测量的样本数不同。[实验设置](https://arxiv.org/html/2405.16444v3#S7.SS1) · [图 14 的基线](./02-results.md)
