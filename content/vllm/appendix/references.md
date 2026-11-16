---
title: "参考资料"
weight: 3
---


本栏的源码解释针对 **vLLM 0.33.0**。调度器和 CLI 以固定版本为准；早期论文解释设计起点，其报告的实验数字应按当时的硬件、模型和对照组理解。

- [0.33.0 架构与进程](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/arch_overview.md)
- [0.33.0 混合 KV 管理器](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/hybrid_kv_cache_manager.md)
- [0.33.0 自动前缀缓存](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/prefix_caching.md)
- [按职责查看源码](/vllm/appendix/code-map/)

---

## 官方资源

### vLLM 官方

- **vLLM GitHub 仓库**
  - https://github.com/vllm-project/vllm
  - 源代码、Issue 讨论、PR 贡献

- **vLLM 官方文档**
  - https://docs.vllm.ai/
  - 安装指南、API 参考、最佳实践

- **vLLM 博客**
  - https://blog.vllm.ai/
  - 技术文章、版本更新、性能分析

---

## 核心论文

### PagedAttention

- **Efficient Memory Management for Large Language Model Serving with PagedAttention**
  - 作者: Woosuk Kwon, Zhuohan Li, et al.
  - 会议: SOSP 2023
  - 链接: https://arxiv.org/abs/2309.06180
  - 要点: vLLM 的核心创新，介绍分页注意力机制

### Transformer 架构

- **Attention Is All You Need**
  - 作者: Vaswani et al.
  - 会议: NeurIPS 2017
  - 链接: https://arxiv.org/abs/1706.03762
  - 要点: Transformer 架构的原始论文

### Flash Attention

- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness**
  - 作者: Tri Dao et al.
  - 会议: NeurIPS 2022
  - 链接: https://arxiv.org/abs/2205.14135
  - 要点: IO 优化的注意力计算

- **FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning**
  - 作者: Tri Dao
  - 链接: https://arxiv.org/abs/2307.08691
  - 要点: Flash Attention 的改进版本

### 投机解码

- **Fast Inference from Transformers via Speculative Decoding**
  - 作者: Yaniv Leviathan et al.
  - 会议: ICML 2023
  - 链接: https://arxiv.org/abs/2211.17192
  - 要点: 投机解码的原始论文

- **EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty**
  - 作者: Yuhui Li et al.
  - 链接: https://arxiv.org/abs/2401.15077
  - 要点: 利用隐藏状态的投机解码

- **Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads**
  - 作者: Tianle Cai et al.
  - 链接: https://arxiv.org/abs/2401.10774
  - 要点: 多头预测投机解码

### 量化技术

- **AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration**
  - 作者: Ji Lin et al.
  - 会议: MLSys 2024
  - 链接: https://arxiv.org/abs/2306.00978
  - 要点: 激活感知权重量化

- **GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers**
  - 作者: Elias Frantar et al.
  - 会议: ICLR 2023
  - 链接: https://arxiv.org/abs/2210.17323
  - 要点: 基于 Hessian 的后训练量化

- **FP8 Formats for Deep Learning**
  - 作者: Paulius Micikevicius et al.
  - 链接: https://arxiv.org/abs/2209.05433
  - 要点: FP8 格式规范

### 分布式并行

- **Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism**
  - 作者: Mohammad Shoeybi et al.
  - 链接: https://arxiv.org/abs/1909.08053
  - 要点: 张量并行和流水线并行

- **GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism**
  - 作者: Yanping Huang et al.
  - 链接: https://arxiv.org/abs/1811.06965
  - 要点: 流水线并行训练

---

## 深度学习基础

### 书籍

- **Deep Learning** (花书)
  - 作者: Ian Goodfellow, Yoshua Bengio, Aaron Courville
  - 链接: https://www.deeplearningbook.org/
  - 要点: 深度学习理论基础

- **Dive into Deep Learning** (动手学深度学习)
  - 作者: Aston Zhang, Zachary C. Lipton, et al.
  - 链接: https://d2l.ai/
  - 要点: 实践导向的深度学习教程

### 在线课程

- **CS231n: Convolutional Neural Networks for Visual Recognition**
  - 学校: Stanford
  - 链接: http://cs231n.stanford.edu/
  - 要点: 神经网络基础

- **CS224n: Natural Language Processing with Deep Learning**
  - 学校: Stanford
  - 链接: https://web.stanford.edu/class/cs224n/
  - 要点: NLP 和 Transformer

---

## GPU 和 CUDA

### NVIDIA 官方

- **CUDA Programming Guide**
  - 链接: https://docs.nvidia.com/cuda/cuda-programming-guide/
  - 要点: CUDA 编程基础

- **NCCL Documentation**
  - 链接: https://docs.nvidia.com/deeplearning/nccl/
  - 要点: 多 GPU 通信

- **cuBLAS Documentation**
  - 链接: https://docs.nvidia.com/cuda/cublas/
  - 要点: GPU 矩阵运算

### 性能优化

- **GPU Performance Background User's Guide**
  - 链接: https://docs.nvidia.com/deeplearning/performance/index.html
  - 要点: GPU 性能优化指南

---

## 相关项目

### 推理引擎

- **TensorRT-LLM**
  - GitHub: https://github.com/NVIDIA/TensorRT-LLM
  - 说明: NVIDIA 官方 LLM 推理引擎

- **Text Generation Inference (TGI)**
  - GitHub: https://github.com/huggingface/text-generation-inference
  - 说明: Hugging Face 的 LLM 推理服务

- **llama.cpp**
  - GitHub: https://github.com/ggerganov/llama.cpp
  - 说明: CPU 推理优化

### 量化工具

- **AutoAWQ**
  - GitHub: https://github.com/casper-hansen/AutoAWQ
  - 说明: AWQ 量化工具

- **AutoGPTQ**
  - GitHub: https://github.com/PanQiWei/AutoGPTQ
  - 说明: GPTQ 量化工具

### 模型库

- **Hugging Face Model Hub**
  - 链接: https://huggingface.co/models
  - 说明: 预训练模型下载

---

## 技术博客

### LLM 推理

- **The Illustrated Transformer**
  - 作者: Jay Alammar
  - 链接: https://jalammar.github.io/illustrated-transformer/
  - 要点: Transformer 可视化解释

- **LLM Inference Performance Engineering: Best Practices**
  - 来源: Databricks
  - 要点: LLM 推理优化最佳实践

### vLLM 相关

- **vLLM: PagedAttention for 24x Faster LLM Inference**
  - 来源: vLLM Blog
  - 链接: https://blog.vllm.ai/2023/06/20/vllm.html
  - 要点: vLLM 介绍博客

---

## 社区资源

### 讨论论坛

- **vLLM 官方论坛**
  - 链接: https://discuss.vllm.ai/
  - 说明: 官方交流社区

- **Hugging Face Forums**
  - 链接: https://discuss.huggingface.co/
  - 说明: 模型和推理讨论

### GitHub Issues

- **vLLM Issues**
  - 链接: https://github.com/vllm-project/vllm/issues
  - 说明: Bug 报告和功能请求

---

## 学习路径建议

### 入门阶段
1. 阅读《动手学深度学习》Transformer 章节
2. 阅读 "The Illustrated Transformer"
3. 了解 vLLM 基本使用

### 进阶阶段
1. 阅读 PagedAttention 论文
2. 阅读 Flash Attention 论文
3. 学习 vLLM 源码中的核心模块

### 深入阶段
1. 阅读量化相关论文（AWQ、GPTQ）
2. 阅读投机解码论文（Speculative Decoding、EAGLE）
3. 了解分布式并行（Megatron-LM）

---

**导航**
- 上一篇：[代码文件索引](code-map.md)
- 返回：[README](/vllm/00-overview/overview/)
