---
title: "SGLang 0.5.23 章节目录"
slug: "readme"
description: "按请求、调度、内存与性能问题查找 SGLang 教程。"
weight: -1
---

第一次使用可先[启动单卡服务](/sglang/01-getting-started/02-quick-start/)。分析已有部署时，从对应瓶颈进入：长输入等待看调度和前缀树，单步执行慢看设备与并行，输出延迟看反分词和流式响应。

### 入门篇

- [从一份共享文档理解 SGLang](/sglang/01-getting-started/01-introduction/)
- [用 SGLang 0.5.23 启动文本服务](/sglang/01-getting-started/02-quick-start/)
- [SGLang 文本服务的进程和状态](/sglang/01-getting-started/03-architecture-overview/)

### 核心架构篇

- [跟踪一次流式生成](/sglang/02-core-architecture/04-request-lifecycle/)
- [TokenizerManager](/sglang/02-core-architecture/05-tokenizer-manager/)
- [Scheduler 怎样选择下一批请求](/sglang/02-core-architecture/06-scheduler-deep-dive/)
- [Detokenizer](/sglang/02-core-architecture/07-detokenizer/)
- [ModelRunner](/sglang/02-core-architecture/08-model-runner/)

### 内存与缓存篇

- [KV Cache 基础](/sglang/03-memory-cache/09-kv-cache-basics/)
- [RadixAttention 如何复用共同前缀](/sglang/03-memory-cache/10-radix-attention/)
- [内存池管理](/sglang/03-memory-cache/11-memory-pool/)
- [哪些前缀可以驱逐](/sglang/03-memory-cache/12-eviction-policy/)

### 批处理与调度篇

- [连续批处理](/sglang/04-batching-scheduling/13-continuous-batching/)
- [Chunked Prefill](/sglang/04-batching-scheduling/14-chunked-prefill/)
- [队列顺序与缓存感知调度](/sglang/04-batching-scheduling/15-schedule-policy/)
- [前向模式](/sglang/04-batching-scheduling/16-forward-modes/)

### 性能优化篇

- [CUDA Graph](/sglang/05-performance/17-cuda-graph/)
- [Torch Compile](/sglang/05-performance/18-torch-compile/)
- [量化](/sglang/05-performance/19-quantization/)
- [投机解码](/sglang/05-performance/20-speculative-decoding/)

### 分布式与并行篇

- [张量并行](/sglang/06-distributed/21-tensor-parallelism/)
- [流水线并行](/sglang/06-distributed/22-pipeline-parallelism/)
- [专家并行](/sglang/06-distributed/23-expert-parallelism/)

### 高级特性篇

- [结构化输出](/sglang/07-advanced-features/24-structured-output/)
- [LoRA 支持](/sglang/07-advanced-features/25-lora-support/)
- [多模态](/sglang/07-advanced-features/26-multimodal/)

### 实战调试篇

- [调试指南](/sglang/08-debugging/27-debugging-guide/)
- [性能分析](/sglang/08-debugging/28-profiling/)
- [问题排查](/sglang/08-debugging/29-troubleshooting/)
