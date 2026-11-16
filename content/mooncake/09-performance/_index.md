---
title: "性能与监控"
linkTitle: "性能优化"
weight: 9
description: >
  基准测试、性能监控与系统调优指南
---

把端到端 TTFT、有效请求容量、外部缓存命中率与某条传输链路的带宽分别测量。比较前要固定输入长度、并发、数据路径和硬件条件。上游将 [vLLM 共享 Store 实验](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/performance/vllm/vllm-v1-mooncake-store.md)与 [P/D 直传实验](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/performance/vllm/vllm-v1-pd-performance.md)分开记录。
