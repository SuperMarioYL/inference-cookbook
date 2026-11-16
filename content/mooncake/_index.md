---
title: "Mooncake"
linkTitle: "Mooncake"
weight: 26
description: "Mooncake v0.3.14 的 KV 直传、共享 Store、前缀索引与部署"
---

Mooncake 可以把一次 prefill 算出的 KV 直接送到 decode 节点，也可以把 KV 放进多个推理实例共享的缓存池。两条路径都要搬数据，区别是 KV 由谁保存、下一次由谁查找。本站按 `v0.3.14` 讲开源组件；FAST ’25 论文描述了它们在 Kimi 系统中要解决的容量和调度问题。[发行说明](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14) · [论文](https://www.usenix.org/conference/fast25/presentation/qin)

第一次使用可以先在一台 Linux 主机上[写入并读回一个对象](./00-beginner-guide/03-quick-start-tutorial.md)。要接推理服务，转到 [vLLM](./08-integration/01-vllm-integration.md) 或 [SGLang](./08-integration/02-sglang-integration.md)。想知道“哪个实例已有这段前缀”，看 [Conductor](./02-architecture/03-conductor-scheduler.md)。
