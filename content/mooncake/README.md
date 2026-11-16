---
title: "Mooncake 阅读指南"
---

读源码前先确定要跟踪哪份数据。Store 管对象名、副本和可读性，Transfer Engine 搬字节，Conductor 收集 KV 事件后回答哪些实例可能已经缓存某个前缀。给实际请求选择哪个实例，由上层路由器决定。[v0.3.14 架构](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/architecture.md) · [Conductor 设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/index.md)

- 查看一次写入。先跑 [Store 的 TCP 示例](./00-beginner-guide/03-quick-start-tutorial.md)，再看 [Master 如何定位副本](./03-mooncake-store/02-master-service.md)。
- 查看一次推理。读 [P/D 直传和共享池对照](./02-architecture/02-request-lifecycle.md)，再选 [vLLM](./08-integration/01-vllm-integration.md) 或 [SGLang](./08-integration/02-sglang-integration.md) 的入口。
- 分析性能。先问等待发生在路由、传输、I/O 还是模型执行。读 [TENT 路径选择](./05-tent/01-design-overview.md)和[监控](./09-performance/02-monitoring.md)，对照相同硬件、负载和服务目标。

论文的 trace 回放和生产部署对比分别测量有效容量。二者的分母、机器和负载不同。[FAST ’25 实验](https://www.usenix.org/conference/fast25/presentation/qin)
