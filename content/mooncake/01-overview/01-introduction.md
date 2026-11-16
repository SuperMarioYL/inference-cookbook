---
title: "从 Kimi 论文到当前开源组件"
weight: 1
---

FAST ’25 的 Mooncake 研究了一个具体问题。面对有长输入和多轮对话的流量，系统怎样在满足首 token 与后续 token 延迟要求的同时，接下更多请求？团队将 prefill 与 decode 分开部署，并利用集群中的主存、SSD 和网络共享 KV。

作者在真实 trace 回放中报告有效请求容量比对照方法提高 **59% 至 498%**。在生产部署对比中，A800 和 H800 集群分别比先前系统多承载 **115% 和 107%** 的请求。这两类实验的负载和基线不同。复现时需要保留每组 SLO、输入分布和设备条件。[论文与演讲](https://www.usenix.org/conference/fast25/presentation/qin)

## 在 v0.3.14 读哪些代码

Transfer Engine 处理字节传递；Store 管对象副本、分配和读取；Conductor 用引擎和 Store 的事件为路由端维护 KV 前缀索引。TENT 为传输增加动态选路、切片和多路径调度。先确定你关注的是一次请求的交接，还是跨请求保存缓存，再选择相应模块。[开源架构](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/architecture.md) · [本版变更](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14)
