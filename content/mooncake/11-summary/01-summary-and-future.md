---
title: "从 v0.3.14 回看存储和传输的取舍"
weight: 1
---

FAST ’25 提出用额外存储承接本可重算的 KV。请求省下多少时间，取决于缓存留下了多少可复用内容、读回要搬多少字节，以及模型会等多久。论文评测看服务目标下有效承载的请求容量，开源版本则可以沿具体组件检查每个环节。[论文](https://www.usenix.org/conference/fast25/presentation/qin)

## v0.3.14 的三个观察点

1. **谁知道缓存位置。** 引擎与 Store 发布事件，Conductor 为路由器维护多层前缀索引。路由查询结果还能结合负载等信息做最终选择。[Conductor](../02-architecture/03-conductor-scheduler.md)
2. **怎样读到数据。** Store 支持可选的 SSD 到 DRAM 预取；开启后，只有磁盘可读副本时，指定选项的存在性查询可以触发后台提升。预取尚未完成时，后续读取仍可能由 SSD 供数。[SSD 预取](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/store/ssd-prefetch.md)
3. **怎样选传输路径。** TENT 根据可达性、拓扑和负载选路；v0.3.14 更新了异步 TCP、多 rail 和大读请求分条。应分别测量传输和应用层 TTFT。[版本记录](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14)

如果部署目标是交接同一个请求，从[直传](../08-integration/01-vllm-integration.md)开始。如果要利用不同请求之间相同的手册或系统提示，先做[跨实例 Store 复用](../10-deployment/01-deployment-guide.md)。
