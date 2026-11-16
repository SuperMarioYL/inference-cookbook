---
title: "Mooncake 能给一次请求提供什么"
weight: 1
---

一台机器读完提示词，另一台机器接着生成回答。让后者从头读同一份提示会重复计算；把已有 KV 送过去则要消耗带宽。Mooncake 的 Transfer Engine 提供数据传输能力，推理框架的 connector 决定何时发、发哪些块、写到哪里。

如果很多请求都会用那份文档，可以让引擎把 KV 写进 Mooncake Store。后来接到请求的实例先查有哪些连续前缀块可用，再把它们读回设备。Store 的 Master 管位置，由客户端和存储节点交换数据。[Store 设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/store/mooncake-store.md)

论文中的 Kimi 系统同时考虑队列、KV 位置和延迟目标。v0.3.14 的开源 Conductor 把缓存事件整理成可查询的前缀索引，路由器可以据此选择实例。[论文](https://www.usenix.org/conference/fast25/presentation/qin) · [开源 Conductor](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/index.md)

先从[单机 put/get](./03-quick-start-tutorial.md)验证存储。部署推理服务时再选直传或共享前缀路径。
