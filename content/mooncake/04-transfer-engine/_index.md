---
title: "Transfer Engine"
linkTitle: "Transfer Engine"
weight: 4
description: >
  高性能数据传输引擎与 RDMA 协议实现
---

一次传输需要源缓冲、目标 segment、地址范围、方向和完成状态。按[经典 Transfer Engine 接口](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/cpp/transfer-engine.md)追踪请求怎样分解为后端操作；动态选路和多 rail 再到 [TENT](../05-tent/01-design-overview.md) 阅读。
