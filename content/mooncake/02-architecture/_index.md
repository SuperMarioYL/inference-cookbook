---
title: "系统架构"
linkTitle: "系统架构"
weight: 2
description: >
  Mooncake 整体架构设计、请求交接、共享存储与 Conductor 前缀索引
---

同一份 KV 可以从 P 直接交给 D，也能先写入共享 Store。先看两条路径中各个组件的职责，接着看上层路由器怎样利用 Conductor 的前缀索引。[数据流对照](./02-request-lifecycle.md) · [Conductor](./03-conductor-scheduler.md)
