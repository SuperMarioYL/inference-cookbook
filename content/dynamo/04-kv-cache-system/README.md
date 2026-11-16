---
title: "KV 缓存与路由导读"
slug: "readme"
description: "后端管理实际 KV，路由器利用事件维护可复用前缀的视图。对照同一个输入在两名 worker 上的命中情况，再看内存压力与驱逐如何改变下一次选择。"
weight: -1
---

后端管理实际 KV，路由器利用事件维护可复用前缀的视图。对照同一个输入在两名 worker 上的命中情况，再看内存压力与驱逐如何改变下一次选择。

1. [KV Cache 原理](/dynamo/04-kv-cache-system/01-transformer-kv-cache/)
2. [优化技术综述](/dynamo/04-kv-cache-system/02-kv-optimization-survey/)
3. [如何从两名 worker 中选择 KV 路由目标](/dynamo/04-kv-cache-system/03-kv-aware-router/)
4. [路由索引怎样表达共享前缀](/dynamo/04-kv-cache-system/04-radix-tree-indexing/)
5. [KV 块管理器与路由索引的职责](/dynamo/04-kv-cache-system/05-kv-cache-manager/)
6. [KV 事件如何更新路由视图](/dynamo/04-kv-cache-system/06-kv-event-system/)

[上一章 · 推理引擎与 worker 导读](/dynamo/03-llm-inference-layer/readme/) · [下一章 · 分布式通信与数据传输导读](/dynamo/05-distributed-communication/readme/)
