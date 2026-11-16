---
title: "推理引擎与 worker 导读"
slug: "readme"
description: "前端和后端 worker 之间传递请求与结果。聚合执行可以沿一名 worker 阅读；P/D 分离则需要同时观察两个角色、交接元信息和 KV 数据实际就绪时间。"
weight: -1
---

前端和后端 worker 之间传递请求与结果。聚合执行可以沿一名 worker 阅读；P/D 分离则需要同时观察两个角色、交接元信息和 KV 数据实际就绪时间。

1. [推理引擎原理](/dynamo/03-llm-inference-layer/01-inference-engine-basics/)
2. [vLLM 技术解析](/dynamo/03-llm-inference-layer/02-vllm-internals/)
3. [多引擎抽象层](/dynamo/03-llm-inference-layer/03-multi-engine-abstraction/)
4. [vLLM Worker 的启动、注册和执行](/dynamo/03-llm-inference-layer/04-worker-design/)
5. [P/D 分离如何交接 KV](/dynamo/03-llm-inference-layer/05-disaggregated-serving/)
6. [请求处理流程](/dynamo/03-llm-inference-layer/06-request-processing/)

[上一章 · 运行时与服务发现导读](/dynamo/02-runtime-layer/readme/) · [下一章 · KV 缓存与路由导读](/dynamo/04-kv-cache-system/readme/)
