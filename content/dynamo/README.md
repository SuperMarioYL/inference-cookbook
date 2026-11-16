---
title: "NVIDIA Dynamo 1.6.2 章节目录"
slug: "readme"
description: "从单机服务到分离部署、KV 路由与 Python 客户端。"
weight: -1
---

初次运行先看[单机启动与验证](/dynamo/06-python-sdk/05-cli-tools/)。追踪一次跨 worker 调用时，可依次阅读架构、运行时、引擎和传输。下表按问题定位到各章。

| 章节 | 阅读目标 |
|---|---|
| [系统架构与部署模式导读](/dynamo/01-architecture-overview/readme/) | 从一条用户请求入手，分清 frontend、路由器与推理引擎的职责，再决定一次完整推理交给聚合 worker，还是拆到 P 与 D。 |
| [运行时与服务发现导读](/dynamo/02-runtime-layer/readme/) | 运行时为异步操作、组件命名和成员发现提供共同基础。追查实例为什么不可见时，先辨明当前是 Kubernetes、文件还是 etcd 发现，再查相应后端。 |
| [推理引擎与 worker 导读](/dynamo/03-llm-inference-layer/readme/) | 前端和后端 worker 之间传递请求与结果。聚合执行可以沿一名 worker 阅读；P/D 分离则需要同时观察两个角色、交接元信息和 KV 数据实际就绪时间。 |
| [KV 缓存与路由导读](/dynamo/04-kv-cache-system/readme/) | 后端管理实际 KV，路由器利用事件维护可复用前缀的视图。对照同一个输入在两名 worker 上的命中情况，再看内存压力与驱逐如何改变下一次选择。 |
| [分布式通信与数据传输导读](/dynamo/05-distributed-communication/readme/) | 比较发现、请求、事件三个运行时平面，再追踪一次 KV 交接。默认 TCP 请求和 ZMQ 事件可独立配置；模型数据使用哪种传输取决于后端与设备。 |
| [Python 运行时与端点导读](/dynamo/06-python-sdk/readme/) | 先从 PyO3 理解调用边界，再以一对 echo 服务与客户端理解 endpoint 的注册、查找及释放。要加载模型，直接阅读本章最后的单机示例。 |
| [HTTP 接口与流式响应导读](/dynamo/07-http-api-layer/readme/) | 从模型发现、聊天请求到 SSE 流，将客户端可见行为与后端执行对应起来。健康端点、本次输出和中途断开的响应分别能证明不同环节。 |
