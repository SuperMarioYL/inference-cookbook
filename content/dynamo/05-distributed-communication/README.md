---
title: "分布式通信与数据传输导读"
slug: "readme"
description: "比较发现、请求、事件三个运行时平面，再追踪一次 KV 交接。默认 TCP 请求和 ZMQ 事件可独立配置；模型数据使用哪种传输取决于后端与设备。"
weight: -1
---

比较发现、请求、事件三个运行时平面，再追踪一次 KV 交接。默认 TCP 请求和 ZMQ 事件可独立配置；模型数据使用哪种传输取决于后端与设备。

1. [通信基础](/dynamo/05-distributed-communication/01-distributed-comm-basics/)
2. [发现、请求与事件三个通信平面](/dynamo/05-distributed-communication/02-three-plane-architecture/)
3. [etcd 模式下的注册与失效](/dynamo/05-distributed-communication/03-etcd-integration/)
4. [NATS 与可选通信平面](/dynamo/05-distributed-communication/04-nats-integration/)
5. [P/D 分离中的 NIXL 传输](/dynamo/05-distributed-communication/05-nixl-data-transfer/)
6. [TCP 请求通道与 KV 传输](/dynamo/05-distributed-communication/06-tcp-pipeline/)

[上一章 · KV 缓存与路由导读](/dynamo/04-kv-cache-system/readme/) · [下一章 · Python 运行时与端点导读](/dynamo/06-python-sdk/readme/)
