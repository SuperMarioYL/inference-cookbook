---
title: "NVIDIA Dynamo"
linkTitle: "Dynamo"
weight: 3
description: "NVIDIA Dynamo 的服务发现、KV 感知路由与后端部署。"
---

NVIDIA Dynamo 1.6.2 负责在 frontend 和推理 worker 之间接入、发现与路由请求。第一次读可以先用 CPU [在两个进程之间发送字符串](./06-python-sdk/03-service-decorators.md)，再[启动带模型的服务](./06-python-sdk/05-cli-tools.md)，跟踪[路由器如何选择 worker](./04-kv-cache-system/03-kv-aware-router.md)。
