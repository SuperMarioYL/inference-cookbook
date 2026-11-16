---
title: "NVIDIA Dynamo 1.6.2 阅读指南"
slug: "overview"
weight: 0
---

Dynamo 在多个推理 worker 前面提供统一服务入口。运行时发现可用实例，路由器选择目标，推理后端执行模型。本文以 **1.6.2** 为基线。该版本官方 CUDA 13 镜像分别采用 vLLM **0.31.0** 或 SGLang **0.5.21**。复现集成示例时，应使用 Dynamo 对应后端的镜像；独立项目栏目的新版有各自的接口与依赖。

1. [在本机启动 frontend 与 worker](/dynamo/06-python-sdk/05-cli-tools/)，查询模型并发一条聊天请求。
2. [区分发现、请求和事件三个平面](/dynamo/05-distributed-communication/02-three-plane-architecture/)。理解为什么单机快速开始可以使用文件发现。
3. [比较不同 worker 的前缀重叠和负载](/dynamo/04-kv-cache-system/03-kv-aware-router/)，再看[预填充与解码如何交接 KV](/dynamo/03-llm-inference-layer/05-disaggregated-serving/)。
4. 没有 GPU 可先用两个 Python 进程运行[已验证的 echo 服务与客户端](/dynamo/06-python-sdk/03-service-decorators/)入手，沿 client、runtime 和 handler 看边界。

集群部署还需要核对 Operator、资源定义、模型存储和设备网络。单机一次正确响应可以验证 HTTP 到执行后端的路径；P/D 实际传输能力与多节点表现需要在目标环境测量。

[单机快速开始](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/cli/getting-started/quickstart.mdx) · [镜像实际依赖](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/container/context.yaml) · [系统架构](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md)
