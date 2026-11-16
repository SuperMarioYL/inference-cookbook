---
title: "UCM"
linkTitle: "UCM"
weight: 3
description: "UCM 0.9.1 的 KV 前缀缓存、分层存储、引擎集成与模型增强"
---

两位用户向同一份长手册提问，推理引擎可以把第一次算出的前缀 KV 写到外部存储，第二次只把命中的块读回设备，再处理新问题。UCM（Unified Cache Manager）为引擎提供这条读写路径，并按模型和引擎版本提供稀疏注意力等扩展。[v0.9.1 发布说明](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

从[两次请求的例子](./01-overview/01-introduction.md)开始，再跑 [vLLM 前缀复用](./01-overview/03-quick-start.md)。关注实现时，跟踪 [Connector](./04-vllm-integration/02-ucm-connector.md)和[存储管线](./02-storage/05-pipeline-composition.md)。部署 SGLang、vLLM-Ascend 或 MindIE 时，从[对应引擎的部署入口](./06-engineering-practice/01-deployment-guide.md)选择适配组合。
