---
title: "运行时与服务发现导读"
slug: "readme"
description: "运行时为异步操作、组件命名和成员发现提供共同基础。追查实例为什么不可见时，先辨明当前是 Kubernetes、文件还是 etcd 发现，再查相应后端。"
weight: -1
---

运行时为异步操作、组件命名和成员发现提供共同基础。追查实例为什么不可见时，先辨明当前是 Kubernetes、文件还是 etcd 发现，再查相应后端。

1. [Rust 异步编程基础](/dynamo/02-runtime-layer/01-async-programming-basics/)
2. [分布式系统基础](/dynamo/02-runtime-layer/02-distributed-system-basics/)
3. [Runtime 双层设计](/dynamo/02-runtime-layer/03-runtime-design/)
4. [Namespace、Component 与 Endpoint](/dynamo/02-runtime-layer/04-component-system/)
5. [Pipeline 架构](/dynamo/02-runtime-layer/05-pipeline-architecture/)
6. [服务发现怎样维护实例集合](/dynamo/02-runtime-layer/06-service-discovery/)

[上一章 · 系统架构与部署模式导读](/dynamo/01-architecture-overview/readme/) · [下一章 · 推理引擎与 worker 导读](/dynamo/03-llm-inference-layer/readme/)
