---
title: "从请求和块跟踪源码"
linkTitle: "代码链路分析"
weight: 12
description: >
  跟踪关键代码执行路径,理解请求处理细节
---

[P 端](./01-prefill-request-trace.md)和 [D 端](./02-decode-request-trace.md)沿 vLLM v0.30.0 的 Mooncake connector 追踪同一笔传输。[Store 生命周期](./03-kvcache-lifecycle.md)与 [Conductor 查询](./05-conductor-scheduling-trace.md)则沿 Mooncake v0.3.14 自身的源码展开。
