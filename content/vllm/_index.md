---
title: "vLLM"
linkTitle: "vLLM"
weight: 1
description: "vLLM V1 的请求调度、KV 块分配与模型执行。"
---

vLLM 0.33.0 把不同请求的 token 合在同一轮执行。可以先[启动 Qwen3-0.6B](./04-code-walkthrough/01-entry-points.md)，再沿[请求生命周期](./04-code-walkthrough/06-request-lifecycle.md)看引擎如何安排一次前向。两条请求共享前缀时，可对照[块池的引用计数](./03-core-modules/03-block-pool.md)。
