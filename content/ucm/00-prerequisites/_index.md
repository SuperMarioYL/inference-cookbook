---
title: "前置知识"
linkTitle: "前置知识"
weight: 10
sidebar:
  order: 10
---

顺着一次长提示的 prefill，先计算 KV 占用，再看引擎怎样为每条请求分配设备块。理解逻辑块和物理块的分工后，外部回填的目标就清楚了。[vLLM 中的一个调度步](./03-vllm-architecture.md)
