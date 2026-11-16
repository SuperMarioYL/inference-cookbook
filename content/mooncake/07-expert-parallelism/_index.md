---
title: "专家并行与 Process Group"
linkTitle: "Expert Parallelism"
weight: 7
description: >
  MoE 模型 Expert 并行化支持
---

MoE 的 dispatch 把 token 分发到目标专家，combine 把结果归并回来。Mooncake EP、PG 与一般 KV 跨机传输复用部分底层通信能力，但调用者、分组和完成条件不同。v0.3.14 新增 EP ElasticBuffer 的 NCCL Device API 后端，并调整了 PG 的 JIT 构建方式。[版本说明](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14) · [PG 设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/mooncake-backend-pg.md)
