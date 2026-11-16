---
title: "Inference Cookbook"
description: "中文 LLM 推理手册。从一个张量操作读到注意力内核、请求调度和跨机器的 KV 缓存。"
lead: "从一个张量操作，到一条请求跨机器完成推理。"
routes:
  - title: "从算子到推理引擎"
    projects:
      - name: PyTorch
        page: /pytorch/00-overview/01-architecture-overview
        detail: "沿着一次张量操作找算子，再看同一段 Python 函数何时需要重新编译。"
      - name: FlashAttention
        page: /flashattention/00-overview/overview
        detail: "手算两个分块的注意力输出，再把中间结果对应到 GPU 内核。"
      - name: vLLM
        page: /vllm/00-overview/overview
        detail: "从服务入口跟踪一个请求，观察调度器怎样分配 KV 块、生成下一个 token。"
      - name: SGLang
        page: /sglang/00-overview/overview
        detail: "发送两条开头相同的请求，看它们如何共享前缀，又在哪里分叉。"
  - title: "当 KV 跨设备移动"
    projects:
      - name: NVIDIA Dynamo
        page: /dynamo/00-overview/overview
        detail: "分别启动 frontend 和 worker，跟踪路由器怎样为请求选择后端。"
      - name: Mooncake
        page: /mooncake/00-beginner-guide/01-what-is-mooncake
        detail: "对照预填充端直传 KV 与读端从 Store 加载 KV 的两条路径。"
      - name: UCM
        page: /ucm/01-overview/01-introduction
        detail: "跟踪两次共享前缀的推理，看缓存块如何写入存储、加载回设备。"
      - name: CacheBlend
        page: /cacheblend/00-overview/overview
        detail: "给一份已缓存的手册换个位置，看哪些计算仍能复用。"
---

## 从一张图开始

- [把 2 × 3 的张量转置，六个数还在原处](/pytorch/01-introduction/01-tensor-basics-tutorial/#fig-torch-strides)
- [沿时间轴看 A 结束后，D 什么时候补入批次](/vllm/01-introduction/01-why-vllm/#fig-连续批处理让-de-补入空位)
- [比较四个 token 的 KV 偏差，选出重算位置](/cacheblend/05-algorithms/02-hkvd-selection/#fig-cacheblend-hkvd-bars)
- [跟着 KV 从预填充端的源块走到解码端的目标块](/mooncake/12-code-walkthrough/02-decode-request-trace/#fig-mooncake-pd-slots)
