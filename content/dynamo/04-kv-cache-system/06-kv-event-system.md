---
title: "KV 事件如何更新路由视图"
weight: 6
---

worker 生成可复用 KV 或驱逐旧块时可以发布事件。路由器把事件用于索引，将“哪些前缀在哪个 worker 上可用”的估计提供给后续请求。事件传送的是元信息，GPU 上的 KV 张量保留在相应的引擎或存储层。

例如 P1 先报告创建前两块，随后报告驱逐第二块。在两条事件之间到达的请求可能看到不同的可复用长度。若事件流不可用，要先确定配置：官方说明中，显式 `--no-router-kv-events` 可从路由决策预测缓存；未提供所需事件且没有启用预测时，KV 模式可能只按负载评分。

1.6.2 事件平面默认 ZMQ，NATS 是配置选项。引擎自身的事件发布与 Dynamo 运行时分发可以有不同端点；诊断宜从 worker 发布、路由接入、索引更新到真实后端命中逐段核对。

## 创建与驱逐之间的一次查询

{{< trace >}}
title: 两个事件之间的一次路由查询
actors: ["P1", "事件通道", "索引", "路由器"]
steps:
  - from: 0
    to: 1
    label: "创建 K1、K2"
    detail: "P1 报告两个连续提示块的缓存。"
  - from: 1
    to: 2
    label: "记录两块"
    detail: "索引更新后，这一前缀在 P1 的重叠量为两块。"
  - from: 2
    to: 3
    label: "回答查询"
    detail: "新请求到达时，路由器用当时的视图估算预填充成本。"
  - from: 0
    to: 1
    label: "驱逐 K2"
    detail: "P1 后来回收第二块，发出相应事件。"
  - from: 1
    to: 2
    label: "只剩 K1"
    detail: "更新送达后，同一提示下次查询只剩一块重叠。"
{{< /trace >}}

## 区分候选重叠、实际命中和节省的计算

同一条 96 token 提示可以在路由器中看到 32 token 的重叠，但 worker 收到它之前正好驱逐了一个 16 token 块。路由选择使用发送当时的估计；最终复用量由目标 worker 确认。评估时对齐同一请求标识、worker 和观察窗口，避免将索引查询数量当作实际命中 token 数。

[事件与无事件模式](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/kv-aware-routing.md) · [路由配置](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/configuration-and-tuning.md) · [发布器](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/kv_router/publisher/mod.rs)
