---
title: "TCP 请求通道与 KV 传输"
weight: 6
---

默认部署下，TCP 请求通道承担 Dynamo 组件之间的 RPC。客户端从发现层获得 endpoint 的实例地址，运行时通过直连连接传递请求。响应流携带生成结果或错误，路由器和客户端按其语义处理。

在 vLLM 分离样例里同时出现 TCP 地址、KV 事件和 NixlConnector。TCP 可能承载组件调用，ZMQ 发布者也可能使用 tcp 地址，而 NIXL 独立承担内存传输。遇到 D 等待交接，查看连接器的完成状态和两端传输描述，确认哪一块还在途中。

## HTTP 流与组件间的 RPC

{{< trace >}}
title: 内部 TCP 流如何写回 HTTP
actors: ["客户端", "Frontend", "Worker"]
steps:
  - from: 0
    to: 1
    label: "提交流式请求"
    detail: "客户端建立 HTTP 请求并要求流式响应。"
  - from: 1
    to: 2
    label: "发起内部 RPC"
    detail: "前端通过请求通道把输入交给已发现的实例。"
  - from: 2
    to: 1
    label: "第一项输出"
    detail: "worker 送回一项可用于生成响应的数据。"
  - from: 1
    to: 0
    label: "写入 SSE"
    detail: "前端把这部分结果写到对应的外部连接。"
  - from: 2
    to: 1
    label: "内部流结束"
    detail: "worker 结束该次调用，前端完成外部响应。"
{{< /trace >}}

客户端看到的是前端输出的 HTTP 数据。组件 RPC 已完成连接时，生成还可能在等待模型、排队或交接。可先发短的非流式请求验证一次完整调用，再用流式请求比较首字节到达与末尾结束事件。

## 阅读实现时检查的边界

从运行时配置确认使用 TCP；观察 endpoint 广告的地址、目的实例、建立连接或复用连接、请求流关闭的时间。若换成 NATS 请求传输，这些连接细节会改变；上层模型请求和响应契约仍然需要验证。

[通信平面](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md) · [TCP 传输实现](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/runtime/src/transports/tcp.rs) · [分离部署配置](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh)
