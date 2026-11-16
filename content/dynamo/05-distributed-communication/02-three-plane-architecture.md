---
title: "发现、请求与事件三个通信平面"
weight: 2
---

同一组 frontend 和 worker 同时有三种流量。发现记录可能的通信对象，请求传递本次调用的数据，事件发布可独立消费的状态变化。分离部署的 KV 数据量通常远大于请求元信息，后端可以另行使用 NIXL 等传输路径。

| 平面 | 在 1.6.2 中怎样选择 | 看到故障时先查什么 |
|---|---|---|
| 发现 | 裸机通常使用 etcd，单机快速开始显式使用 file，Operator 环境使用 Kubernetes。 | 上下游是否处于同一命名空间、是否看到实例。 |
| 请求 | `DYN_REQUEST_PLANE` 默认 TCP，也可配置 NATS。 | worker 已注册时，调用能否连上其实际请求地址、是否有响应。 |
| 事件 | `DYN_EVENT_PLANE` 默认 ZMQ，也可配置 NATS。 | KV 事件发布者、订阅者和路由器索引是否同步。 |

假如请求可以正常生成，但路由多次选到没有相同缓存的 worker，先看事件与索引；如果 frontend 从来没有发现任何后端，先看发现层。更换 RPC 传输不会自动改变 KV 跨设备的传输方式。

## 如何跟踪一个请求穿过三个平面

{{< trace >}}
title: 注册、事件与 RPC 各传什么
actors: ["Worker", "发现后端", "路由器", "Frontend"]
steps:
  - from: 0
    to: 1
    label: "注册端点"
    detail: "Worker 发布自己能接收请求的地址。"
  - from: 1
    to: 2
    label: "成员变化"
    detail: "路由器看到这个模型的新候选实例。"
  - from: 0
    to: 2
    label: "发布 KV 事件"
    detail: "路由器从事件平面获知该实例可能拥有的前缀。"
  - from: 3
    to: 2
    label: "选择实例"
    detail: "前端提交请求，路由器结合候选与负载挑选目标。"
  - from: 3
    to: 0
    label: "发送推理 RPC"
    detail: "实际输入经请求平面交给被选中的 worker。"
  - from: 0
    to: 3
    label: "返回生成结果"
    detail: "输出沿请求的返回路径到达前端。"
{{< /trace >}}

发现后端短时不可用、事件晚到和 RPC 连接中断是三种不同故障。可以在相同请求标识下查看实例何时被发现、事件何时被摄入、实际转发给谁。这个顺序比只观察一个 `/health` 响应更能说明请求在哪一段停下。

[官方通信平面说明](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md) · [运行时配置](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/reference/components/runtime-configuration.mdx)
