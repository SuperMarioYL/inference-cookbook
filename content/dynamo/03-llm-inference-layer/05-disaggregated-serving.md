---
title: "P/D 分离如何交接 KV"
weight: 5
---

在分离服务里，prefill worker 用提示词建立 KV 状态，decode worker 使用相同上下文继续生成。两者可以位于不同 GPU；显存地址只在各自的进程和设备上有意义，需后端协议把远端状态与本地槽位对应起来。

以 Dynamo 1.6.2 的 vLLM 双 GPU 启动脚本为例，P 配置 `NixlConnector` 与 `kv_producer`，D 配置 `kv_consumer`。P 处理请求后返回含块与远端连接信息的交接元数据，D 根据这些信息协调数据传输。前端传递的是描述本次交接的信息，大块 KV 经连接器传输。D 使用已就绪的 KV 继续解码。两端模型和传输配置必须相容。

SGLang 的实现会提供 bootstrap 地址和 room 等信息，允许后台 prefill 与传输协调有不同的时序。阅读跨后端流程图时，应先注明后端，再判断哪一步等待什么。评估分离部署可同时记录 P 排队和计算、KV 交接等待、D 排队和输出间隔；总时延的变化未必来自 GPU 算力。

## 两个池各自可以推进哪些工作

{{< stateboard >}}
title: P 计算 B 时，D 可以生成 A
rows: ["Prefill 池", "交接中的 KV", "Decode 池"]
steps:
  - title: "A 进入 P"
    detail: "先处理请求 A 的提示，D 此时空闲。"
    cells: [["A"], ["·"], ["·"]]
    focus: [[0, 0]]
  - title: "A 交接"
    detail: "A 的前缀 KV 开始交接，P 可以在资源可用时接入 B。"
    cells: [["B"], ["A"], ["·"]]
    focus: [[0, 0], [1, 0]]
  - title: "两阶段并行"
    detail: "A 的 KV 已在 D 就绪。D 生成 A 的后续 token，P 继续预填充 B。"
    cells: [["B"], ["·"], ["A"]]
    focus: [[0, 0], [2, 0]]
  - title: "B 也完成"
    detail: "B 准备交接；D 上已有 A，B 的后续等待取决于 D 的容量。"
    cells: [["·"], ["B"], ["A"]]
    focus: [[1, 0], [2, 0]]
{{< /stateboard >}}

同时到达的两个长提示可能排在同一个 P 上，而多个 decode 请求可以在 D 上继续生成。两个池排队时间不同。即使总 GPU 数不变，P 处理得更快也可能迫使 D 积压；扩容时先看哪个阶段超出目标。

## 检查一次传输的双方配置

| 项目 | P 侧检查 | D 侧检查 |
|---|---|---|
| 进程与设备 | `--disaggregation-mode prefill` 对应的模型与设备 | `--disaggregation-mode decode` 对应的模型与设备 |
| 连接器 | 该示例的 `NixlConnector` 和 producer 角色 | consumer 角色、可达的远端连接信息 |
| 请求关联 | prefill 完成、传出元信息与请求标识 | 取到元信息，KV 可用后开始解码 |

处于解码状态却一直没有后续 token 时，可从 D 的等待点回查交接信息，区分元数据未到、传输未完成和模型前向慢。

要看一批 KV 在 P 和 D 之间实际去了哪里，可对照 [Mooncake 的直传与 Store 读写](/mooncake/02-architecture/02-request-lifecycle/)。两张图分别回答请求由谁执行、数据由谁交付。

[各后端交接契约](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md) · [vLLM 双 GPU 实例](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh)
