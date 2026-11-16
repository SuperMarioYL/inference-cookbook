---
title: "聚合、P/D 分离与部署环境"
weight: 5
---

部署要回答两个独立问题：一条请求由多少类 worker 执行，实例又通过什么环境发现彼此。单机也能演示分离，Kubernetes 也能运行聚合 worker。

| 选择 | 资源和数据如何流动 | 验证点 |
|---|---|---|
| 聚合 | 被选中的 worker 完成 prefill 和 decode，同一模型执行环境持有 KV。 | 先打通 frontend → worker → HTTP 响应。 |
| P/D 分离 | P 计算提示词并给出后端相关的交接信息，D 获取 KV 后续写。两池可以分别扩缩。 | 观察同一请求的 P、D 记录、传输完成和第一个解码输出。 |
| 本地文件发现 | 同机组件读写同一发现位置。 | 两端使用 `--discovery-backend file`，能看到同一实例。 |
| Kubernetes | Operator 利用 Kubernetes 资源管理实例发现与生命周期。 | 同时查看 Pod 就绪、Dynamo 端点注册、模型是否出现在前端。 |

从[单机聚合示例](/dynamo/06-python-sdk/05-cli-tools/)起步，再决定是否需要拆开 P 和 D。长提示词挤压 decode、两阶段资源比差异较大时分离值得测量；交接 KV 需要额外设备、带宽和协调时间。应在相同的输入长度、并发和模型精度下比较 TTFT 与逐 token 延迟。

## 比较请求路径与扩容单位

| 问题 | 聚合 worker | 分离的 P 与 D |
|---|---|---|
| 增加副本后谁能使用新资源？ | 新副本可以处理完整请求。 | 取决于新增的是预填充还是解码，另一个池仍可能是瓶颈。 |
| 相同前缀能在哪里复用？ | 在被选中的 worker 上查询本地缓存。 | P 和 D 可使用各自可见状态，数据交接还取决于后端。 |
| 延迟应拆成哪些部分？ | 排队、预填充、逐步解码。 | 两池各自的排队、工作及中间的传输和同步。 |

## 对照官方启动脚本

聚合脚本启动 frontend 与一个 worker。再看分离脚本，按 P、D 对照设备、`disaggregation-mode`、`kv_role`、事件端口与侧信道端口。从同一个请求 ID 找出 P 完成时间、D 的 KV 就绪时间和首个输出，就能量出交接阶段。

[部署与发现](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md) · [聚合示例](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/agg.sh) · [分离示例](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh)
