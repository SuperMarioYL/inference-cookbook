---
title: "P/D 分离中的 NIXL 传输"
weight: 5
---

预填充与解码位于不同 GPU 时，P 上形成的 KV 需要出现在 D 可以使用的内存中。Dynamo 传递这个请求的交接元信息；vLLM 的 `NixlConnector` 与 NIXL 在后端协调数据传输。单机文件发现、默认 TCP RPC 和 KV 传输解决的是不同问题。

## 一次交接经过哪些状态

{{< mechanism "dynamo-transfer-ready" >}}

两端使用相同模型和兼容的缓存布局，各自持有本地设备内存。交接参数标明远端引擎、连接和源块，D 预留目的槽；连接器报告传输就绪后，D 读取这部分 KV。

## 对照官方双 GPU 脚本

| 脚本中观察的内容 | 在现场核对什么 |
|---|---|
| P 的 `--disaggregation-mode prefill` 与 `kv_producer` | 使用哪一块 GPU、哪个模型、哪些块可以提供给远端。 |
| D 的 `--disaggregation-mode decode` 与 `kv_consumer` | 目标设备能否建立连接、收到与该请求对应的数据。 |
| `VLLM_NIXL_SIDE_CHANNEL_PORT` 与两侧系统端口 | 两端元信息能否互通，多个实例有无端口冲突。 |
| P 的 KV 事件配置 | 事件供路由和索引观察，日志中的“已发布”不能替代传输完成通知。 |

估算流量时，用本次迁移 token 数乘以当前布局下的每 token 字节数，再算上块对齐与元数据。对照连接器日志可找到实际使用的传输插件与路径。Dynamo 的 TCP 请求平面另行承载组件间 RPC。

[vLLM 双 GPU 启动脚本](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh) · [后端交接契约](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/disaggregated-serving.md) · [运行时通信平面](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md)
