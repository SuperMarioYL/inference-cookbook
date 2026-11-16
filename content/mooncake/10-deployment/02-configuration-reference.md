---
title: "配置分别由谁读取"
weight: 2
---

同一个部署里至少有三类配置读取者。Master 从自己的命令行读服务与驱逐设置，Store 客户端在 setup 时读网络和容量，vLLM 的共享池与 P/D 直传又有两个独立 connector。把值填在实际消费它的进程上，才能从启动日志看出设置是否生效。

## 先确定两个地址与三类端口

| 谁监听 | 设置与有效默认值 | 对端怎样使用 |
|---|---|---|
| Store Master RPC | `--rpc_port`，最终默认 50051 | Python `master_server_addr`，服务或 vLLM JSON 的 `master_server_address` |
| Master 指标 HTTP | `--metrics_port=9003` | `curl -fsS http://127.0.0.1:9003/metrics/summary` 检查控制面 |
| 可选的传输元数据 HTTP | `--enable_http_metadata_server=true --http_metadata_server_port=8080` | 客户端 `metadata_server="http://HOST:8080/metadata"`；使用 `P2PHANDSHAKE` 时去掉此项 |
| Store Python 服务 REST | `python -m mooncake.mooncake_store_service --port=8081` | [部署实验](/mooncake/10-deployment/01-deployment-guide/)在 8081 写入与读回文本 |
| vLLM 直传 P bootstrap | `VLLM_MOONCAKE_BOOTSTRAP_PORT`，示例默认 8998 | 代理对 P 查询 `/query`，将地址传给 D；P/D 的 OpenAI HTTP 端口由 `vllm serve --port` 决定 |

例如 Master 与 Python REST 服务同机且都想使用 8080 时会抢同一监听端口。示例把 REST 显式设为 8081。Master 地址、传输元数据地址和对外的模型 HTTP 地址可以指向同一主机，但调用的是不同协议。[Master 与客户端参数](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/mooncake-store-deployment-guide.md) · [直传 bootstrap](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/disagg-prefill-decode.md)

## 怎样从环境生成这份客户端配置

[部署实验](/mooncake/10-deployment/01-deployment-guide/)使用环境变量。要改用文件，可把同样设置保存为 `store.json`，停掉旧服务，再运行 `python -m mooncake.mooncake_store_service --config store.json --port=8081`。这份配置只适用于单机 TCP 示例。

```json
{
  "local_hostname": "127.0.0.1",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "127.0.0.1:50051",
  "protocol": "tcp",
  "device_name": "",
  "global_segment_size": 134217728,
  "local_buffer_size": 67108864,
  "tenant_id": "default"
}
```

`global_segment_size` 是进程贡献给对象池的字节数；`local_buffer_size` 为该进程准备传输缓冲。读客户端也需要工作空间。直接调用 `store.setup(...)` 时两项显式传入；独立 reader 可以贡献 0 字节。Python 服务的 `MooncakeConfig` 先使用 `--config`，未给这个参数时按 `MOONCAKE_CONFIG_PATH` 或 `MOONCAKE_*` 环境变量读取，最后应用 `-D key=value`。存在文件时不要假设普通环境变量会覆盖文件中的每个字段。[服务 CLI](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-wheel/mooncake/mooncake_store_service.py) · [MooncakeConfig](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/python/mooncake/mooncake_config.py)

## vLLM 的两个入口

`MooncakeStoreConnector` 的 v0.30.0 worker 自己解析 `MOONCAKE_CONFIG_PATH`。它的 `mode="embedded"` 要求 `global_segment_size > 0`，每个 rank 在进程内贡献容量；`mode="standalone-store"` 要求该字段为 0，外部进程提供池容量，两种模式都要求 `local_buffer_size > 0`。前述 JSON 在 Python 服务中绑定的 HTTP 端口不是 vLLM 的配置项。对照解析器时要同时检查 connector 名和 tag。[vLLM Store 配置](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py)

`MooncakeConnector` 使用 P/D 双方的 `--kv-transfer-config` 及代理传递的 `kv_transfer_params`。给 P 设置 `kv_role=kv_producer`，给 D 设置 `kv_role=kv_consumer`。配置再完整也不能代替每请求唯一的 `transfer_id`、已注册 P 的 `remote_engine_id` 和可达的 `remote_bootstrap_addr`。[跟踪一次直传](/mooncake/12-code-walkthrough/01-prefill-request-trace/)

## 容量紧张、SSD 与控制面恢复

| 现象或目标 | 核对设置及其后果 |
|---|---|
| Master 有空闲，某个租户写不进去 | 分开看全局 `--eviction_high_watermark_ratio` 与 `--tenant_eviction_high_watermark_ratio`，二者默认均为 0.90，后者作用于租户有效预算。也要确认请求的放置位置及是否存在可驱逐对象。 |
| 读取后的对象一时不能驱逐 | `--default_kv_lease_ttl` 默认 10,000 ms。读取成功会刷新租约；缩短它会改变对正在读的对象的保护窗口。 |
| 已设置 SSD 目录却没有落盘 | 对照 Master 的 `--enable_offload` 与持有方的 `MOONCAKE_OFFLOAD_ENABLED`、`MOONCAKE_OFFLOAD_FILE_STORAGE_PATH`。`--offload_on_evict` 将部分写入推迟到驱逐阶段；检查具体副本的存储类型。 |
| 磁盘对象读后没有回升到 DRAM | 按部署分别查 `--promotion_on_hit` 与 `--enable_ssd_prefetch`。读后提升和存在性检查触发的预取是不同路径。 |
| 需要 Master 故障转移 | 按已构建的 HA 后端与端点配置发现当前 leader。切换后检查恢复的元数据、对象所指的 segment 是否仍在线，以及先前写入的固定字节是否能读回。 |

这些值来自 Mooncake v0.3.14，配置模板不要跨版本照抄。更改租约、驱逐与 SSD 策略前，先在独立实验池用一个已知 key 观察查找结果、读回字节与副本类型。[版本参数表](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/mooncake-store-deployment-guide.md) · [SSD 预取语义](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/store/ssd-prefetch.md)
