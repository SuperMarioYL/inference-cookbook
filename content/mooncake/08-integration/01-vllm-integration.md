---
title: "vLLM 的直传与共享 Store 两种接法"
weight: 1
---

`MooncakeConnector` 把同一个请求的 KV 在 prefill 和 decode 两端直接交接。`MooncakeStoreConnector` 把 KV 放进共享存储，供引擎按前缀重新读取。这两个名字分别选择不同的数据路径。[路径对照](../02-architecture/02-request-lifecycle.md)

## 在两台主机间直传

以下片段采用 v0.3.14 集成指南中的 vLLM V1 接口。两台机器需安装支持该 connector 的同一 vLLM 版本、匹配设备的 Mooncake 包和同一模型。下面的命令分别在 P 和 D 主机执行。

```bash
# P 主机
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8010 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
# D 主机
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8020 \
  --kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_consumer"}'
```

路由器负责生成 `transfer_id` 并把 P 的 bootstrap 地址带给 D。用与已安装 vLLM 相同版本的源码 checkout，在该目录启动配套示例代理。以下 `P_HOST` 和 `D_HOST` 为两台可达主机的地址，8998 为 P 的默认 bootstrap 端口。

```bash
python examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py \
  --port 8000 --prefill "http://${P_HOST}:8010" 8998 --decode "http://${D_HOST}:8020"
```

向代理 8000 端口发请求，并核对两端实际传输记录。只看到正确生成文本时，还要排除 D 端重算。[上游直传部署与代理约定](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/disagg-prefill-decode.md)

## 由 Store 共享前缀

先用 [put/get](../00-beginner-guide/03-quick-start-tutorial.md)验证 Store 并保证池中有节点贡献空间。按固定版配置填写 Master 地址、元数据握手方式、所用网卡和 buffer 容量。已安装该 connector 的 vLLM 实例可用 `kv_both` 读写同一池。

```bash
export MOONCAKE_CONFIG_PATH="$PWD/mooncake_config.json"
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --kv-transfer-config '{"kv_connector":"MooncakeStoreConnector","kv_role":"kv_both"}'
```

用一个实例预热，再让另一个实例读取相同长前缀。检查外部命中块和网卡或读回字节；同实例再次发送相同请求还可能命中显存中的前缀缓存。[Store 配置与生产者、消费者示例](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/kv-cache-storage.md)

如果关心引擎如何决定哪些 token 可以跳过计算，可以从 [vLLM 的 KV 缓存查询](/vllm/03-core-modules/02-kv-cache-manager/) 读起，再对照 [UCM 的外部前缀缓存示例](/ucm/01-overview/03-quick-start/) 看另一种连接方式。
