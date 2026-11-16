---
title: "沿 vLLM 源码跟踪一次 prefill 交接"
weight: 1
---

本篇和[下一篇](/mooncake/12-code-walkthrough/02-decode-request-trace/)追踪同一条请求：假设一个普通全注意力模型的提示已经切成 48 个 token，每块 16 个，TP=1、PP=1，只有一个 KV group。P 上已计算的三个物理块号是 `[7,11,12]`；D 本地已经命中第一块，给后两块分配 `[31,32]`。这些编号是帮助走读的例子，真实运行时从两边的 metadata 取得。

这里跟踪 vLLM `v0.30.0` 的 `MooncakeConnector`，Mooncake 安装和 P/D 入口参考 `v0.3.14` 指南。后者的 V1 安装说明使用“latest”，并未给出这两个版本与任意驱动组合的实测兼容承诺。先确认 vLLM、Mooncake wheel、CUDA/驱动和目标 GPU 匹配，再在真实环境测试。代码阅读不需要 GPU。[P/D 指南](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/disagg-prefill-decode.md) · [connector 固定源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)

## 从 HTTP 请求跟进 P

代理在启动时读取 P bootstrap 的 `/query`，建立 `dp_rank → engine_id` 映射。一条外部请求进入 `_handle_completions` 后，代理自己生成 UUID。发给 P 的请求副本将 `stream` 关掉，把 `max_tokens` 限为 1，并附上 `do_remote_decode=True` 与 `transfer_id="xfer-<UUID>"`。另一条给 D 的请求携带同一个 transfer ID、P 的 engine ID 和 bootstrap 地址。

代理用 `asyncio.create_task` 发起 P 的 HTTP 请求，同时准备 D 的响应流。它没有从 P 的 HTTP 响应体提取 KV 或块号，最终客户端收到的响应来自 D。双方各自的内部 request ID 用于本地记账，跨端对齐用 transfer ID。[代理两条 HTTP 路径](https://github.com/vllm-project/vllm/blob/v0.30.0/examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py)

{{< trace >}}
title: "同一条交接从 HTTP 请求走到源块"
actors: ["代理", "P 调度器", "P worker", "D worker"]
steps:
  - from: 0
    to: 1
    label: "标记请求"
    detail: "代理为这次交接生成 transfer ID，让 P 保留相应的发送信息。"
  - from: 1
    to: 2
    label: "留住源块"
    detail: "P 完成 prefill 后，把源块及请求状态交给 worker；发送源在传输中保持有效。"
  - from: 3
    to: 2
    label: "提供目标"
    detail: "D worker 携带同一 transfer ID，告知 P 已分配的目的块。"
  - from: 2
    to: 3
    label: "发出 KV"
    detail: "P 对齐源、目的范围，通过引擎写入 D。"
  - from: 2
    to: 1
    label: "报告完成"
    detail: "发送侧完成后，P 可按引擎的生命周期释放相应的源资源。"
{{< /trace >}}

图中控制消息可以先后互换。D 的目的块先到时，P 可以先建立 `ready` 未置位的发送记录；源块先到时则等 D 的写入请求。重要的是传输开始前两个条件都已满足。

## P 为什么不在模型计算完就释放块

| 跟踪位置 | 在 48-token 算例中观察的字段 |
|---|---|
| `MooncakeConnectorScheduler.update_state_after_alloc` | 对 `do_remote_decode` 的请求，先将 `(request, [])` 加入 `_reqs_need_send`。这里还没有可以发送的块号。 |
| `build_connector_meta`、`record_send_reqs` | 当前调度步的 `reqs_to_send` 发给 worker，worker 用 transfer ID 建立 `SendBlockMeta`，其中有 `ready`、`local_block_ids`、`p_req_id`。 |
| `request_finished` | P 端请求因 `FINISHED_LENGTH_CAPPED` 结束且有非空块时，登记源块并返回 `(True, None)`。第一个值要求引擎延迟释放，第二个值表明没有要塞进 HTTP 响应的交接数据。 |
| 下一步的 `record_send_reqs` | 写入内部 request ID 和各组块表，在本例中为 `[[7,11,12]]`，设置到期时间，调用 `ready.set()`。 |
| `send_kv_to_decode`、`_send_blocks` | 按 D 提交的目的分组与源分组配对，后台调用 `batch_transfer_sync_write`。只有对应写入成功、所需发送数完成后，正常路径才把 P 的 ID 加入 `finished_sending_reqs`。 |
| `get_finished` | 返回该 worker 的已发送与已接收集合，引擎按完成报告回收资源。异常、中止及等待超时另有清理路径，清理了资源并不自动表示 D 收到了有效数据。 |

`send_meta.sent` 计数的是发给配对对端的完成任务。单卡例子里只需一份；TP 与 PP 改变时还要看 `resolve_need_send`、层和 rank 的匹配，不能用“看到了第一次发送”作为整条请求完成条件。

## 在固定源码中定位

```bash
git clone --depth 1 --branch v0.30.0 https://github.com/vllm-project/vllm.git
cd vllm
proxy=examples/disaggregated/mooncake_connector/mooncake_connector_proxy.py
conn=vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
rg -n 'dp_engine_id|def _handle_completions|xfer-|create_task' "$proxy"
rg -n 'class SendBlockMeta|def (request_finished|record_send_reqs|send_kv_to_decode)|ready.set|finished_sending_reqs.add' "$conn"
```

在支持的部署上，用 vLLM 的 debug 日志对照 P 的 `request_finished`、D 的 `update_state_after_alloc`、各 rank 的 `get_finished`。在 P 的模型服务端口和 bootstrap 端口分别检查 `/health` 与 `/query`。一条“缺 transfer ID”的普通请求也可能在 D 本地重算并产出文本；确认是直传需要把双方的 transfer ID、目标范围和完成结果连起来。[D 端怎样消费这些块](/mooncake/12-code-walkthrough/02-decode-request-trace/)
