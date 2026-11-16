---
title: "D 节点怎样接收并消费 KV"
weight: 2
---

沿用[P 端的 48-token 提示](/mooncake/12-code-walkthrough/01-prefill-request-trace/)。单组、普通全注意力、TP=PP=1，每块 16 个 token。P 的三块为 `[7,11,12]`，D 的本地前缀已覆盖第一块。D 为外部数据分配的两块是 `[31,32]`。D 要把“这个请求应有的前缀”“自己的目的槽位”“哪些字节真的到了”依次对应起来。

## 从 D 的调度线程到接收线程

| vLLM v0.30.0 入口 | 状态如何改变 |
|---|---|
| `get_num_new_matched_tokens` | `do_remote_prefill` 为真时，普通全注意力路径按 prompt 长度减去本地已算 token 数。本例给 connector 的两个数是 48 和 16，产生 `(32, True)`。True 表示异步接收，此时没有发生网络写入。 |
| `update_state_after_alloc` | 只有 `remote_engine_id`、`remote_bootstrap_addr`、`transfer_id` 都存在时，才记录分组的未命中目标块。将 `do_remote_prefill` 置为 False，防止下一步再次登记同一接收任务。 |
| `build_connector_meta` | 将请求放入 `MooncakeConnectorMetadata.reqs_to_recv`，按 P 引擎组织；其中 `PullReqMeta` 保存 D 的 request ID、跨端 transfer ID 和 `local_block_ids=[[31,32]]`。 |
| `start_load_kv` | 在接收事件循环提交协程。第一次遇到某个 P 引擎时连接 bootstrap `/query` 发现 worker，之后可复用已知地址。 |
| `receive_kv`、`MooncakeXferMetadata` | D 给配对的 P worker 发送自身地址、rank、分组块号、已注册区域基址与跨度，供 P 按布局写入这些范围。 |
| `process_pulling_result`、`get_finished` | 正常响应扣减 `pull_tasks_count`，减到 0 才报告本 worker 收完。失败路径也会唤醒调度器，还要查加载错误块。 |

匹配数是 connector 回答调度器的候选数，真实调度还有序列末尾及引擎计算的约束。Mamba/GDN 等模型对最后一个 token、状态槽与分组有专门处理，这些路径按各自的状态与缓存组计算可用长度。[调度与 worker 源码](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py)

## 手算一次地址映射

P 先检查 group 数，随后按 group 对齐源块和目标块。本例中 D 命中一个本地前缀，P 对其源块表取后缀，得到 `11→31`、`12→32`。如果 P 的源块少于 D 所要求的块数，代码报告错误。

对某一注册区域，假设源基址为 `0x1000`，目的基址为 `0x8000`，物理块跨度和单块传输长度均为 256 字节，无额外 rank 偏移、padding 或 kernel block 拆分。下例只做地址算术，可在 CPU 运行。

```python
source = [7, 11, 12]
destination = [31, 32]
stride = 256
copied = list(zip(source[-len(destination):], destination))
regions = [(0x1000 + p * stride, 0x8000 + d * stride, stride) for p, d in copied]
assert regions == [(0x1B00, 0x9F00, 256), (0x1C00, 0xA000, 256)]
assert regions[0][0] + stride == regions[1][0]
assert regions[0][1] + stride == regions[1][1]
print("合并后的源、目的和字节数", hex(regions[0][0]), hex(regions[0][1]), 2 * stride)
```

源、目标都连续且复制完整块，可以生成 `(0x1b00, 0x9f00, 512)` 描述符。若 D 的块是 `[31,34]`，目的地中间有空洞，两段要分别提交。若 TP 规模不同，某个 rank 可能仅写部分 KV 头，即使块号相邻也不能跨过不属于自己的区间。源码的 `_align_transfer_regions`、`_compute_sender_transfer_plan` 和 `_can_coalesce_block_transfers` 分别处理区域匹配、每个 rank 传的字节段和能否合并。

## 什么时候可以消费目标块

{{< mechanism "mooncake-pd-slots" >}}

`get_finished` 返回二元组，先是发送完成、随后是接收完成。接收任务失败时，`_handle_failed_recv` 记录失效目标块，并让等待这些块的请求退出等待。读取完成集合时，同时检查 `get_block_ids_with_load_errors`、错误日志和上层调度行为。若 D 本地前缀全部命中，零目标块的通知仍可用来释放 P 的源资源。

## 定位及验证

在上一章检出的 vLLM 源码根目录执行：

```bash
conn=vllm/distributed/kv_transfer/kv_connector/v1/mooncake/mooncake_connector.py
rg -n 'req_blocks|def (_build_transfer_params|receive_kv|process_pulling_result|_handle_failed_recv)|get_block_ids_with_load_errors' "$conn"
rg -n 'def test_(build_transfer_params_multi_group_trimming|build_transfer_params_group_count_mismatch|request_finished_with_hma_groups)' tests/v1/kv_connector/unit/test_mooncake_connector_hma.py
```

准备两条共享长手册、问题不同的请求，用确定性采样对照完整 prefill，再按单条 transfer ID 对照 P 的源、D 的目的块、传输返回值与 D 的后续 decode。只见到文本仍不能区分交接与本地重算。上面的 CPU 算例可核对分组原则，跨卡带宽、时延和 GPU 输出则需要实际 P/D 部署。[分组裁剪与错误测试](https://github.com/vllm-project/vllm/blob/v0.30.0/tests/v1/kv_connector/unit/test_mooncake_connector_hma.py) · [部署入口](/mooncake/08-integration/01-vllm-integration/)
