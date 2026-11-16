---
title: "一份 KV 缓存从生成到驱逐"
weight: 3
---

前两篇跟踪 P/D 直传：P 持有源块 `[7,11,12]`，D 为未命中部分分配 `[31,32]`。P 上注册的内存由 vLLM 的发送完成报告解除占用。如果部署选用 `MooncakeStoreConnector` 来跨请求复用同一份 48-token 手册，就出现第三位持有者：Store 中可按 key 读取的副本。P 的 GPU 地址、Store 对象名和 D 的块编号分别属于不同的生命周期。

## 三种标识如何对应

| 标识 | 谁持有 | 在本例中意味着什么 |
|---|---|---|
| 请求与块号 | vLLM scheduler、block pool | `[7,11,12]` 可在请求结束后分给其他 token，不能用作跨请求对象名。 |
| 池 key | `ChunkedTokenDatabase`、`StoreLayout` | rank-local 布局组合模型、配置的 namespace、并行 rank、group 和块 hash 等。跨 TP 的布局有自己的规则，同一份文本未必有跨布局通用的 key。 |
| 副本与地址 | Mooncake Master 元数据、segment 所有者 | Master 返回可读副本描述，字节由持有者保存。驱逐一个副本时，其他合格副本仍可提供同一对象。 |

[分组、rank 与 key 结构](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/data.py)

## 从计算完成到 Store 副本可读

下图只跟踪一个写入 DRAM 的候选副本，省略 SSD 与复制旁路。一个对象可以同时有已完成的旧副本与正在创建的新副本。

{{< stateboard >}}
title: "同一份 KV 在设备、Store 和下一次请求之间"
rows: ["最初设备", "Store 副本", "另一实例"]
steps:
  - title: "P 计算完成"
    detail: "本次请求在设备块中有 KV，对象尚未发布到共享池。"
    cells:
      - ["有 KV"]
      - ["·"]
      - ["等待"]
    focus: [[0, 0]]
  - title: "写入处理中"
    detail: "Store 为对象安排副本，数据正在搬运。"
    cells:
      - ["有 KV"]
      - ["写入中"]
      - ["等待"]
    focus: [[1, 0]]
  - title: "发布完成"
    detail: "写入结束后 Master 返回可读的副本位置。"
    cells:
      - ["可释放"]
      - ["可读"]
      - ["查到"]
    focus: [[1, 0], [2, 0]]
  - title: "读入 D"
    detail: "接收实例分配自己的设备块并完成回填，随后用它执行模型。"
    cells:
      - ["·"]
      - ["可读"]
      - ["能使用"]
    focus: [[1, 0], [2, 0]]
{{< /stateboard >}}

1. **vLLM 选择要保存的块。** `MooncakeStoreScheduler.build_connector_meta` 将可保存 token 范围和对应块表交给 worker。算完一个 decode token 之后，worker 依据保存边界、角色和 `save_decode_cache` 决定哪些完整块可以写出。
2. **worker 发起保存。** `MooncakeStoreWorker.wait_for_save` 根据 metadata 向发送线程提交工作；线程根据 key、布局和来源区域构造多缓冲写入。`batch_put_from_multi_buffers` 返回各项结果。正在写的源块不能提前拿给其他请求使用。
3. **Master 分配与发布。** `PutStart` 分配副本位置，`PROCESSING` 副本不能当完整内容交给读者。数据路径成功后 `PutEnd` 将这次提交的目标副本转为 `COMPLETE`；失败可以沿 `PutRevoke` 撤销未完成写入。Master 知道地址与状态，字节没有通过 Master RPC 转发。
4. **另一个请求查同一块。** `GetReplicaList` 查 tenant 与 key 对应的对象，筛出可读副本。没有对象时返回 `OBJECT_NOT_FOUND`；有元数据但没有已完成的可读副本时可返回 `REPLICA_IS_NOT_READY`。成功取得描述会刷新对象的读租约。

[Store scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/scheduler.py) · [读写线程](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store/worker.py) · [Master 发布与租约](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-store/src/master_service.cpp)

## 一次命中怎样成为可用的 GPU KV

假设第二个请求本地已有 16 个 token，Store 查到的连续前缀有 48 个。scheduler 的 `get_num_new_matched_tokens` 算出待载入量 `48−16=32`。启用异步查找且本轮没有结果时，函数可返回 `(None, False)`，等待下一调度步；它与明确的 0 命中不同。

命中后，scheduler 保留 `LoadSpec`：`vllm_cached_tokens=16`、`kvpool_cached_tokens=48`、`can_load=False`。确认目标块已分配且数量与计划一致后，`update_state_after_alloc` 才把 `can_load` 打开。worker 的接收线程按分组 key 与目标 buffer 调用 `batch_get_into_multi_buffers`。因此应分别记录“命中 token 数”“分配了哪些目标块”“传输是否完成”，然后观察请求是否跳过对应 prefill。这里假设了相同模型、token 化结果、兼容布局与命名空间。

## 副本为什么消失

持有者退出后，其 DRAM segment 可能离场；空间压力也会触发驱逐。Master 会考虑对象是否正被处理、是否有有效租约或其他保护。租约到期表示保护窗口结束，对象可能继续留在原位。启用 SSD 时，某个对象可能只剩磁盘副本，读回和后台提升到 DRAM 可以是两个过程；存在性探测不会证明数据已到 GPU。[SSD 预取流程](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/store/ssd-prefetch.md)

用[三进程部署](/mooncake/10-deployment/01-deployment-guide/)独立验证持有者关系：让唯一贡献容量的服务保存 `manual-v1`，由零贡献读者读回 `shared-manual`。在专用实验池中停止贡献进程，观察 Master 仍可达，等待 segment 离场后再次查询该 key。若还有副本或 SSD，应继续检查剩余描述和实际返回内容。用 GPU 跑同一组 token 前缀时，再观察引擎记录的外部命中与读回量。

在两个项目各自的固定 tag 根目录，可用以下命令定位实现：

```bash
# vLLM v0.30.0
store_src=vllm/distributed/kv_transfer/kv_connector/v1/mooncake/store
rg -n 'class (PoolKey|RankLocalStoreLayout|ChunkedTokenDatabase|LoadSpec)' "$store_src/data.py"
rg -n 'def (get_num_new_matched_tokens|update_state_after_alloc|wait_for_save|get_finished)|batch_(put_from|get_into)_multi_buffers' "$store_src"
```

```bash
# Mooncake v0.3.14
rg -n 'MasterService::(PutStart|PutEnd|PutRevoke|GetReplicaList|ReadReplicaList)|GrantReadLease' mooncake-store/src/master_service.cpp
```
