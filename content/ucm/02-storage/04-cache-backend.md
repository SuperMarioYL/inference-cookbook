---
title: "Host Cache 在读取和写回时的职责"
weight: 4
---

在 `Cache|Posix` 中，Cache 层位于设备与 POSIX 后端之间。读取时，Host 层已有的数据可以直接送到设备；缺失的数据还要等待后端读取。保存时，设备缓冲必须在相应复制结束后才可复用。[管线用法](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/pipeline_store.md)

|调试入口|观察内容|
|---|---|
|[CacheStore](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/cache_store.cc)|查询、加载和写出如何委派给具体组件|
|[load 队列](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/load_queue.cc)|哪些 shard 已在本层，哪些仍需等待后端|
|[dump 队列](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/dump_queue.cc)|写入和缓冲释放的顺序|
|[传输缓冲](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/trans_buffer.cc)|任务操作的主机缓冲及生命周期|

调优时重复同一请求三次并按时间记录设备、Host、文件后端命中。第一次一般写外部存储，后面两次可能在不同层已命中。只比较总 TTFT 很难知道 Host 缓冲究竟省去了哪次磁盘读取。应结合[后端加载份额](../05-shared-infrastructure/02-metrics-observability.md)解释数据。

## 谁发起下层读，谁等同一份数据

`LoadQueue::DispatchOneTask` 将一次 load 按 shard 展开。对每个 shard，从 `TransBuffer` 取出句柄：缓冲已就绪可以进入 H2D；尚未就绪且本任务是 owner 时，才提交从后端填充这块 Host 缓冲的请求。其他使用同一份缓冲的任务可以等待相应的 ready 状态。随后 `TransferOneTask` 处理等待和 H2D。即使上层只有一个 block，内部也可能产生多个 shard，指标分母必须与观察的层一致。[加载队列](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/load_queue.cc)

{{< trace >}}
title: "两个加载任务同时需要同一份 shard"
actors: ["任务 A", "Host 缓冲", "后端", "任务 B"]
steps:
  - from: 0
    to: 1
    label: "获取句柄"
    detail: "A 发现缓冲尚未就绪，取得当前填充责任。"
  - from: 1
    to: 2
    label: "发起读取"
    detail: "A 的路径为这个 shard 提交一次后端读取。"
  - from: 3
    to: 1
    label: "等待同一缓冲"
    detail: "B 也需要该数据；遇到已有的填充过程，等待对应句柄就绪。"
  - from: 2
    to: 1
    label: "填充完成"
    detail: "数据到达 Host 后，等待方才能继续各自的 H2D。"
  - from: 1
    to: 3
    label: "继续复制"
    detail: "B 从就绪的 Host 缓冲取数，具备复制到其目的块的条件。"
{{< /trace >}}

## 写回端的两个队列

`DumpQueue` 接收任务后由 dispatch 线程处理设备到 Host 的复制；对带 prerequisite 的任务，先确认源设备的计算已到达指定事件。按 shard 获取缓冲后，已经形成的后端描述会送往 `BackendDumpStage`。由此可以分别计时队列等待、等待前置事件、准备缓冲、传输和下层持久化。一次 `dump` 提交很快，只能说明调用者迅速排入队列。[写出队列](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/dump_queue.cc)

## 看指标时先找读的是哪一份

例如统计窗口中本层处理 8 个 shard，其中 2 个需要等待后端，分层指标可以估计其中 6 个由 Cache 层直接供应。它是这组已发生加载的来源分布；只有再结合外部总命中率，才能讨论整个请求流里 Host cache 的贡献。增大 `cache_buffer_capacity_gb` 前，先检查缓存容量是否真的限制了热工作集，以及多个 DP 实例在主机上共占多少 Host 内存。
