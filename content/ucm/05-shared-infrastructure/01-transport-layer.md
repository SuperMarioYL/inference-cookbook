---
title: "一次设备与主机之间的 KV 复制"
weight: 1
---

以 `Cache|Posix` 回填为例，存储层先把数据读到主机缓冲，再将字节复制到 worker 已分配的设备块。在 v0.9.1，公共传输抽象位于 C++ 的 `ucm/shared/trans`。`Stream` 同时提供同步、异步和批量复制，以及事件等待与同步操作。[Stream 接口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/shared/trans/stream.h)

{{< trace >}}
title: "每个阶段谁能读这块内存"
actors: ["文件后端", "主机缓冲", "设备 stream", "模型层"]
steps:
  - from: 0
    to: 1
    label: "读入主存"
    detail: "外部 KV 的字节先落到这次传输占用的主机缓冲。"
  - from: 1
    to: 2
    label: "提交 H2D"
    detail: "传输流取得目标设备范围，开始异步复制。"
  - from: 2
    to: 2
    label: "到达完成点"
    detail: "相应事件或任务报告复制完成；此时目标块才满足消费条件。"
  - from: 2
    to: 3
    label: "进入计算"
    detail: "模型层读取已经回填的数据，源缓冲按自身生命周期回收。"
{{< /trace >}}

## 提交成功与复制完成

假设一批包含两个目标块。`HostToDeviceAsync` 返回后，调用方仍需按所用执行流的契约确认完成，才能释放源缓冲或消费目标块。遇到偶发数据错误时，记录源/目标地址、长度、所属设备、执行流和等待位置；仅比较张量形状无法发现过早复用缓冲。

## 找到对应平台的实现

- [CUDA stream](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/shared/trans/cuda/cuda_stream.cc) 和 [Ascend stream](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/shared/trans/ascend/ascend_stream.cc)各自负责调用平台运行时。
- [缓冲池](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/shared/pool/buffer_pool.h)决定缓冲的分配与回收。Host cache 的容量和传输所需的临时缓冲有不同的占用周期。
- 跨进程或跨节点的 P2P 另有 [transport 管理器](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/transport/p2p/src/core/transport_manager.cpp)。从本机 H2D 数据流推导网络传输行为时，需要确认实际走的后端。

## 传输前的依赖与传输后的完成各管什么

写出 KV 时，模型先把源数据写进设备。若 D2H 使用另一条 stream，需要一个 prerequisite 使复制等待前面的设备写入；复制完成之后，Host 缓冲才可以交给另一层持久化或回收。加载时顺序相反：后端读入缓冲后提交 H2D，使用该设备块的 attention 要等待相应事件。两处等待覆盖的是不同时间窗口。

`UcmPipelineStore.load_data` 向原生后端传递先决事件句柄和每次任务的分发选项。`RankConsistencyManager` 把 task 与返回它的 Store 绑定，后续的 `wait_load` 由同一个 Store 接收。对异步故障，仅观察提交时的返回值不够；有些错误要到 wait 才能报告。[管线地址接口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/pipeline/connector.py) · [异步记录](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/rank_consistency.py)

如果只某一个 rank 挂住，在同一请求上分别记录源地址范围、各 rank 的目标、分发选项和等待者。传输字节总数可能正确，某个 rank 的私有状态仍可能没有对应的数据供应。这与单纯提高 buffer 数或切换链路是不同层面的问题。[GLM5.3 加载分析](../04-vllm-integration/05-patch-anatomy.md)
