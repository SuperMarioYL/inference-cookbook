---
title: "从 vLLM 的一个调度步理解 UCM"
weight: 3
---

vLLM V1 把请求切成若干调度步。scheduler 选择本步要处理的 token，KV cache manager 分配缓存块，worker 根据调度结果运行模型。在前缀缓存场景中，UCM 参与的时刻是“本地已命中多少、外部还命中多少、分配的目标块何时可读”。

{{< trace >}}
title: "本地查完之后，外部命中怎样进入一次调度"
actors: ["调度器", "块管理器", "UCM", "worker"]
steps:
  - from: 0
    to: 1
    label: "查本地前缀"
    detail: "块管理器先返回设备里可复用的前缀长度。"
  - from: 0
    to: 2
    label: "查新增命中"
    detail: "UCM 从这段本地前缀之后开始计算外部能补上的范围。"
  - from: 0
    to: 1
    label: "分配目标块"
    detail: "本步为外部命中的数据分配设备位置。"
  - from: 0
    to: 3
    label: "下发本步"
    detail: "worker 收到调度输出与加载元数据。"
  - from: 3
    to: 2
    label: "启动加载"
    detail: "UCM 把存储中的块搬到指定位置，按当前 connector 的等待条件交给计算。"
{{< /trace >}}

假设每块装 128 个 token，请求开头有 384 个 token 与上次相同。其中第一块仍留在 HBM，第二、三块只在外部存储。调度器需要知道新增的是 **256** 个 token，再给外部数据分配目标块。末尾不同的问题仍需执行 prefill。这个例子的块长是为了便于计算，实际值由模型和引擎配置决定。

从支持 UCM 0.9.1 的 vLLM 0.30.0 代码，可以并排跟踪 [V1 scheduler](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/sched/scheduler.py)、[KV cache manager](https://github.com/vllm-project/vllm/blob/v0.30.0/vllm/v1/core/kv_cache_manager.py) 和 [UCM 回调](../04-vllm-integration/01-connector-interface.md)。

## 分清逻辑序号与物理块号

下图用块长 16 举例，同一请求的逻辑序列连续，设备块号可以分散。
{{< mechanism "ucm-physical-slot" >}}

下面用字典举例说明如何读块表。
```python
request_1 = {
    "block_table": [5, 12, 3, None, None],  # 逻辑块 → 物理块
    "num_tokens": 45,  # 当前 token 数
    "max_blocks": 5    # 最大 Block 数
}

外部存储按内容或前缀查对象，回填时则按这次请求刚分配的物理块找目标。
