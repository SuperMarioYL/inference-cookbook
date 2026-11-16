---
title: "KVStar 怎样跨解码步检索缓存块"
weight: 6
---

KVStar 在一次生成过程中持续选择要载入的历史块。当前步需要哪些块、上一轮候选是否已经就绪、哪些块要留在设备上，这些状态都随请求和层而变化。本版对应的实现是 `KVStarMultiStep`，位于 [multistep.py](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/kvstar/multistep.py)。

## 以最后一个 prefill chunk 为边界

`ReqMeta.stage` 根据是否已产生输出 token 区分 prefill 与 decode。`is_last_chunk` 比较已经计算与本步调度的 token 总数是否覆盖 prompt。假设 prompt 有 384 个 token，先调度 256 个，再调度 128 个，第二次才到达这个边界。读取 `query_offload_info` 时要结合这两个条件，才能解释当前究竟在保存哪段 query。

每一层的 `ReqPerLayerState` 保存检索和换入所需状态。`retrieval_async` 提交检索，`load_retrieve_result_async` 使用相应步数和目标块安排加载。把“当前 decode 步”“生成的候选块”和“可消费的结果”放在三列，能看清上一轮任务是否在这一步用到。

{{< stateboard >}}
title: "同一层的候选如何跨两个 decode 步"
rows: ["时刻", "当前步", "后续候选"]
steps:
  - title: "刚进入第 t 步"
    detail: "第 t 步使用已准备好的工作集，同时准备后续候选。"
    cells:
      - ["t", "t+1"]
      - ["使用 A", "·"]
      - ["检索 B", "·"]
    focus: [[1, 0], [2, 0]]
  - title: "异步结果到达"
    detail: "候选 B 的结果返回，换入对应设备位置。"
    cells:
      - ["t", "t+1"]
      - ["使用 A", "·"]
      - ["B 就绪", "·"]
    focus: [[2, 0]]
  - title: "进入下一步"
    detail: "第 t+1 步按该层实际状态消费 B，同时可以安排新的检索。"
    cells:
      - ["t", "t+1"]
      - ["完成", "使用 B"]
      - ["B 就绪", "检索 C"]
    focus: [[1, 1], [2, 1]]
{{< /stateboard >}}

`examples/offline_inference_kvstar.py` 给出了所用参数、模型和运行环境的入口。比较完整注意力与检索路径时，同时记录答案、每步选择的块与 I/O 等待。[上游示例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/offline_inference_kvstar.py)
