---
title: "ReRoPE 如何处理远距离的位置差"
weight: 8
---

在因果注意力里，某个 query 会接触到相隔很远的 key。ReRoPE 保留窗口内的相对位置，对超过窗口的距离采用经过限制的位置关系。UCM 的实现用 Triton 内核计算对应的注意力分数，对照不同距离选择相应的计算路径。[原理与实验条件](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/rerope/rerope.md)

设窗口为 `w`，查询位置为 `i`、键位置为 `j`。在 `i − j < w` 时使用常规的相对位置关系；达到窗口距离后使用限制后的远距离得分路径。它改变了模型看到的相对位置，因此长上下文任务需要重新检查正确性和耗时。

|想跟踪的步骤|v0.9.1 源码|
|---|---|
|读取窗口和训练长度配置|[rerope_utils.py](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/rerope/rerope_utils.py)|
|根据模型的注意力路径准备 Q、K、V|[attn_forward_utils.py](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/rerope/attn_forward_utils.py)|
|用 Triton 计算注意力|[triton_unified_attention_rerope.py](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/rerope/triton_unified_attention_rerope.py)|

上游增强特性矩阵把 ReRoPE 列在 vLLM 0.11.0。复现实验从匹配版本的 `offline_inference_rerope.py` 入手，记录上下文长度、窗口、模型、输出质量和吞吐。[支持矩阵](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md) · [示例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/offline_inference_rerope.py)
