---
title: "UCMBlendConnector 如何描述换了位置的文本块"
weight: 3
---

`UCMBlendConnector` 同时处理连续前缀命中与按文档 chunk 的命中。一次请求会先判断已有前缀，再构建尚未覆盖的 chunk 描述。每个 `ChunkMetaData` 记录它在新请求中的起点、缓存时的起点、对应块、每块的命中情况。[Blend connector 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/blend_connector.py)

## 跟踪一个位移

若手册原来起于 token 128，现在起于 token 384，`position_offset` 为 256。若手册开头的若干完整块已经被普通前缀覆盖，`update_meta_partial_pc` 会一起移动新旧起点，并裁掉相应的哈希和命中列表。这样剩下的区间不会被重复计入。

{{< stateboard >}}
title: "新位置与旧位置如何得到块位移"
rows: ["块", "旧起点", "新起点"]
steps:
  - title: "记录原位置"
    detail: "手册起点在 token 128。示例用 128 token 表示一个块。"
    cells:
      - ["M0", "M1"]
      - ["128", "256"]
      - ["·", "·"]
    focus: [[1, 0], [1, 1]]
  - title: "前面插入文档"
    detail: "这次手册起点变成 384，后面一块也右移相同距离。"
    cells:
      - ["M0", "M1"]
      - ["128", "256"]
      - ["384", "512"]
    focus: [[2, 0], [2, 1]]
  - title: "计算差值"
    detail: "两个块的位置差均为 +256，加载后的 K 用对应位移校正。"
    cells:
      - ["M0", "M1"]
      - ["128", "256"]
      - ["384", "512"]
    focus: [[1, 0], [2, 0]]
{{< /stateboard >}}

通过 `BlendStage` 可以区分构建 chunk 缓存、构建前缀缓存和执行融合的请求阶段。调试时先查 `get_num_new_matched_tokens`、`build_connector_meta`，再沿 worker 侧的块回填与计算掩码检查同一请求。这个实现服务于文本文档融合；适配版本与 CUDA 条件见[算法说明](../03-sparse-attention/07-blend-algorithm.md)。
