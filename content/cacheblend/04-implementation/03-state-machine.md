---
title: "状态机"
weight: 3
---

`LlamaModel` 用 `temp_status` 告诉每层该执行完整 prefill、检查偏差还是只更新入选位置。普通 decode 另走状态 −1。

## temp_status 状态定义

CacheBlend 使用 `temp_status` 变量控制每层的处理模式：

| 状态值 | 名称 | 说明 |
|--------|------|------|
| -1 | Decode | 解码阶段，仅使用缓存 |
| 0 | Full Prefill | 完整预填充，计算所有 token |
| 1 | Check | 检查层，执行 HKVD 选择 + 计算差异 |
| 2 | After Check | 检查后层，使用选定的 HKVD token 更新 KV |

{{< stateboard >}}
title: "一次融合前向的层状态"
rows: ["步骤", "状态", "处理位置"]
steps:
  - title: "完整前向"
    detail: "进入融合前向时，较早层仍按整段序列处理。"
    cells:
      - ["早期层", "检查层", "后续层"]
      - ["0", "·", "·"]
      - ["全部", "·", "·"]
    focus: [[1, 0]]
  - title: "检查层"
    detail: "到达 check_layers 中的层，后端比较新旧 V 并选出位置。"
    cells:
      - ["早期层", "检查层", "后续层"]
      - ["0", "1", "·"]
      - ["全部", "筛选", "·"]
    focus: [[1, 1], [2, 1]]
  - title: "后续层"
    detail: "状态切换为 2，后端只更新入选位置，其他位置沿用旧 KV。"
    cells:
      - ["早期层", "检查层", "后续层"]
      - ["0", "1", "2"]
      - ["全部", "筛选", "更新"]
    focus: [[1, 2], [2, 2]]
{{< /stateboard >}}

## 状态转换逻辑

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaModel.forward() 第 330-356 行

def forward(self, input_ids, positions, kv_caches, attn_metadata, ...):
    # 初始状态判断
    if attn_metadata.prefill_metadata:
        temp_status = 0  # Full Prefill
        if self.cache_fuse_metadata["check"]:
            # 初始化检查模式
            self.cache_fuse_metadata["org_seq_len"] = input_ids.shape[0]
            self.cache_fuse_metadata["fake_q"] = None
            self.cache_fuse_metadata["attn_bias"] = None
            self.cache_fuse_metadata["imp_indices"] = None
            self.cache_fuse_metadata['org_pos'] = positions[:]
    else:
        temp_status = -1  # Decode

    # 层循环中的状态转换
    for i in range(len(self.layers)):
        if self.cache_fuse_metadata["check"]:
            if i in self.cache_fuse_metadata["check_layers"]:
                temp_status = 1  # Check 层
            elif i > self.cache_fuse_metadata["check_layers"][0]:
                temp_status = 2  # After Check

        # 执行该层
        hidden_states, residual = layer(
            ...,
            status=temp_status,
            cache_fuse_metadata=self.cache_fuse_metadata,
            old_kv=self.old_kvs[i]
        )

        # Check 层后更新 positions
        if temp_status == 1:
            positions = positions[self.cache_fuse_metadata["imp_indices"]]
```

## 状态转换流程图

普通 decode 使用 `-1`。融合 prefill 从 `0` 进入检查层的 `1`，随后用 `2` 处理较少的位置；未启用检查的完整 prefill 则沿 `0` 继续。

---

**上一步**: [核心数据结构](./02-data-structures.md)

**下一步**: [LlamaModel 实现](./04-llama-model.md)
