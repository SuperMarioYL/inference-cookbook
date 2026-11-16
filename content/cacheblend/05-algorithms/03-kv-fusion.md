---
title: "KV 融合算法"
weight: 3
---

假设只重算了位置 1 和 3。它们产生的新 K、V 要写回原来的两处；后续层收到的隐藏状态和残差也要按 1、3 这一相同顺序排列。

---

## 5.3 KV Cache 融合算法

### 5.3.1 算法描述

在 status=2 的层中，将新计算的 HKVD token 的 KV 更新到旧 KV Cache 中。

### 5.3.2 核心代码

```python
# 文件: vllm_blend/vllm/attention/backends/xformers.py
# 位置: XFormersImpl.forward() 第 240-245 行

if status in [2]:
    # 获取 HKVD 索引
    imp_indices = cache_fuse_metadata["imp_indices"]

    # 核心：部分更新
    # 只更新 HKVD token 对应位置的 KV
    key_old[imp_indices] = key      # key 只包含 HKVD token
    value_old[imp_indices] = value  # value 只包含 HKVD token

    # 使用融合后的 KV 进行后续计算
    key = key_old
    value = value_old
```

### 5.3.3 融合过程图示

{{< stateboard >}}
title: "位置 1 和 3 怎样写回旧缓存"
rows: ["位置", "旧 KV", "融合结果"]
steps:
  - title: "读取旧数据"
    detail: "整行旧 KV 保持原序列的四个位置。"
    cells:
      - ["0", "1", "2", "3"]
      - ["K0", "K1", "K2", "K3"]
      - ["·", "·", "·", "·"]
    focus: [[1, 0], [1, 1], [1, 2], [1, 3]]
  - title: "写回位置 1"
    detail: "检查层选中了 1 和 3。先看位置 1，它接收本步重算出的新 KV。"
    cells:
      - ["0", "1", "2", "3"]
      - ["K0", "K1", "K2", "K3"]
      - ["K0", "新1", "K2", "K3"]
    focus: [[2, 1]]
  - title: "写回位置 3"
    detail: "位置 3 同样更新，位置 0、2 仍保留原值。"
    cells:
      - ["0", "1", "2", "3"]
      - ["K0", "K1", "K2", "K3"]
      - ["K0", "新1", "K2", "新3"]
    focus: [[2, 1], [2, 3]]
{{< /stateboard >}}

### 5.3.4 内存效率分析

这种部分更新策略非常高效：
- 只需要存储 HKVD token 的新 KV（约 15%）
- 直接在 old_kv 上原地更新
- 不需要额外的大内存分配

---

## 5.4 层间传播算法

### 5.4.1 残差连接处理

在 Check 层（status=1）之后，残差连接也需要过滤：

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaDecoderLayer.forward() 第 264-265 行

if status == 1:
    # 过滤残差，只保留 HKVD token 对应的部分
    residual = residual[cache_fuse_metadata["imp_indices"]]
```

### 5.4.2 位置编码过滤

在 LlamaModel 中，位置编码也需要过滤：

```python
# 文件: vllm_blend/vllm/model_executor/models/llama.py
# 位置: LlamaModel.forward() 第 373-376 行

if temp_status == 1:
    # 过滤位置编码，只保留 HKVD token 的位置
    positions = positions[self.cache_fuse_metadata["imp_indices"]]
```

### 5.4.3 层间数据一致性

{{< stateboard >}}
title: "用同一索引压缩位置、残差和隐藏状态"
rows: ["positions", "hidden", "residual"]
steps:
  - title: "筛选前"
    detail: "三行均按原 token 顺序排列。"
    cells:
      - ["0", "1", "2", "3"]
      - ["h0", "h1", "h2", "h3"]
      - ["r0", "r1", "r2", "r3"]
    focus: [[0, 0], [0, 1], [0, 2], [0, 3]]
  - title: "取相同索引"
    detail: "选中 1、3，三行要取相同的两个位置。"
    cells:
      - ["0", "1", "2", "3"]
      - ["h0", "h1", "h2", "h3"]
      - ["r0", "r1", "r2", "r3"]
    focus: [[0, 1], [0, 3], [1, 1], [1, 3], [2, 1], [2, 3]]
  - title: "形成紧凑张量"
    detail: "后续层的第 0 行属于原 token 1，第 1 行属于原 token 3。"
    cells:
      - ["1", "3", "·", "·"]
      - ["h1", "h3", "·", "·"]
      - ["r1", "r3", "·", "·"]
    focus: [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0], [2, 1]]
{{< /stateboard >}}

### 5.4.4 完整的层间传播流程

下一层用紧凑张量继续前向，注意力后端仍通过原序列位置访问整段 KV。位置向量、隐藏状态和残差同步选取，才会保持 token 对应关系。

---

## 下一步

- [实验设置](../06-experiments/01-setup.md) - 了解 CacheBlend 的实验评估方法
