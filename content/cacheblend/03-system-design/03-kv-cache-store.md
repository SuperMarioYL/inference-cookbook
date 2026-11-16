---
title: "KV Cache Store"
weight: 3
---

同一份文档第二次进入提示时，Store 找出按文档保存的 KV，Fusor 在模型前向中挑选并更新受新上文影响的位置。

---

## 3.3 KV Cache Store（KV缓存存储）

### 3.3.1 文本分块策略

KV Cache Store 将 LLM 输入分割成多个文本块，每个块可以是复用的或新的：
- RAG 输入通常由多个检索的上下文块（通常是固定长度）和用户输入组成
- LLM 输入的分割是特定于应用的
- CacheBlend 实现了与近期工作相同的策略

### 3.3.2 Hash 映射机制

一旦输入被分割成文本块，每个块被哈希以找到其对应的 KV Cache：
- 与 vLLM 中的块哈希实现方式相同
- 如果找到匹配，复用 KV Cache
- 如果未找到，需要计算新的 KV Cache

{{< trace >}}
title: "文档第一次出现与第二次出现"
actors: ["文档 M", "内容索引", "KV Store", "推理进程"]
steps:
  - from: 0
    to: 1
    label: "查 M"
    detail: "服务根据文档的 token 内容定位已有缓存。"
  - from: 1
    to: 3
    label: "未命中"
    detail: "第一次没有对应条目，推理进程计算 M 的 KV。"
  - from: 3
    to: 2
    label: "写入 KV"
    detail: "缓存存下本次算出的张量，供后续请求查找。"
  - from: 1
    to: 2
    label: "再次查到"
    detail: "M 再出现时，索引定位旧数据；调用方按新位置与上文决定如何融合。"
{{< /trace >}}

### 3.3.3 LRU 驱逐策略

当存储设备满时，使用 LRU（Least Recently Used）策略驱逐 KV Cache：
- 驱逐最近最少使用的 KV Cache
- 保留最常用的 KV Cache

### 3.3.4 存储层级

CacheBlend 支持在单一存储设备层级中存储 KV Cache：
- CPU RAM
- SSD

哈希表保存在 CPU 中，因为其大小相对较小（一百万块约 16MB）。

---

## 3.4 Fusor（缓存融合器）

### 3.4.1 核心功能

Fusor（缓存融合器）通过选择性重计算合并预计算的 KV Cache：

{{< stateboard >}}
title: "Fusor 怎样处理一层中的三段输入"
rows: ["位置", "旧 KV", "本层结果"]
steps:
  - title: "加载命中文档"
    detail: "例子中 A、B 已有缓存，Q 是新后缀。"
    cells:
      - ["A", "B", "Q"]
      - ["旧A", "旧B", "·"]
      - ["·", "·", "·"]
    focus: [[1, 0], [1, 1]]
  - title: "重算受影响位置"
    detail: "检查层选择 B，新后缀 Q 也参与计算。"
    cells:
      - ["A", "B", "Q"]
      - ["旧A", "旧B", "·"]
      - ["·", "新B", "新Q"]
    focus: [[2, 1], [2, 2]]
  - title: "组合完整上下文"
    detail: "当前层继续用旧 A，并使用刚得到的新 B、Q。"
    cells:
      - ["A", "B", "Q"]
      - ["旧A", "旧B", "·"]
      - ["旧A", "新B", "新Q"]
    focus: [[2, 0], [2, 1], [2, 2]]
{{< /stateboard >}}

### 3.4.2 层级处理流程

从算法 4.3 回顾，哪些 token 需要在一层重计算取决于前一层的重计算结果。因此：

1. Fusor 等待前一层的重计算完成
2. 第 $L$ 层的 KV Cache 被加载到 GPU 内存的队列中
3. 使用 Loading Controller 计算的重计算比例执行选择性重计算
4. Fusor 重复此过程直到所有层都被重计算

### 3.4.3 接口设计

CacheBlend 通过三个接口执行逐层部分 Prefill：

```python
# 接口 1: 获取 KV Cache
fetch_kv(text, layer_id) -> KVCache
# 给定文本和层 ID，从 KV Store 获取对应的 KV Cache
# 如果 KV Cache 不在系统中，返回 -1

# 接口 2: 部分 Prefill
prefill_layer(input_dict, KVCache) -> output_dict
# 接收输入和当前层的 KV Cache，执行该层的部分 Prefill
# 输出用作下一层的输入

# 接口 3: 同步
synchronize()
# 在每层 Prefill 前需要同步，确保该层的 KV Cache 已加载到 GPU
```

---

**上一步**: [Loading Controller](./02-loading-controller.md)

**下一步**: [Pipeline 优化](./04-pipeline-optimization.md)
