---
title: "路由索引怎样表达共享前缀"
weight: 4
---

路由器需要回答“每个候选 worker 上能复用多少连续前缀”。例如 P1 记录块序列 A、B、C，P2 记录 A、X；新输入 A、B、D 对 P1 有两块重叠，对 P2 只有一块。P2 走到 X 的分支就停下。连续重叠只计到两条路径第一次分叉的位置。

索引记录块键、前缀关系和所属 worker，不持有设备张量。压缩 radix tree 可以合并无分支的连续路径，减少元数据；具体索引布局和分片取决于所选实现。1.6.2 将多种索引实现置于 `lib/kv-router/src/indexer`，LLM 侧负责 ingress、路由连接或远端索引等集成。读性能讨论时先确认选中的实现及索引部署位置。

创建、驱逐和 worker 退出改变可匹配前缀。索引传播滞后时路由器可能高估某 worker 的可复用长度，应对照后端真实命中与排队负载。

## Trie 与压缩前缀树

### Trie 与 Radix Tree

**Trie（前缀树）** 是一种用于存储字符串集合的树形数据结构，每个节点代表一个字符：

```
普通 Trie
root → t → e → s → t
             ↘ a → m
         ↘ o → m
```

压缩树把无分支的连续路径合在一条边。三个字符串第一个字符相同，因此压缩树根节点也只有一个 t 分支。

```text
root → "t" ┬ "e" ┬ "st"
           │     └ "am"
           └ "om"
```

### 优势

| 特性 | Trie | Radix Tree |
|------|------|------------|
| 元数据 | 无分支路径上仍按字符分节点 | 合并无分支路径，减少节点；标签仍须表示原序列 |
| 查询复杂度 | O(m) | O(m) |
| 前缀查询 | 支持 | 支持 |
| 适用场景 | 密集键 | 稀疏键、公共前缀多 |

## 对包含分叉的请求进行查找

{{< mechanism "dynamo-prefix-members" >}}

树节点保存块键与 worker 成员。选定 worker 后还要确认其缓存是否已发生变化。一个索引节点可以被多个 worker 报告，释放其中一个 worker 的副本时需要保留其他实例的成员关系。

[重叠匹配的语义](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/kv-aware-routing.md) · [索引实现](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/kv-router/src/indexer/README.md) · [路由集成](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/kv_router/indexer/mod.rs)
