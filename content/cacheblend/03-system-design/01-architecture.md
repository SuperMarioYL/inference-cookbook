---
title: "整体架构"
weight: 1
---

一条新提示到达时，系统先确定哪些文档已有 KV，再安排逐层读取。模型一边消费本层缓存，一边重算受新上下文影响较大的位置。

---

## 3.1.1 CacheBlend 系统组件

CacheBlend 系统由三个主要组件组成：

```mermaid
graph TB
    subgraph "CacheBlend 系统架构"
        User["用户查询"] --> Retriever["检索器"]
        Retriever --> Controller["Loading Controller<br>加载控制器"]

        subgraph "KV Cache Store"
            CPU["CPU RAM"]
            SSD["SSD"]
            Disk["Slower Disks"]
        end

        Controller --> Store["KV Cache Store<br>KV缓存存储"]
        Store --> CPU
        Store --> SSD
        Store --> Disk

        Controller --> Fusor["Fusor<br>缓存融合器"]
        Store --> Fusor

        Fusor --> LLM["LLM 推理引擎"]
        LLM --> Response["生成响应"]
    end

    style Controller fill:#87CEEB
    style Fusor fill:#90EE90
    style Store fill:#FFD700
```

## 3.1.2 与 vLLM 的集成

CacheBlend 在 vLLM 之上实现，约 3000 行 Python 代码。集成点包括：

```mermaid
graph TB
    subgraph "vLLM + CacheBlend 集成"
        subgraph "vLLM 核心"
            Engine["LLM Engine"]
            Executor["Model Executor"]
            Runner["Model Runner"]
        end

        subgraph "CacheBlend 扩展"
            Meta["cache_fuse_metadata"]
            OldKV["old_kvs"]
            Status["temp_status"]
        end

        subgraph "修改的模块"
            LlamaModel["LlamaModel"]
            LlamaAttn["LlamaAttention"]
            XFormers["XFormers Backend"]
        end

        Engine --> Executor
        Executor --> Runner
        Runner --> LlamaModel
        LlamaModel --> Meta
        LlamaModel --> OldKV
        LlamaModel --> Status
        LlamaAttn --> XFormers
    end
```

## 3.1.3 数据流与控制流

{{< trace >}}
title: "检索到的两份文档如何进入模型"
actors: ["检索器", "控制器", "缓存存储", "Fusor", "模型"]
steps:
  - from: 0
    to: 1
    label: "返回文档"
    detail: "检索器按本次问题返回 M 与 R，控制器保留它们在提示中的顺序。"
  - from: 1
    to: 2
    label: "查询缓存"
    detail: "控制器查询哪份文档的 KV 可用，并安排逐层读取。"
  - from: 2
    to: 3
    label: "加载当前层"
    detail: "命中的数据先送给 Fusor，未命中的文档部分仍需计算。"
  - from: 3
    to: 4
    label: "拼出本层 KV"
    detail: "Fusor 结合复用与重算的位置，让模型继续这一层的注意力。"
{{< /trace >}}

---

**下一步**: [Loading Controller（加载控制器）](./02-loading-controller.md)
