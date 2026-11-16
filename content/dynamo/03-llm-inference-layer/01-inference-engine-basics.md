---
title: "推理引擎原理"
weight: 1
---

## 先区分部署调度与引擎调度

Dynamo 为请求选 worker，推理引擎再在 worker 内给 token 分配一轮运行机会。要解释一个长请求的等待时间，先从前端记录分配时间，再比较后端是否有可用 KV 槽与 token 预算。 [路由与引擎的边界](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/kv-aware-routing.md)

## 1. 推理引擎核心功能

LLM 推理引擎是将预训练模型部署为可用服务的核心组件：

```mermaid
graph TB
    subgraph engine["LLM 推理引擎核心功能"]
        subgraph load["模型加载"]
            L1["权重加载"]
            L2["模型分片"]
            L3["量化处理"]
        end

        subgraph batch["批处理调度"]
            B1["请求排队"]
            B2["动态批处理"]
            B3["抢占与恢复"]
        end

        subgraph mem["内存管理"]
            M1["权重内存"]
            M2["激活内存"]
            M3["KV Cache 内存"]
        end

        subgraph opt["计算优化"]
            O1["算子融合"]
            O2["FlashAttention"]
            O3["并行策略"]
        end
    end
```

---

## 2. 模型加载与初始化

### 2.1 权重加载

推理引擎首先需要将预训练模型从存储加载到 GPU 内存中：

1. **权重加载**：从磁盘读取模型权重（safetensors 或 PyTorch 格式）
2. **模型分片**：将模型切分到多个 GPU（Tensor Parallelism）
3. **量化处理**：可选地将 FP16/BF16 权重转换为 INT8/INT4/FP8

### 2.2 并行策略

```mermaid
graph TB
    subgraph parallel["并行策略"]
        subgraph tp["Tensor Parallelism"]
            GPU1_TP["GPU 1: 部分权重"]
            GPU2_TP["GPU 2: 部分权重"]
            GPU1_TP <--> GPU2_TP
        end

        subgraph pp["Pipeline Parallelism"]
            GPU1_PP["GPU 1: Layer 1-20"]
            GPU2_PP["GPU 2: Layer 21-40"]
            GPU1_PP --> GPU2_PP
        end
    end
```

| 并行策略 | 说明 | 适用场景 |
|---------|------|----------|
| **Tensor Parallelism** | 将每层权重切分到多个 GPU | 单节点多卡 |
| **Pipeline Parallelism** | 将不同层分配到不同 GPU | 多节点部署 |

---

## 3. 批处理策略

### 3.1 静态批处理

传统方法：等待凑齐一个 batch 再一起处理。

**问题**：最长请求决定整个 batch 的延迟。

### 3.2 动态批处理 (Continuous Batching)

{{< stateboard >}}
title: 同一 worker 怎样补入下一条请求
rows: ["执行槽位", "等待请求"]
steps:
  - title: "首轮"
    detail: "一个聚合 worker 本轮执行 A、B，C 等待。"
    cells: [["A", "B"], ["C", "·"]]
    focus: [[0, 0], [0, 1], [1, 0]]
  - title: "A 提前结束"
    detail: "A 完成退出，引擎下一次选批时获得一个可用执行位置。"
    cells: [["·", "B"], ["C", "·"]]
    focus: [[0, 0]]
  - title: "C 进入"
    detail: "把 C 纳入本地执行，B 保留之前的上下文并继续解码。"
    cells: [["C", "B"], ["·", "·"]]
    focus: [[0, 0], [0, 1]]
{{< /stateboard >}}

**优势**：
- 请求完成即释放，新请求可立即加入
- GPU 利用率更高
- 减少平均延迟

---

## 4. 内存管理

### 4.1 内存类型

LLM 推理中的内存主要分为三部分：

| 内存类型 | 特点 | 大小估算（Llama-70B） |
|---------|------|----------------------|
| **权重内存** | 固定，与模型大小相关 | ~140GB (FP16) |
| **激活内存** | 临时，与 batch_size 相关 | 可变 |
| **KV Cache** | 动态增长，与序列长度相关 | 见下方公式 |

### 4.2 KV Cache 内存计算

```
KV Cache 内存 = 2 × L × H × D × S × B × dtype_size
```

其中：
- L = 层数 (80)
- H = 注意力头数 (64)
- D = 每头维度 (128)
- S = 序列长度
- B = 批大小
- dtype_size = 数据类型字节数 (FP16 = 2)

### 4.3 GPU HBM 容量限制

```mermaid
graph LR
    subgraph memory["GPU 内存分配 - H100 80GB"]
        WEIGHT["模型权重<br/>~70GB"]
        KV["KV Cache<br/>~10GB"]
        ACT["激活内存<br/>动态"]
    end

    WEIGHT --> LIMIT["容量受限"]
    KV --> LIMIT
```

---

## 5. 计算优化

### 5.1 算子融合

将多个连续算子合并为一个 CUDA kernel：

```mermaid
graph LR
    subgraph before["融合前"]
        Q["Q 投影"]
        K["K 投影"]
        V["V 投影"]
        Q --> ATTN["Attention"]
        K --> ATTN
        V --> ATTN
    end

    subgraph after["融合后"]
        QKV["QKV 融合投影"] --> ATTN2["Fused Attention"]
    end
```

### 5.2 FlashAttention

FlashAttention 通过优化内存访问模式，显著提升 Attention 计算效率：

- **Tiling**：将计算分块，充分利用 GPU 高速缓存
- **减少 HBM 访问**：避免存储完整的 attention 矩阵
- **IO 感知**：优化内存访问模式

---

## 6. 采样策略

### 6.1 采样参数

| 参数 | 说明 | 典型值 |
|------|------|--------|
| **Temperature** | 控制随机性，越高越随机 | 0.0-2.0 |
| **Top-P** | 核采样，累积概率阈值 | 0.9-0.95 |
| **Top-K** | 只考虑前 K 个 token | 40-100 |
| **Repetition Penalty** | 重复惩罚 | 1.0-1.2 |

### 6.2 采样流程

```mermaid
graph LR
    LOGITS["Logits"] --> TEMP["Temperature 缩放"]
    TEMP --> TOPK["Top-K 过滤"]
    TOPK --> TOPP["Top-P 过滤"]
    TOPP --> SAMPLE["采样"]
    SAMPLE --> TOKEN["输出 Token"]
```

---

## 下一篇

继续阅读 [02-vLLM 技术解析](02-vllm-internals.md)，深入了解 vLLM 的核心技术。
