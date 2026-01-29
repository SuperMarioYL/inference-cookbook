---
title: "KV Cache 概念"
weight: 4
---

# KV Cache 概念

> 本章将详细介绍 KV Cache 的概念、作用和实现原理，这是理解 vLLM 核心优化的关键。

---

## 引言

KV Cache 是 LLM 推理中最重要的优化技术之一。它通过缓存历史计算结果，避免重复计算，显著提升推理速度。理解 KV Cache 对于理解 vLLM 的 PagedAttention 至关重要。

---

## 1. 为什么需要 KV Cache

### 1.1 自回归生成的特点

LLM 生成文本是**自回归**的：每次只生成一个 token，然后将其加入输入，继续生成下一个。

```mermaid
sequenceDiagram
    participant User as 用户
    participant LLM as LLM

    User->>LLM: "今天天气"
    LLM-->>LLM: 计算所有 token 的 Attention
    LLM->>User: "很"

    User->>LLM: "今天天气很"
    LLM-->>LLM: 重新计算所有 token 的 Attention?
    LLM->>User: "好"

    User->>LLM: "今天天气很好"
    LLM-->>LLM: 又重新计算所有?
    LLM->>User: "。"
```

### 1.2 没有 KV Cache 时的重复计算

在注意力计算中，每个 token 需要：
1. 计算自己的 Q（Query）
2. 计算自己的 K（Key）和 V（Value）
3. 用 Q 与**所有** K 计算注意力
4. 用注意力加权**所有** V

**问题**：历史 token 的 K 和 V 每次都要重新计算！

```mermaid
flowchart TD
    subgraph step1["Step 1 - 处理 'Hello'"]
        A1[Hello] --> K1["计算 K₁"]
        A1 --> V1["计算 V₁"]
        A1 --> Q1["计算 Q₁"]
    end

    subgraph step2["Step 2 - 处理 'Hello World'"]
        B1[Hello] --> K1_2["重新计算 K₁"]
        B1 --> V1_2["重新计算 V₁"]
        B2[World] --> K2["计算 K₂"]
        B2 --> V2["计算 V₂"]
        B2 --> Q2["计算 Q₂"]
    end

    subgraph step3["Step 3 - 处理 'Hello World !'"]
        C1[Hello] --> K1_3["再次计算 K₁"]
        C1 --> V1_3["再次计算 V₁"]
        C2[World] --> K2_3["再次计算 K₂"]
        C2 --> V2_3["再次计算 V₂"]
        C3[!] --> K3["计算 K₃"]
        C3 --> V3["计算 V₃"]
        C3 --> Q3["计算 Q₃"]
    end

    style K1_2 fill:#ffcdd2
    style V1_2 fill:#ffcdd2
    style K1_3 fill:#ffcdd2
    style V1_3 fill:#ffcdd2
    style K2_3 fill:#ffcdd2
    style V2_3 fill:#ffcdd2
```

### 1.3 计算量分析

生成 N 个 token，不使用 KV Cache：

| Step | 需要计算的 K/V | 累计 K/V 计算次数 |
|------|---------------|------------------|
| 1 | 1 | 1 |
| 2 | 2（重新计算 1 + 新的 1） | 1 + 2 = 3 |
| 3 | 3（重新计算 2 + 新的 1） | 3 + 3 = 6 |
| ... | ... | ... |
| N | N | 1 + 2 + ... + N = N(N+1)/2 |

**时间复杂度**：$O(N^2)$

---

## 2. KV Cache 工作原理

### 2.1 核心思想

**观察**：在自回归生成中，历史 token 的 K 和 V 不会改变。

**解决方案**：计算一次后缓存起来，后续直接使用。

```mermaid
flowchart TD
    subgraph use_kv_cache["使用 KV Cache"]
        subgraph cache_step1["Step 1"]
            S1A[Hello] --> S1K["计算 K₁"]
            S1A --> S1V["计算 V₁"]
            S1K --> Cache1[("缓存 K₁")]
            S1V --> Cache1
        end

        subgraph cache_step2["Step 2"]
            Cache1 --> Use1["使用缓存的 K₁, V₁"]
            S2A[World] --> S2K["计算 K₂"]
            S2A --> S2V["计算 V₂"]
            S2K --> Cache2[("缓存 K₁, K₂")]
            S2V --> Cache2
        end

        subgraph cache_step3["Step 3"]
            Cache2 --> Use2["使用缓存的 K₁, K₂, V₁, V₂"]
            S3A[!] --> S3K["计算 K₃"]
            S3A --> S3V["计算 V₃"]
        end
    end

    style Use1 fill:#c8e6c9
    style Use2 fill:#c8e6c9
```

### 2.2 计算量对比

使用 KV Cache 后：

| Step | 需要计算的 K/V | 累计 K/V 计算次数 |
|------|---------------|------------------|
| 1 | 1 | 1 |
| 2 | 1（只计算新的） | 1 + 1 = 2 |
| 3 | 1（只计算新的） | 2 + 1 = 3 |
| ... | ... | ... |
| N | 1 | N |

**时间复杂度**：$O(N)$

**加速比**：从 $O(N^2)$ 到 $O(N)$，生成 1000 个 token 时加速约 500 倍！

### 2.3 图解对比

```mermaid
graph TD
    subgraph no_kv_cache["无 KV Cache"]
        A1[Token 1] --> C1["计算全部 K,V"]
        A2["Token 1,2"] --> C2["计算全部 K,V"]
        A3["Token 1,2,3"] --> C3["计算全部 K,V"]
        A4["Token 1,2,3,4"] --> C4["计算全部 K,V"]
        style A1 fill:#ffcdd2
        style A2 fill:#ffcdd2
        style A3 fill:#ffcdd2
        style A4 fill:#ffcdd2
    end

    subgraph with_kv_cache["有 KV Cache"]
        B1[Token 1] --> D1["计算 K₁,V₁ + 缓存"]
        B2[Token 2] --> D2["计算 K₂,V₂ + 读缓存"]
        B3[Token 3] --> D3["计算 K₃,V₃ + 读缓存"]
        B4[Token 4] --> D4["计算 K₄,V₄ + 读缓存"]
        D1 --> Cache[(KV Cache)]
        D2 --> Cache
        D3 --> Cache
        D4 --> Cache
        Cache --> D2
        Cache --> D3
        Cache --> D4
        style B1 fill:#c8e6c9
        style B2 fill:#c8e6c9
        style B3 fill:#c8e6c9
        style B4 fill:#c8e6c9
    end
```

---

## 3. KV Cache 的数据结构

### 3.1 基本形状

KV Cache 需要存储每层的 K 和 V：

```python
# KV Cache 形状
# 方式 1: 分开存储
k_cache = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, head_dim)
v_cache = torch.zeros(num_layers, batch_size, num_heads, max_seq_len, head_dim)

# 方式 2: 合并存储
kv_cache = torch.zeros(num_layers, 2, batch_size, num_heads, max_seq_len, head_dim)
# kv_cache[:, 0, ...] 是 K
# kv_cache[:, 1, ...] 是 V
```

### 3.2 维度解释

| 维度 | 含义 | 示例值 |
|------|------|--------|
| num_layers | Transformer 层数 | 32 |
| 2 | K 和 V | 2 |
| batch_size | 批次大小 | 1-64 |
| num_heads | 注意力头数（或 KV heads） | 32 或 8 |
| max_seq_len | 最大序列长度 | 4096 |
| head_dim | 每个头的维度 | 128 |

### 3.3 代码示例

```python
class KVCache:
    def __init__(self, num_layers, num_heads, head_dim, max_seq_len, dtype=torch.float16):
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # 预分配 K 和 V 缓存
        # 形状: [num_layers, 2, max_batch, num_heads, max_seq_len, head_dim]
        self.cache = None
        self.current_len = 0

    def allocate(self, batch_size):
        self.cache = torch.zeros(
            self.num_layers, 2, batch_size, self.num_heads,
            self.max_seq_len, self.head_dim,
            dtype=self.dtype, device='cuda'
        )
        self.current_len = 0

    def update(self, layer_idx, new_k, new_v):
        """添加新的 K, V 到缓存"""
        # new_k, new_v: [batch, num_heads, new_len, head_dim]
        new_len = new_k.shape[2]
        start_pos = self.current_len
        end_pos = start_pos + new_len

        self.cache[layer_idx, 0, :, :, start_pos:end_pos, :] = new_k
        self.cache[layer_idx, 1, :, :, start_pos:end_pos, :] = new_v

        if layer_idx == self.num_layers - 1:
            self.current_len = end_pos

    def get(self, layer_idx):
        """获取当前层的完整 K, V"""
        k = self.cache[layer_idx, 0, :, :, :self.current_len, :]
        v = self.cache[layer_idx, 1, :, :, :self.current_len, :]
        return k, v
```

---

## 4. 显存占用详细计算

### 4.1 计算公式

```
KV Cache 显存 = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch_size × bytes_per_element
```

简化版（使用 hidden_dim）：

```
KV Cache 显存 = 2 × num_layers × hidden_dim × seq_len × batch_size × bytes_per_element
```

注意：如果使用 GQA，num_kv_heads 可能小于 num_attention_heads。

### 4.2 LLaMA-2-7B 示例

**模型参数**：
- num_layers: 32
- hidden_dim: 4096
- num_kv_heads: 32（MHA）
- head_dim: 128
- 精度: FP16（2 bytes）

**单个请求不同序列长度的 KV Cache**：

| 序列长度 | 计算 | 大小 |
|---------|------|------|
| 512 | 2 × 32 × 4096 × 512 × 2 | 256 MB |
| 1024 | 2 × 32 × 4096 × 1024 × 2 | 512 MB |
| 2048 | 2 × 32 × 4096 × 2048 × 2 | 1 GB |
| 4096 | 2 × 32 × 4096 × 4096 × 2 | 2 GB |
| 8192 | 2 × 32 × 4096 × 8192 × 2 | 4 GB |

### 4.3 LLaMA-2-70B 示例（使用 GQA）

**模型参数**：
- num_layers: 80
- hidden_dim: 8192
- num_kv_heads: 8（GQA，原本是 64 个 attention heads）
- head_dim: 128
- 精度: FP16

**单个请求 4096 序列长度**：

```
KV Cache = 2 × 80 × 8 × 128 × 4096 × 2 = 1.34 GB
```

对比 MHA（如果 kv_heads = 64）：
```
KV Cache = 2 × 80 × 64 × 128 × 4096 × 2 = 10.7 GB
```

**GQA 节省了 8 倍显存！**

### 4.4 显存占用可视化

```mermaid
pie title 7B 模型显存分布（单请求 2048 tokens）
    "模型权重 (14GB)" : 14
    "KV Cache (1GB)" : 1
    "激活值等 (1GB)" : 1
```

```mermaid
pie title 7B 模型显存分布（32 并发 × 2048 tokens）
    "模型权重 (14GB)" : 14
    "KV Cache (32GB)" : 32
    "激活值等 (2GB)" : 2
```

---

## 5. KV Cache 管理的挑战

### 5.1 动态序列长度

KV Cache 的大小随着生成过程动态增长：

```mermaid
graph LR
    subgraph gen_process["生成过程"]
        S1["Step 1<br/>KV: 10 tokens"]
        S2["Step 2<br/>KV: 11 tokens"]
        S3["Step 3<br/>KV: 12 tokens"]
        SN["Step N<br/>KV: N+10 tokens"]
        S1 --> S2 --> S3 --> SN
    end
```

**问题**：在请求开始时，我们不知道最终会生成多少 token！

### 5.2 预分配策略的问题

**传统方案**：预分配最大可能长度（如 4096 tokens）

```
预分配: 4096 tokens × 每token 0.5MB = 2GB
实际使用: 100 tokens × 0.5MB = 50MB
浪费: 1.95GB (97.5%)
```

```mermaid
graph TB
    subgraph prealloc_waste["预分配的浪费"]
        Alloc["预分配 2GB"]
        Used["实际使用 50MB"]
        Waste["浪费 1.95GB"]
        Alloc --> Used
        Alloc --> Waste
    end

    style Waste fill:#ffcdd2
```

### 5.3 显存碎片化

当多个请求同时运行时，问题更加严重：

```
显存状态：
+--------+--------+--------+--------+--------+
| Req A  | Req B  | Req C  | Req D  | 空闲   |
| 2GB    | 2GB    | 2GB    | 2GB    | 碎片   |
| 用50MB | 用100MB| 用30MB | 用200MB|        |
+--------+--------+--------+--------+--------+

实际使用: 380MB
预分配: 8GB
浪费: 7.62GB (95%!)
```

### 5.4 这就是 PagedAttention 要解决的问题！

传统方案的问题：
1. **预分配浪费**：每个请求预留最大空间
2. **内部碎片**：实际使用远小于预分配
3. **外部碎片**：释放后的空间不连续

PagedAttention 的解决方案（下一部分详细介绍）：
1. **按需分配**：用多少分配多少
2. **分块管理**：固定大小的块，减少碎片
3. **非连续存储**：块可以不连续

---

## 6. Prefill 和 Decode 中的 KV Cache

### 6.1 Prefill 阶段

处理输入 prompt，一次性计算所有输入 token 的 K、V：

```mermaid
flowchart LR
    subgraph prefill_stage["Prefill"]
        I["输入: 'Hello, how are you?'<br/>5 tokens"]
        C["并行计算 K₁...K₅, V₁...V₅"]
        S["存入 KV Cache"]
        I --> C --> S
    end
```

**特点**：
- 批量计算，效率高
- 计算密集型
- KV Cache 从 0 增长到输入长度

### 6.2 Decode 阶段

逐个生成 token，每次只计算新 token 的 K、V：

```mermaid
flowchart TD
    subgraph decode_loop["Decode 循环"]
        R["读取 KV Cache"]
        N["新 token"]
        C["计算 K_new, V_new"]
        A["Attention: Q_new x [K_cache; K_new]"]
        U["更新 KV Cache"]
        O["输出 token"]

        R --> A
        N --> C --> A
        A --> U --> O
        O -.->|下一轮| N
    end
```

**特点**：
- 增量计算，每次只算 1 个
- 内存密集型（需要读取整个 KV Cache）
- KV Cache 每步增长 1

### 6.3 两阶段的 KV Cache 操作对比

| 操作 | Prefill | Decode |
|------|---------|--------|
| K/V 计算 | 批量（N 个） | 单个（1 个） |
| KV Cache 读取 | 无 | 全部 |
| KV Cache 写入 | N 个 | 1 个 |
| 计算/访存比 | 高 | 低 |

---

## 7. vLLM 中的 KV Cache 相关代码

### 7.1 关键文件位置

| 功能 | 文件 |
|------|------|
| KV Cache 管理 | `vllm/v1/core/kv_cache_manager.py` |
| 块池 | `vllm/v1/core/block_pool.py` |
| 块表 | `vllm/v1/worker/block_table.py` |
| KV Cache 接口 | `vllm/v1/kv_cache_interface.py` |

### 7.2 数据结构预览

```python
# vllm/v1/core/block_pool.py 中的块定义
@dataclass
class KVCacheBlock:
    block_id: int          # 块 ID
    ref_cnt: int           # 引用计数
    block_hash: Optional[BlockHash]  # 用于前缀缓存

# vllm/v1/worker/block_table.py 中的块表
class BlockTable:
    """管理逻辑块到物理块的映射"""
    def __init__(self, ...):
        self.block_table: torch.Tensor  # 形状: [max_blocks]
```

---

## 8. 本章小结

### 核心概念

1. **KV Cache 的作用**：缓存历史 token 的 K、V，避免重复计算
2. **加速效果**：从 $O(N^2)$ 降到 $O(N)$，约 500 倍加速（N=1000）
3. **显存占用**：随序列长度线性增长，可能成为主要显存消耗

### 关键公式

```
KV Cache = 2 × num_layers × num_kv_heads × head_dim × seq_len × bytes
```

### 管理挑战

- **动态增长**：序列长度在生成过程中不断增加
- **预分配浪费**：传统方案浪费 60-80% 显存
- **碎片化**：多请求并发时问题更严重

### 与 vLLM 的关联

- **PagedAttention**：解决 KV Cache 的显存浪费问题
- **分块管理**：将 KV Cache 分成固定大小的块
- **按需分配**：用多少分配多少，不预留

---

## 思考题

1. 如果一个模型使用 GQA，KV heads 是 attention heads 的 1/8，KV Cache 显存会减少多少？
2. 为什么 Decode 阶段是"内存密集型"而不是"计算密集型"？
3. 如果 vLLM 要支持无限长度的上下文，KV Cache 管理会面临什么额外挑战？

---

## 下一步

了解了 KV Cache 后，让我们来看看 LLM 完整的生成过程：

👉 [下一章：LLM 生成过程](05-llm-generation-process.md)
