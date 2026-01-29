---
title: "PagedAttention 分页注意力"
weight: 1
---

# PagedAttention 分页注意力

> 本章将详细介绍 vLLM 的核心创新——PagedAttention，包括设计思想、数据结构和实现原理。

---

## 引言

PagedAttention 是 vLLM 最重要的创新，它借鉴了操作系统虚拟内存管理的思想，革命性地解决了 KV Cache 的显存浪费问题。本章将深入剖析其设计原理和实现细节。

---

## 1. 传统 KV Cache 的问题回顾

### 1.1 连续内存分配的要求

传统方案要求每个请求的 KV Cache 存储在**连续的内存空间**中：

```
传统 KV Cache 布局:
+----------------------------------------------------------+
| Request A 的 KV Cache (预分配 max_seq_len)                 |
| [K0,V0][K1,V1][K2,V2]...[Kn,Vn][   空闲预留空间   ]        |
+----------------------------------------------------------+
| Request B 的 KV Cache (预分配 max_seq_len)                 |
| [K0,V0][K1,V1]...[Km,Vm][      空闲预留空间      ]         |
+----------------------------------------------------------+
```

### 1.2 显存碎片化图解

当多个请求并发时，显存碎片化问题严重：

```mermaid
graph TB
    subgraph time_t1["时间 T1 - 三个请求开始"]
        M1["Request A<br/>预分配 2GB<br/>实际用 0.1GB"]
        M2["Request B<br/>预分配 2GB<br/>实际用 0.2GB"]
        M3["Request C<br/>预分配 2GB<br/>实际用 0.1GB"]
        M4["空闲 2GB"]
    end

    subgraph time_t2["时间 T2 - Request B 完成"]
        N1["Request A<br/>预分配 2GB<br/>实际用 0.5GB"]
        N2["空洞 2GB<br/>外部碎片"]
        N3["Request C<br/>预分配 2GB<br/>实际用 0.3GB"]
        N4["空闲 2GB"]
    end

    subgraph time_t3["时间 T3 - 新请求 D 到来"]
        O1["Request A<br/>2GB"]
        O2["空洞 2GB"]
        O3["Request C<br/>2GB"]
        O4["空闲 2GB"]
        O5["Request D 需要 3GB<br/>失败!"]
    end

    style N2 fill:#ffcdd2
    style O2 fill:#ffcdd2
    style O5 fill:#ffcdd2
```

### 1.3 量化浪费

| 问题类型 | 说明 | 浪费比例 |
|---------|------|---------|
| 内部碎片 | 预分配 >> 实际使用 | 40-60% |
| 外部碎片 | 空洞无法利用 | 20-30% |
| **总计** | **综合浪费** | **60-80%** |

---

## 2. PagedAttention 核心思想

### 2.1 灵感来源：操作系统虚拟内存

操作系统如何管理内存？

```mermaid
graph LR
    subgraph virtual_memory["虚拟内存"]
        VA["虚拟地址空间<br/>程序看到的连续空间"]
    end

    subgraph page_table["页表"]
        PT["Page Table<br/>虚拟页 → 物理页"]
    end

    subgraph physical_memory["物理内存"]
        PA["物理内存<br/>实际不连续的页"]
    end

    VA --> PT --> PA
```

**关键特性**：
1. 程序看到连续的地址空间
2. 物理内存可以不连续
3. 按需分配（用到才分配）
4. 页面可以共享

### 2.2 PagedAttention 的类比

将操作系统的思想应用到 KV Cache 管理：

| 操作系统概念 | PagedAttention 对应 |
|-------------|-------------------|
| 页（Page） | Block（块） |
| 页表（Page Table） | Block Table（块表） |
| 虚拟地址 | 逻辑块索引 |
| 物理地址 | 物理块 ID |
| 页帧 | KV Cache 块 |

### 2.3 核心改进

```mermaid
graph LR
    subgraph traditional["传统方案"]
        T1["预分配连续空间"]
        T2["大量浪费"]
        T1 --> T2
    end

    subgraph paged_attention["PagedAttention"]
        P1["按需分配"]
        P2["分块存储"]
        P3["非连续"]
        P4["高利用率"]
        P1 --> P2 --> P3 --> P4
    end

    style T2 fill:#ffcdd2
    style P4 fill:#c8e6c9
```

---

## 3. 关键数据结构详解

### 3.1 Block（块）

Block 是 KV Cache 的基本存储单元：

```python
# 概念定义
class KVCacheBlock:
    block_id: int              # 物理块 ID
    ref_cnt: int               # 引用计数（支持共享）
    block_hash: Optional[int]  # 用于前缀缓存匹配
```

**Block 的特点**：
- **固定大小**：每个 block 存储固定数量的 token（如 16 个）
- **独立分配**：不需要连续
- **可复用**：释放后可分配给其他请求

### 3.2 Block 的存储内容

每个 Block 存储若干 token 的 K 和 V：

```
Block 结构 (block_size = 16):
┌─────────────────────────────────────────────────┐
│ Token 0:  K[layers, heads, head_dim]            │
│           V[layers, heads, head_dim]            │
├─────────────────────────────────────────────────┤
│ Token 1:  K[layers, heads, head_dim]            │
│           V[layers, heads, head_dim]            │
├─────────────────────────────────────────────────┤
│ ...                                             │
├─────────────────────────────────────────────────┤
│ Token 15: K[layers, heads, head_dim]            │
│           V[layers, heads, head_dim]            │
└─────────────────────────────────────────────────┘
```

**实际存储形状**：

```python
# 单个 Block 的 KV Cache 形状
k_block = torch.zeros(num_layers, num_heads, block_size, head_dim)
v_block = torch.zeros(num_layers, num_heads, block_size, head_dim)

# 整个 KV Cache 池
kv_cache = torch.zeros(num_blocks, 2, num_layers, num_heads, block_size, head_dim)
```

### 3.3 Block Table（块表）

Block Table 记录逻辑块到物理块的映射：

```mermaid
classDiagram
    class BlockTable {
        +block_ids: List[int]
        +num_blocks: int
        +append(block_id)
        +get_physical_block(logical_idx) int
    }
```

**示例**：

```
Request A 的 Block Table:
逻辑块索引:  0    1    2    3
            ↓    ↓    ↓    ↓
物理块 ID:  [5]  [2]  [8]  [12]

解释:
- 逻辑块 0 → 物理块 5
- 逻辑块 1 → 物理块 2
- 逻辑块 2 → 物理块 8
- 逻辑块 3 → 物理块 12
```

### 3.4 Slot Mapping（槽位映射）

Slot Mapping 将 token 位置映射到具体的缓存槽位：

```python
def get_slot_mapping(token_position, block_size, block_table):
    """
    token_position: token 在序列中的位置（如 35）
    block_size: 每个 block 的 token 数（如 16）
    block_table: 块表
    """
    logical_block_idx = token_position // block_size  # 35 // 16 = 2
    block_offset = token_position % block_size        # 35 % 16 = 3

    physical_block_id = block_table[logical_block_idx]  # 假设是 8
    slot_id = physical_block_id * block_size + block_offset  # 8 * 16 + 3 = 131

    return slot_id
```

**图解**：

```mermaid
graph LR
    subgraph token_pos["Token 位置 35"]
        T[token_position = 35]
    end

    subgraph calculation["计算"]
        LB["逻辑块 = 35 // 16 = 2"]
        OFF["偏移 = 35 % 16 = 3"]
        PB["物理块 = block_table[2] = 8"]
        SLOT["slot = 8 × 16 + 3 = 131"]
    end

    T --> LB --> PB
    T --> OFF --> SLOT
    PB --> SLOT
```

---

## 4. 内存管理优势

### 4.1 减少显存碎片

```mermaid
graph TB
    subgraph traditional_approach["传统方案"]
        A1["Request A: 2GB 预分配<br/>实际 0.1GB"]
        A2["Request B: 2GB 预分配<br/>实际 0.2GB"]
        A3["Request C: 2GB 预分配<br/>实际 0.1GB"]
        A4["空洞和碎片"]
    end

    subgraph paged_attention_approach["PagedAttention"]
        B1["Block Pool<br/>统一管理所有块"]
        B2["Request A: 2 blocks"]
        B3["Request B: 4 blocks"]
        B4["Request C: 2 blocks"]
        B5["空闲块可立即复用"]
    end

    style A4 fill:#ffcdd2
    style B5 fill:#c8e6c9
```

### 4.2 按需分配

```mermaid
sequenceDiagram
    participant R as Request
    participant S as Scheduler
    participant BP as BlockPool

    R->>S: 开始生成（0 tokens）
    S->>BP: 分配 1 个 block
    BP-->>S: Block 5

    loop 每 16 个 token
        R->>S: 需要新空间
        S->>BP: 分配 1 个 block
        BP-->>S: Block N
    end

    R->>S: 生成完成
    S->>BP: 释放所有 blocks
    Note over BP: 块立即可用于其他请求
```

### 4.3 支持 Copy-on-Write

当多个请求共享相同前缀时，可以共享 Block：

```mermaid
graph TB
    subgraph shared_scenario["共享场景"]
        P["共同前缀<br/>'System prompt...'"]
    end

    subgraph shared_blocks["共享的 Blocks"]
        B1[Block 0]
        B2[Block 1]
        B3[Block 2]
    end

    subgraph request_a["Request A"]
        A["继续生成 A 的内容"]
        AB[Block 10]
    end

    subgraph request_b["Request B"]
        B["继续生成 B 的内容"]
        BB[Block 15]
    end

    P --> B1
    P --> B2
    P --> B3
    B3 --> AB
    B3 --> BB

    style B1 fill:#c8e6c9
    style B2 fill:#c8e6c9
    style B3 fill:#c8e6c9
```

**引用计数**：
- Block 0, 1, 2 的 ref_cnt = 2（被两个请求共享）
- 只有当 ref_cnt = 0 时才真正释放

### 4.4 支持前缀缓存

相同前缀的请求可以直接复用已计算的 KV Cache：

```python
# 前缀缓存示例
request_1 = "你好，请问" + "天气怎么样？"
request_2 = "你好，请问" + "今天星期几？"

# "你好，请问" 的 KV Cache 可以复用！
```

---

## 5. PagedAttention 计算流程

### 5.1 写入 KV Cache

```mermaid
sequenceDiagram
    participant M as Model
    participant PA as PagedAttention
    participant BT as Block Table
    participant KVC as KV Cache Memory

    M->>PA: 新 token 的 K, V
    PA->>BT: 查询 slot_mapping
    BT-->>PA: slot_id = 131
    PA->>KVC: kv_cache[131] = (K, V)
```

### 5.2 读取并计算 Attention

```mermaid
flowchart TD
    subgraph input["输入"]
        Q["Query: 新 token 的 Q"]
        BT["Block Table: 物理块列表"]
    end

    subgraph paged_attn_compute["PagedAttention 计算"]
        FETCH["从各个物理块获取 K, V"]
        COMPUTE["计算 Attention<br/>Q × K^T / √d"]
        SOFTMAX[Softmax]
        WEIGHTED["加权求和 V"]
    end

    subgraph output["输出"]
        O["Attention 输出"]
    end

    Q --> COMPUTE
    BT --> FETCH
    FETCH --> COMPUTE
    COMPUTE --> SOFTMAX
    SOFTMAX --> WEIGHTED
    WEIGHTED --> O
```

### 5.3 代码实现概览

```python
# vllm/v1/attention/ops/paged_attn.py (简化版)

class PagedAttention:
    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,           # [num_tokens, num_heads, head_dim]
        value: torch.Tensor,         # [num_tokens, num_heads, head_dim]
        key_cache: torch.Tensor,     # [num_blocks, block_size, num_heads, head_dim]
        value_cache: torch.Tensor,   # [num_blocks, block_size, num_heads, head_dim]
        slot_mapping: torch.Tensor,  # [num_tokens]
    ):
        """将新的 K, V 写入缓存"""
        # 使用 slot_mapping 确定写入位置
        # slot_mapping[i] 告诉我们 token i 应该写入哪个槽位
        pass

    @staticmethod
    def forward(
        query: torch.Tensor,         # [num_tokens, num_heads, head_dim]
        key_cache: torch.Tensor,     # KV Cache
        value_cache: torch.Tensor,
        block_tables: torch.Tensor,  # [batch, max_blocks] 块表
        context_lens: torch.Tensor,  # [batch] 每个请求的上下文长度
        ...
    ) -> torch.Tensor:
        """执行 PagedAttention 计算"""
        # 1. 根据 block_tables 定位 K, V
        # 2. 计算 Attention
        # 3. 返回输出
        pass
```

---

## 6. 块的动态管理

### 6.1 块的生命周期

```mermaid
stateDiagram-v2
    [*] --> Free: 初始化

    Free --> Allocated: 分配给请求
    Allocated --> Free: 请求完成<br/>ref_cnt=0

    Allocated --> Cached: 启用前缀缓存
    Cached --> Allocated: 缓存命中<br/>ref_cnt++

    Cached --> Free: LRU 驱逐<br/>或缓存失效

    note right of Allocated : ref_cnt >= 1
    note right of Cached : 保留以备复用
```

### 6.2 块分配流程

```python
def allocate_blocks_for_request(request, kv_cache_manager):
    """为请求分配所需的 blocks"""
    num_tokens = len(request.prompt_tokens) + request.num_generated_tokens
    num_blocks_needed = (num_tokens + block_size - 1) // block_size

    blocks = []
    for i in range(num_blocks_needed):
        # 尝试获取空闲块
        block = kv_cache_manager.get_free_block()
        if block is None:
            # 没有空闲块，触发驱逐或返回失败
            return None
        blocks.append(block)

    # 更新请求的块表
    request.block_table = blocks
    return blocks
```

### 6.3 块增长过程

```mermaid
gantt
    title 请求的块分配时间线
    dateFormat X
    axisFormat %s

    section 块分配
    Block 0 (prefill) :done, b0, 0, 16
    Block 1 (prefill 续) :done, b1, 16, 32
    Block 2 (decode) :active, b2, 32, 48
    Block 3 (decode 续) :active, b3, 48, 64
```

---

## 7. CUDA 内核实现

### 7.1 文件位置

- Python 接口：`vllm/v1/attention/ops/paged_attn.py`
- CUDA 内核：`csrc/attention/paged_attention_v1.cu`、`paged_attention_v2.cu`

### 7.2 V1 vs V2 内核

| 特性 | V1 | V2 |
|------|----|----|
| 适用场景 | 短序列 | 长序列 |
| 分块策略 | 简单 | 两级分块 |
| 性能 | 中等 | 更优 |

### 7.3 内核参数

```cpp
// paged_attention_v2.cu (简化)
template<typename T, int BLOCK_SIZE, int NUM_THREADS>
__global__ void paged_attention_v2_kernel(
    T* __restrict__ out,              // 输出
    const T* __restrict__ q,          // Query
    const T* __restrict__ k_cache,    // Key Cache
    const T* __restrict__ v_cache,    // Value Cache
    const int* __restrict__ block_tables,  // 块表
    const int* __restrict__ context_lens,  // 上下文长度
    const float scale,                // 缩放因子
    ...
) {
    // 1. 确定当前线程处理的 query
    // 2. 遍历 block_table 中的所有块
    // 3. 计算 Attention 分数
    // 4. Softmax 和加权求和
}
```

---

## 8. 性能对比

### 8.1 显存效率

| 方案 | 显存利用率 | 最大并发 |
|------|-----------|---------|
| 传统预分配 | 20-40% | 低 |
| PagedAttention | 96%+ | 高 2-4 倍 |

### 8.2 吞吐量提升

```mermaid
graph LR
    subgraph throughput_compare["吞吐量对比"]
        T1["传统方案<br/>1x 基准"]
        T2["PagedAttention<br/>2-4x 提升"]
    end

    style T2 fill:#c8e6c9
```

### 8.3 碎片率

```
传统方案:
- 内部碎片: 50-70%
- 外部碎片: 10-20%
- 总碎片: 60-80%

PagedAttention:
- 内部碎片: < 4% (最后一个块)
- 外部碎片: 0% (固定大小块)
- 总碎片: < 4%
```

---

## 9. 本章小结

### 核心创新

1. **分块存储**：将 KV Cache 分成固定大小的 Block
2. **非连续分配**：Block 可以分散在显存任意位置
3. **按需分配**：生成新 token 时才分配新 Block
4. **块表映射**：通过 Block Table 管理逻辑到物理的映射

### 关键数据结构

| 结构 | 作用 |
|------|------|
| Block | KV Cache 的基本存储单元 |
| Block Table | 逻辑块 → 物理块映射 |
| Slot Mapping | Token 位置 → 缓存槽位 |
| BlockPool | 管理所有空闲块 |

### 优势总结

- **显存效率**：从 20-40% 提升到 96%+
- **减少碎片**：从 60-80% 降到 4% 以下
- **支持共享**：多请求可共享相同前缀的 Block
- **按需增长**：不需要预分配最大长度

### 代码位置

| 功能 | 文件 |
|------|------|
| Python 接口 | `vllm/v1/attention/ops/paged_attn.py` |
| CUDA 内核 | `csrc/attention/paged_attention_v2.cu` |
| 块管理 | `vllm/v1/core/block_pool.py` |
| 块表 | `vllm/v1/worker/block_table.py` |

---

## 思考题

1. 为什么选择固定大小的 Block 而不是可变大小？
2. 前缀缓存和 Copy-on-Write 有什么区别和联系？
3. 如果 block_size 设置得太大或太小，会有什么影响？

---

## 下一步

了解了 PagedAttention 的原理后，让我们来看看 KV Cache Manager 是如何管理这些 Block 的：

👉 [下一章：KV Cache 管理器](02-kv-cache-manager.md)
