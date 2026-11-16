---
title: "KV Cache 基础原理"
weight: 4
---

模型读完手册后留下各层的 K 和 V。下一次请求能复用多少，取决于输入 token 和它们之前的上下文。

---


## 1. 为什么需要 KV Cache

### 1.1 自回归生成的重复计算问题

在没有 KV Cache 的情况下，每生成一个 token 都需要重新计算所有之前 token 的 K 和 V：

```mermaid
graph TB
    subgraph "Without KV Cache"
        Step1["Step 1: 计算 [A] 的 KV"]
        Step2["Step 2: 重新计算 [A, B] 的 KV"]
        Step3["Step 3: 重新计算 [A, B, C] 的 KV"]
        Step4["Step 4: 重新计算 [A, B, C, D] 的 KV"]

        Step1 --> Step2 --> Step3 --> Step4

        Note1["大量重复计算!"]
    end
```

### 1.2 KV Cache 的解决方案

缓存已经计算过的 K 和 V，新 token 只需计算自己的 KV 并追加：

{{< stateboard >}}
title: "生成第三个词时复用前两个位置"
rows: ["输入", "已存 KV", "本步新增"]
steps:
  - title: "第一个词"
    detail: "处理 A 后留下它在各层的 K、V。"
    cells:
      - ["A", "·", "·"]
      - ["A", "·", "·"]
      - ["A", "·", "·"]
    focus: [[2, 0]]
  - title: "第二个词"
    detail: "B 读取 A 的缓存，并把自己的 K、V 追加到第二个位置。"
    cells:
      - ["A", "B", "·"]
      - ["A", "B", "·"]
      - ["·", "B", "·"]
    focus: [[1, 1], [2, 1]]
  - title: "第三个词"
    detail: "C 读取 A、B 的缓存，更新的位置只有第三格。"
    cells:
      - ["A", "B", "C"]
      - ["A", "B", "C"]
      - ["·", "·", "C"]
    focus: [[1, 2], [2, 2]]
{{< /stateboard >}}

### 1.3 计算量对比

| 步骤 | 无 KV Cache | 有 KV Cache | 节省 |
|------|------------|------------|------|
| Step 1 | 1 token | 1 token | 0% |
| Step 2 | 2 tokens | 1 token | 50% |
| Step 3 | 3 tokens | 1 token | 67% |
| Step n | n tokens | 1 token | (n-1)/n |

生成 1000 tokens 时，KV Cache 减少了约 **99.9%** 的重复计算。

---

## 2. KV Cache 的存储结构

### 2.1 每层的 KV Cache

每个 Transformer 层都有独立的 KV Cache：

```mermaid
graph TB
    subgraph "KV Cache 结构"
        subgraph "Layer 0"
            K0["K: [seq_len, num_kv_heads, head_dim]"]
            V0["V: [seq_len, num_kv_heads, head_dim]"]
        end

        subgraph "Layer 1"
            K1["K: [seq_len, num_kv_heads, head_dim]"]
            V1["V: [seq_len, num_kv_heads, head_dim]"]
        end

        subgraph "Layer N-1"
            KN["K: [seq_len, num_kv_heads, head_dim]"]
            VN["V: [seq_len, num_kv_heads, head_dim]"]
        end
    end
```

### 2.2 维度说明

以 LLaMA-7B 为例：

| 参数 | 值 | 说明 |
|------|---|------|
| num_layers | 32 | 层数 |
| hidden_size | 4096 | 隐藏维度 |
| num_attention_heads | 32 | 注意力头数 |
| num_kv_heads | 32 (MHA) / 8 (GQA) | KV 头数 |
| head_dim | 128 | 每个头的维度 |

### 2.3 PyTorch 中的表示

```python
# KV Cache 的典型存储方式
class KVCache:
    def __init__(self, num_layers, max_seq_len, num_kv_heads, head_dim, dtype):
        self.k_cache = torch.zeros(
            num_layers, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.v_cache = torch.zeros(
            num_layers, max_seq_len, num_kv_heads, head_dim,
            dtype=dtype, device='cuda'
        )
        self.seq_len = 0

    def update(self, layer_idx, new_k, new_v):
        """追加新的 KV"""
        batch_len = new_k.shape[0]
        self.k_cache[layer_idx, self.seq_len:self.seq_len + batch_len] = new_k
        self.v_cache[layer_idx, self.seq_len:self.seq_len + batch_len] = new_v
        if layer_idx == 0:  # 只在第一层更新 seq_len
            self.seq_len += batch_len

    def get(self, layer_idx):
        """获取当前层的 KV"""
        return (
            self.k_cache[layer_idx, :self.seq_len],
            self.v_cache[layer_idx, :self.seq_len]
        )
```

---

## 3. KV Cache 的内存计算

### 3.1 内存公式

KV Cache 的内存占用：

$$
\text{Memory} = 2 \times L \times S \times H_{kv} \times D \times \text{sizeof(dtype)}
$$

其中：
- $2$: K 和 V 两个张量
- $L$: 层数
- $S$: 序列长度
- $H_{kv}$: KV 头数
- $D$: 每个头的维度
- $\text{sizeof(dtype)}$: 数据类型大小（FP16=2, FP32=4）

### 3.2 实际计算示例

```python
def calculate_kv_cache_memory(
    num_layers=32,
    seq_len=2048,
    num_kv_heads=8,  # GQA
    head_dim=128,
    dtype_bytes=2,  # FP16
    batch_size=1
):
    """计算 KV Cache 内存占用"""
    memory_bytes = (
        2  # K and V
        * num_layers
        * seq_len
        * num_kv_heads
        * head_dim
        * dtype_bytes
        * batch_size
    )
    return memory_bytes

# 一个 32 层 GQA 模型，2048 tokens，batch_size=1
memory = calculate_kv_cache_memory()
print(f"KV Cache Memory: {memory / 1024**2:.2f} MiB")
# 256 MiB

# LLaMA-70B, 8192 tokens, batch_size=8
memory_70b = calculate_kv_cache_memory(
    num_layers=80,
    seq_len=8192,
    num_kv_heads=8,
    head_dim=128,
    batch_size=8
)
print(f"KV Cache Memory (70B): {memory_70b / 1024**3:.2f} GiB")
# 20 GiB
```

### 3.3 KV Cache 内存占用趋势

{{< stateboard >}}
title: "每个输入位置占 128 KiB 时"
rows: ["窗口长度", "有效 KV"]
steps:
  - title: "一个窗口"
    detail: "32 层、8 个 KV 头、每头 128 维、FP16，一条序列每个 token 用 128 KiB。"
    cells:
      - ["1 Ki"]
      - ["128 MiB"]
    focus: [[0, 0], [1, 0]]
  - title: "窗口延长四倍"
    detail: "同一条序列增长到 4096 token，各层的有效 KV 一起增加为 512 MiB。"
    cells:
      - ["4 Ki"]
      - ["512 MiB"]
    focus: [[0, 0], [1, 0]]
  - title: "长序列"
    detail: "16384 token 对应 2 GiB。这里计算张量有效字节，块对齐另算。"
    cells:
      - ["16 Ki"]
      - ["2 GiB"]
    focus: [[0, 0], [1, 0]]
{{< /stateboard >}}

**关键观察**: KV Cache 内存随序列长度**线性增长**，成为长上下文的主要瓶颈。

---

## 4. KV Cache 复用

### 4.1 复用的动机

在很多场景下，多个请求共享相同的上下文：

```mermaid
graph TB
    subgraph "KV Cache 复用场景"
        System["System Prompt<br>通用指令"] --> Q1["Query 1"]
        System --> Q2["Query 2"]
        System --> Q3["Query 3"]

        Note["System Prompt 的 KV 可以复用"]
    end
```

### 4.2 Prefix Caching

**原理**: 缓存相同前缀的 KV Cache

{{< stateboard >}}
title: "相同前缀在哪一块结束"
rows: ["请求甲", "请求乙", "匹配"]
steps:
  - title: "比较第一块"
    detail: "示例每格是一块。两个请求从相同的系统提示开始。"
    cells:
      - ["系统", "手册", "甲问"]
      - ["系统", "手册", "乙问"]
      - ["命中", "·", "·"]
    focus: [[0, 0], [1, 0], [2, 0]]
  - title: "继续向后"
    detail: "手册前面具有相同上下文，第二个完整块也命中。"
    cells:
      - ["系统", "手册", "甲问"]
      - ["系统", "手册", "乙问"]
      - ["命中", "命中", "·"]
    focus: [[0, 1], [1, 1], [2, 1]]
  - title: "遇到分歧"
    detail: "两个问题的 token 不同，乙的引擎复用前两块，计算自己的问题。"
    cells:
      - ["系统", "手册", "甲问"]
      - ["系统", "手册", "乙问"]
      - ["命中", "命中", "计算"]
    focus: [[0, 2], [1, 2], [2, 2]]
{{< /stateboard >}}

**限制**: 只能复用**连续的前缀**，中间的 chunk 无法复用。

### 4.3 CacheBlend 的创新

CacheBlend 突破了 Prefix Caching 的限制，支持**任意位置**的 KV Cache 复用：

```mermaid
graph TB
    subgraph "CacheBlend vs Prefix Caching"
        subgraph "Prefix Caching"
            P1["Chunk A"] --> P2["Chunk B"]
            P2 --> P3["Query"]
            Note1["只能复用 Chunk A"]
        end

        subgraph "CacheBlend"
            C1["Chunk A"] --> C2["Chunk B"]
            C2 --> C3["Query"]
            Note2["可以复用 Chunk A + B"]
        end
    end
```

---

## 5. KV Cache 优化技术

### 5.1 优化方法总览

```mermaid
graph TB
    subgraph "KV Cache 优化技术"
        A["GQA/MQA<br>减少 KV 头数"] --> Benefit1["4-8x 内存节省"]
        B["KV 量化<br>降低精度"] --> Benefit2["2-4x 内存节省"]
        C["KV 压缩<br>稀疏化/淘汰"] --> Benefit3["动态调整"]
        D["PagedAttention<br>按需分配"] --> Benefit4["减少碎片"]
    end
```

### 5.2 Grouped Query Attention (GQA)

```python
# MHA: 32 个 Q heads, 32 个 KV heads
# GQA: 32 个 Q heads, 8 个 KV heads (每 4 个 Q 共享 1 个 KV)

class GQA(nn.Module):
    def __init__(self, hidden_size, num_q_heads, num_kv_heads):
        self.num_q_heads = num_q_heads      # 32
        self.num_kv_heads = num_kv_heads    # 8
        self.num_groups = num_q_heads // num_kv_heads  # 4

        self.q_proj = nn.Linear(hidden_size, num_q_heads * head_dim)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 更小
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim)  # 更小

    def forward(self, x, kv_cache=None):
        q = self.q_proj(x)  # [batch, seq, 32 * head_dim]
        k = self.k_proj(x)  # [batch, seq, 8 * head_dim]
        v = self.v_proj(x)  # [batch, seq, 8 * head_dim]

        # 扩展 K, V 以匹配 Q 的头数
        k = k.repeat_interleave(self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_groups, dim=2)

        return attention(q, k, v)
```

### 5.3 KV Cache 量化

```python
# FP16 -> INT8 量化
def quantize_kv_cache(kv_cache, scale=None):
    """将 KV Cache 从 FP16 量化到 INT8"""
    if scale is None:
        scale = kv_cache.abs().max() / 127.0

    quantized = (kv_cache / scale).round().clamp(-128, 127).to(torch.int8)
    return quantized, scale

def dequantize_kv_cache(quantized, scale):
    """反量化"""
    return quantized.to(torch.float16) * scale
```

### 5.4 KV Cache 压缩（H2O, StreamingLLM）

```python
class H2OCache:
    """Heavy-Hitter Oracle: 保留高注意力权重的 KV"""

    def __init__(self, max_size, heavy_ratio=0.2):
        self.max_size = max_size
        self.heavy_ratio = heavy_ratio

    def update(self, k, v, attn_weights):
        if self.cache_len + 1 > self.max_size:
            # 根据累积注意力权重淘汰
            importance = attn_weights.sum(dim=0)  # 累积注意力
            keep_idx = importance.topk(int(self.max_size * self.heavy_ratio)).indices

            # 保留重要的 + 最近的
            self.k_cache = torch.cat([self.k_cache[keep_idx], k])
            self.v_cache = torch.cat([self.v_cache[keep_idx], v])
```

---

## 6. RAG 场景的 KV Cache 挑战

### 6.1 RAG 场景描述

RAG（Retrieval-Augmented Generation）需要将多个检索到的文本块作为上下文：

```mermaid
graph TB
    subgraph "RAG 场景"
        Query["用户查询"] --> Retriever["检索器"]
        Retriever --> Chunk1["文本块 1"]
        Retriever --> Chunk2["文本块 2"]
        Retriever --> Chunk3["文本块 3"]

        Chunk1 --> Concat["拼接上下文"]
        Chunk2 --> Concat
        Chunk3 --> Concat
        Query --> Concat

        Concat --> LLM["LLM 推理"]
        LLM --> Answer["生成答案"]
    end
```

### 6.2 现有方案的问题

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Full Recompute** | 质量最高 | TTFT 延迟大 |
| **Prefix Caching** | 可复用前缀 | 非前缀 chunk 无法复用 |
| **Full KV Reuse** | 速度最快 | 丢失 Cross-Attention，质量差 |

### 6.3 Cross-Attention 的问题

当直接复用 KV Cache 时，**Cross-Attention 会丢失**：

{{< stateboard >}}
title: "M 的位置和上文同时改变"
rows: ["首次输入", "新输入", "M 的上下文"]
steps:
  - title: "收集手册"
    detail: "缓存来自单独输入的手册 M。"
    cells:
      - ["M", "问题"]
      - ["·", "·"]
      - ["起始", "·"]
    focus: [[0, 0]]
  - title: "检索顺序变化"
    detail: "下一次请求把检索材料 R 拼在 M 前面。"
    cells:
      - ["M", "问题"]
      - ["R", "M"]
      - ["R 在前", "·"]
    focus: [[1, 0], [1, 1]]
  - title: "比较旧缓存"
    detail: "M 的起点右移，也可以读到 R。位置修正和上下文重算分别处理这两个变化。"
    cells:
      - ["M", "问题"]
      - ["R", "M"]
      - ["新位置", "新上文"]
    focus: [[2, 0], [2, 1]]
{{< /stateboard >}}

### 6.4 CacheBlend 的解决方案

CacheBlend 通过**选择性重计算**恢复 Cross-Attention：

```mermaid
graph TB
    subgraph "CacheBlend 解决方案"
        Step1["1. 加载预计算的 KV Cache"]
        Step2["2. 识别 HKVD Tokens<br>(KV 偏差最大的 tokens)"]
        Step3["3. 仅重计算 HKVD Tokens 的 KV"]
        Step4["4. 融合新旧 KV Cache"]
        Step5["5. 恢复大部分 Cross-Attention"]

        Step1 --> Step2 --> Step3 --> Step4 --> Step5

        Result["结果: 15% 重计算量<br>恢复 >95% 质量"]
    end
```

---

## 总结

1. **为什么需要 KV Cache**: 避免自回归生成中的重复计算
2. **存储结构**: 每层独立存储 K 和 V 张量
3. **内存计算**: 随序列长度线性增长
4. **复用技术**: Prefix Caching 的限制
5. **优化方法**: GQA、量化、压缩、PagedAttention
6. **RAG 挑战**: Cross-Attention 丢失问题

CacheBlend 正是为了解决 RAG 场景中 KV Cache 复用导致的 Cross-Attention 丢失问题而设计的。

---

## 下一步

现在你已经具备了理解 CacheBlend 的所有前置知识！

- [CacheBlend 背景与动机](../01-introduction/01-background.md)
- [CacheBlend 核心概念](../02-theory/01-core-concepts.md)
