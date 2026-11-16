---
title: "稀疏注意力理论基础"
weight: 1
---


> **前置要求**: [Transformer 与注意力机制](../00-prerequisites/01-transformer-attention.md)

---

## 概述

一次 decode 通常要读取之前各个位置的 KV。稀疏注意力先选择本步会用到的历史位置，再让模型读取这部分数据。

---

## 1. 为什么需要稀疏注意力

### 1.1 标准注意力的问题

```mermaid
graph TB
    subgraph problem["标准注意力的挑战"]
        P1["O(n²) 计算复杂度"]
        P2["O(n) KV Cache 内存"]
        P3["长序列带宽瓶颈"]
    end

    subgraph impact["影响"]
        I1["序列长度受限"]
        I2["并发数减少"]
        I3["推理延迟增加"]
    end

    P1 --> I1
    P2 --> I2
    P3 --> I3
```
**具体数据**:
- 4K 序列长度: 需要计算 16M 个注意力分数
- 32K 序列长度: 需要计算 1B 个注意力分数
- 128K 序列长度: 需要计算 16B 个注意力分数

### 1.2 稀疏注意力的核心洞察

研究发现，注意力权重在实践中通常是**高度稀疏**的：

```mermaid
graph TB
    subgraph observation["观察到的现象"]
        O1["大多数注意力权重接近 0"]
        O2["少数 token 获得大部分注意力"]
        O3["存在可预测的注意力模式"]
    end

    subgraph conclusion["结论"]
        C1["可以只计算重要的注意力"]
        C2["可以只加载重要的 KV"]
        C3["精度损失可控"]
    end

    O1 --> C1
    O2 --> C2
    O3 --> C3
```
---
## 2. 注意力稀疏模式

### 2.1 局部注意力（Local Attention）

相邻的 token 倾向于相互关注：

```
注意力矩阵示意（对角带状）：
     T1  T2  T3  T4  T5  T6  T7  T8
T1 [  1   ·   ·   ·   ·   ·   ·   · ]
T2 [  1   1   ·   ·   ·   ·   ·   · ]
T3 [  ·   1   1   ·   ·   ·   ·   · ]
T4 [  ·   ·   1   1   ·   ·   ·   · ]
T5 [  ·   ·   ·   1   1   ·   ·   · ]
T6 [  ·   ·   ·   ·   1   1   ·   · ]
T7 [  ·   ·   ·   ·   ·   1   1   · ]
T8 [  ·   ·   ·   ·   ·   ·   1   1 ]

局部窗口大小 = 2
```

### 2.2 Sink Token（锚点 Token）

序列开始的 token 通常获得较高注意力：

```
     T1  T2  T3  T4  T5  T6  T7  T8
T8 [  1   ·   ·   ·   ·   ·   1   1 ]
      ↑                       ↑   ↑
    Sink Token            局部 Token
```

### 2.3 语义锚点

关键词和重要实体获得更多注意力：

```
输入: "北京是中国的首都，它有着悠久的历史"

注意力分布（对于"它"）:
  "北京" → 高
  "是" → 低
  "中国" → 中
  "的" → 低
  "首都" → 高
  ...
```

### 2.4 稀疏模式分类

```mermaid
mindmap
  root["稀疏注意力模式"]
    static["静态模式"]
      s1["固定窗口"]
      s2["固定步长"]
      s3["预定义模式"]
    dynamic["动态模式"]
      d1["Top-K 选择"]
      d2["阈值过滤"]
      d3["学习预测"]
    hybrid["混合模式"]
      h1["局部 + 全局"]
      h2["Sink + 局部"]
      h3["分层稀疏"]
```

---
## 3. 稀疏注意力的优化原理
### 3.1 减少计算
```mermaid
graph LR
    subgraph full["标准注意力"]
        F1["计算所有 n² 个分数"]
    end
    subgraph sparse["稀疏注意力"]
        S1["只计算 k 个分数"]
        S2["k << n²"]
    end

    full --> |"O(n²) → O(nk)"| sparse
```
### 3.2 减少内存读取
{{< stateboard >}}
title: "本步需要从内存读取哪些 KV"
rows: ["历史位置", "完整读取", "稀疏读取"]
steps:
  - title: "完整工作集"
    detail: "对单个 query，四个历史位置都参与完整注意力。"
    cells:
      - ["A", "B", "C", "D"]
      - ["读", "读", "读", "读"]
      - ["·", "·", "·", "·"]
    focus: [[1, 0], [1, 1], [1, 2], [1, 3]]
  - title: "选中位置"
    detail: "示例筛选器保留 B、D。它们的索引对应真实的历史 KV。"
    cells:
      - ["A", "B", "C", "D"]
      - ["读", "读", "读", "读"]
      - ["·", "B", "·", "D"]
    focus: [[2, 1], [2, 3]]
  - title: "读取选中块"
    detail: "在支持按块稀疏读取的实现中，其余位置的实际 KV 可跳过，这一步仍有选择开销。"
    cells:
      - ["A", "B", "C", "D"]
      - ["读", "读", "读", "读"]
      - ["·", "读", "·", "读"]
    focus: [[2, 1], [2, 3]]
{{< /stateboard >}}

### 3.3 精度与效率权衡

| 稀疏比例 | 计算节省 | 内存节省 | 精度影响 |
|----------|----------|----------|----------|
| 10% | 10x | 10x | 可能较大 |
| 30% | 3.3x | 3.3x | 通常可接受 |
| 50% | 2x | 2x | 较小 |
| 70% | 1.4x | 1.4x | 很小 |
---
## 4. UCM 稀疏算法概览

### 4.1 算法分类

```mermaid
graph TB
    subgraph ucm_sparse["UCM 稀疏算法"]
        subgraph retrieval["检索类"]
            ESA["ESA<br/>Essential Sparse Attention"]
            GSA["GSA<br/>Gather-Scatter Attention"]
        end

        subgraph ondevice["GPU 端"]
            GSAOD["GSA On-Device<br/>GPU 端检索"]
        end

        subgraph reuse["复用类"]
            Blend["Blend<br/>非前缀 KV 复用"]
        end

        subgraph consistency["一致性"]
            KVStar["KVStar<br/>多步一致性"]
        end

        subgraph position["位置"]
            RERoPE["RERoPE<br/>位置外推"]
        end
    end
```

### 4.2 算法对比

| 算法 | 核心思想 | 适用场景 | 稀疏选择方式 |
|------|----------|----------|--------------|
| **ESA** | 检索重要 Block | 长上下文 | CPU 端预计算 |
| **GSA** | 收集-分散注意力 | 通用 | Prefetch 引擎 |
| **GSA On-Device** | GPU 端检索 | 低延迟 | Hash + Top-K |
| **Blend** | 非前缀复用 | RAG、多轮对话 | 位置校正 |
| **KVStar** | 多步一致 | 推测解码 | 状态跟踪 |
| **RERoPE** | 位置外推 | 超长序列 | RoPE 修正 |

---
## 5. Block 表示法
### 5.1 为什么用 Block
UCM 以 Block（而非单个 Token）为单位管理 KV Cache：
```mermaid
graph TB
    subgraph token_level["Token 级别"]
        T1["优点: 精细控制"]
        T2["缺点: 开销大、碎片化"]
    end

    subgraph block_level["Block 级别"]
        B1["优点: 批量操作、对齐友好"]
        B2["缺点: 粒度较粗"]
    end
    block_level --> |"UCM 选择"| Winner["Block 级别"]
```

### 5.2 Block 表示的稀疏选择

{{< stateboard >}}
title: "用分数选择两个历史块"
rows: ["候选块", "示例分数", "所选块"]
steps:
  - title: "准备块级表示"
    detail: "历史有四个完整块，每个块保留供筛选用的表示。"
    cells:
      - ["A", "B", "C", "D"]
      - ["·", "·", "·", "·"]
      - ["·", "·", "·", "·"]
    focus: [[0, 0], [0, 1], [0, 2], [0, 3]]
  - title: "计算相关性"
    detail: "图中是假定分数，用于说明排序选择。"
    cells:
      - ["A", "B", "C", "D"]
      - ["0.2", "0.8", "0.1", "0.6"]
      - ["·", "·", "·", "·"]
    focus: [[1, 0], [1, 1], [1, 2], [1, 3]]
  - title: "取前两名"
    detail: "预算为两块，B 与 D 进入本次注意力的候选工作集。"
    cells:
      - ["A", "B", "C", "D"]
      - ["0.2", "0.8", "0.1", "0.6"]
      - ["·", "B", "·", "D"]
    focus: [[2, 1], [2, 3]]
{{< /stateboard >}}
### 5.3 Block 表示方法
| 方法 | 说明 | UCM 应用 |
|------|------|----------|
| 均值池化 | Block 内 KV 平均 | ESA |
| Hash 编码 | 二进制编码 | GSA On-Device |
| 首 Token | 使用首个 Token | 简单场景 |
| 加权 | 注意力加权平均 | 高精度场景 |

---

## 6. 稀疏注意力的挑战

### 6.1 选择开销

选择哪些 Token/Block 是重要的，本身也有计算开销：

```mermaid
graph TB
    subgraph challenge["选择开销"]
        C1["需要某种形式的全局信息"]
        C2["选择算法本身消耗时间"]
        C3["可能需要额外的数据结构"]
    end

    subgraph solution["UCM 解决方案"]
        S1["预计算 Block 表示"]
        S2["异步预取"]
        S3["GPU 端快速检索"]
    end

    challenge --> solution
```
### 6.2 精度保证
如何确保稀疏不会损害生成质量：
| 策略 | 说明 |
|------|------|
| 保留 Sink Token | 始终包含序列开始的 token |
| 保留局部窗口 | 始终包含最近的 token |
| 自适应稀疏比例 | 根据任务动态调整 |
| 质量监控 | 监控输出质量指标 |

---

## 7. 关键概念总结

| 概念 | 说明 | 重要性 |
|------|------|--------|
| 注意力稀疏性 | 权重集中在少数 token | 稀疏优化的理论基础 |
| 局部注意力 | 关注相邻 token | 保证局部一致性 |
| Sink Token | 序列开始的锚点 | 避免信息丢失 |
| Block 表示 | 块级别的稀疏选择 | UCM 的管理单位 |
| Top-K 选择 | 选择最相关的 K 个 | 常用的稀疏方法 |

---

## 延伸阅读

- [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150)
- [BigBird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062)
- [Efficient Attention: Attention with Linear Complexities](https://arxiv.org/abs/1812.01243)
