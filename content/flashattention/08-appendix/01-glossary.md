---
title: "术语表"
weight: 1
---

## 目录

- [A-D](#a-d)
- [E-H](#e-h)
- [I-N](#i-n)
- [O-S](#o-s)
- [T-Z](#t-z)

---

## A-D

### Attention Sink
注意力沉积。指在滑动窗口注意力中，保留序列开头若干 token 的注意力连接，即使它们不在窗口范围内。由 `sink_token_length` 参数控制。常用于流式推理场景。

### AppendKV
追加 KV。在推理中将新的 K/V token 追加到已有的 KV Cache 中。`flash_attn_with_kvcache` 通过 `k` 和 `v` 参数实现原地更新。

### Block Table
页表。Paged KV Cache 中用于记录每个序列使用了哪些物理页的索引表。形状为 `(batch_size, max_num_blocks_per_seq)`，`int32` 类型。

### Causal Masking
因果遮蔽。限制位置 $i$ 只能注意到位置 $j \le i$ 的注意力模式。用于自回归语言模型。Flash Attention 中对齐到注意力矩阵的右下角。

### ClusterShape
集群形状。Hopper 架构中 Thread Block Cluster 的配置，用于 TMA 多播优化。常见值 `(1, 1, 1)` 或 `(2, 1, 1)`。

### Consumer (Warp Group)
消费者 Warp Group。Warp Specialization 中负责执行 GMMA 矩阵乘法和 Softmax 计算的 Warp Group。与 Producer 协作。

### CpAsync
异步全局内存到共享内存的拷贝指令（SM80+）。在 PackGQA 模式下用于加载不连续的 Q 数据，因为 TMA 要求数据连续。

### cu_seqlens
累积序列长度。变长 API 中用于标记各序列在拼接张量中的起止位置。形状 `(batch_size + 1,)`，`int32`。例如序列长度 `[5, 3]` 对应 `cu_seqlens = [0, 5, 8]`。

### Decode Phase
解码阶段。自回归生成中逐 token 生成的阶段，`seqlen_q = 1`，从 KV Cache 中读取历史 KV。属于 Memory Bound 任务。

### Descale
反量化缩放因子。FP8 推理中用于恢复量化前的数值范围。Flash Attention 支持 per-head 的 Q/K/V descale，形状 `(batch_size, num_heads_k)`。

---

## E-H

### E4M3 (FP8)
8 位浮点格式，4 位指数 + 3 位尾数。最大值 448，适用于推理。Flash Attention 使用 `torch.float8_e4m3fn`。

### Epilogue
收尾阶段。CUDA 内核中主循环结束后的处理，包括 Softmax 归一化、LSE 存储、输出写回等。实现在 `hopper/epilogue_fwd.hpp` 和 `hopper/epilogue_bwd.hpp`。

### FastDivmod
快速整除取模。CUTLASS 提供的优化工具，用硬件乘法指令替代开销大的整数除法。在 PackGQA 和 Paged KV 中频繁使用。

### Flash Decoding
闪存解码。Split-KV 的别名。将 KV 序列分成多个 split 并行计算，通过 Combine Kernel 合并结果。解决 Decode 阶段 GPU 利用率低的问题。

### GMMA
Group Matrix Multiply-Accumulate。Hopper 架构的矩阵乘法指令，操作 Warp Group 级别（128 线程）的数据。支持 FP16/BF16/FP8 输入，FP32 累加。

### GQA (Grouped-Query Attention)
分组查询注意力。每 $G$ 个 Q 头共享 1 个 KV 头。LLaMA-2/3、Mistral 等模型使用。减少 KV Cache 大小和推理带宽消耗。

### Head Dimension (headdim)
注意力头维度。单个注意力头的特征维度 $d$。Flash Attention 支持最大 256，内部自动对齐到 8 的倍数。

---

## I-N

### IntraWGOverlap
Warp Group 内重叠。一种流水线优化，在同一个 Consumer Warp Group 内重叠 GEMM(QK) 和 GEMM(PV) 的执行，隐藏延迟。

### KV Cache
键值缓存。推理中存储历史 K/V 的 GPU 内存区域，避免重复计算。支持连续和分页两种模式。

### LSE (Log-Sum-Exp)
对数-求和-指数。Softmax 的副产品 $\text{LSE}_i = \log \sum_j \exp(S_{ij})$。在 Split-KV 合并和反向传播中使用。形状 `(batch, num_heads, seqlen_q)`。

### Max_offset
最大偏移量。FP8 模式下的 Softmax 优化技巧。通过从 max 中减去固定偏移（默认 8），将 $e^{x-\max}$ 的范围从 $[0, 1]$ 扩展到 $[0, 256]$，提升 E4M3 表示精度。

### MHA (Multi-Head Attention)
多头注意力。标准 Transformer 注意力机制，Q/K/V 拥有相同数量的头。

### MQA (Multi-Query Attention)
多查询注意力。GQA 的极端形式，所有 Q 头共享 1 个 KV 头（`num_heads_kv = 1`）。Falcon、PaLM 使用。

### n_block_max / n_block_min
KV 块范围上下界。`get_n_block_min_max()` 计算的值，定义每个 Q 块需要遍历的 KV 块范围 $[n\_block\_min, n\_block\_max)$。块级跳过的核心。

---

## O-S

### Online Softmax
在线 Softmax。在单次遍历中同时计算 max、sum 和归一化结果的算法。Flash Attention 的核心算法基础，避免了对 $N \times N$ 矩阵的多次扫描。

### PackGQA
GQA 打包。将同一 KV 头组下的多个 Q 头沿 M 维度打包到同一个 Tile 中，提高 GPU 利用率。由 `should_pack_gqa()` 启发式决策是否启用。

### Paged KV Cache
分页 KV 缓存。将 KV Cache 分成固定大小的页（page），通过 Block Table 间接寻址。类似操作系统的虚拟内存。避免为每个序列预分配最大长度的内存。

### Pipeline (SMEM Pipeline)
共享内存流水线。Hopper 架构的异步 Pipeline 机制，协调 Producer（TMA 加载）和 Consumer（GMMA 计算）的数据交接。通常有 2-3 个 stage。

### Prefill Phase
预填充阶段。推理中处理完整 prompt 的阶段，`seqlen_q` 等于 prompt 长度。属于 Compute Bound 任务。

### Producer (Warp Group)
生产者 Warp Group。Warp Specialization 中负责通过 TMA 从全局内存加载 K/V 到共享内存的 Warp Group。

### qhead_per_khead
每个 KV 头对应的 Q 头数。即 GQA 的 group size $G = H / H_k$。

### Rescale O
输出重缩放。Online Softmax 中，当新的 KV 块改变了 row_max 时，需要对之前累积的输出 $O$ 乘以修正因子 $e^{max_{old} - max_{new}}$。

### Softcap
分数截断。通过 $\text{softcap} \cdot \tanh(S / \text{softcap})$ 限制注意力分数的幅度。Gemma 2 等模型使用。在 masking 之前应用。

### Split-KV
KV 分裂。将 KV 序列分成多个 split，每个 split 由独立的 Thread Block 并行处理。通过 Combine Kernel 使用 Online Softmax 合并部分结果。

### SMEM (Shared Memory)
共享内存。GPU SM 内的高速暂存器（~164-228 KB per SM on Hopper）。Flash Attention 将 Q/K/V 的 Tile 加载到 SMEM 以减少 GMEM 访问。

---

## T-Z

### Tiling
分块。将大矩阵分割成小的 Tile 块进行计算。Flash Attention 的核心策略——将 $N \times N$ 的注意力矩阵分成 $kBlockM \times kBlockN$ 的小块，逐块计算并在线累积结果。

### Tile Scheduler
Tile 调度器。决定各 Thread Block 处理哪些 Tile 的策略。Flash Attention 有 5 种调度器：SingleTile、StaticPersistent、DynamicPersistent、VarlenDynamic、SingleBwdLPT。

### TMA (Tensor Memory Accelerator)
张量内存加速器。Hopper 架构的硬件单元，支持异步、多维、带 swizzle 的内存拷贝。Flash Attention 用 TMA 加载 K/V（但 PackGQA 模式下的 Q 不使用 TMA）。

### Varlen (Variable Length)
变长。支持 batch 内不同序列拥有不同长度的模式。所有序列拼接为连续张量，通过 `cu_seqlens` 标记边界。消除 padding 浪费。

### V_colmajor
V 列主序。V 矩阵以列主序（K-major）存储在内存中。FP8 模式下可以跳过内核中的 V 转置操作，提升性能。

### Warp Specialization
Warp 专化。Hopper 架构的编程模型，将同一 Thread Block 中的 Warp Group 分为 Producer 和 Consumer 角色，分别负责数据加载和计算，形成流水线。

### window_size
窗口大小。滑动窗口注意力的参数 `(left, right)`。`-1` 表示不限制。例如 `(256, 0)` 表示因果 + 向左 256 个位置的滑动窗口。

---

## 导航

- 上一篇：[性能测试](../07-usage-tutorial/04-benchmarking.md)
- 下一篇：[论文阅读指南](02-paper-reading-guide.md)
- [返回目录](../README.md)
