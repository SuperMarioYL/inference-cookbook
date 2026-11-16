---
title: "术语表"
weight: 1
---

读到 **V1** 时，要先区分引擎架构、模型执行器和 attention backend。vLLM 0.33.0 中 V1 引擎可以选择不同的 GPU model runner，attention 实现还取决于平台与模型。对照源码入口可查[固定版本代码索引](/vllm/appendix/code-map/)和[运行器架构](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/model_runner_v2.md)。

本术语表按字母顺序列出了 vLLM 文档中使用的关键术语及其解释。

---

## A

### Activation（激活值）
神经网络中间层的输出张量。在推理过程中，激活值存储在 GPU 显存中，占用一定的显存空间。

### AllGather（全收集）
分布式通信原语，将所有进程的数据收集到每个进程。用于张量并行中收集分片的输出。

### AllReduce（全归约）
分布式通信原语，将所有进程的数据进行归约（如求和）并将结果分发到每个进程。是张量并行中最常用的通信操作。

### Attention（注意力）
Transformer 架构的核心机制，用于计算序列中不同位置之间的关联性。通过 Query、Key、Value 三个矩阵计算注意力权重。

### AWQ (Activation-aware Weight Quantization)
一种激活感知的权重量化方法，通过保护对输出影响大的通道来减少量化误差。

### Async（异步）
vLLM 中的异步编程模式，允许在等待 I/O 或计算完成时处理其他任务，提高整体效率。

---

## B

### Batch Size（批大小）
同时处理的请求数量。更大的批大小通常能提高 GPU 利用率和吞吐量。

### Block（块）
PagedAttention 中 KV Cache 的基本分配单位。每个块包含固定数量的 token 的 K 和 V 值。

### Block Pool（块池）
管理所有物理块的组件，负责块的分配、释放和 LRU 驱逐。

### Block Table（块表）
记录逻辑块到物理块映射关系的数据结构。类似于操作系统的页表。

### BF16 (Brain Floating Point 16)
Google 开发的 16 位浮点格式，指数位与 FP32 相同，精度略低于 FP16 但动态范围更大。

---

## C

### Causal Mask（因果掩码）
在自回归生成中使用的掩码，防止模型看到未来的 token。也称为 Attention Mask。

### Chunked Prefill（分块预填充）
将长输入分成多个小块进行处理的技术，可以与 Decode 阶段交错执行，降低延迟。

### Continuous Batching（连续批处理）
vLLM 的核心调度策略，允许在每个迭代动态添加或移除请求，提高 GPU 利用率。

### Copy-on-Write（写时复制）
内存管理技术，多个请求可以共享相同的 KV Cache 块，只在需要修改时才创建副本。

### CUDA
NVIDIA 的并行计算平台和编程模型，用于 GPU 加速计算。

### CUDA Graph
NVIDIA 的优化技术，将一系列 CUDA 操作捕获为图形，减少 kernel launch 开销。

---

## D

### Data Parallelism（数据并行）
分布式策略，将数据分配到多个设备，每个设备持有完整的模型副本。

### Decode（解码阶段）
LLM 生成过程的第二阶段，逐个生成输出 token。特点是计算量小但依赖 KV Cache 读取。

### Draft Model（草稿模型）
投机解码中使用的小型模型，快速生成候选 token 供目标模型验证。

---

## E

### EAGLE
一种高效的投机解码方法，利用目标模型的隐藏状态来预测 draft token。

### Embedding（嵌入）
将离散的 token 映射到连续的向量空间的过程，或指嵌入向量本身。

### EngineCore
vLLM V1 中的核心引擎组件，负责调度、执行和状态管理。

### Executor（执行器）
负责管理 Worker 进程并协调模型执行的组件。

---

## F

### FFN (Feed-Forward Network)
Transformer 中的前馈网络层，通常由两个线性层和一个激活函数组成。

### Flash Attention
一种 IO 优化的注意力计算方法，通过减少 GPU 内存访问显著提高效率。

### FP8 (8-bit Floating Point)
8 位浮点数格式，有 E4M3 和 E5M2 两种变体，用于高效量化。

### FP16 (16-bit Floating Point)
16 位浮点数格式，是 LLM 推理中常用的精度。

---

## G

### GELU (Gaussian Error Linear Unit)
一种激活函数，比 ReLU 更平滑，在 Transformer 中广泛使用。

### GPTQ
一种基于二阶信息的后训练量化方法，可以将模型量化到 INT4 精度。

### GPU Utilization（GPU 利用率）
GPU 计算资源的使用程度。Continuous Batching 的目标之一就是提高 GPU 利用率。

---

## H

### Head（头）
多头注意力中的一个注意力头。每个头独立计算注意力，捕获不同类型的关系。

### Hidden Size（隐藏层大小）
Transformer 中间表示的维度，也称为模型维度（d_model）。

### Hidden States（隐藏状态）
模型中间层的输出，在 EAGLE 等方法中用于指导 draft token 生成。

---

## I

### INT4/INT8
4 位或 8 位整数量化格式，用于减少模型显存占用和加速计算。

### Iteration-Level Scheduling（迭代级调度）
每个推理迭代重新进行调度决策的策略，是 Continuous Batching 的基础。

---

## K

### Key（键）
注意力机制中的 Key 矩阵，与 Query 矩阵相乘计算注意力分数。

### KV Cache
存储已计算的 Key 和 Value 的缓存，避免重复计算，是 LLM 推理优化的关键。

### KVCacheManager
vLLM 中管理 KV Cache 分配和释放的组件。

---

## L

### Latency（延迟）
从请求发送到收到响应的时间。包括 TTFT（首 token 延迟）和 TPOT（单 token 延迟）。

### LayerNorm（层归一化）
一种归一化技术，用于稳定训练和提高模型性能。

### Linear Layer（线性层）
执行矩阵乘法和可选偏置加法的神经网络层。

### LLM (Large Language Model)
大语言模型，通常指参数量在数十亿以上的语言模型。

### LRU (Least Recently Used)
最近最少使用的缓存驱逐策略，用于 Block Pool 管理。

---

## M

### Marlin
一套高度优化的 CUDA 内核，用于 INT4/INT8 矩阵乘法加速。

### Memory Bandwidth（内存带宽）
GPU 内存的数据传输速率，是 Decode 阶段的主要瓶颈。

### MLP (Multi-Layer Perceptron)
多层感知机，在 Transformer 中通常指 FFN 层。

### Multi-Head Attention（多头注意力）
将注意力分成多个头并行计算，捕获不同类型的依赖关系。

---

## N

### NCCL
NVIDIA Collective Communications Library，用于多 GPU 间高效通信。

### num_heads（头数）
多头注意力中的头数量，影响模型的表达能力和计算量。

### num_layers（层数）
Transformer 中的解码器层数量。

---

## O

### Output Processing（输出处理）
将模型输出转换为用户可读格式的过程，包括采样、去分词等。

---

## P

### PagedAttention
vLLM 将 KV 分块存放，通过块表访问非连续的物理位置。

### Pipeline Parallelism（流水线并行）
将模型的层分配到不同设备的并行策略，适用于多节点部署。

### Position Encoding（位置编码）
向输入添加位置信息的方法，使模型能够理解序列顺序。

### Preemption（抢占）
当内存不足时，暂停低优先级请求，释放资源给高优先级请求的机制。

### Prefill（预填充阶段）
LLM 生成过程的第一阶段，并行处理所有输入 token 并初始化 KV Cache。

### Prefix Caching（前缀缓存）
缓存相同前缀的 KV Cache，供后续请求复用，提高效率。

---

## Q

### Quantization（量化）
将高精度数值转换为低精度的技术，用于减少模型大小和加速计算。

### Query（查询）
注意力机制中的 Query 矩阵，用于查询与其他位置的相关性。

---

## R

### Ray
分布式计算框架，vLLM 使用它进行多节点分布式推理。

### Rejection Sampling（拒绝采样）
投机解码中验证 draft token 的方法，确保输出分布与只用目标模型一致。

### Request（请求）
用户发送的推理请求，包含输入 prompt 和采样参数。

### RMSNorm (Root Mean Square Normalization)
一种简化的归一化方法，计算效率比 LayerNorm 更高。

### RoPE (Rotary Position Embedding)
旋转位置编码，通过旋转操作编码位置信息，支持长度外推。

---

## S

### Sampler（采样器）
根据模型输出的 logits 选择下一个 token 的组件。

### Sampling Parameters（采样参数）
控制文本生成的参数，如 temperature、top_k、top_p 等。

### Scale（缩放因子）
量化中用于映射浮点值和整数值的比例因子。

### Scheduler（调度器）
决定哪些请求被执行、分配多少资源的核心组件。

### Self-Attention（自注意力）
序列对自身进行注意力计算，捕获序列内部的依赖关系。

### Sequence Length（序列长度）
输入或输出的 token 数量。

### Slot Mapping（槽位映射）
将 token 位置映射到 KV Cache 存储位置的机制。

### Softmax
将任意数值转换为概率分布的函数，在注意力计算中用于归一化。

### Speculative Decoding（投机解码）
使用小模型预测、大模型验证的加速技术。

### Streaming（流式输出）
边生成边返回结果的输出方式，降低用户感知延迟。

---

## T

### Temperature（温度）
采样参数，控制输出分布的平滑度。较高温度使输出更随机。

### Tensor Parallelism（张量并行）
将模型的权重矩阵切分到多个设备的并行策略。

### Throughput（吞吐量）
单位时间内处理的 token 数量，通常以 tokens/s 表示。

### Token（词元）
文本的基本单位，由分词器生成。

### Tokenization（分词）
将文本转换为 token 序列的过程。

### Top-K Sampling
只从概率最高的 K 个 token 中采样的策略。

### Top-P Sampling（Nucleus Sampling）
从累积概率达到 P 的 token 集合中采样的策略。

### Transformer
基于注意力机制的神经网络架构，是现代 LLM 的基础。

### TTFT (Time To First Token)
首 token 延迟，从请求发送到收到第一个输出 token 的时间。

---

## V

### Value（值）
注意力机制中的 Value 矩阵，根据注意力权重聚合信息。

### vLLM
高效的大语言模型推理引擎，核心创新是 PagedAttention。

### Vocab Size（词表大小）
模型支持的不同 token 数量。

---

## W

### Weight（权重）
模型的可学习参数，存储在模型文件中。

### Worker
执行模型计算的工作进程，在分布式设置中运行在各个 GPU 上。

---

## Z

### Zero-Point（零点）
量化中的偏移值，用于非对称量化。

---

**导航**
- 上一篇：[分布式推理](../05-advanced-topics/03-distributed-inference.md)
- 下一篇：[代码文件索引](code-map.md)
