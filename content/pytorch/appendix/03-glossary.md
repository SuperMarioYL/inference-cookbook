---
title: "术语表"
weight: 3
---

## 概述

本术语表收录了 PyTorch 源码分析中常见的专业术语和缩写，按字母顺序排列。

## A

### Allocator（分配器）
负责管理 Tensor 内存分配和释放的组件。PyTorch 支持多种分配器，包括 CPU 分配器和 CUDA 缓存分配器。

### AOT（Ahead-of-Time）
提前编译，与 JIT（即时编译）相对。AOT Autograd 是 PyTorch 2.0 编译器栈的一部分。

### Autograd（自动微分）
PyTorch 的自动求导系统，负责自动计算梯度。核心组件包括计算图、反向传播引擎。

## B

### Backward（反向传播）
从损失函数开始，沿计算图反向计算梯度的过程。

### Boxing（装箱）
将具体类型的参数转换为通用类型（如 IValue）的过程，用于算子分发系统。

## C

### C10（Core 10）
PyTorch 的核心 C++ 库，提供基础数据结构和工具（Tensor、Device、DispatchKey 等）。

### CUDA Caching Allocator（CUDA 缓存分配器）
PyTorch 用于管理 GPU 内存的高性能分配器，通过缓存机制减少 `cudaMalloc/cudaFree` 调用。

### Computational Graph（计算图）
表示张量操作和依赖关系的有向无环图（DAG），是 Autograd 的核心数据结构。

## D

### DDP（DistributedDataParallel）
PyTorch 的数据并行分布式训练策略，通过 AllReduce 同步梯度。

### Device（设备）
Tensor 存储的硬件位置，如 `cpu`、`cuda:0`、`mps`。

### DispatchKey（分发键）
算子分发系统中用于路由算子实现的键，如 `CPU`、`CUDA`、`Autograd`。

### Dispatcher（分发器）
PyTorch 的算子分发系统核心，根据 DispatchKey 选择正确的算子实现。

### Dtype（数据类型）
Tensor 元素的数据类型，如 `float32`、`int64`、`bfloat16`。

### Dynamo（TorchDynamo）
PyTorch 编译栈的图捕获前端，通过拦截 Python 字节码动态追踪模型执行。

## E

### Engine（引擎）
Autograd 的反向传播引擎，负责执行 backward pass。

## F

### Forward（前向传播）
模型从输入到输出的计算过程。

### FSDP（FullyShardedDataParallel）
全分片数据并行，一种更高效的分布式训练策略，分片模型参数、梯度和优化器状态。

### FX（Torch FX）
PyTorch 的图表示和变换框架，提供 Python-level 的 IR（中间表示）。

## G

### Grad（梯度）
损失函数对参数的导数，通过反向传播计算得到。

### GradFn（梯度函数）
计算图中每个操作对应的反向传播函数。

### Graph（图）
FX 中表示计算流程的数据结构，由 Node 组成。

### Guard（保护条件）
TorchDynamo 中用于验证编译图有效性的条件，如输入形状、类型检查。

## H

### Hook（钩子）
在特定事件发生时执行的回调函数。PyTorch 支持 Tensor hook、Module hook、Optimizer hook 等。

## I

### Inductor（TorchInductor）
默认编译后端，负责 lowering、融合与设备代码生成，可以选择 Triton、C++、外部算子库或特定设备模板。

### In-place Operation（原地操作）
直接修改 Tensor 数据而不创建副本的操作，如 `add_()`, `relu_()`。

### IR（Intermediate Representation）
中间表示，编译器中介于源码和机器码之间的表示形式。FX Graph 是 PyTorch 的一种 IR。

## J

### JIT（Just-In-Time）
即时编译，在运行时动态编译代码。`torch.jit` 是 PyTorch 早期的 JIT 编译器。

## K

### Kernel（核函数）
在 GPU 或 CPU 上执行的底层计算函数。

### KV Cache（键值缓存）
在 Transformer 模型推理中缓存注意力机制的 Key 和 Value，避免重复计算。

## L

### Layout（布局）
Tensor 的内存布局方式，如 `strided`（跨步）、`sparse_coo`（稀疏坐标格式）。

### LR Scheduler（学习率调度器）
动态调整学习率的组件，如 `StepLR`、`CosineAnnealingLR`。

## M

### Meta Tensor（元张量）
不包含实际数据，仅存储形状、类型等元信息的 Tensor，用于编译优化。

### Mixed Precision（混合精度）
结合使用不同精度（如 FP16 和 FP32）进行训练，以加速计算并减少显存占用。

### Module（模块）
PyTorch 中构建神经网络的基础单元，`nn.Module` 是所有网络层的基类。

## N

### Node（节点）
FX Graph 中的基本单元，表示一个操作或占位符。

## O

### Operator（算子）
PyTorch 中的基本计算单元，如 `torch.add`、`torch.matmul`。

### Optimizer（优化器）
更新模型参数的算法实现，如 `SGD`、`Adam`、`AdamW`。

## P

### Parameter（参数）
模型中需要学习的权重，继承自 `Tensor` 并自动注册到 `Module`。

## R

### Requires Grad（需要梯度）
标记 Tensor 是否需要计算梯度，控制 Autograd 是否追踪该 Tensor。

## S

### Storage（存储）
Tensor 的底层数据存储，多个 Tensor 可以共享同一个 Storage。

### Stride（步长）
访问 Tensor 多维数组时，每个维度的偏移量。

### Symbolic Shape（符号化形状）
用符号表示 Tensor 形状而非具体数值，支持动态形状推导。

## T

### Tensor（张量）
PyTorch 的核心数据结构，多维数组的抽象。

### TensorImpl（张量实现）
Tensor 的底层实现类，包含 Storage、Stride、Dtype 等信息。

### Tracing（追踪）
通过执行模型并记录操作序列来构建计算图的方法。

### Triton
NVIDIA 开发的 Python-based GPU 编程语言，TorchInductor 用其生成 GPU kernel。

## V

### View（视图）
共享底层 Storage 但具有不同形状或步长的 Tensor。

## 缩写对照表

| 缩写 | 全称 | 中文 |
|------|------|------|
| AOT | Ahead-of-Time | 提前编译 |
| AMP | Automatic Mixed Precision | 自动混合精度 |
| BF16 | Brain Floating Point 16 | 16位脑浮点数 |
| C10 | Core 10 | 核心库 |
| CUDA | Compute Unified Device Architecture | 统一计算设备架构 |
| DAG | Directed Acyclic Graph | 有向无环图 |
| DDP | DistributedDataParallel | 分布式数据并行 |
| FP16/32 | Floating Point 16/32 | 16/32位浮点数 |
| FSDP | FullyShardedDataParallel | 全分片数据并行 |
| FX | Torch FX | 图变换框架 |
| IR | Intermediate Representation | 中间表示 |
| JIT | Just-In-Time | 即时编译 |
| LR | Learning Rate | 学习率 |
| SGD | Stochastic Gradient Descent | 随机梯度下降 |

## 参考资源

- [PyTorch 官方术语表](https://pytorch.org/docs/stable/glossary.html)
- [PyTorch Internals 文档](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [PyTorch 源码注释](https://github.com/pytorch/pytorch)
