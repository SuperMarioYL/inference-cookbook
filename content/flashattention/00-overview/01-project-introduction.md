---
title: "Flash Attention 项目全景介绍"
weight: 1
---

一次完整的注意力前向计算需要得到 `softmax(QKᵀ / √d) V`。直接实现会把形状为 `[Nq, Nk]` 的分数或概率写入显存，再读出来与 V 相乘。FlashAttention 在片上处理小块；每行保留最大值、指数和及加权输出，最终作一次除法。对于密集自注意力，计算量仍随序列长度平方增长。

## 四代工作解决了哪些问题

| 工作 | 数据或工作分配如何改变 | 可以直接读的原始资料 |
|---|---|---|
| FA1 | 分块融合分数、softmax 与乘 V，反向时重算概率，降低显存流量 | [FA1 论文](https://arxiv.org/abs/2205.14135) |
| FA2 | 将不同 Q 行块分给更多线程块，调整 warp 分工，减少非矩阵乘运算和线程间通信 | [FA2 论文](https://arxiv.org/abs/2307.08691) |
| FA3 | 在 Hopper 上让数据搬运、异步矩阵乘和 softmax 重叠，提供低精度前向路径 | [FA3 论文](https://arxiv.org/abs/2407.08691) |
| FA4 | 针对 Blackwell 各计算单元增长不均衡的问题，重新安排矩阵乘、指数运算与片上存储的流水线；CuTeDSL 实现也覆盖 Hopper | [FA4 论文](https://arxiv.org/abs/2603.05451) · [项目说明](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/README.md) |

## 一个具体的内存差异

批大小 1、单头、序列长 4096 时，一张 FP16 分数矩阵占 `4096 × 4096 × 2 = 32 MiB`。显式前向可能还要保存概率矩阵。分块内核仍会计算所有未屏蔽的分数，但可以在消费完一个块后复用片上空间。Q/K/V、输出与反向所需统计量仍占内存。实际延迟还取决于序列形状、头维度、精度和设备。

## 原理、软件包和框架入口

研究论文描述的是计算方法。应用代码选择的包、导入名和硬件决定实际执行的内核。FA2 的 `flash-attn`、FA3 的 Hopper 扩展和 FA4 的 `flash-attn-4` 分别维护入口。PyTorch 的 `scaled_dot_product_attention` 又有自己的后端选择；可在 profiler 的 kernel 事件中核对 SDPA 实际命中的实现。[分代架构](/flashattention/00-overview/02-architecture-overview/) · [快速开始](/flashattention/00-overview/03-quick-start/)
