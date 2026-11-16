---
title: "PyTorch"
linkTitle: "PyTorch"
weight: 1
description: "从张量执行追到算子分发、图编译、FSDP 与 AMP 的 PyTorch 源码指南"
---

一行 `torch.mm(a, b)` 会触发张量布局检查、算子分发和设备内核；给函数加上 `torch.compile` 后，Dynamo 还能捕获其中的张量计算，让编译后端优化。这里以 **PyTorch 2.15.0** 为代码阅读基线（2026-11-16 核对），从用户手中的调用一路追到实现。[2.15 发布说明](https://pytorch.org/blog/pytorch-2-15-release-blog/) · [固定源码](https://github.com/pytorch/pytorch/tree/v2.15.0)

| 想回答的问题 | 阅读入口 |
|---|---|
| 如何安装并核对本机版本 | [开发环境搭建](/pytorch/00-overview/02-build-and-dev-env/) |
| 为什么两个视图能共享同一块内存 | [Tensor、stride 与 view](/pytorch/01-introduction/01-tensor-basics-tutorial/) |
| 算子由谁派发，梯度在哪里记录 | [Dispatcher](/pytorch/03-core-modules/07-dispatcher-core/)与[计算图](/pytorch/02-fundamentals/05-computational-graph/) |
| 输入长度变化时为什么重新编译 | [编译栈与可运行示例](/pytorch/05-advanced-topics/01-compiler-stack-overview/) |
| 多卡训练如何分片，混合精度选哪个入口 | [DDP 与 FSDP](/pytorch/05-advanced-topics/10-ddp-fsdp/)与[AMP](/pytorch/05-advanced-topics/12-amp-mixed-precision/) |

按主题循序阅读，可查看[阅读指南](/pytorch/appendix/01-reading-guide/)。
