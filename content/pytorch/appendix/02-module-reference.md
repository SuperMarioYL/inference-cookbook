---
title: "主题索引"
weight: 2
---

按现在的章节目录查找具体文章。若从一个函数名出发，请转到[源码地图](/pytorch/appendix/04-code-map/)。

## 架构与源码入口

- [PyTorch 源码架构全景](/pytorch/00-overview/01-architecture-overview/)
- [开发环境搭建](/pytorch/00-overview/02-build-and-dev-env/)
- [核心概念速览](/pytorch/00-overview/03-key-concepts/)
- [执行流程概览](/pytorch/00-overview/04-execution-flow-overview/)
- [代码导航技巧](/pytorch/00-overview/05-code-navigation-tips/)

## 使用入门

- [Tensor 基础教程](/pytorch/01-introduction/01-tensor-basics-tutorial/)
- [Autograd 基础教程](/pytorch/01-introduction/02-autograd-basics-tutorial/)
- [nn.Module 教程](/pytorch/01-introduction/03-nn-module-tutorial/)
- [Optimizer 教程](/pytorch/01-introduction/04-optimizer-tutorial/)

## 数据结构与执行

- [TensorImpl 深度解析](/pytorch/02-fundamentals/01-tensorimpl-deep-dive/)
- [Storage 和内存分配](/pytorch/02-fundamentals/02-storage-and-allocator/)
- [数据类型与设备](/pytorch/02-fundamentals/03-dtype-device-layout/)
- [Tensor 创建代码路径](/pytorch/02-fundamentals/04-tensor-creation-codepath/)
- [计算图构建](/pytorch/02-fundamentals/05-computational-graph/)
- [引擎执行机制](/pytorch/02-fundamentals/06-engine-execution/)
- [符号化形状推导](/pytorch/02-fundamentals/07-symbolic-shapes/)

## 核心模块

- [Module 核心机制](/pytorch/03-core-modules/01-module-core-mechanism/)
- [Module Hooks 系统](/pytorch/03-core-modules/02-module-hooks-system/)
- [Optimizer 基类设计](/pytorch/03-core-modules/03-optimizer-base-class/)
- [Adam 算法分析](/pytorch/03-core-modules/04-adam-algorithm-analysis/)
- [学习率调度器](/pytorch/03-core-modules/05-lr-scheduler/)
- [DispatchKey 系统](/pytorch/03-core-modules/06-dispatch-key-system/)
- [Dispatcher 核心实现](/pytorch/03-core-modules/07-dispatcher-core/)
- [算子注册机制](/pytorch/03-core-modules/08-operator-registration/)

## 代码走读

- [Backward 代码路径调试](/pytorch/04-code-walkthrough/01-backward-codepath-debug/)
- [Dispatch 代码路径调试](/pytorch/04-code-walkthrough/02-dispatch-codepath-debug/)
- [常见层实现分析](/pytorch/04-code-walkthrough/03-common-layers-analysis/)
- [Functional vs Module](/pytorch/04-code-walkthrough/04-functional-vs-module/)
- [Module 序列化](/pytorch/04-code-walkthrough/05-module-serialization/)
- [自定义 Autograd Function](/pytorch/04-code-walkthrough/06-custom-autograd-function/)
- [梯度模式与推理](/pytorch/04-code-walkthrough/07-grad-mode-and-inference/)
- [Saved Tensors Hooks](/pytorch/04-code-walkthrough/08-saved-tensors-hooks/)
- [Boxing 和 Fallthrough](/pytorch/04-code-walkthrough/09-boxing-and-fallthrough/)

## 编译与分布式

- [编译器栈概览](/pytorch/05-advanced-topics/01-compiler-stack-overview/)
- [FX 图系统教程](/pytorch/05-advanced-topics/02-fx-tutorial/)
- [Graph 和 Node 深度解析](/pytorch/05-advanced-topics/03-graph-and-node/)
- [TorchDynamo 教程](/pytorch/05-advanced-topics/04-dynamo-tutorial/)
- [字节码分析机制](/pytorch/05-advanced-topics/05-bytecode-analysis/)
- [Guard 保护机制](/pytorch/05-advanced-topics/06-guard-mechanism/)
- [TorchInductor 概览](/pytorch/05-advanced-topics/07-inductor-overview/)
- [Triton 代码生成](/pytorch/05-advanced-topics/08-triton-codegen/)
- [分布式训练概览](/pytorch/05-advanced-topics/09-distributed-overview/)
- [DDP 和 FSDP 实现](/pytorch/05-advanced-topics/10-ddp-fsdp/)
- [CUDA 内存管理教程](/pytorch/05-advanced-topics/11-cuda-memory-tutorial/)
- [混合精度训练](/pytorch/05-advanced-topics/12-amp-mixed-precision/)
