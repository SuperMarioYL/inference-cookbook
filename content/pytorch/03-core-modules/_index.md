---
title: "核心模块详解"
linkTitle: "核心模块详解"
weight: 3
description: >
  探索 PyTorch 核心系统：nn.Module、Optimizer 和算子分发机制
---

跟踪模块如何管理参数、优化器如何更新它们，再观察 Dispatcher 如何为算子选择内核。调用时的 dispatch key 集合与注册时的 schema、实现是两份不同的信息。[Dispatcher](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/core/dispatch/Dispatcher.h)
