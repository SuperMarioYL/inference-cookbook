---
title: "基础知识"
linkTitle: "基础知识"
weight: 2
description: >
  深入理解 PyTorch 底层机制：TensorImpl、Storage、计算图和自动微分引擎
---

同一块存储可以有不同的形状、stride 与偏移。TensorImpl 把这些元数据与设备、类型结合起来；Autograd 再为需要梯度的操作记录图节点和边。[TensorImpl](https://github.com/pytorch/pytorch/blob/v2.15.0/c10/core/TensorImpl.h)
