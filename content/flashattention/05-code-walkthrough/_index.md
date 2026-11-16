---
title: "代码链路追踪"
linkTitle: "代码链路追踪"
weight: 5
description: >
  前向/反向调用链追踪与调试指南
---

从 FA2 用户函数经过 Autograd 与 C++ 绑定，追到 CUDA 分发，再对照 FA3 与 FA4 各自的入口。数值不一致时，先核对布局、mask 对齐、dtype 与 dropout，再缩小到单个 kernel。
