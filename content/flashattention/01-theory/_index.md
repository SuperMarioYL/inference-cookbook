---
title: "理论基础"
linkTitle: "理论基础"
weight: 1
description: >
  Attention 机制的数学推导与 IO-Awareness 分析
---

用一行 Q 对所有 K 打分，先写出标准 softmax，再看显式分数矩阵如何占用显存。最后把同一行拆成两个块，检验运行最大值改变时的缩放因子。这里的推导对多代内核都适用。[原始论文](https://arxiv.org/abs/2205.14135)
