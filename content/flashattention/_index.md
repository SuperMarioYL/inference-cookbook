---
title: "Flash Attention"
linkTitle: "Flash Attention"
weight: 1
description: "Flash Attention 原理深度解析"
---

FlashAttention 把一行注意力所需的 K、V 分块搬到 GPU 片上，边计算边合并归一化统计量。输出仍是缩放点积注意力；省下的是反复读写完整分数矩阵的开销。数值舍入、mask 对齐与 dropout 均会影响比较结果。

## 选择阅读入口

| 想解决的问题 | 从这里开始 |
|---|---|
| 安装哪个包，如何验证结果 | [分代安装与一次前向计算](/flashattention/00-overview/03-quick-start/) |
| 为什么可以分块算出同一个结果 | [用两块数据手算在线 softmax](/flashattention/02-core-algorithm/01-online-softmax/) |
| 函数实际进入哪套内核 | [分代架构](/flashattention/00-overview/02-architecture-overview/)与[前向调用链](/flashattention/05-code-walkthrough/01-forward-call-trace/) |
| KV Cache 推理、比较不同形状下的表现 | [KV Cache 的写入和读取](/flashattention/06-advanced-features/02-kv-cache-inference/)与[计时方法](/flashattention/07-usage-tutorial/04-benchmarking/) |

本栏分别采用 FA2 的稳定版 `v2.8.3.post2`、FA3 的 Hopper 源码和 FA4 的 CuTeDSL 预览版 `fa4-v4.0.0.beta39`。后两项的源码定位统一使用该预览 tag。核对日期为 2026-11-16。[稳定版](https://github.com/Dao-AILab/flash-attention/releases/tag/v2.8.3.post2) · [预览版及各代安装入口](https://github.com/Dao-AILab/flash-attention/blob/fa4-v4.0.0.beta39/README.md)
