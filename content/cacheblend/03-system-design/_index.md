---
title: "论文的缓存与加载设计"
linkTitle: "系统设计"
weight: 3
description: "整体架构、Loading Controller、KV Cache Store、Pipeline 优化"
---

论文用 Loading Controller、KV Cache Store 和 Cache Fusor 分解检索、读入和选择性更新，并分析加载与计算能否重叠。本章沿论文的设计命名讲这三个职责；现行 LMCache 使用独立的多进程 Blend 模块。[论文](https://arxiv.org/abs/2405.16444v3) · [LMCache 模块协议](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/design/v1/multiprocess/modules/blend.md)
