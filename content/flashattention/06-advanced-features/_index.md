---
title: "高级特性"
linkTitle: "高级特性"
weight: 6
description: >
  因果遮蔽、KV Cache、GQA/MQA 与 FP8 支持
---

Mask 决定哪些分数可见；GQA 决定 Q 头如何映射到 KV 头；页表决定去哪个物理页读取 KV。本章把三者分开解释。FP8 篇使用 FA3 Hopper 的独立接口。
