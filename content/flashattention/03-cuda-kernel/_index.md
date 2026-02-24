---
title: "CUDA 内核解析"
linkTitle: "CUDA 内核解析"
weight: 3
description: >
  GPU 编程基础与 Hopper/Ampere 内核实现
---

本部分从 GPU 硬件架构出发，逐行解析 Flash Attention 在 SM90 (Hopper) 和 SM80 (Ampere) 架构上的 CUDA 内核实现。
