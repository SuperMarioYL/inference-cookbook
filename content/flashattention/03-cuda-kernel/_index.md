---
title: "CUDA 内核解析"
linkTitle: "CUDA 内核解析"
weight: 3
description: >
  GPU 编程基础与 Hopper/Ampere 内核实现
---

本章跟踪 C++/CUDA 内核中的 tile、warp 分工和同步。读 SM80 段落时核对文件位于 `csrc/flash_attn/src/` 还是 `hopper/`；两者有独立入口。需要看 Blackwell CuTeDSL 的 TMEM 和代码生成路径，可转到[FA4 分代架构](/flashattention/00-overview/02-architecture-overview/)。
