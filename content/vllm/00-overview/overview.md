---
title: "vLLM 0.33.0 阅读指南"
slug: "overview"
weight: 0
---

vLLM 把来自不同请求的 token 放进同一轮模型执行，并按需分配 KV 缓存。这个栏目以 **0.33.0** 为源码基线。第一次使用可以先启动一个单卡文本服务，熟悉运行过程后，再沿同一个请求阅读调度与内存管理。

1. [启动服务并发送一条请求](/vllm/04-code-walkthrough/01-entry-points/)。比较在线服务与 Python 离线接口的返回形式。
2. [跟踪请求的一生](/vllm/04-code-walkthrough/06-request-lifecycle/)。找出请求在哪个进程排队、在哪一步得到输出 token。
3. [观察两个请求如何共享和释放 KV 块](/vllm/03-core-modules/03-block-pool/)。弄清 token 的逻辑位置、物理块编号与引用计数。
4. [按职责查询源码](/vllm/appendix/code-map/)。修改参数之前，查看本版配置和运行路径。

神经网络、注意力与自回归生成章节提供原理背景。源码涉及不同硬件时，本栏的单卡例子采用 NVIDIA CUDA，其他设备要使用相应发行包和注意力后端。

[官方快速开始](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/getting_started/quickstart.md) · [进程架构](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/arch_overview.md)
