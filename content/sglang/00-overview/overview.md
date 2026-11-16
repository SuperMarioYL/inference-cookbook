---
title: "SGLang 0.5.23 阅读指南"
slug: "overview"
weight: 0
---

SGLang 的文本推理服务把传入请求转成 token，按显存和调度策略组成批次，再将生成的 token 逐步还原成文本。以下按 **0.5.23** 的文本生成路径说明。

| 读者问题 | 从哪里开始 |
|---|---|
| 如何启动并请求一个服务？ | [单卡服务与验证请求](/sglang/01-getting-started/02-quick-start/) |
| 一条请求由谁接手？ | [从输入 token 到响应](/sglang/02-core-architecture/04-request-lifecycle/) |
| 第二个请求能省下哪部分计算？ | [前缀树匹配与 HiCache](/sglang/03-memory-cache/10-radix-attention/) |
| 内存不够时先回收谁？ | [驱逐候选与请求锁](/sglang/03-memory-cache/12-eviction-policy/) |
| 为什么有空闲 GPU 却仍在排队？ | [每一轮如何选择批次](/sglang/02-core-architecture/06-scheduler-deep-dive/) |

多模态输入、投机解码和 P/D 分离会增加处理路径。阅读对应章节时，应先记录实际模型、硬件、启动参数和选中的执行后端，再对照固定版本源码。

[上游快速开始](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/quickstart.mdx) · [服务参数](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)
