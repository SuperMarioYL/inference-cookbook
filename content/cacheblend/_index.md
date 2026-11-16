---
title: "CacheBlend"
linkTitle: "CacheBlend"
weight: 4
description: "CacheBlend 的跨文档 KV 复用、论文原型和 LMCache 多进程实现"
---

同一份手册可能出现在几百个问题里。手册放在提示词开头时，推理引擎可以复用已经算过的前缀。一旦在它前面插入新的检索材料，手册里的词就看到了不同的上文。CacheBlend 挑出受影响较大的 token 重新计算，再与留下的 KV 合并。

想看懂这个差别，先读[两次请求的对照](./01-introduction/02-problem-definition.md)。[原型源码](./04-implementation/01-project-structure.md)使用论文作者的 `vllm_blend`，固定在 `55ad026`。[部署指南](./07-advanced/03-deployment-guide.md)讲 LMCache `v0.5.7.post1` 的独立服务和引擎接入。[论文](https://arxiv.org/abs/2405.16444v3) · [作者仓库](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581) · [LMCache 版本](https://github.com/LMCache/LMCache/releases/tag/v0.5.7.post1)
