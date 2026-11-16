---
title: "论文原型的源码实现"
linkTitle: "实现详解"
weight: 4
description: "项目结构、数据结构、状态机、模型实现"
---

从作者仓库的 `example/blend.py` 进入，沿 `vllm_blend` 的模型层和 xFormers 后端读到选择、旋转与更新旧 KV 的位置。这里的内部属性和类属于固定提交 `55ad026`。[原型源码](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581) · [新版工程入口](../07-advanced/03-deployment-guide.md)
