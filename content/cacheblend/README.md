---
title: "CacheBlend 阅读指南"
---

遇到同一段文档在提示中换位置的问题，从[问题定义](./01-introduction/02-problem-definition.md)看两次请求。接着读[高偏差 token 的选择](./05-algorithms/02-hkvd-selection.md)和 [KV 融合](./05-algorithms/03-kv-fusion.md)，可以看到原型用什么代理量做筛选，以及最后写回了哪些位置。

复现 EuroSys 论文时请使用作者保留的 `vllm_blend` 和对应数据集；工程接入使用 LMCache `v0.5.7.post1`。两条路径各有自己的配置、连接器和测试方式。[原型入口](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581) · [部署指南](./07-advanced/03-deployment-guide.md) · [论文实验解读](./06-experiments/02-results.md)
