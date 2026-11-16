---
title: "CacheBlend 的论文、原型与工程实现"
slug: "overview"
weight: 0
---

读 CacheBlend 时可以沿着一次缓存命中追踪三个问题。引擎从哪找到重复文本，哪些 KV 还能用，以及为了补上新上文究竟要算哪些位置。论文、作者公开的研究代码、LMCache 服务分别回答了其中不同的部分。

|要观察的行为|起点|
|---|---|
|重新排列文档时，普通前缀缓存在哪里失配|[手册与检索段落的例子](../01-introduction/02-problem-definition.md)|
|原型在哪里挑 token，哪一层开始更新旧 KV|[模型中的状态切换](../04-implementation/03-state-machine.md)与[注意力后端](../04-implementation/06-xformers-backend.md)|
|当前服务怎样发现位移后的文本、回填缓存|[LMCache 部署](../07-advanced/03-deployment-guide.md)与[固定版本源码索引](../appendix/A-code-index.md)|
|一组请求究竟省下多少时间、答案是否变化|[论文结果与实验边界](../06-experiments/02-results.md)|

这里的跨块注意力是同一条因果自注意力序列内的跨文档依赖。论文用 *cross-attention* 描述后一个文本块对前面块的关注。编码器和解码器之间的交叉注意力是另一类网络结构。[论文](https://arxiv.org/abs/2405.16444v3)
