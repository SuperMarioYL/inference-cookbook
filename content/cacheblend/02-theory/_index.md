---
title: "偏差与选择性重计算"
linkTitle: "核心理论"
weight: 2
description: "核心概念、数学基础、算法设计"
---

如果完整重算每一层才能知道哪些位置变化最大，选择本身就耗掉了预期节省的计算。论文利用层间高偏差位置的相关性来近似选择；公开原型采用具体检查层和 V 差值实现。[论文](https://arxiv.org/abs/2405.16444v3) · [原型选择代码](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py)
