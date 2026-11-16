---
title: "CacheBlend 常见问题"
weight: 7
---

## 为什么插入一段话会影响后面的手册？

因果注意力允许手册中的 token 读取它前面的文本。增加材料后，手册的隐藏状态、K 和 V 都可能变化。位置修正可以处理一部分位置编码变化，选择性重算用来补上上下文变化带来的影响。[原理论文](https://arxiv.org/abs/2405.16444v3)

## 已经启动缓存进程，为什么只看到了普通前缀命中？

先发两条具有相同长前缀的请求，确认已有缓存能读回。再加入移位文档。如果只在第一组命中，检查服务是否使用 `blend`，推理侧是否装入匹配的 CacheBlend 插件，以及 tokenizer 产生的文档片段是否仍然一致。[MP 接入说明](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/mp/operator.rst)

## 能把原型的重算比例直接用于生产吗？

用自己的模型和检索数据做完整重算对照。较低的比例能减少前向计算，也可能让答案遗漏跨文档信息。分别记录质量、TTFT 和实际命中量。原型的参数入口与现在的独立服务分别见[配置参考](./B-config-reference.md)和[部署指南](../07-advanced/03-deployment-guide.md)。

## 检查层怎样决定重算位置？

研究原型把当前 V 与旧 V 的逐元素差平方后，沿 KV 头和 head 维度求和，得到每个文档位置一个分数。假设三个候选 token 的分数是 0.1、3.0 和 0.5，预算只够更新一个，原型会选择中间的位置，并把新问题的后缀加入重算集合。这个分数反映模型内部状态变化，无法直接代替答案质量指标。[筛选代码](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py)
