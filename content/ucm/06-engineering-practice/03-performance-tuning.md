---
title: "把外部缓存收益拆成命中、传输和计算"
weight: 3
---

给同一模型准备三组同长度请求：冷前缀、仍在 HBM 的热前缀，以及已离开 HBM 但仍在外部存储的前缀。让三组使用同一并发、输出长度和采样设置。每组分别记录 TTFT、中尾分位的读取耗时、传输字节，以及未命中的 token 数。

|现象|下一步测什么|可能需要调整的部分|
|---|---|---|
|HBM 热请求快，外部热请求慢|load 的字节数和等待时间；Host 与 POSIX 实际供数份额|存储介质、传输、Host 缓冲|
|未出现外部命中|共同前缀是否形成完整块；是否已有设备命中|测试流量、tokenizer 与块尺寸|
|高并发的 p99 恶化|队列、I/O 并发、设备空闲时间；冷请求是否挤占写入|入流并发、后端执行队列|
|命中提高但答案变化|同模型完整重算结果、layout、rank 和读回的数据|缓存键/布局或稀疏模型配置|

## 有出处的参考值

上游 Pipeline Store 示例在关闭 HBM 缓存、设定 SSD 80% 命中的条件下测 QwQ-32B 和 DeepSeek-R1-awq。QwQ-32B、4×H100、输入长度 8,000、并发 1 时，表中完整计算为 350.47 ms，SSD80% 为 228.27 ms。换用其他设备、模型或负载，应先重复同一对照再量化收益。[实验条件与完整数据](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/pipeline_store.md)

## 让每次改动有可解释的结果

先调预留容量和 I/O 路径，再考虑逐层加载或其他模型特性。增加 Host cache 时记录源自 POSIX 的读取份额有无下降；调整逐层等待时同时观察模型前向停顿和 GPU 利用率；改稀疏比例时加入质量评估。每轮保存完整配置和测试输入。[指标定义](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md) · [配置入口](./02-configuration-reference.md)
