---
title: "缓存容量、质量与运行时间的取舍"
weight: 2
---

同一台机器的 HBM 用于权重、执行工作区和 KV。把冷 KV 搬到主存或存储可以释放设备空间，但下一次命中要付读回成本。需要比较“重算同一段提示所花时间”和“找到、读取、回填这些块所花时间”。前缀长度、并发和后端带宽一变，两者关系也会变。

## 普通前缀复用与稀疏计算

普通前缀复用的目标是恢复同一模型、同一上文的缓存。稀疏方法改变模型实际关注或重算的位置，质量也成为实验变量。用带参考答案的请求集分别报告任务分数、TTFT 和吞吐。某个输入上的最合适比例很难从另一台机器的 I/O 性能推导。[增强特性支持范围](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)

## 整块与逐层加载

整组传输可减少任务调度开销；逐层传输能让一层计算与其他层加载重叠，同时引入更多等待边界。若存储供数稳定且传输短，细分任务的收益可能有限；若慢 I/O 正好处于关键路径，重叠可能改善首 token 延迟。两条路线用相同输入，比较每层等待时间、传输字节和端到端 TTFT。[层级 Connector](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py)

## 如何解释结果

HBM 已经命中九成输入时，外部存储最多只服务剩余的一成。总 TTFT 还包括排队、设备计算和输入末尾的处理。用同一时间窗口和相同的 token 计数单位比较各层命中，再决定把容量留给哪一层。[监控指标](../05-shared-infrastructure/02-metrics-observability.md) · [实测对照](../06-engineering-practice/03-performance-tuning.md)
