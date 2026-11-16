---
title: "稀疏方法的选择、块集合与输出对照"
weight: 9
---

在 v0.9.1，从 `UcmSparseFactory` 确认配置名是否已注册。该版本注册 `ESA`、`GSAOnDevice`、`KVStarMultiStep` 和 `Blend`，普通 `GSA` 的注册行处于注释状态。配置项应填写工厂实际注册的名字。[工厂源码](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/factory.py)

## 让同一个请求走一遍生命周期

1. 用相同模型、输入 token、采样与设备先保留完整注意力输出。记录实际的 prompt 长度、调度步和层。
2. 追 `request_begin` 与 `build_sparse_meta`，核对本步的请求元数据。chunked prefill 要看这一调度步是不是最后一段。
3. 到 `attention_begin` 与 `attention_finished`，对齐本层读取的块、候选索引与本轮计算的输出；异步检索先确认提交，再确认取到的是哪一轮结果。
4. 对照完整注意力的答案，而后才比较 TTFT 和每步时延。若数量、mask 或输出在首个检查层已分歧，继续跟这一层比从最后的生成文本倒推更快。

```bash
rg -n 'register_sparse_method' ucm/sparse/factory.py
rg -n 'def (request_begin|build_sparse_meta|attention_begin|attention_finished)' ucm/sparse
```

|路径|适合追踪的内容|
|---|---|
|[ESA](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/esa/esa.py)|请求/每层状态、块表示和检索|
|[GSAOnDevice](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa_on_device/gsa_on_device.py)|实际 batch 描述、设备检索与输出|
|[KVStarMultiStep](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/kvstar/multistep.py)|每层的异步检索、候选及工作步骤|
|[Blend](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/blend/blend.py)|文档换位后的块、重算位置和新后缀|

启用 `UCM_LOG_LEVEL=debug` 后从当前运行的日志收集上下文，具体适配以[模型/框架特性矩阵](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)和各自示例为准。
