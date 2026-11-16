---
title: "从一次外部命中定位 Store 故障"
weight: 7
---

用一条足够长的固定 token 前缀做两次请求。首次保存完成后，比较第二次的块 ID、HBM 命中、外部查询、实际 load 字节和模型前向。任意一层已经命中，都会影响下一层究竟收到多少查询。[复现实例](../01-overview/03-quick-start.md)

## 在对应层停下来

|代码位置|实际值得观察的状态|
|---|---|
|[UcmPipelineStore](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/pipeline/connector.py)|Store 类型、管线名、批次 block ID、目标数据描述与任务|
|[POSIX 查询与读写](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/posix/cc/posix_store.cc)|返回的最后连续命中下标、空间管理器的结果、load/dump 的状态|
|[Host cache 队列](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/cache/cc/load_queue.cc)|本层可直接供应的 shard，以及仍在等后端的 shard|
|[Connector 路径](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py)|同一请求的匹配长度、目标设备块和 worker 等待点|

```bash
rg -n 'def (lookup_on_prefix|load_data|dump_data|check|wait)' ucm/store/pipeline/connector.py
rg -n 'LookupOnPrefix|Expected<Detail::TaskHandle> (Load|Dump)' ucm/store/posix/cc/posix_store.cc
```

Python 调试器能看到传入 native 层之前的参数。要查看后端 I/O 队列，在包含调试符号的匹配构建里按上面的函数名定位 C++；先以一个 rank、一个请求复现，再增大并发。`UCM_LOG_LEVEL=debug` 要在启动该服务前设置。[日志环境变量](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/logger.py)

## 如何区分容易混淆的结果

- **查不到第一块**。比较模型、分词后的前缀、布局和确实保存的数据。文件数大于零只能证明目录里有数据。
- **能查到若干块，未必都是连续前缀**。例如存在性向量为真、假、真时，普通前缀可覆盖的范围在第二块停止。不要把总命中数直接当作可跳过的 token。
- **已经宣布命中，worker 还在等待**。对照实际读入字节、目标缓冲、层和 rank。共享与 rank 私有状态要按对应布局处理。
- **写入慢**。分别观察异步任务提交、等待完成、Host 缓冲占用和文件系统 I/O。只有保留下来的等待进入请求关键路径时，影响才会直接表现在端到端延迟上。

单次失败的现场至少保留引擎及 UCM 版本、有效 YAML、模型和设备、请求 token、rank 与相关时间窗口。[指标定义](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md)
