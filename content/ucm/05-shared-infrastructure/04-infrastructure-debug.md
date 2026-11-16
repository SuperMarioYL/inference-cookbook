---
title: "区分日志、连接器和存储层问题"
weight: 4
---

诊断一条超时请求时，先问它卡在排队、查找、搬运还是模型执行。用带相同前缀的冷、热请求做对照，并给日志和指标选择同一时间窗口。

1. 在 worker 所处环境检查 `UCM_LOG_PATH`、`UCM_LOG_LEVEL` 与实际文件的更新时间；容器中的相对路径跟随启动目录。[日志实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/logger.py)
2. 看 HBM、UCM 的命中和成功加载的字节数。命中长度与总字节数用于不同层次的核验。[计数器定义](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md)
3. 收集实际存储后端对应的状态。如果 `Cache|Posix` 的 Host 层已经命中，文件系统没有新增读取是合理现象。若需要观察 POSIX 路径，选择 Host 缓存之外但存储仍可访问的请求。
4. 对照引擎回调的提交与等待位置；以写任务的完成通知记录落盘时刻。[集成调试](../04-vllm-integration/06-integration-debug.md)

保留故障发生时的引擎版本、容器标签、后端 YAML、模型布局和异常 rank。后端、模型或并行组不同，即使同一个状态码也可能发生在不同层。
