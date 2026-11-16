---
title: "沿着缓存读写路径定位问题"
weight: 4
---

## Connector 启动前报错

先收集实际进程的 Python ABI、vLLM/vLLM-Ascend 与 UCM 版本、CUDA 或 CANN 与芯片。import 成功只代表模块能找到；原生库仍可能与运行时不匹配。从发行矩阵查对应 wheel 或镜像，再核对 worker 内的安装位置。[v0.9.1 发行矩阵](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

## 能生成回答，外部命中仍为零

第一次请求要足够长，能够写出至少一个完整缓存块。等保存任务完成，第二次发送相同分词前缀，末尾换问题，并分别看设备和 UCM 命中。若全在设备命中，外部查询可能无需进行。比较两个实例前，确认它们共享后端数据、模型、分词器、块尺寸和布局。[前缀复用示例](../01-overview/03-quick-start.md)

## 已查询到块，模型没有向前推进

按“分配目标块、把元数据交给 worker、提交加载、等待完成”逐项对齐同一请求。同一模型的共享状态和每 rank 私有状态可能采用不同分发路径。v0.9.1 的 GLM5.3 修复提供了两个有已知根因的例子。[补丁分析](../04-vllm-integration/05-patch-anatomy.md)

## 外部命中率高，首 token 延迟仍高

比较相同输入长度和并发下的首 token 延迟，再分开看排队、后端读取、搬到设备及模型前向。统计时给 token、block 与 shard 各自配相同单位的分母。[上游指标与单位](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md)

## 在容器里验证实际生效的配置

在 worker 使用的容器内运行下面的检查。`UCM_CONFIG_FILE` 若在 vLLM 启动 JSON 中指定，先把同一个实际路径赋给命令行变量。先确认容器内能读取配置和存储位置，再观察第一次完整写入。

```bash
python -m pip show uc-manager vllm
python -c 'import sys, ucm; print(sys.executable, ucm.__file__)'
ls -l "$UCM_CONFIG_FILE"
df -h /cache /dev/shm
curl -fsS http://localhost:7800/metrics | grep -E '^(ucm|kv):'
```

需要观察 `/metrics` 时在顶层 UCM 配置加 `enable_metrics: true`。API Server 向 Prometheus 暴露已经归集的数据；vLLM 处理推理请求时才会从 scheduler/worker 同步新的 UCM 统计。只轮询 HTTP 端点而没有新请求，不一定能看到内部刚产生的变化。[指标数据流](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics.md)

## submit 与 wait 报错的区别

上游排障资料将提交失败和等待写出完成失败分别列出。提交阶段优先检查后端是否可用、目标路径/空间和原生 API 错误；等待阶段说明任务已经提交，还需检查执行队列、文件 I/O 和传输依赖。仓库中的错误码表区分一般错误、系统调用失败、重复 key 等，保留原错误码和附带信息比“读盘失败”更有定位价值。

有 GPU 环境时从全量重算和外部命中两条路径对照。缺少设备时，可以先运行 [POSIX 主机缓冲区测试](../02-storage/03-posix-backend.md)，逐字节排除存储库本身的问题。该测试不会运行模型，也不能检验 H2D、稀疏 attention 或并行布局。[上游错误码与诊断](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/troubleshooting.md)
