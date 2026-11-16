---
title: "把日志对应到请求、worker 和缓存块"
weight: 3
---

v0.9.1 的 `ucm.logger` 从 `UCM_LOG_LEVEL` 读取日志级别；如果未设置，会尝试 `UC_LOGGER_LEVEL`。默认级别为 info。`UCM_LOG_PATH` 决定文件位置，默认是启动进程目录下的 `log`。容器内多进程运行时，应事先确定该目录可写、日志卷可在主机读取。[Python 日志入口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/logger.py)

```bash
export UCM_LOG_LEVEL=debug
export UCM_LOG_PATH=/tmp/ucm-trace
mkdir -p "$UCM_LOG_PATH"
```

在同一进程环境启动推理，记录模型、rank、请求到达时刻和后端配置。先定位 connector 创建，再找 lookup、load 与 dump 的先后关系。多 rank 同时输出时，按 rank 和请求分组，再按时间对照。一个请求返回了文本，既可能来自设备命中，也可能来自外部回填或重新计算，因此还要核对命中与加载指标。[指标定义](./02-metrics-observability.md)

运行时通过一次后端初始化建立日志 bridge，在进程退出时 flush。`UCM_LOG_MAX_FILES` 与 `UCM_LOG_MAX_SIZE` 配置轮转。定位失败时先确认日志路径、级别和挂载状态，随后再提高单个实例的记录量。
