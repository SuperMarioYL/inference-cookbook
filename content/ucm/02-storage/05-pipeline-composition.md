---
title: "一条存储管线如何组装和执行"
weight: 5
---

`UcmPipelineStore` 让上层用同一组查询、加载、写出和等待接口访问不同的存储路径。配置里的 `store_pipeline` 是已注册组合的名字。Python 的 builder 按该名字装配对应原生后端，调用方无须自己串起每次拷贝。[注册表与实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/pipeline/connector.py)

```mermaid
flowchart LR
    G["设备 KV 块"] <--> C["Cache - 主机缓冲"]
    C <--> P["Posix - 文件后端"]
```

## 以 Cache + POSIX 为例

1. `lookup_on_prefix` 查询已保存的连续块范围，`load_data` 或 `load` 接收实际目的数据描述。
2. 命中主机的 shard 可以直接复制；其他 shard 需要等下层数据进入可传输缓冲。`wait` 等待传输任务，`check` 用于查询完成状态。
3. 写出时，`dump_data` 或 `dump` 接收源缓冲描述；设备缓冲和中间缓冲的回收以实际任务的完成条件为准。

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /mnt/test
      posix_capacity_gb: 32
      posix_io_engine: psync
```

## 确认组合确实存在

在 v0.9.1 注册表中能找到 `Cache|Posix`、`Cache|Ds3fs`、`Cache|Compress|Posix`、`Mooncake|Posix` 等具体名字。需要 NFS 时区分 POSIX 后端使用已经挂载的 NFS 目录与独立 NFS store 接口。`Cache|NFS` 并未注册为这版的管线名。不同构建能加载哪些库，还要核对发行环境。[注册表](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/pipeline/connector.py) · [发布的基线配置](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/ucm_config_example.yaml)
