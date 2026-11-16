---
title: "Connector 与存储后端分别读取哪些配置"
weight: 2
---

连接器从 vLLM 的 `--kv-transfer-config` 获取 `kv_connector`、Python 模块、角色和 `UCM_CONFIG_FILE`；UCM 打开后者并选择存储后端。当前运行镜像中的示例位于 `/workspace/ucm_config_example.yaml`，源码 checkout 的示例是 `examples/ucm_config_example.yaml`。[固定版 YAML](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/ucm_config_example.yaml)

|作用域|配置|含义|
|---|---|---|
|vLLM|`kv_role: kv_both`|本实例的配置允许读取和写出 KV|
|UCM|`ucm_connectors[].ucm_connector_name`|选择 Store 实现，例如 `UcmPipelineStore`|
|Pipeline Store|`store_pipeline: "Cache\|Posix"`|Host 缓存与 POSIX 后端协同|
|POSIX|`storage_backends`、`posix_capacity_gb`|文件系统位置与容量预算（GB）|
|POSIX|`posix_io_engine`|选择该后端的 I/O 引擎，发布示例用 `psync`|

当前实例的选择还与引擎缓存布局、模型、并行配置和是否逐层传输有关。先用[可运行的前缀缓存例子](../01-overview/03-quick-start.md)对齐这两层配置。更换 Mooncake Store、DS3FS 或启用压缩时，改用对应后端的配置契约。[Prefix Cache 后端指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/index.md)

## 同样写“容量”，实际限制的资源不同

`cache_buffer_capacity_gb` 管 Host 侧缓存缓冲，多个数据并行实例可能各分配一份；`posix_capacity_gb` 管文件系统后端预算，大于零会启用相应的空间回收。先列出每台主机的实例数和实际可用 Host 内存、共享内存及文件系统空间，再按实例分配。只扩容 POSIX 目录无法解决 Host 缓冲不足。

`use_layerwise` 属于 UCM 顶层配置，`posix_io_engine` 则位于 `ucm_connectors[].ucm_connector_config`。前者影响何时传输与等待，后者影响 POSIX 读写方式，两者产生的耗时会出现在不同阶段。`io_direct` 打开后还需检查缓冲和文件系统的对齐条件。[Pipeline 参数](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/pipeline_store.md)

## 变更后端时核对什么

从 `Cache|Posix` 改为 `Mooncake|Posix` 时，实例还要能连接同一个 Mooncake 元数据服务和实际存储池。建议先单独读写一个对象，随后比较具有同一前缀的两次推理。保存配置时记录各进程可见的路径；同样的容器内路径在两台机器上可能对应不同的本地目录。[Mooncake 后端示例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/mooncake_store.md)
