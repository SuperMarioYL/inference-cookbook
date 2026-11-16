---
title: "多实例怎样访问共享 KV 存储"
weight: 6
---

两个实例想共享 KV，除了相同模型、分词器、块尺寸和缓存布局，还要让它们访问同一个存储命名空间。不同后端的连接方式、数据路径及可见时刻由相应接口定义。

|路径|要确认的共享对象|实际入口|
|---|---|---|
|POSIX 对接共享文件系统|每个节点的挂载指向同一存储、权限允许读写|[Pipeline 与 POSIX](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/pipeline_store.md)|
|DS3FS|指定的文件路径和后端依赖在参与节点一致|[DS3FS 指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/ds3fs_store.md) · [后端实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/ds3fs/cc/ds3fs_store.cc)|
|Mooncake Store|客户端连接相同的 Master 和可用的存储池|[Mooncake 后端指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/prefix-cache/mooncake_store.md)|
|独立 NFS Store|NFS 挂载、域及后端配置|[Python 入口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/nfsstore/nfsstore_connector.py)|

## 做一次跨实例读取

在实例 A 上运行长前缀，等保存完成；让实例 B 在自己的设备缓存还没有该前缀时执行相同 token 前缀。先核对 B 侧外部命中，再看实际加载字节，最后比较答案。只在 A 上重复请求，容易测到本地显存的收益。

切换到 P/D 直传时，关注点会变成“同一个请求由谁触发交接、接收端何时可读”。共享存储的确认条件不能直接代替直传协议的完成条件。[Mooncake 两条路径](/mooncake/02-architecture/02-request-lifecycle/)
