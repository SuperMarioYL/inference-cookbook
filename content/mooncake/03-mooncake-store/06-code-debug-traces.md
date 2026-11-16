---
title: "沿 Put、Get 和驱逐三条链路调试 Store"
weight: 6
---

先用[单机读写实验](../00-beginner-guide/03-quick-start-tutorial.md)得到一个可重现的 key，然后在 v0.3.14 的符号上设断点。这里涉及两个进程：Master 持有元数据，Python 客户端调用原生库搬运字节。只调试 Master 看不到客户端的数据传输分支。

## 准备带符号的构建

在具备官方构建依赖的 Linux 环境中使用同一 tag，保留本次使用的传输后端。调试前确认运行的 Master 和客户端绑定的共享库都来自该构建。要在 GDB 中查看局部变量，使用保留调试信息的构建。

```bash
git clone --branch v0.3.14 --depth 1 https://github.com/kvcache-ai/Mooncake.git
cd Mooncake
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --target mooncake_master --parallel
gdb --args "$(find build -type f -name mooncake_master -print -quit)"
```

```gdb
set breakpoint pending on
break mooncake::MasterService::PutStart
break mooncake::MasterService::PutEnd
break mooncake::MasterService::ReadReplicaList
run
```

多线程程序中优先用 `info args`、`bt` 和 `info threads` 观察当前停在哪个 RPC。对短租约或高并发负载，长时间停在断点本身就可能改变心跳或驱逐行为；可用单请求、日志和固定 key 缩小范围。[构建选项](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/getting_started/build.md)

## Trace 1：从 PutStart 到 PutEnd

从 [Client::Put](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-store/src/client_service.cpp) 开始，写者告诉 Master 要写哪个 key、多少数据及需要何种副本。`PutStart` 返回描述符。跟踪客户端 `TransferWrite` 和 [TransferSubmitter](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-store/src/transfer_task.cpp)，看被选中的描述符、缓冲范围和传输结果；成功后才进入 `PutEnd`。如果数据操作失败，回到 `PutRevoke`，确认该写者撤销的是仍在处理中的目标。

{{< trace >}}
title: "一次 Put 在哪里跨过可读边界"
actors: ["写客户端", "Master", "存储持有者", "读客户端"]
steps:
  - from: 0
    to: 1
    label: "PutStart"
    detail: "写客户端为 key 申请副本。Master 返回目标描述符。"
  - from: 0
    to: 2
    label: "写入 payload"
    detail: "客户端通过所选传输路径把字节放入目标缓冲。"
  - from: 0
    to: 1
    label: "PutEnd"
    detail: "写者收到成功结果后，通知 Master 发布已完成副本。"
  - from: 3
    to: 1
    label: "查询副本"
    detail: "另一客户端此时可获得用于读取的有效描述符。"
{{< /trace >}}

**值得保存的现场**包括租户、对象名、client ID、传输方向、源/目的长度和最终错误码。若写方成功但另一个进程找不到对象，核对两端是否在同一命名空间、写者是否仍在线、`PutEnd` 是否执行完毕。[Master 写入与撤销](./02-master-service.md)

## Trace 2：从 Get 到实际读取

在客户端依次跟 `Client::Query`、`Client::Get` 和副本选择。Master 走 `GetReplicaList`，最后筛出可读副本并返回租约。客户端面对的可能是主存、节点本地 SSD 或其他介质。内存传输在某些单机配置可走 memcpy；在读进程不贡献 segment、关闭本地 memcpy 的实验中，可以再判断是否真的发生跨进程 TCP。磁盘路径则需检查对象持有者、实际文件或存储后端。[当前存储位置的表示](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/store/metadata-management.md)

```gdb
set breakpoint pending on
break mooncake::Client::Query
break mooncake::Client::Get
break mooncake::Client::FindFirstCompleteReplica
break mooncake::TransferSubmitter::submit
```

把这些断点设置在客户端进程。若 `GetReplicaList` 返回 `REPLICA_IS_NOT_READY`，应回到写端确认发布流程；若已经返回可读描述符而数据搬运失败，则转到连接、内存注册或持有者侧的 I/O。

## Trace 3：内存压力、租约与离场

`EvictionThreadFunc` 触发后台驱逐，`BatchEvict` 处理批量候选；`EvictNamespaceMemory` 则针对某个租户空间。先查容量、水位和租户配额，再核对候选的读租约、pin、正在进行的任务和其他副本。启用 SSD offload 时，还要看这次回收是否伴随后台落盘，以及磁盘队列是否有容量。[驱逐实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-store/src/eviction/evictor.cpp)

```gdb
break mooncake::MasterService::EvictionThreadFunc
break mooncake::MasterService::EvictNamespaceMemory
break mooncake::MasterService::ClientMonitorFunc
```

另外两种现象看起来也像驱逐：第一，持有者进程提前结束，segment 不再可访问；第二，客户端从心跳或 remount 中恢复时，Master 里的可读位置发生变化。对照心跳、有效 segment、对象副本与驱逐指标的同一时间窗口，能够区分这三种情况。[运行时观测](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/getting_started/observability.md)
