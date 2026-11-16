---
title: "跟踪一次 Conductor 前缀查询"
weight: 5
---

从缓存事件出发更容易理解一次命中。某个引擎保存一组完整 KV 块，发布 `BlockStored`。Conductor 先要知道这是谁的事件：服务可以从配置静态注册，也可以通过 HTTP 或 RPC 动态注册。每个注册项带有发布端地址、实例、DP rank 和哈希规则。[源码设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/conductor-architecture-design.md)

1. `EventManager` 为已注册的发布方创建订阅连接。`ZMQClient` 读取消息序号和批次。
2. `KVEventHandler` 将事件转为该上下文中某个存储层的所有者记录。`PrefixCacheTable` 维护这些记录，引擎后来驱逐块时会更新这张表。
3. 路由器调用 `Query`。服务按注册的哈希规则计算传入前缀并查表，返回实例和层级命中结果。路由器结合实际负载进行决策。
4. 如果路由上的命中和最终实际加载不同，按事件序号、查询时刻与引擎内的驱逐记录对齐。查询与使用分属两个时刻。

接口示例从 [Python Conductor API](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/conductor.md) 起步，事件格式见[发布方协议](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/publisher-design.md)。

## 用空索引建立对照

在带 `WITH_CONDUCTOR=ON` 的构建中启动 `mooncake_conductor`，确认默认 RPC 端口 13334 可访问。此时还没有注册发布者，可用下面的查询验证 RPC 本身。随后注册实际引擎、写入同一前缀并等事件到达，再查询同一批 token。[Python API](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/conductor.md)

```python
from mooncake.conductor import ConductorClient, OK

with ConductorClient() as client:
    assert client.setup("127.0.0.1:13334") == OK
    result = client.query("my-model", block_size=16, token_ids=list(range(32)))
    assert result["ret"] == OK
    print(result["hits"])
```

若调用成功却一直返回空集合，在 [event_handler.cpp](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-conductor/src/kvevent/event_handler.cpp) 查事件接收和介质映射，再对照 [prefix_indexer.cpp](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-conductor/src/prefixindex/prefix_indexer.cpp) 的索引更新与命中查询。输入端与发布端要使用相同的模型名、块尺寸、租户和哈希约定。
