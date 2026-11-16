---
title: "KV 块管理器与路由索引的职责"
weight: 5
---

worker 的推理引擎把逻辑 token 映射到本地缓存槽。路由索引记录各实例拥有的前缀。启用分层卸载后，存储组件还要维护设备、主机与存储层上的副本。

Dynamo KVBM 提供逻辑块生命周期、物理存储与传输的组件。是否启用、可用存储层以及与引擎原生 offload 的组合要按当前后端与配置核对。普通的本地聚合示例可以直接用推理引擎自己的缓存工作；仅勾选 KV-aware routing 不会自动给每个 worker 启用 KVBM。

例如路由器报告“潜在命中两个块”时，应向 engine 查询实际复用了多少、向卸载组件查询是否有加载成本，再结合输入排队时间判断请求是否受益。

## 一次块迁移的资源账

一块 KV 从设备写到主机时，至少涉及源块占用、目的空间和传输完成通知。如果只是增添副本，传输期间源块仍可能处于使用中；只有拥有其生命周期的一方确认可释放，设备容量才真正回升。容量指标应按设备、主机和外部存储各自理解。

## 从逻辑句柄跟到物理存储

跟踪一次 offload/reload 时，先查调用者持有可写块、已完成块还是活动引用，再记录源层、目的层和块键。复制完成时间说明何时能在目的端读取，释放最后一个源端引用的时间解释容量何时回升。

## 用逻辑块类型跟踪两个引用

上游 `kvbm-logical` 的教程给出逻辑状态如何体现在 Rust 类型上。假设请求刚分配一个可写块，并且填好了该页 KV：

| 句柄 | 此时能做的事情 | 丢掉最后一个此类句柄后的行为 |
|---|---|---|
| `MutableBlock` | 写入新内容 | 未完成的块回到 reset 池。 |
| `CompleteBlock` | 已携带序列哈希，等待注册 | 尚未注册的块可回到 reset 池。 |
| `ImmutableBlock` | 通过强引用保护注册后的内容 | 最后的强引用释放后，块可以进入不活跃缓存。 |
| `WeakBlock` | 保留非拥有查询句柄，尝试升级 | 它自身不阻止回收，升级要重新确认内容仍可用。 |

若 A 与 B 都持有已注册块的强引用，A 完成只减掉一个使用者，B 仍可读取。等二者都释放后，新的查询还可能通过缓存重新取得块。把“某请求结束”“最后一个强引用释放”和“物理内存被重新使用”分开，是排查容量和错误命中的第一步。类型名和上游示例见 [KVBM 逻辑块说明](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/kvbm-logical/README.md)。

[KVBM 配置](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/reference/components/kvbm-configuration.mdx) · [逻辑块管理](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/kvbm-logical/README.md) · [路由与卸载支持矩阵](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/offloading-support-matrix.md)
