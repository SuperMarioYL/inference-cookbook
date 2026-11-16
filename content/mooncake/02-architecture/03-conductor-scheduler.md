---
title: "Conductor 怎样为路由器查找 KV 前缀"
weight: 3
---

假设两台推理实例都能处理下一条请求，只有其中一台保留了手册的前半段。路由器可以把 token 序列交给 Conductor，查询各实例在设备、主机和磁盘层上有多少前缀可复用，再结合队列和自己的策略选择目标。

v0.3.14 已公开 `mooncake_conductor`。它订阅 vLLM、SGLang 和 Mooncake Store Master 的增加、删除和清空事件，并把这些事件换算成一致的前缀索引。FAST ’25 论文还讨论了完整服务的调度、预测和负载控制，两者的责任边界不同。[开源设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/conductor-architecture-design.md) · [论文](https://www.usenix.org/conference/fast25/presentation/qin)

{{< stateboard >}}
title: "路由器何时能查到块 B"
rows: ["事件来源", "索引里的 B", "查询结果"]
steps:
  - title: "尚未发布"
    detail: "设备已有块 B，此时 Conductor 还没有收到事件，查询结果为空。"
    cells:
      - ["设备有 B"]
      - ["·"]
      - ["空"]
    focus: [[0, 0]]
  - title: "接收事件"
    detail: "引擎发布 BlockStored，索引记录实例 A 的设备持有 B。"
    cells:
      - ["已发布"]
      - ["A 设备"]
      - ["空"]
    focus: [[1, 0]]
  - title: "路由器查询"
    detail: "路由器带着相同 token 和上下文查询，得到 A 的前缀覆盖范围。"
    cells:
      - ["已发布"]
      - ["A 设备"]
      - ["命中 A"]
    focus: [[2, 0]]
  - title: "块被驱逐"
    detail: "删除事件到达后，这个实例的位置记录消失。"
    cells:
      - ["已驱逐"]
      - ["·"]
      - ["空"]
    focus: [[0, 0], [1, 0], [2, 0]]
{{< /stateboard >}}

## 为什么同样的 token 仍可能查不到

查询、发布端要使用相同的模型、租户、LoRA、块尺寸和哈希方案。缓存被驱逐时对应记录会删除。事件序号发生跳跃时，能否补齐取决于发布端是否提供重放。路由器根据索引选择实例，持有目标块的引擎再发起读取。[事件订阅与查询](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/subscriber-guide.md)

## 一次查询怎样比较不同层

查询上下文由 tenant、model、LoRA 和块长决定，还绑定来源采用的哈希方案。Conductor 根据 query 的 token 重算前缀哈希，然后沿连续前缀查询：某一层级第一次失配即停止该层级的累计。返回的 device、cpu_local、cpu_share、disk 是可覆盖范围逐步放宽得到的累计 token 数，不能相加。例如得到 64、128、128、192，只代表最长可覆盖 192 个 token。多个 DP rank 汇总到实例时，每一层取该实例各 rank 的最大值。[查询流程](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/conductor/conductor-architecture-design.md)

## 事件缺口会怎样影响路由

引擎发布端可以配置 replay endpoint。Conductor 发现序号跳跃时，暂存新事件并请求缺失范围，按顺序恢复。Mooncake Store Master 没有这条 replay 接口；无法恢复时，索引清掉来自该来源的记录并标记 stale，直到重新注册。此时返回命中减少，可能是路由可见性变化，而实际 Store 中的字节仍然存在。

## 论文里的调度目标怎样映射到开源组件

论文同时关注 TTFT、逐 token 延迟和符合 SLO 的容量；开源查询给路由端提供的是缓存局部性。以两个候选实例说明：A 在设备上多命中一个块但队列很长，B 要多做一次 prefill 却能立即服务。选择哪个实例要看路由器的排队、计算和传输估计。Conductor 没有向查询者承诺某个固定权重公式或必达的 SLO。[FAST ’25 调度与实验](https://www.usenix.org/conference/fast25/presentation/qin)
