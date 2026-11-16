---
title: "TENT 怎样选择传输路径"
weight: 1
---

一台主机可能有多张网卡，本地 GPU 也可能通过不同链路连接。TENT（Transfer Engine NEXT）接收源位置、目标位置和数据长度，再从当前拓扑与可用 transport 中选择路径。调用方追踪传输状态，路径选择由运行时承担。[TENT 设计](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/overview.md)

## 从请求到切片

较大的数据请求可以拆成多个 slice。合适的多 rail 配置允许它们分布到不同链路。可达性、负载与链路状态会影响选择；路径退化后，后续请求可能采取不同路由。`v0.3.14` 增加异步 TCP、多 rail 路由及大读请求分条等能力。是否实际启用某条路径，要看构建的后端、GPU/NIC 拓扑和运行配置。[传输选择](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/transport-selector.md) · [版本说明](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14)

看性能时，记录单条传输的大小、各 rail 实际分到的字节和尾延迟。把每条链路最后一个 slice 的完成时刻对齐，可以看出尾延迟集中在哪张网卡。[TENT 指标](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/metrics.md)

## 直连不可用时，数据经过哪里

应用程序使用 TENT 时**不直接选择传输后端**。它们提交的传输请求只描述需要移动的数据（源地址、目标 Segment、偏移量、长度），而非如何移动。

运行时会依次处理下面的决策。
1. 确定源和目的地之间有哪些传输后端和路径可用
2. 在运行时选择执行计划
3. 如果条件变化，可以为后续请求变更决策

当直接路径不可用时，TENT 自动构建**分级传输 (staged transfer)**，例如通过主机内存中转数据。这一逻辑完全在运行时内部处理，不需要应用层修改。

比较 GPU 直连与主存中转时，记录源/目标 segment、内存位置和实际选中的路径。传输字节量相同，中转仍会额外占用 Host 带宽。

## 多 rail 的两种选择

在 RDMA slice 分配中，基础模式先选最近的非空 NUMA 层级，再在该层选择网卡。打开智能选择后，selector 结合在途字节、slice 大小和 EWMA 带宽估算完成时间，并加上 NUMA 距离权重。这样，最近的网卡已经拥堵时，更远的可达路径也可能更早完成。应从实际配置确认采用哪个模式。[Slice spraying 设计与参数](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/slice-spraying.md)

## 失败发生在提交还是完成队列

某条 rail 的坏 completion 可以触发故障计数和冷却；试探成功后路径重新参与调度。另一层机制允许一项已提交的任务在符合条件时改走其他 transport。提交阶段直接返回非 OK 时，该任务进入失败状态。对照提交返回值和后续 completion，可找到出错阶段。[故障恢复与限制](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/failover.md)
