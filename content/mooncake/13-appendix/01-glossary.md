---
title: "术语表与缩略词"
weight: 1
---

遇到 segment、replica 或 lease 时，先看哪个组件持有它，再查它在何时失效。

---

## 1. 架构术语

### 1.1 核心概念

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Prefill | 预填充 | - | LLM 推理的首阶段,处理完整输入 Prompt 并生成首个 Token 的过程 |
| Decode | 解码 | - | LLM 推理的第二阶段,逐个生成后续 Token 的自回归过程 |
| KVCache | 键值缓存 | - | 存储 Transformer 中每层 Attention 的 Key 和 Value 张量,避免重复计算 |
| Prefill-Decode Disaggregation | P/D 分离 | P/D | 将 Prefill 和 Decode 阶段部署在不同的物理节点上执行 |
| Mooncake Store | - | - | Mooncake 的分布式 KVCache 存储引擎 |
| Transfer Engine | 传输引擎 | TE | 负责高性能跨节点数据传输的模块 |
| Conductor | 指挥者 | - | 开源前缀缓存索引，按事件维护 KV 位置并供路由器查询 |

### 1.2 系统组件

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Prefill Worker | Prefill 工作节点 | 执行 Prefill 计算的节点,通常配置 GPU |
| Decode Worker | Decode 工作节点 | 执行 Decode 计算的节点,通常配置 GPU |
| Storage Worker | 存储工作节点 | 提供 CPU 内存/SSD 存储空间的节点 |
| Master Node | 主控节点 | 管理 Mooncake Store 元数据的中心节点 |
| Cache-aware Scheduling | 缓存感知调度 | 根据 KVCache 位置优化任务分配的调度策略 |
| Hot Prefix Migration | 热前缀迁移 | 将高频访问的前缀 KVCache 迁移到需求节点 |

---

## 2. 网络与传输术语

### 2.1 RDMA 核心术语

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Remote Direct Memory Access | 远程直接内存访问 | RDMA | 绕过 CPU 直接访问远程节点内存的网络技术 |
| Queue Pair | 队列对 | QP | RDMA 通信的基本单元,包含发送队列和接收队列 |
| Completion Queue | 完成队列 | CQ | 存储 RDMA 操作完成事件的队列 |
| Work Request | 工作请求 | WR | 描述 RDMA 操作的数据结构 (Read/Write/Send) |
| Memory Region | 内存区域 | MR | 注册给 RDMA NIC 的内存区域,可被远程访问 |
| Local Key | 本地密钥 | lkey | 用于本地访问 MR 的密钥 |
| Remote Key | 远程密钥 | rkey | 用于远程访问 MR 的密钥 |
| Scatter-Gather Entry | 分散聚集条目 | SGE | 描述内存缓冲区位置和长度的结构 |

### 2.2 RDMA 操作类型

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| RDMA Read | RDMA 读 | 从远程节点读取数据到本地,单边操作 |
| RDMA Write | RDMA 写 | 将本地数据写入远程节点,单边操作 |
| RDMA Send/Recv | RDMA 发送/接收 | 双边通信操作,需要接收方配合 |
| GPU Direct RDMA | GPU 直接 RDMA | 绕过 CPU,直接在 GPU 显存和 RDMA NIC 间传输数据 |

### 2.3 网络拓扑术语

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Top-of-Rack Switch | 机架顶部交换机 | ToR | 连接同机架内所有节点的交换机 |
| Spine Switch | 主干交换机 | - | 连接多个 ToR 交换机的上层交换机 |
| Leaf-Spine Topology | 叶脊拓扑 | - | 数据中心常用的二层网络拓扑结构 |
| Network Interface Card | 网络接口卡 | NIC | 物理网络硬件,RDMA NIC 支持 RDMA 协议 |
| InfiniBand | InfiniBand | IB | 高性能 RDMA 网络协议,常见于 HPC 场景 |
| RoCE | RDMA over Converged Ethernet | RoCE | 基于以太网的 RDMA 协议 (RoCE v2 为主流) |

---

## 3. 性能与监控术语

### 3.1 性能指标

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Service Level Objective | 服务等级目标 | SLO | 系统需满足的性能指标 (如延迟、可用性) |
| Time Between Tokens | 每 Token 间隔时间 | TBT | Decode 阶段生成相邻两个 Token 的时间间隔 |
| Time To First Token | 首 Token 时间 | TTFT | 从请求提交到返回首个 Token 的时间 (Prefill 延迟) |
| Throughput | 吞吐量 | - | 单位时间内系统处理的请求数 (req/s) 或 Token 数 (tokens/s) |
| Latency | 延迟 | - | 请求从提交到完成的时间 |
| P50/P95/P99 Latency | 延迟分位数 | - | 50%/95%/99% 请求的延迟不超过该值 |
| Effective Request Capacity | 有效请求容量 | - | 在满足 SLO 约束下系统能处理的最大请求量 |

### 3.2 资源指标

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| GPU Utilization | GPU 利用率 | GPU 计算单元的使用率 (0-100%) |
| Memory Bandwidth | 内存带宽 | 内存读写速度 (GB/s) |
| Network Bandwidth | 网络带宽 | 网络传输速度 (Gbps) |
| Queue Length | 队列长度 | 等待处理的请求数 |
| Compute-bound | 计算受限 | 性能瓶颈在 GPU 计算能力 |
| Memory-bound | 内存受限 | 性能瓶颈在显存带宽 |

---

## 4. 存储术语

### 4.1 存储介质

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| GPU Memory (VRAM) | GPU 显存 | GPU 板载的高速内存 (HBM),带宽 ~2TB/s |
| CPU Memory (DRAM) | CPU 内存 | 主机内存,带宽 ~100GB/s |
| NVMe SSD | NVMe 固态硬盘 | 基于 PCIe 的高速 SSD,带宽 ~7GB/s |
| Memory Hierarchy | 存储层次 | GPU 显存 > CPU 内存 > SSD (速度递减,容量递增) |

### 4.2 存储策略

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Least Recently Used | 最近最少使用 | LRU | 驱逐最久未访问的缓存条目 |
| Least Frequently Used | 最少使用频率 | LFU | 驱逐访问次数最少的缓存条目 |
| Time To Live | 生存时间 | TTL | 缓存条目的有效期,超时后自动失效 |
| Segment | 分段 | - | Mooncake Store 中存储数据的基本单元 |
| Replica | 副本 | - | 数据的备份拷贝,用于高可用 |
| Metadata | 元数据 | - | 描述数据位置、大小、状态等信息的数据 |

---

## 5. 深度学习术语

### 5.1 模型架构

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Transformer | Transformer | 基于 Self-Attention 的神经网络架构 |
| Self-Attention | 自注意力 | Transformer 的核心机制,计算序列内 Token 间关系 |
| Multi-Head Attention | 多头注意力 | MHA | 并行计算多组 Attention,增强表达能力 |
| Grouped Query Attention | 分组查询注意力 | GQA | 减少 KVCache 显存占用的 Attention 变体 |
| Mixture of Experts | 混合专家模型 | MoE | 由多个专家子网络组成的模型架构 |
| Expert Parallelism | 专家并行 | EP | 将 MoE 的不同 Expert 分布到不同节点 |

### 5.2 推理优化

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Continuous Batching | 连续批处理 | 动态组合不同长度的请求进行批量推理 |
| Paged Attention | 分页注意力 | 将 KVCache 分页管理,减少显存碎片 |
| FlashAttention | FlashAttention | 优化 Attention 计算的内存访问模式 |
| Speculative Decoding | 推测解码 | 用小模型预测多个 Token,大模型验证 |
| Prefix Caching | 前缀缓存 | 缓存共享的 Prompt 前缀,避免重复计算 |

---

## 6. 分布式系统术语

### 6.1 通信协议

| 英文术语 | 中文翻译 | 缩写 | 解释 |
|---------|---------|------|------|
| Remote Procedure Call | 远程过程调用 | RPC | 跨进程/跨节点调用函数的通信机制 |
| gRPC | - | gRPC | Google 开发的高性能 RPC 框架 |
| Protocol Buffers | 协议缓冲区 | Protobuf | 高效的结构化数据序列化格式 |
| Message Passing Interface | 消息传递接口 | MPI | HPC 领域的标准通信协议 |

### 6.2 一致性与可用性

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Fault Tolerance | 容错性 | 系统在部分组件故障时仍能正常工作的能力 |
| High Availability | 高可用性 | HA | 系统长期稳定运行的能力 (通常 99.9%+ 在线率) |
| Consistency | 一致性 | 数据在多个副本间保持同步的程度 |
| Eventual Consistency | 最终一致性 | 数据副本最终会达到一致,但可能短期不一致 |
| Replication Factor | 副本因子 | 每份数据存储的副本数量 |

---

## 7. Mooncake 特有术语

### 7.1 核心模块

| 英文术语 | 缩写 | 解释 |
|---------|------|------|
| Mooncake Store | - | 分布式 KVCache 存储系统,支持 CPU 内存 + SSD |
| Transfer Engine | TE | 基于 RDMA 的高性能数据传输引擎 |
| TENT | TENT | Topology-aware Engine for Network Transfer,拓扑感知传输引擎 |
| P2P Store | - | 点对点对象存储,用于 Checkpoint 和大文件传输 |

### 7.2 优化技术

| 英文术语 | 解释 |
|---------|------|
| KVCache-centric Architecture | 以 KVCache 为中心的架构设计理念 |
| Cache-aware Scheduling | 根据 KVCache 位置优化调度的策略 |
| Topology-aware Transfer | 基于网络拓扑优化传输路径的技术 |
| Hot Prefix Migration | 将高频访问的前缀 KVCache 迁移到需求节点 |
| Batch Transfer | 批量传输多个 Segment,减少 RDMA 开销 |

---

## 8. 性能分析术语

### 8.1 性能剖析

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Profiling | 性能剖析 | 分析程序执行时间分布的过程 |
| Bottleneck | 瓶颈 | 限制系统整体性能的关键环节 |
| Critical Path | 关键路径 | 决定总延迟的最长执行路径 |
| Hotspot | 热点 | 消耗 CPU/GPU 时间最多的代码段或数据 |

### 8.2 优化指标

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Speedup | 加速比 | 优化后性能相对优化前的提升倍数 |
| Overhead | 开销 | 引入新机制带来的额外成本 (时间/空间) |
| Trade-off | 权衡 | 在多个优化目标间的平衡 (如 Mooncake 用存储换计算) |
| Scalability | 可扩展性 | 增加资源时性能提升的能力 |

---

## 9. 使用场景术语

### 9.1 应用场景

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Long Context | 长上下文 | Prompt 长度 >8K Token 的场景 |
| Multi-turn Dialogue | 多轮对话 | 连续对话场景,上下文累积增长 |
| Batch Inference | 批量推理 | 同时处理多个请求以提升吞吐量 |
| Online Serving | 在线服务 | 实时响应用户请求的推理场景 |
| Offline Batch | 离线批处理 | 非实时场景,吞吐优先于延迟 |

### 9.2 工作负载

| 英文术语 | 中文翻译 | 解释 |
|---------|---------|------|
| Compute-intensive | 计算密集型 | Prefill 阶段,需要大量矩阵运算 |
| Memory-intensive | 内存密集型 | Decode 阶段,受限于 KVCache 内存访问 |
| Bursty Traffic | 突发流量 | 请求量短时间内剧烈波动 |
| Steady Traffic | 稳定流量 | 请求量相对均匀 |

---

## 10. 缩略词速查表

按字母顺序排列:

| 缩写 | 全称 | 中文 |
|------|------|------|
| API | Application Programming Interface | 应用程序接口 |
| CQ | Completion Queue | 完成队列 |
| CUDA | Compute Unified Device Architecture | 统一计算设备架构 |
| DRAM | Dynamic Random Access Memory | 动态随机存取内存 |
| EP | Expert Parallelism | 专家并行 |
| GQA | Grouped Query Attention | 分组查询注意力 |
| HA | High Availability | 高可用性 |
| HBM | High Bandwidth Memory | 高带宽内存 |
| HPC | High Performance Computing | 高性能计算 |
| IB | InfiniBand | InfiniBand 网络协议 |
| KV | Key-Value | 键值 |
| LFU | Least Frequently Used | 最少使用频率 |
| LLM | Large Language Model | 大语言模型 |
| LRU | Least Recently Used | 最近最少使用 |
| MHA | Multi-Head Attention | 多头注意力 |
| MoE | Mixture of Experts | 混合专家模型 |
| MPI | Message Passing Interface | 消息传递接口 |
| MR | Memory Region | 内存区域 |
| NIC | Network Interface Card | 网络接口卡 |
| NVMe | Non-Volatile Memory Express | 非易失性内存主机控制器接口 |
| P/D | Prefill-Decode | 预填充-解码 |
| P2P | Peer-to-Peer | 点对点 |
| P99 | 99th Percentile | 99 分位数 |
| QP | Queue Pair | 队列对 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问 |
| RoCE | RDMA over Converged Ethernet | 融合以太网 RDMA |
| RPC | Remote Procedure Call | 远程过程调用 |
| SGE | Scatter-Gather Entry | 分散聚集条目 |
| SLO | Service Level Objective | 服务等级目标 |
| SSD | Solid State Drive | 固态硬盘 |
| TBT | Time Between Tokens | 每 Token 间隔时间 |
| TCP | Transmission Control Protocol | 传输控制协议 |
| TE | Transfer Engine | 传输引擎 |
| TENT | Topology-aware Engine for Network Transfer | 拓扑感知网络传输引擎 |
| ToR | Top-of-Rack | 机架顶部 (交换机) |
| TTL | Time To Live | 生存时间 |
| TTFT | Time To First Token | 首 Token 时间 |
| VRAM | Video Random Access Memory | 显存 |
| WR | Work Request | 工作请求 |

---

## 参考资料

- Mooncake 论文: [FAST'25 Paper](https://www.usenix.org/conference/fast25/presentation/qin)
- vLLM 文档: [vLLM Documentation](https://docs.vllm.ai)
- RDMA 编程指南: [RDMA Programming Guide](https://www.rdmamojo.com/)

---

**提示**: 本术语表会随文档更新持续完善。如有遗漏或错误,欢迎提交 Issue 或 PR。
