---
title: "通信基础"
weight: 1
---

RPC 适合传递与当前调用相关的请求和响应，发布订阅适合推送可异步处理的状态变化，设备内存复制还涉及传输完成与所有权。下面用这些区别解释 [Dynamo 1.6.2 的通信平面](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md)。

---

## 1. 分布式通信模式

### 1.1 通信模式对比

```
┌─────────────────────────────────────────────────────────────────────┐
│                     分布式通信模式对比                                │
├─────────────────┬─────────────────┬─────────────────────────────────┤
│     模式        │     特点         │         适用场景               │
├─────────────────┼─────────────────┼─────────────────────────────────┤
│ 同步请求-响应   │ 阻塞等待响应     │ 需要立即结果的操作             │
│ 异步请求-响应   │ 非阻塞，回调通知 │ 长时间运行的操作               │
│ 发布-订阅       │ 解耦生产者消费者 │ 事件通知、状态广播             │
│ 点对点          │ 一对一通信       │ 直接数据传输                   │
│ 广播            │ 一对多通信       │ 状态同步、配置分发             │
└─────────────────┴─────────────────┴─────────────────────────────────┘
```

### 1.2 请求-响应模式

```mermaid
sequenceDiagram
    participant C as Client
    participant S as Server

    Note over C,S: 同步模式
    C->>S: Request
    Note over C: 阻塞等待
    S-->>C: Response

    Note over C,S: 异步模式
    C->>S: Request
    Note over C: 继续执行其他任务
    S-->>C: Callback/Future
```

### 1.3 发布-订阅模式

```mermaid
graph TB
    subgraph pubsub["发布-订阅模式"]
        P1["Publisher 1"] --> T["Topic/Subject"]
        P2["Publisher 2"] --> T
        T --> S1["Subscriber 1"]
        T --> S2["Subscriber 2"]
        T --> S3["Subscriber 3"]
    end
```

**优势**：
- 解耦：发布者不需要知道订阅者
- 灵活：动态添加/移除订阅者
- 扩展：支持多对多通信

---

## 2. 服务发现与配置中心

### 2.1 为什么需要服务发现

在微服务架构中，服务实例是动态的：
- IP 地址可能变化
- 实例数量可能伸缩
- 实例可能故障重启

服务发现解决这些问题：

```mermaid
graph TB
    subgraph discovery["服务发现模式"]
        subgraph client["客户端发现"]
            C1["Client"] --> R1["Registry"]
            R1 --> C1
            C1 --> S1["Service A"]
            C1 --> S2["Service B"]
        end

        subgraph server["服务端发现"]
            C2["Client"] --> LB["Load Balancer"]
            LB --> R2["Registry"]
            LB --> S3["Service A"]
            LB --> S4["Service B"]
        end
    end
```

### 2.2 服务发现的核心功能

| 功能 | 说明 |
|------|------|
| 服务注册 | 服务启动时注册自己的地址 |
| 服务发现 | 客户端查询可用服务实例 |
| 健康检查 | 定期检测服务是否存活 |
| 负载均衡 | 在多个实例间分配请求 |

### 2.3 Dynamo 的选择

**Dynamo 采用客户端发现模式**：
- 客户端直接查询 etcd 获取服务列表
- 客户端负责选择目标实例
- 减少中间层，降低延迟

---

## 3. 消息队列基础

### 3.1 消息队列的作用

{{< stateboard >}}
title: 生产和消费短时不同速
rows: ["消息队列", "正在处理"]
steps:
  - title: "生产两项"
    detail: "生产者连续写入 A、B，消费者暂时空闲。"
    cells: [["A", "B"], ["·", "·"]]
    focus: [[0, 0], [0, 1]]
  - title: "取出 A"
    detail: "消费者开始处理 A，队列里还剩 B。"
    cells: [["B", "·"], ["A", "·"]]
    focus: [[0, 0], [1, 0]]
  - title: "C 抵达"
    detail: "A 仍在计算，队列保存后来的 C。"
    cells: [["B", "C"], ["A", "·"]]
    focus: [[0, 1]]
  - title: "继续消费"
    detail: "A 处理完后轮到 B，C 等待下一次读取。"
    cells: [["C", "·"], ["B", "·"]]
    focus: [[0, 0], [1, 0]]
{{< /stateboard >}}

### 3.2 核心功能

| 功能 | 说明 |
|------|------|
| 异步解耦 | 生产者和消费者不需要同时在线 |
| 流量削峰 | 缓冲突发流量，保护下游服务 |
| 可靠传递 | 消息持久化，确保不丢失 |
| 广播分发 | 一条消息可被多个消费者处理 |

### 3.3 消息模式

**点对点（Queue）**：
```
Producer → Queue → Consumer (一个消息只被一个消费者处理)
```

**发布-订阅（Topic）**：
```
Publisher → Topic → Subscriber 1
                 → Subscriber 2
                 → Subscriber N
```

---

## 4. 高性能数据传输

### 4.1 传统 TCP 的瓶颈

```
传统 TCP 数据传输：
┌────────┐    copy    ┌────────┐    copy    ┌────────┐
│ App    │ ────────→ │ Kernel │ ────────→ │ NIC    │
│ Buffer │           │ Buffer │           │ Buffer │
└────────┘           └────────┘           └────────┘
    ↑                    ↑                    ↑
  用户态               内核态               硬件层
```

**问题**：
- 多次内存拷贝
- 频繁上下文切换
- CPU 开销大

### 4.2 RDMA 零拷贝

```
RDMA 零拷贝传输：
┌────────┐  DMA direct  ┌────────┐
│ App    │ ──────────→ │ NIC    │
│ Buffer │             │ Buffer │
└────────┘             └────────┘
    ↑                      ↑
  用户态                 硬件层
```

**RDMA（Remote Direct Memory Access）优势**：
- 绕过 CPU，直接访问远程内存
- 零拷贝，减少内存带宽消耗
- 微秒级延迟

### 4.3 性能对比

| 指标 | TCP | RDMA |
|------|-----|------|
| 延迟 | ~100μs | ~1μs |
| 吞吐 | ~10 Gbps | ~100+ Gbps |
| CPU 占用 | 高 | 极低 |
| 适用场景 | 通用 | 高性能计算 |

---

## 5. 一致性模型

### 5.1 CAP 定理

分布式系统无法同时满足：
- **C**onsistency（一致性）
- **A**vailability（可用性）
- **P**artition tolerance（分区容错）

```mermaid
graph TB
    subgraph cap["CAP 定理"]
        C["Consistency<br/>一致性"]
        A["Availability<br/>可用性"]
        P["Partition Tolerance<br/>分区容错"]

        CA["CA 系统<br/>单机数据库"]
        CP["CP 系统<br/>etcd, ZooKeeper"]
        AP["AP 系统<br/>Cassandra, DynamoDB"]

        C --> CA
        A --> CA
        C --> CP
        P --> CP
        A --> AP
        P --> AP
    end
```

### 5.2 Dynamo 的选择

| 组件 | 一致性级别 | 原因 |
|------|-----------|------|
| etcd | 强一致性 (CP) | 服务注册需要准确 |
| NATS | 最终一致性 (AP) | 事件通知允许短暂延迟 |
| KV Transfer | N/A | 数据传输，不涉及一致性 |

---

## 6. 与 Dynamo 的关联

### 三类运行时流量

```mermaid
flowchart LR
    W["Worker"] --> D["发现服务与成员注册"]
    FE["Frontend"] --> RPC["请求与响应"]
    RPC --> W
    W --> E["KV 事件与遥测"]
    E --> R["路由索引"]
```

从发现得到成员地址后，运行时默认用 TCP 传递请求，以 ZMQ 发布事件。这两者可独立改为 NATS。跨 GPU 的 KV 由后端及其连接器传输。排查缺块时应看连接器的完成通知。具体部署条件见[通信平面](/dynamo/05-distributed-communication/02-three-plane-architecture/)。

## 参考资料

- [Designing Data-Intensive Applications](https://dataintensive.net/)
- [RDMA over Converged Ethernet](https://en.wikipedia.org/wiki/RDMA_over_Converged_Ethernet)
- [CAP Theorem](https://en.wikipedia.org/wiki/CAP_theorem)
