---
title: "Rust 异步编程基础"
weight: 1
---

一次服务调用可能等待成员发现、网络或下一项流式输出。Rust Future 在等待期间通过 waker 请求唤醒，使 Tokio 可以推进其他就绪任务。以下将这些基础机制与 [Dynamo 1.6.2 的运行时](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/runtime/src/runtime.rs) 对照。

---

## 1. 同步 vs 异步

### 1.1 同步 I/O 的问题

传统的同步 I/O 模型中，当程序执行 I/O 操作（如网络请求、文件读写）时，线程会被阻塞，直到操作完成：

```mermaid
sequenceDiagram
    participant T1 as Thread 1
    participant IO as I/O 操作

    T1->>IO: 发起请求
    Note over T1: 阻塞等待...
    IO-->>T1: 完成
    T1->>T1: 继续执行
```

**问题**：
- 每个并发任务需要一个线程
- 频繁线程切换会占用 CPU 时间，开销随系统与负载变化
- 线程需要独立栈与调度状态，实际占用依赖线程栈配置与使用情况

### 1.2 异步 I/O 的优势

异步 I/O 模型允许在等待 I/O 时切换到其他任务：

```mermaid
graph LR
    subgraph thread["单线程执行"]
        A["任务 A<br/>发起 I/O"] --> B["任务 B<br/>发起 I/O"]
        B --> C["任务 A<br/>I/O 完成"]
        C --> D["任务 C<br/>发起 I/O"]
        D --> E["任务 B<br/>I/O 完成"]
    end
```

**优势**：
- 单线程可处理大量并发
- 无线程切换开销
- 内存占用低

---

## 2. Future 与 Poll 机制

### 2.1 Future Trait

Rust 的异步模型基于 `Future` trait，这是一种 **惰性求值** 的设计：

```rust
pub trait Future {
    type Output;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output>;
}

pub enum Poll<T> {
    Ready(T),    // 计算完成，返回结果
    Pending,     // 计算未完成，需要稍后再试
}
```

### 2.2 Poll 状态机

{{< trace >}}
title: 一个等待 I/O 的 Future 如何再被 poll
actors: ["执行器", "Future", "I/O"]
steps:
  - from: 0
    to: 1
    label: "首次 poll"
    detail: "执行器把当前任务的 waker 交给 Future。"
  - from: 1
    to: 2
    label: "预约唤醒"
    detail: "此时数据还没到，Future 登记等待条件。"
  - from: 1
    to: 0
    label: "返回 Pending"
    detail: "执行器可以转去推进其他已就绪的任务。"
  - from: 2
    to: 0
    label: "数据就绪"
    detail: "I/O 完成使任务重新就绪，执行器安排下一次 poll。"
  - from: 1
    to: 0
    label: "返回 Ready"
    detail: "第二次 poll 能取得结果，等待这项数据的调用继续。"
{{< /trace >}}

### 2.3 关键概念

| 概念 | 说明 |
|------|------|
| **惰性（Lazy）** | Future 创建后不会立即执行，必须被 poll 才会推进 |
| **协作式（Cooperative）** | Future 主动让出控制权，而非被抢占 |
| **零成本抽象** | 编译器将 async 函数转换为状态机，调度、唤醒与 I/O 轮询仍有成本 |

---

## 3. async/await 语法糖

### 3.1 基本用法

`async/await` 是对 Future 的语法糖，使异步代码看起来像同步代码：

```rust
// 使用 async/await
async fn fetch_and_process() -> Result<Data> {
    let response = client.get("http://api.example.com").await?;
    let data = response.json::<Data>().await?;
    Ok(data)
}
```

### 3.2 状态机转换

编译器将 async 函数转换为状态机：

```mermaid
stateDiagram-v2
    Start --> WaitingForResponse: client.get() 发起请求
    WaitingForResponse --> WaitingForJson: 响应到达
    WaitingForJson --> Done: JSON 解析完成

    note right of WaitingForResponse: 网络 I/O 进行中
    note right of WaitingForJson: JSON 解析中
```

**编译器生成的状态机**：

```rust
enum FetchAndProcessState {
    Start,
    WaitingForResponse { client: Client },
    WaitingForJson { response: Response },
    Done,
}
```

---

## 4. Tokio 异步运行时

### 4.1 核心组件

Tokio 是 Rust 生态中最流行的异步运行时，Dynamo 选择 Tokio 作为底层运行时。

```mermaid
graph TB
    subgraph tokio["Tokio Runtime"]
        subgraph scheduler["Scheduler 调度器"]
            MT["Multi-Thread<br/>多线程调度器"]
            CT["Current-Thread<br/>单线程调度器"]
        end

        subgraph driver["I/O Driver"]
            EP["Epoll/Kqueue/IOCP<br/>系统事件循环"]
            TIM["Timer<br/>定时器"]
        end

        subgraph task["Task System"]
            Q1["Work Queue 1"]
            Q2["Work Queue 2"]
            GQ["Global Queue"]
        end

        BP["Blocking Pool<br/>阻塞线程池"]
    end

    MT --> Q1
    MT --> Q2
    MT --> GQ
    EP --> MT
    TIM --> MT
```

### 4.2 Work-Stealing 调度

Tokio 的多线程调度器使用 **Work-Stealing** 算法来平衡负载：

```mermaid
graph TB
    subgraph workers["Worker 线程"]
        W1["Worker 1<br/>任务 A, B, C"]
        W2["Worker 2<br/>任务 D, E"]
        W3["Worker 3<br/>空闲"]
    end

    W3 --> |steal| W1

    GQ["Global Inject Queue<br/>新任务首先进入"]

    GQ --> W1
    GQ --> W2
    GQ --> W3
```

**Work-Stealing 的优势**：

| 优势 | 说明 |
|------|------|
| **负载均衡** | 空闲 Worker 主动获取任务 |
| **缓存友好** | 任务优先在创建它的 Worker 上执行 |
| **低竞争** | 每个 Worker 有自己的队列 |

### 4.3 Blocking 线程池

某些操作无法异步化（如文件系统操作、CPU 密集计算），Tokio 提供 Blocking 线程池：

```rust
// 在 blocking 线程池中执行同步操作
let result = tokio::task::spawn_blocking(|| {
    // 这里的代码在独立线程中运行
    std::fs::read_to_string("large_file.txt")
}).await?;
```

```mermaid
graph LR
    subgraph async_rt["Async Runtime"]
        AT1["Async Task 1"]
        AT2["Async Task 2"]
    end

    subgraph blocking["Blocking Pool"]
        BT1["Blocking Task 1"]
        BT2["Blocking Task 2"]
    end

    AT1 --> |spawn_blocking| BT1
    BT1 --> |JoinHandle| AT1
```

---

## 5. Dynamo 中的应用

### 5.1 Runtime 配置

Dynamo 基于 Tokio 构建运行时：

```rust
// Dynamo Runtime 的核心结构
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,      // 主任务运行时
    secondary: RuntimeType,    // 后台任务运行时
    cancellation_token: CancellationToken,
}
```

### 5.2 双运行时设计

```mermaid
graph TB
    subgraph primary["Primary Runtime - 多线程"]
        P1["推理请求处理"]
        P2["HTTP 请求处理"]
        P3["数据预处理"]
    end

    subgraph secondary["Secondary Runtime - 单线程"]
        S1["etcd Keep-Alive"]
        S2["NATS 订阅"]
        S3["健康检查"]
    end

    Client["客户端请求"] --> P1
    ETCD[("etcd")] <--> S1
    NATS[("NATS")] <--> S2
```

**分离的原因**：

1. **防止阻塞**：后台任务不会抢占业务任务的 CPU
2. **资源控制**：可以独立配置两个运行时的线程数
3. **故障隔离**：后台任务的问题不会直接影响主业务

---

## 下一篇

继续阅读 [02-分布式系统基础](02-distributed-system-basics.md)，了解分布式系统的核心概念。
