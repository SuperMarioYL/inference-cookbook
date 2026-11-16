---
title: "Runtime 双层设计"
weight: 3
---

## 本地执行器与分布式成员怎样配合

每个进程独立创建 runtime，以相同 namespace 和发现后端联通。Dynamo 1.6.2 的开发例子可使用文件发现，默认请求传输与事件传输分别是 TCP、ZMQ。以下 etcd/NATS 的配置对应明确选用这些后端的情况。 [DistributedRuntime 源码](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/runtime/src/distributed.rs)

## 1. 设计理念

Dynamo 的运行时采用双层设计：**Local Runtime** 和 **Distributed Runtime**。这种设计遵循了 **关注点分离** 原则：

```mermaid
graph TB
    subgraph app["应用层"]
        App["用户应用"]
    end

    subgraph distributed["分布式运行时层"]
        DRT["DistributedRuntime"]
        ETCD["etcd Client"]
        NATS["NATS Client"]
        TCP["TCP Server"]
        REG["Component Registry"]
    end

    subgraph local["本地运行时层"]
        RT["Runtime"]
        Primary["Primary Runtime<br/>主业务线程池"]
        Secondary["Secondary Runtime<br/>后台任务线程池"]
        CT["CancellationToken"]
    end

    subgraph tokio["Tokio 层"]
        TokioRT["Tokio Runtime"]
    end

    App --> DRT
    DRT --> RT
    DRT --> ETCD
    DRT --> NATS
    DRT --> TCP
    DRT --> REG

    RT --> Primary
    RT --> Secondary
    RT --> CT

    Primary --> TokioRT
    Secondary --> TokioRT
```

**设计目标**：

| 目标 | 实现方式 |
|------|----------|
| **分层解耦** | Local Runtime 不依赖分布式组件，可独立测试 |
| **资源隔离** | 业务逻辑和后台任务使用不同的线程池 |
| **生命周期管理** | 统一的取消令牌机制 |

---

## 2. Local Runtime

### 2.1 核心结构

`Runtime` 是本地运行时，管理 Tokio 线程池和取消令牌：

```rust
// 源码位置: lib/runtime/src/lib.rs:54-67

pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,
    secondary: RuntimeType,
    cancellation_token: CancellationToken,
}

enum RuntimeType {
    Shared(Arc<tokio::runtime::Runtime>),
    External(tokio::runtime::Handle),
}
```

### 2.2 RuntimeType 枚举

`RuntimeType` 支持两种模式：

```mermaid
graph LR
    subgraph shared["Shared 模式"]
        S1["Dynamo 创建 Runtime"]
        S2["Dynamo 管理生命周期"]
        S3["Dynamo 关闭 Runtime"]
        S1 --> S2 --> S3
    end

    subgraph external["External 模式"]
        E1["外部提供 Handle"]
        E2["Dynamo 使用 Handle"]
        E3["外部管理生命周期"]
        E1 --> E2 --> E3
    end
```

| 模式 | 说明 | 使用场景 |
|------|------|----------|
| **Shared** | Dynamo 创建和管理 Tokio Runtime | 独立运行的服务 |
| **External** | 使用外部提供的 Runtime Handle | 嵌入到其他应用中 |

### 2.3 Primary vs Secondary Runtime

Dynamo 使用两个运行时来隔离不同类型的任务：

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
        S4["日志上报"]
    end

    Client["客户端请求"] --> P1
    ETCD[("etcd")] <--> S1
    NATS[("NATS")] <--> S2
```

| 运行时 | 用途 | 特点 |
|--------|------|------|
| **Primary** | 主业务逻辑 | 多线程，处理推理请求 |
| **Secondary** | 后台任务 | 单线程，处理 etcd/NATS 通信 |

**为什么要分离？**

1. **防止阻塞**：后台任务（如 Keep-Alive）不会抢占业务任务的 CPU
2. **资源控制**：可以独立配置两个运行时的线程数
3. **故障隔离**：后台任务的问题不会直接影响主业务

### 2.4 Runtime 创建方式

```rust
impl Runtime {
    /// 从当前 Tokio 运行时创建（嵌入模式）
    pub fn from_current() -> Result<Runtime>;

    /// 从外部 Handle 创建
    pub fn from_handle(handle: tokio::runtime::Handle) -> Result<Runtime>;

    /// 从配置文件创建（推荐）
    pub fn from_settings() -> Result<Runtime>;

    /// 创建单线程运行时（测试用）
    pub fn single_threaded() -> Result<Runtime>;
}
```

---

## 3. Distributed Runtime

### 3.1 核心结构

`DistributedRuntime` 扩展了 `Runtime`，增加了分布式系统所需的组件：

```rust
// 源码位置: lib/runtime/src/lib.rs:68-86

pub struct DistributedRuntime {
    // 本地运行时
    runtime: Runtime,

    // 分布式组件
    etcd_client: transports::etcd::Client,
    nats_client: transports::nats::Client,
    tcp_server: Arc<OnceCell<Arc<TcpStreamServer>>>,

    // 本地组件注册表
    component_registry: component::Registry,
}
```

### 3.2 类图

```mermaid
classDiagram
    class Runtime {
        -id: Arc~String~
        -primary: RuntimeType
        -secondary: RuntimeType
        -cancellation_token: CancellationToken
        +primary() Handle
        +secondary() Handle
        +child_token() CancellationToken
        +shutdown()
    }

    class DistributedRuntime {
        -runtime: Runtime
        -etcd_client: etcd::Client
        -nats_client: nats::Client
        -tcp_server: Arc~OnceCell~
        -component_registry: Registry
        +namespace(name) Namespace
        +tcp_server() TcpStreamServer
        +nats_client() nats::Client
        +etcd_client() etcd::Client
    }

    DistributedRuntime *-- Runtime
```

### 3.3 初始化流程

{{< trace >}}
title: 启动时如何接好运行时组件
actors: ["应用进程", "运行时", "发现后端", "目标实例"]
steps:
  - from: 0
    to: 1
    label: "读取配置"
    detail: "应用指定 namespace、发现方式以及请求和事件的传输选择。"
  - from: 1
    to: 2
    label: "接入发现服务"
    detail: "运行时初始化相应后端。例如单机可选文件，裸机亦可使用 etcd。"
  - from: 2
    to: 1
    label: "提供成员集合"
    detail: "已有实例和后续成员变化成为客户端选择目标的输入。"
  - from: 1
    to: 3
    label: "发起首次调用"
    detail: "一个已发现的目标被选中，运行时向其请求端点发送消息。"
{{< /trace >}}

### 3.4 组件注册表 (Registry)

`Registry` 用于管理本地组件，避免重复创建资源：

```mermaid
graph TB
    subgraph registry["Component Registry"]
        REG["Registry"]

        subgraph services["Services"]
            SVC1["Service: dynamo|frontend"]
            SVC2["Service: dynamo|processor"]
        end

        subgraph handlers["Stats Handlers"]
            SH1["Handler Map for frontend"]
            SH2["Handler Map for processor"]
        end
    end

    REG --> SVC1
    REG --> SVC2
    REG --> SH1
    REG --> SH2
```

**Registry 解决的问题**：

1. **Service 复用**：同一个 Component 的多个 Endpoint 共享一个 NATS Service
2. **Stats 聚合**：统一管理每个端点的统计信息处理器
3. **资源控制**：防止创建重复的 Watcher 和 Connection

---

## 4. 生命周期管理

### 4.1 CancellationToken 层级

`CancellationToken` 是 Tokio 提供的取消信号机制，Dynamo 使用层级结构管理：

```mermaid
graph TB
    subgraph tokens["CancellationToken 层级"]
        Root["Root Token<br/>Runtime 级别"]
        Child1["Child Token 1<br/>Component A"]
        Child2["Child Token 2<br/>Component B"]
        GC1["Grandchild Token<br/>Endpoint A1"]
        GC2["Grandchild Token<br/>Endpoint A2"]
        GC3["Grandchild Token<br/>Endpoint B1"]
    end

    Root --> Child1
    Root --> Child2
    Child1 --> GC1
    Child1 --> GC2
    Child2 --> GC3
```

**取消传播规则**：

| 规则 | 说明 |
|------|------|
| 父取消 → 子取消 | 父 Token 取消时，所有子 Token 自动取消 |
| 子取消 ≠ 父取消 | 子 Token 取消时，不影响父 Token 和兄弟 Token |
| 异步等待 | 可以通过 `cancelled()` 异步等待取消信号 |

### 4.2 与 etcd Lease 的绑定

Dynamo 将 `CancellationToken` 与 etcd Lease 绑定：

```mermaid
sequenceDiagram
    participant App as 应用
    participant RT as Runtime
    participant Lease as etcd Lease
    participant etcd as etcd

    App->>RT: shutdown()
    RT->>RT: cancellation_token.cancel()

    Note over RT,Lease: Token 取消传播

    Lease->>Lease: 检测到 cancel
    Lease->>etcd: Revoke Lease

    Note over etcd: Lease 过期<br/>所有绑定的 Key 自动删除
```

### 4.3 Graceful Shutdown 流程

{{< trace >}}
title: 关闭信号如何走到进程退出
actors: ["系统", "服务进程", "运行时", "客户端"]
steps:
  - from: 0
    to: 1
    label: "送出终止信号"
    detail: "进程进入关闭流程。"
  - from: 1
    to: 2
    label: "启动清理"
    detail: "服务停止接收新工作，运行时通知相关任务关闭。"
  - from: 1
    to: 3
    label: "排空已接请求"
    detail: "已经进入处理的请求有机会结束，直到完成或达到配置的关闭期限。"
  - from: 2
    to: 2
    label: "清理成员记录"
    detail: "相应的发现后端处理注销；在 etcd 模式也需处理租约。"
  - from: 1
    to: 0
    label: "退出进程"
    detail: "服务和后台任务收尾后结束进程。"
{{< /trace >}}

---

## 5. Worker 执行框架

### 5.1 Worker 设计

Worker 是运行用户应用的入口点：

```rust
pub struct Worker {
    runtime: Runtime,
}

impl Worker {
    /// 从配置创建 Worker
    pub fn from_settings() -> Result<Worker>;

    /// 执行应用
    pub fn execute<F, Fut>(self, f: F) -> Result<()>
    where
        F: FnOnce(Runtime) -> Fut + Send + 'static,
        Fut: Future<Output = Result<()>> + Send + 'static;
}
```

### 5.2 配置管理

配置优先级（从低到高）：

```mermaid
graph LR
    Default["代码默认值<br/>最低优先级"]
    Defaults["/opt/dynamo/defaults/*.toml"]
    Etc["/opt/dynamo/etc/*.toml"]
    Env["环境变量 DYN_RUNTIME_*<br/>最高优先级"]

    Default --> Defaults --> Etc --> Env --> Final["最终配置"]
```

---

## 下一篇

继续阅读 [04-Component 系统详解](04-component-system.md)，了解 Dynamo 的组件模型。
