---
title: "Dynamo 设计理念"
weight: 3
---

## 把部署需求映射到可观测边界

在 1.6.2，发现后端决定哪些 worker 进入候选，请求平面承载组件调用，事件平面传播异步状态。架构取舍可以先用一次请求解释：模型是否可见、为什么选中该实例、它何时返回第一个 token。 [运行时架构](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md)

## 1. 设计目标

NVIDIA Dynamo 的设计目标是解决传统推理框架的所有挑战，打造一个**生产级分布式 LLM 推理框架**：

```mermaid
graph TB
    subgraph goals["Dynamo 设计目标"]
        G1["高吞吐"] --> GPU["最大化 GPU 利用率"]
        G2["低延迟"] --> TTFT["优化 TTFT"]
        G2 --> ITL["优化 ITL"]
        G3["弹性扩展"] --> SCALE["支持动态扩缩容"]
        G4["引擎无关"] --> ENGINES["支持多种推理后端"]
    end
```

### 怎么衡量这些设计的效果

| 关心的问题 | 用同一条请求轨迹比较 |
|---|---|
| 路由是否省了预填充？ | 各 worker 实际复用的 token、排队时间与客户端首 token 延迟。 |
| 分离后解码是否更平稳？ | 聚合与分离时的逐 token 间隔，分别记录 P/D 排队与 KV 传输时间。 |
| 增加设备后容量如何变化？ | 在同一服务延迟目标下，测每秒能完成多少请求，记录 GPU 数、模型与并行布局。 |

## 2. 技术栈选择

### 2.1 Rust 核心层

Dynamo 的核心运行时使用 **Rust** 实现，原因如下：

#### 性能优势
- **零成本抽象**：高级语法不带来运行时开销
- **无 GC 暂停**：确定性延迟
- **LLVM 优化**：生成高效机器码

#### 内存安全
- **所有权系统**：编译时防止内存错误
- **借用检查器**：防止数据竞争
- **无空指针异常**

#### 并发支持
- `async/await`：原生异步支持
- `Send/Sync` trait：编译时并发安全检查
- Tokio 生态：成熟的异步运行时

#### 与 C++/Go 的对比

| 特性 | Rust | C++ | Go |
|------|------|-----|-----|
| 内存安全 | 编译时保证 | 需要手动管理 | GC |
| 性能 | 最优 | 最优 | 略低 |
| 并发模型 | 所有权 + async | 手动 + 线程 | Goroutine |
| 学习曲线 | 陡峭 | 陡峭 | 平缓 |
| 生态成熟度 | 快速发展 | 最成熟 | 成熟 |

#### Tokio 异步运行时

Dynamo 基于 Tokio 构建异步系统：

```rust
// Dynamo Runtime 的核心结构
pub struct Runtime {
    id: Arc<String>,
    primary: RuntimeType,      // 主任务运行时
    secondary: RuntimeType,    // 后台任务运行时
    cancellation_token: CancellationToken,
}
```

```mermaid
graph TB
    subgraph runtime["Tokio 运行时架构"]
        MAIN["Main Runtime"] --> EXEC["执行器"]
        EXEC --> TASK1["任务 1"]
        EXEC --> TASK2["任务 2"]
        EXEC --> TASKN["任务 N"]

        BG["Background Runtime"] --> BGEXEC["后台执行器"]
        BGEXEC --> TIMER["定时任务"]
        BGEXEC --> IO["I/O 任务"]
    end
```

---

### 2.2 Python 应用层

虽然核心用 Rust 实现，但 Dynamo 提供完整的 Python API：

#### PyO3 绑定机制
- Rust 类直接暴露为 Python 类
- 异步方法支持 Python `async/await`
- 类型安全的参数传递

```python
# Python 中使用 Dynamo
from dynamo.runtime import DistributedRuntime

async def main():
    runtime = await DistributedRuntime.create()
    namespace = runtime.namespace("myapp")
    component = namespace.component("worker")
```

#### 开发效率与生态
- 快速原型开发
- 丰富的 ML/AI 生态
- 与 vLLM、SGLang 等无缝集成

#### 性能关键路径的取舍

```mermaid
graph LR
    subgraph layers["代码分层"]
        PY["Python 层"] --> |PyO3| RUST["Rust 层"]
        PY --> |业务逻辑| CONFIG["配置/编排"]
        RUST --> |性能关键| CORE["核心运行时"]
        RUST --> |性能关键| NET["网络传输"]
        RUST --> |性能关键| ROUTE["路由调度"]
    end
```

---

## 3. 核心设计原则

### 3.1 分层解耦

Dynamo 采用严格的分层架构：

```mermaid
graph TB
    subgraph layers["分层解耦"]
        L1["应用层"] --> |接口| L2["调度层"]
        L2 --> |接口| L3["推理层"]
        L3 --> |接口| L4["运行时层"]
        L4 --> |接口| L5["通信层"]
    end
```

每层：
- 只依赖下层接口
- 不关心下层实现
- 可独立演进

---

### 3.2 事件驱动

系统状态通过事件传播：

{{< trace >}}
title: 路由器怎样得知一个块出现又消失
actors: ["Worker", "事件通道", "路由索引"]
steps:
  - from: 0
    to: 1
    label: "发布块 A"
    detail: "P1 计算出可复用的块 A，向配置的事件通道发布相应状态。"
  - from: 1
    to: 2
    label: "加入 P1"
    detail: "索引更新 A 的成员集合，后续查询可以估计 P1 有一块重叠。"
  - from: 0
    to: 1
    label: "P1 驱逐 A"
    detail: "本地空间被回收，worker 发布驱逐事件。"
  - from: 1
    to: 2
    label: "移除 P1"
    detail: "索引接收事件后更新成员；两个事件之间到达的查询看到不同快照。"
{{< /trace >}}

**优势**：
- **松耦合**：组件通过事件通信
- **可扩展**：新组件只需订阅相关事件
- **异步处理**：事件异步传递不阻塞

---

### 3.3 零拷贝传输

KV Cache 传输使用 RDMA：

```mermaid
graph LR
    subgraph rdma["零拷贝传输"]
        GPU1["GPU 1 HBM"] --> |RDMA| GPU2["GPU 2 HBM"]
    end

    NOTE1["无需 CPU 参与"]
    NOTE2["无需数据拷贝"]
```

**NIXL 传输层**：
- 直接 GPU-to-GPU 传输
- 利用 NVLink/NVSwitch/InfiniBand
- 最小化传输延迟

---

### 3.4 渐进式复杂度

用户可以根据需求选择复杂度：

```mermaid
graph LR
    subgraph progression["渐进式复杂度"]
        A["单机聚合"] --> B["单机 + 路由"]
        B --> C["多机分离"]
        C --> D["多机分离 + 路由"]
    end

    SIMPLE["简单<br/>配置少<br/>性能基准"]
    COMPLEX["复杂<br/>配置多<br/>性能最优"]

    A --> SIMPLE
    D --> COMPLEX
```

| 模式 | 配置复杂度 | 性能 | 适用场景 |
|------|-----------|------|----------|
| 单机聚合 | 低 | 基准 | 开发测试 |
| 单机 + 路由 | 中 | TTFT 优化 | 多轮对话 |
| 多机分离 | 中高 | 吞吐提升 | 高吞吐需求 |
| 多机分离 + 路由 | 高 | 最优 | 生产环境 |

---

## 4. 设计理念总结

```mermaid
graph TB
    subgraph philosophy["Dynamo 设计理念"]
        PERF["性能优先<br/>Rust 核心"] --> GOAL["目标"]
        SAFE["安全可靠<br/>内存安全"] --> GOAL
        FLEX["灵活扩展<br/>多引擎支持"] --> GOAL
        EASY["易用性<br/>Python API"] --> GOAL

        GOAL --> PROD["生产级分布式<br/>LLM 推理框架"]
    end
```

**核心理念**：
1. **性能是第一优先级**：Rust 核心 + 零拷贝传输
2. **安全不妥协**：编译时内存安全 + 类型安全
3. **灵活性为王**：多引擎支持 + 渐进式复杂度
4. **开发者友好**：Python API + 丰富生态

---

## 下一篇

继续阅读 [04-系统架构总览](04-system-architecture.md)，了解 Dynamo 的五层架构和核心概念。
