---
title: "HTTP Server 设计"
weight: 2
---

## 前端准备好与模型准备好是不同状态

1.6.2 的 /health 在没有发现 worker 时也可能返回 200；应查看正文中的 endpoints 和 instances，再核对模型就绪端点和一次实际生成。进入 draining 状态会改变 readiness，已有请求仍可能在返回。 [就绪与排空状态](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/reference/observability/health-checks.mdx)

## 1. 整体架构

### 1.1 架构图

```mermaid
graph TB
    subgraph http["HTTP 服务层"]
        A["HttpService"] --> B["axum::Router"]
        B --> C["/v1/chat/completions"]
        B --> D["/v1/completions"]
        B --> E["/v1/models"]
        B --> F["/metrics"]
    end

    subgraph state["状态管理"]
        G["DeploymentState"] --> H["ChatCompletionEngines"]
        G --> I["CompletionEngines"]
        G --> J["Metrics"]
    end

    subgraph model["模型管理"]
        K["ModelManager"] --> G
        K --> L["add_model"]
        K --> M["remove_model"]
        K --> N["list_models"]
    end

    subgraph engine["引擎层"]
        H --> O["Engine 1"]
        H --> P["Engine 2"]
    end

    C --> G
    D --> G
    E --> G
```

### 1.2 核心组件

| 组件 | 职责 |
|------|------|
| HttpService | HTTP 服务入口，管理路由和生命周期 |
| DeploymentState | 全局状态，存储引擎和指标 |
| ModelManager | 模型管理，增删查模型引擎 |
| ModelEngines | 模型到引擎的映射 |

---

## 2. HttpService 实现

### 2.1 结构定义

```rust
// lib/llm/src/http/service/service_v2.rs

#[derive(Clone)]
pub struct HttpService {
    models: ModelManager,
    router: axum::Router,
    port: u16,
    host: String,
}
```

### 2.2 Builder 模式

```rust
#[derive(Clone, Builder)]
#[builder(pattern = "owned", build_fn(private, name = "build_internal"))]
pub struct HttpServiceConfig {
    #[builder(default = "8787")]
    port: u16,

    #[builder(setter(into), default = "String::from(\"0.0.0.0\")")]
    host: String,

    #[builder(default = "true")]
    enable_chat_endpoints: bool,

    #[builder(default = "true")]
    enable_cmpl_endpoints: bool,
}
```

### 2.3 构建流程

```mermaid
sequenceDiagram
    participant User as 用户代码
    participant Builder as HttpServiceConfigBuilder
    participant MM as ModelManager
    participant Router as axum::Router

    User->>Builder: HttpService::builder()
    User->>Builder: .port(8080)
    User->>Builder: .host("0.0.0.0")
    User->>Builder: .build()

    Builder->>MM: ModelManager::new()
    Builder->>Router: 创建路由

    alt enable_chat_endpoints
        Builder->>Router: 添加 /v1/chat/completions
    end

    alt enable_cmpl_endpoints
        Builder->>Router: 添加 /v1/completions
    end

    Builder->>Router: 添加 /v1/models
    Builder->>Router: 添加 /metrics

    Builder-->>User: HttpService
```

### 2.4 服务启动

```rust
impl HttpService {
    /// 运行 HTTP 服务
    pub async fn run(&self, cancel_token: CancellationToken) -> Result<()> {
        let address = format!("{}:{}", self.host, self.port);
        tracing::info!(address, "Starting HTTP service on: {address}");

        let listener = tokio::net::TcpListener::bind(address.as_str())
            .await
            .unwrap_or_else(|_| panic!("could not bind to address: {address}"));

        let router = self.router.clone();
        let observer = cancel_token.child_token();

        // 使用 Axum 的优雅关闭
        axum::serve(listener, router)
            .with_graceful_shutdown(observer.cancelled_owned())
            .await
            .inspect_err(|_| cancel_token.cancel())?;

        Ok(())
    }

    /// 后台运行服务
    pub async fn spawn(&self, cancel_token: CancellationToken) -> JoinHandle<Result<()>> {
        let this = self.clone();
        tokio::spawn(async move { this.run(cancel_token).await })
    }
}
```

---

## 3. DeploymentState 状态管理

### 3.1 结构定义

```rust
// lib/llm/src/http/service.rs

/// 全局部署状态，跨所有请求共享
pub struct DeploymentState {
    completion_engines: Arc<Mutex<ModelEngines<OpenAICompletionsStreamingEngine>>>,
    chat_completion_engines: Arc<Mutex<ModelEngines<OpenAIChatCompletionsStreamingEngine>>>,
    metrics: Arc<Metrics>,
}
```

### 3.2 引擎获取

```rust
impl DeploymentState {
    fn new() -> Self {
        Self {
            completion_engines: Arc::new(Mutex::new(ModelEngines::default())),
            chat_completion_engines: Arc::new(Mutex::new(ModelEngines::default())),
            metrics: Arc::new(Metrics::default()),
        }
    }

    fn get_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAICompletionsStreamingEngine, ServiceHttpError> {
        self.completion_engines
            .lock()
            .unwrap()
            .get(model)
            .cloned()
            .ok_or(ServiceHttpError::ModelNotFound(model.to_string()))
    }

    fn get_chat_completions_engine(
        &self,
        model: &str,
    ) -> Result<OpenAIChatCompletionsStreamingEngine, ServiceHttpError> {
        self.chat_completion_engines
            .lock()
            .unwrap()
            .get(model)
            .cloned()
            .ok_or(ServiceHttpError::ModelNotFound(model.to_string()))
    }
}
```

### 3.3 状态结构图

```mermaid
classDiagram
    class DeploymentState {
        +completion_engines: Mutex~ModelEngines~
        +chat_completion_engines: Mutex~ModelEngines~
        +metrics: Arc~Metrics~
        +get_completions_engine()
        +get_chat_completions_engine()
    }

    class ModelEngines {
        +default: Option~String~
        +engines: HashMap~String, Engine~
        +add()
        +remove()
        +get()
        +list()
        +contains()
    }

    class Metrics {
        +request_counter: IntCounterVec
        +inflight_gauge: IntGaugeVec
        +request_duration: HistogramVec
    }

    DeploymentState --> ModelEngines
    DeploymentState --> Metrics
```

---

## 4. ModelManager 模型管理

### 4.1 结构定义

```rust
#[derive(Clone)]
pub struct ModelManager {
    state: Arc<DeploymentState>,
}
```

### 4.2 核心方法

```rust
impl ModelManager {
    pub fn new() -> Self {
        let state = Arc::new(DeploymentState::new());
        Self { state }
    }

    /// 检查是否存在指定模型
    pub fn has_model_any(&self, model: &str) -> bool {
        self.state.chat_completion_engines.lock().unwrap().contains(model)
            || self.state.completion_engines.lock().unwrap().contains(model)
    }

    /// 列出所有 Chat Completion 模型
    pub fn list_chat_completions_models(&self) -> Vec<String> {
        self.state.chat_completion_engines.lock().unwrap().list()
    }

    /// 列出所有 Completion 模型
    pub fn list_completions_models(&self) -> Vec<String> {
        self.state.completion_engines.lock().unwrap().list()
    }

    /// 添加 Chat Completion 模型引擎
    pub fn add_chat_completions_model(
        &self,
        model: &str,
        engine: OpenAIChatCompletionsStreamingEngine,
    ) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.chat_completion_engines.lock().unwrap();
        clients.add(model, engine)
    }

    /// 添加 Completion 模型引擎
    pub fn add_completions_model(
        &self,
        model: &str,
        engine: OpenAICompletionsStreamingEngine,
    ) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.completion_engines.lock().unwrap();
        clients.add(model, engine)
    }

    /// 移除模型
    pub fn remove_chat_completions_model(&self, model: &str) -> Result<(), ServiceHttpError> {
        let mut clients = self.state.chat_completion_engines.lock().unwrap();
        clients.remove(model)
    }

    /// 获取指标对象
    pub fn metrics(&self) -> Arc<Metrics> {
        self.state.metrics.clone()
    }

    /// 获取内部状态（用于路由处理器）
    pub fn state(&self) -> Arc<DeploymentState> {
        self.state.clone()
    }
}
```

### 4.3 模型引擎映射

```rust
pub struct ModelEngines<T> {
    default: Option<String>,
    engines: HashMap<String, T>,
}

impl<T: Clone> ModelEngines<T> {
    pub fn add(&mut self, model: &str, engine: T) -> Result<(), ServiceHttpError> {
        if self.engines.contains_key(model) {
            return Err(ServiceHttpError::ModelAlreadyExists(model.to_string()));
        }

        // 第一个添加的模型成为默认模型
        if self.default.is_none() {
            self.default = Some(model.to_string());
        }

        self.engines.insert(model.to_string(), engine);
        Ok(())
    }

    pub fn remove(&mut self, model: &str) -> Result<(), ServiceHttpError> {
        if self.engines.remove(model).is_none() {
            return Err(ServiceHttpError::ModelNotFound(model.to_string()));
        }

        // 如果移除的是默认模型，重新选择
        if self.default.as_deref() == Some(model) {
            self.default = self.engines.keys().next().cloned();
        }

        Ok(())
    }

    pub fn get(&self, model: &str) -> Option<&T> {
        self.engines.get(model)
    }

    pub fn list(&self) -> Vec<String> {
        self.engines.keys().cloned().collect()
    }

    pub fn contains(&self, model: &str) -> bool {
        self.engines.contains_key(model)
    }
}
```

---

## 5. 路由注册

### 5.1 路由结构

```mermaid
graph TB
    subgraph router["axum::Router"]
        R["根路由"]
        R --> C["/v1/chat/completions"]
        R --> CP["/v1/completions"]
        R --> M["/v1/models"]
        R --> ME["/metrics"]
    end

    subgraph handlers["处理器"]
        C --> CH["chat_completions()"]
        CP --> CPH["completions()"]
        M --> MH["list_models()"]
        ME --> MEH["handler_metrics()"]
    end

    subgraph state["共享状态"]
        S["Arc~DeploymentState~"]
    end

    CH --> S
    CPH --> S
    MH --> S
```

### 5.2 路由创建

```rust
impl HttpServiceConfigBuilder {
    pub fn build(self) -> Result<HttpService, anyhow::Error> {
        let config = self.build_internal()?;

        let model_manager = ModelManager::new();

        // 启用 Prometheus 指标
        let registry = metrics::Registry::new();
        model_manager.metrics().register(&registry)?;

        let mut router = axum::Router::new();
        let mut routes = vec![
            metrics::router(registry, None),
            openai::list_models_router(model_manager.state(), None),
        ];

        if config.enable_chat_endpoints {
            routes.push(openai::chat_completions_router(
                model_manager.state(),
                None,
            ));
        }

        if config.enable_cmpl_endpoints {
            routes.push(openai::completions_router(
                model_manager.state(),
                None,
            ));
        }

        for (route_docs, route) in routes.into_iter() {
            router = router.merge(route);
        }

        Ok(HttpService {
            models: model_manager,
            router,
            port: config.port,
            host: config.host,
        })
    }
}
```

---

## 6. Axum 框架特性

### 6.1 为什么选择 Axum

| 特性 | 说明 |
|------|------|
| Tokio 生态 | 与 Dynamo 运行时无缝集成 |
| 类型安全 | 编译时路由检查 |
| 高性能 | 零成本抽象 |
| 模块化 | 易于组合和扩展 |

### 6.2 状态共享模式

```rust
// 使用 State 提取器共享状态
async fn chat_completions(
    State(state): State<Arc<DeploymentState>>,
    Json(request): Json<NvCreateChatCompletionRequest>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    // state 在所有请求间共享
    let engine = state.get_chat_completions_engine(&request.inner.model)?;
    // ...
}
```

### 6.3 优雅关闭

```rust
axum::serve(listener, router)
    .with_graceful_shutdown(cancel_token.cancelled_owned())
    .await?;
```

关闭流程：

{{< trace >}}
title: Frontend 如何排空一个已接受的请求
actors: ["系统", "Frontend", "请求 R7", "就绪探针"]
steps:
  - from: 0
    to: 1
    label: "收到关闭信号"
    detail: "前端进入 draining 阶段。"
  - from: 1
    to: 3
    label: "返回未就绪"
    detail: "readiness 转为未就绪，负载均衡器可据此移除新流量。"
  - from: 1
    to: 2
    label: "等待已有请求"
    detail: "已接纳的 R7 在关闭期限内继续生成。"
  - from: 2
    to: 1
    label: "返回末项"
    detail: "R7 完成，待排空的活动请求减少。"
  - from: 1
    to: 0
    label: "退出服务"
    detail: "排空结束或达到期限后，进程继续清理并关闭。"
{{< /trace >}}

---

## 7. 动态模型发现

### 7.1 etcd Watch 机制

以下为配置 etcd 时的订阅示意。HTTP 服务接收的成员信息由当前发现后端提供：

```rust
// components/src/dynamo/frontend/main.py

async fn app(runtime: Runtime) -> Result<()> {
    let http_service = HttpService::builder()
        .port(args.port)
        .host(args.host)
        .build()?;
    let manager = http_service.model_manager().clone();

    // Watch 模型注册
    let etcd_path = format!("{}/models/chat/", etcd_root);
    let models_watcher = etcd_client
        .kv_get_and_watch_prefix(etcd_path)
        .await?;

    tokio::spawn(model_watcher(manager, models_watcher));

    http_service.run(cancel_token).await
}
```

### 7.2 模型发现流程

```mermaid
sequenceDiagram
    participant HTTP as HTTP Service
    participant etcd as etcd
    participant Worker as LLM Worker

    Note over HTTP,etcd: 初始化阶段
    HTTP->>etcd: Watch /models/chat/

    Note over Worker,etcd: Worker 注册
    Worker->>etcd: PUT /models/chat/llama-3-70b

    etcd-->>HTTP: Watch Event (PUT)
    HTTP->>HTTP: add_chat_completions_model()

    Note over HTTP: 模型可用

    Note over Worker,etcd: Worker 下线
    Worker->>etcd: DELETE /models/chat/llama-3-70b

    etcd-->>HTTP: Watch Event (DELETE)
    HTTP->>HTTP: remove_chat_completions_model()
```

---

## 8. 使用示例

### 8.1 创建 HTTP 服务

```rust
let http_service = HttpService::builder()
    .port(8080)
    .host("0.0.0.0")
    .enable_chat_endpoints(true)
    .enable_cmpl_endpoints(false)
    .build()?;
```

### 8.2 添加模型

```rust
let manager = http_service.model_manager();

// 添加 Chat Completion 模型
manager.add_chat_completions_model(
    "llama-3-70b",
    chat_engine,
)?;

// 添加 Completion 模型
manager.add_completions_model(
    "codellama-34b",
    completion_engine,
)?;
```

### 8.3 启动服务

```rust
let cancel_token = CancellationToken::new();

// 方式 1: 阻塞运行
http_service.run(cancel_token.clone()).await?;

// 方式 2: 后台运行
let handle = http_service.spawn(cancel_token.clone()).await;

// 关闭服务
cancel_token.cancel();
handle.await??;
```

---

## 参考文件

- [lib/llm/src/http/service/service_v2.rs](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/http/service/service_v2.rs) - HttpService 实现
- [lib/llm/src/http/service.rs](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/http/service.rs) - 服务模块定义
- [前端进程入口](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/components/src/dynamo/frontend/main.py) - HTTP 服务入口
