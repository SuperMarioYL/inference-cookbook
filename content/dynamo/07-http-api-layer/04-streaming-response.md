---
title: "流式响应处理"
weight: 4
---

## 从网络事件还原模型输出

在同一请求上分别记录开始响应、首个包含文本的增量以及终止事件。客户端断开后，前端需要把取消传播到下游；这与完整流按 finish reason 正常结束是不同路径。 [取消传播](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/fault-tolerance/request-cancellation-architecture.md)

## 1. SSE 基础

### 1.1 Server-Sent Events 协议

SSE 是 HTML5 规范的一部分，用于服务端向客户端推送事件：

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache
Connection: keep-alive

data: {"content": "Hello"}

data: {"content": " World"}

data: [DONE]
```

### 1.2 事件格式

| 字段 | 说明 | 示例 |
|------|------|------|
| `data:` | 数据内容 | `data: {"text": "hi"}` |
| `event:` | 事件类型 | `event: message` |
| `id:` | 事件 ID | `id: 123` |
| `retry:` | 重连间隔 | `retry: 3000` |
| `:` | 注释 | `: keep-alive` |

---

## 2. 事件转换

### 2.1 EventConverter

```rust
struct EventConverter<T>(Annotated<T>);

impl<T> From<Annotated<T>> for EventConverter<T> {
    fn from(annotated: Annotated<T>) -> Self {
        EventConverter(annotated)
    }
}

/// 将 Annotated 转换为 SSE Event
impl<T: Serialize> TryFrom<EventConverter<T>> for Event {
    type Error = axum::Error;

    fn try_from(annotated: EventConverter<T>) -> Result<Self, Self::Error> {
        let annotated = annotated.0;
        let mut event = Event::default();

        // 设置 data 字段
        if let Some(data) = annotated.data {
            event = event.json_data(data)?;
        }

        // 设置 event 字段（错误处理）
        if let Some(msg) = annotated.event {
            if msg == "error" {
                let msgs = annotated
                    .comment
                    .unwrap_or_else(|| vec!["unspecified error".to_string()]);
                return Err(axum::Error::new(msgs.join(" -- ")));
            }
            event = event.event(msg);
        }

        // 设置注释
        if let Some(comments) = annotated.comment {
            for comment in comments {
                event = event.comment(comment);
            }
        }

        Ok(event)
    }
}
```

### 2.2 Annotated 结构

```rust
/// 带注解的响应
pub struct Annotated<T> {
    pub data: Option<T>,        // 数据内容
    pub event: Option<String>,  // 事件类型
    pub comment: Option<Vec<String>>,  // 注释
}
```

### 2.3 转换流程

```mermaid
graph LR
    subgraph input["输入"]
        A["Annotated~T~"]
    end

    subgraph convert["EventConverter"]
        B["提取 data"]
        C["提取 event"]
        D["提取 comment"]
    end

    subgraph output["输出"]
        E["SSE Event"]
    end

    A --> B
    A --> C
    A --> D
    B --> E
    C --> E
    D --> E
```

---

## 3. 断开检测与取消

### 3.1 检测机制

当客户端断开连接时，需要：
1. 检测断开事件
2. 通知后端停止生成
3. 释放计算资源

### 3.2 实现代码

```rust
/// 监控客户端断开并传播取消信号
async fn monitor_for_disconnects(
    stream: Pin<Box<dyn Stream<Item = Result<Event, axum::Error>> + Send>>,
    context: Arc<dyn AsyncEngineContext>,
    inflight: InflightGuard,
) -> ReceiverStream<Result<Event, axum::Error>> {
    let (tx, rx) = tokio::sync::mpsc::channel(8);

    tokio::spawn(async move {
        let mut inflight = inflight;
        let mut stream = stream;

        while let Some(event) = stream.next().await {
            let event = match event {
                Ok(event) => Ok(event),
                Err(err) => Ok(Event::default()
                    .event("error")
                    .comment(err.to_string())),
            };

            // 如果发送失败，说明客户端已断开
            if (tx.send(event).await).is_err() {
                tracing::trace!("Forwarding SSE stream was dropped; breaking loop");
                // 通知后端停止生成
                context.stop_generating();
                break;
            }
        }

        // 流正常完成，发送 [DONE] 并标记成功
        if tx.send(Ok(Event::default().data("[DONE]"))).await.is_ok() {
            inflight.mark_ok();
        }
    });

    ReceiverStream::new(rx)
}
```

### 3.3 断开处理流程

```mermaid
flowchart TD
    A["开始流式响应"] --> B{"客户端连接?"}
    B -->|是| C["发送 Token"]
    C --> D{"发送成功?"}
    D -->|是| E{"还有更多 Token?"}
    E -->|是| B
    E -->|否| F["发送 DONE"]
    F --> G["标记成功"]
    G --> H["结束"]

    D -->|否| I["检测到断开"]
    I --> J["调用 stop_generating()"]
    J --> K["释放资源"]
    K --> H

    B -->|否| I
```

### 3.4 AsyncEngineContext 接口

```rust
/// 异步引擎上下文接口
pub trait AsyncEngineContext: Send + Sync {
    /// 停止生成
    fn stop_generating(&self);

    /// 检查是否已停止
    fn is_stopped(&self) -> bool;
}
```

---

## 4. Keep-Alive 机制

### 4.1 为什么需要 Keep-Alive

长时间生成过程中，需要防止连接超时：

| 问题 | 解决方案 |
|------|----------|
| 代理超时 | 定期发送心跳 |
| 连接断开 | Keep-Alive 包 |
| 客户端等待 | 减少等待焦虑 |

### 4.2 Axum SSE Keep-Alive

```rust
// 使用 Axum 的 SSE Keep-Alive
Ok(Sse::new(stream)
    .keep_alive(KeepAlive::default())
    .into_response())
```

Keep-Alive 默认配置：

| 参数 | 默认值 |
|------|--------|
| 间隔 | 15 秒 |
| 内容 | 空注释 (`:`) |

### 4.3 Keep-Alive 格式

```
: keep-alive

data: {"content": "token1"}

: keep-alive

data: {"content": "token2"}
```

---

## 5. 流式响应管道

### 5.1 完整管道

```mermaid
graph LR
    subgraph engine["引擎层"]
        A["LLM Engine"]
        B["Token Stream"]
    end

    subgraph convert["转换层"]
        C["Annotated"]
        D["EventConverter"]
        E["SSE Event"]
    end

    subgraph monitor["监控层"]
        F["monitor_for_disconnects"]
        G["mpsc Channel"]
    end

    subgraph output["输出层"]
        H["ReceiverStream"]
        I["Sse~Stream~"]
    end

    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F --> G
    G --> H
    H --> I
```

### 5.2 代码组装

```rust
async fn chat_completions(...) -> Result<Response, ...> {
    // 1. 调用引擎获取流
    let stream = engine.generate(request).await?;

    // 2. 获取上下文用于取消
    let ctx = stream.context();

    // 3. 转换为 SSE 事件流
    let sse_stream = stream.map(|response| {
        Event::try_from(EventConverter::from(response))
    });

    // 4. 添加断开检测
    let monitored = monitor_for_disconnects(
        sse_stream.boxed(),
        ctx,
        inflight
    ).await;

    // 5. 返回 SSE 响应
    Ok(Sse::new(monitored)
        .keep_alive(KeepAlive::default())
        .into_response())
}
```

---

## 6. 错误处理

### 6.1 流中的错误

```rust
impl<T: Serialize> TryFrom<EventConverter<T>> for Event {
    fn try_from(annotated: EventConverter<T>) -> Result<Self, Self::Error> {
        // 检查是否是错误事件
        if let Some(msg) = annotated.event {
            if msg == "error" {
                let msgs = annotated.comment
                    .unwrap_or_else(|| vec!["unspecified error".to_string()]);
                return Err(axum::Error::new(msgs.join(" -- ")));
            }
        }
        // ...
    }
}
```

### 6.2 错误转换

```rust
// monitor_for_disconnects 中的错误处理
while let Some(event) = stream.next().await {
    let event = match event {
        Ok(event) => Ok(event),
        // 将错误转换为 SSE 错误事件
        Err(err) => Ok(Event::default()
            .event("error")
            .comment(err.to_string())),
    };
    // ...
}
```

### 6.3 错误事件格式

```
event: error
: Internal server error: model unavailable

data: {"error": "model unavailable"}
```

---

## 7. 性能优化

### 7.1 缓冲策略

```rust
// 使用有界 channel 防止背压
let (tx, rx) = tokio::sync::mpsc::channel(8);
```

| 参数 | 值 | 说明 |
|------|-----|------|
| 缓冲大小 | 8 | 平衡内存和延迟 |

### 7.2 背压处理

{{< stateboard >}}
title: 慢客户端如何让有界队列填满
rows: ["缓冲槽", "发送端"]
steps:
  - title: "收到 A"
    detail: "示例队列能容纳两项，A 已就绪，客户端还没读取。"
    cells: [["A", "·"], ["可写", "·"]]
    focus: [[0, 0]]
  - title: "再到 B"
    detail: "现在两个槽都满，再写 C 时发送端需要等待消费。"
    cells: [["A", "B"], ["等待", "C"]]
    focus: [[0, 1], [1, 0]]
  - title: "客户端读 A"
    detail: "腾出的槽允许 C 入队，原来的 B 仍排在前面。"
    cells: [["B", "C"], ["可写", "·"]]
    focus: [[0, 0], [0, 1]]
  - title: "客户端继续读"
    detail: "B、C 依序交给消费者，缓冲重新空出。"
    cells: [["·", "·"], ["可写", "·"]]
    focus: [[0, 0], [0, 1]]
{{< /stateboard >}}

### 7.3 零拷贝序列化

```rust
// 使用 json_data 直接序列化到输出
event = event.json_data(data)?;
```

---

## 8. 客户端使用

### 8.1 JavaScript（EventSource）

```javascript
const eventSource = new EventSource('/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        model: 'llama-3-70b',
        messages: [{role: 'user', content: 'Hello'}],
        stream: true
    })
});

eventSource.onmessage = (event) => {
    if (event.data === '[DONE]') {
        eventSource.close();
        return;
    }
    const data = JSON.parse(event.data);
    console.log(data.choices[0].delta.content);
};

eventSource.onerror = (error) => {
    console.error('Stream error:', error);
    eventSource.close();
};
```

### 8.2 Python（requests）

```python
import requests
import json

response = requests.post(
    'http://localhost:8080/v1/chat/completions',
    json={
        'model': 'llama-3-70b',
        'messages': [{'role': 'user', 'content': 'Hello'}],
        'stream': True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        line = line.decode('utf-8')
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            content = chunk['choices'][0]['delta'].get('content', '')
            print(content, end='', flush=True)
```

---

## 参考文件

- [lib/llm/src/http/service/openai.rs](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/http/service/openai.rs) - SSE 实现
- [断开和取消处理](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/llm/src/http/service/disconnect.rs) - SSE 工具
