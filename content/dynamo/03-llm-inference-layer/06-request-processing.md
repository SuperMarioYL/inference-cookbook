---
title: "请求处理流程"
weight: 6
---

## 追踪 HTTP 请求而非固定进程数

从一次短请求开始，关联前端模板与分词、选择 worker、后端执行、输出解码和客户端接收的时间。不同后端与输入形式下预处理所在位置可能变化，客户端的请求标识和返回格式仍是观察入口。 [请求处理架构](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/developer-guide/knowledge-base/concepts/system-architecture/architecture.md)

## 1. 处理流程概览

```mermaid
graph TB
    subgraph flow["请求处理流程"]
        HTTP["1. HTTP Server"]
        PREPROC["2. Preprocessor"]
        ROUTER["3. KV-Aware Router"]
        BACKEND["4. Backend"]
        ENGINE["5. LLM Engine"]
        DECODER["6. Decoder"]
        RESPONSE["7. HTTP Response"]
    end

    HTTP --> PREPROC --> ROUTER --> BACKEND --> ENGINE --> DECODER --> RESPONSE
```

---

## 2. Tokenization

### 2.1 Tokenizer 接口

```rust
pub trait Tokenizer: Send + Sync {
    fn encode(&self, text: &str) -> Result<Vec<u32>>;
    fn decode(&self, ids: &[u32]) -> Result<String>;
    fn vocab_size(&self) -> usize;
    fn eos_token_id(&self) -> Option<u32>;
    fn bos_token_id(&self) -> Option<u32>;
}
```

### 2.2 增量解码

```rust
pub struct DecodeStream {
    tokenizer: Arc<dyn Tokenizer>,
    ids: Vec<u32>,      // 累积的 token IDs
    prefix: String,     // 已解码的文本前缀
}

impl DecodeStream {
    /// 添加新 token 并返回增量文本
    pub fn add_tokens(&mut self, new_ids: &[u32]) -> Result<String> {
        self.ids.extend(new_ids);

        // 解码完整序列
        let full_text = self.tokenizer.decode(&self.ids)?;

        // 返回增量部分
        let delta = &full_text[self.prefix.len()..];
        self.prefix = full_text;

        Ok(delta.to_string())
    }
}
```

### 2.3 增量解码流程

{{< trace >}}
title: 增量解码只发新增文字
actors: ["引擎", "解码状态", "客户端"]
steps:
  - from: 0
    to: 1
    label: "第一个片段"
    detail: "收到第一个输出 ID，对应可显示的 Hello。"
  - from: 1
    to: 2
    label: "发送 Hello"
    detail: "记录已经送出的文本边界。"
  - from: 0
    to: 1
    label: "第二个片段"
    detail: "新 ID 解码后，完整可见文本变为 Hello world。"
  - from: 1
    to: 2
    label: "发送空格与 world"
    detail: "这次只发上次边界之后的部分，客户端拼接得到 Hello world。"
{{< /trace >}}

---

## 3. Preprocessor

### 3.1 处理流程

```mermaid
flowchart LR
    subgraph preproc["Preprocessor"]
        CHAT["Chat Template<br/>渲染"]
        TOKEN["Tokenizer<br/>编码"]
        SAMPLE["SamplingOptions<br/>提取"]
        STOP["StopConditions<br/>提取"]
    end

    INPUT["OpenAI Request"] --> CHAT --> TOKEN --> SAMPLE --> STOP --> OUTPUT["BackendInput"]
```

### 3.2 Chat Template 渲染

```python
# 输入消息
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# 渲染后的 prompt (Llama 3 格式)
prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello!<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
```

### 3.3 BackendInput 结构

```rust
pub struct BackendInput {
    pub token_ids: Vec<u32>,
    pub sampling_options: SamplingOptions,
    pub stop_conditions: StopConditions,
    pub mdcsum: String,  // 模型卡校验和
}
```

---

## 4. 采样参数

### 4.1 SamplingOptions

```rust
pub struct SamplingOptions {
    pub temperature: f32,        // 温度 (0.0-2.0)
    pub top_p: f32,              // Top-P 采样
    pub top_k: Option<u32>,      // Top-K 采样
    pub max_tokens: u32,         // 最大生成长度
    pub min_tokens: Option<u32>, // 最小生成长度
    pub repetition_penalty: Option<f32>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}
```

### 4.2 采样流程

```mermaid
graph LR
    LOGITS["Logits"] --> TEMP["Temperature<br/>缩放"]
    TEMP --> TOPK["Top-K<br/>过滤"]
    TOPK --> TOPP["Top-P<br/>过滤"]
    TOPP --> SAMPLE["采样"]
    SAMPLE --> TOKEN["输出 Token"]
```

---

## 5. 停止条件检测

### 5.1 StopConditions

```rust
pub struct StopConditions {
    pub stop_sequences: Vec<String>,
    pub include_stop_str_in_output: bool,
}
```

### 5.2 检测逻辑

```rust
impl Decoder {
    pub fn should_stop(&self, token_ids: &[u32], stop_sequences: &[String]) -> bool {
        // 1. 检查 EOS token
        if let Some(&last_token) = token_ids.last() {
            if self.eos_token_ids.contains(&last_token) {
                return true;
            }
        }

        // 2. 检查自定义停止序列
        for stop_seq in stop_sequences {
            if self.tok_trie.matches_suffix(token_ids, stop_seq) {
                return true;
            }
        }

        false
    }
}
```

### 5.3 FinishReason

```rust
pub enum FinishReason {
    EoS,            // 遇到 EOS token
    Length,         // 达到最大长度
    Stop,           // 匹配停止序列
    Error(String),  // 发生错误
    Cancelled,      // 被取消
}
```

---

## 6. 完整请求处理流程

```mermaid
sequenceDiagram
    participant Client
    participant HTTP as HTTP Server
    participant Preproc as Preprocessor
    participant Router as KV Router
    participant Backend
    participant Engine as LLM Engine
    participant Decoder

    Client->>HTTP: POST /v1/chat/completions
    HTTP->>Preproc: 解析请求

    Note over Preproc: 1. Chat Template 渲染
    Note over Preproc: 2. Tokenization
    Note over Preproc: 3. 提取采样参数

    Preproc->>Router: BackendInput

    Note over Router: 查询 KV Index
    Note over Router: 计算匹配分数
    Note over Router: 选择最优 Worker

    Router->>Backend: 路由请求
    Backend->>Engine: 提交推理

    loop Token 生成
        Engine-->>Backend: LLMEngineOutput
        Backend->>Decoder: 停止条件检测
        Decoder->>Decoder: 增量解码
        Backend-->>HTTP: BackendOutput
        HTTP-->>Client: SSE Event
    end

    Engine-->>Backend: finish_reason
    Backend-->>HTTP: [DONE]
    HTTP-->>Client: 完成
```

---

## 7. HTTP 响应格式

### 7.1 SSE 流式响应

```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"llama","choices":[{"delta":{"content":"Hi"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"llama","choices":[{"delta":{"content":"!"}}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","model":"llama","choices":[{"finish_reason":"stop"}]}

data: [DONE]
```

### 7.2 非流式响应

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "model": "llama",
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "Hi!"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 2,
    "total_tokens": 12
  }
}
```

---

## 8. 错误处理

### 8.1 错误类型

| 错误类型 | HTTP 状态码 | 说明 |
|---------|-------------|------|
| 模型不存在 | 404 | 请求的模型未部署 |
| 请求格式错误 | 400 | JSON 解析失败 |
| 超时 | 504 | 推理超时 |
| 引擎错误 | 500 | 推理引擎内部错误 |
| 取消 | 499 | 客户端断开连接 |

### 8.2 错误响应格式

```json
{
  "error": {
    "type": "invalid_request_error",
    "message": "Model 'gpt-5' not found",
    "code": "model_not_found"
  }
}
```

---

## 下一章

完成本章阅读后，建议继续阅读 [第四章：KV Cache 管理与路由](../04-kv-cache-system/README.md)，了解 Dynamo 的 KV Cache 优化技术。
