---
title: "请求生命周期"
weight: 6
---

# 请求完整生命周期

本章将完整跟踪一个请求从用户提交到最终返回的全过程，将前面章节的知识串联起来，帮助读者建立完整的认知图景。

---

## 1. 生命周期概览

```mermaid
graph TD
    subgraph phase1["1. 提交阶段"]
        A1[用户调用 generate]
        A2[Tokenize]
        A3[创建请求]
        A4[加入 waiting 队列]
    end

    subgraph phase2["2. 调度阶段"]
        B1[查找前缀缓存]
        B2[分配 KV Cache]
        B3[加入 running 队列]
    end

    subgraph phase3["3. 执行阶段"]
        C1[准备输入]
        C2[模型前向传播]
        C3[采样]
    end

    subgraph phase4["4. 更新阶段"]
        D1[追加 token]
        D2[检查停止条件]
        D3[更新状态]
    end

    subgraph phase5["5. 返回阶段"]
        E1[Detokenize]
        E2[构建输出]
        E3[返回用户]
    end

    A1 --> A2 --> A3 --> A4
    A4 --> B1 --> B2 --> B3
    B3 --> C1 --> C2 --> C3
    C3 --> D1 --> D2 --> D3
    D3 -->|未完成| C1
    D3 -->|完成| E1 --> E2 --> E3
```

---

## 2. 阶段 1：请求提交

### 2.1 用户调用

```python
# 用户代码
from vllm import LLM, SamplingParams

llm = LLM(model="meta-llama/Llama-2-7b-hf")

prompts = ["The capital of France is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=50)

outputs = llm.generate(prompts, sampling_params)
```

### 2.2 Tokenize

```python
# vllm/entrypoints/llm.py

def generate(self, prompts, sampling_params, ...):
    # 1. 处理输入
    for prompt in prompts:
        # Tokenize prompt
        prompt_token_ids = self.tokenizer.encode(prompt)

        # 创建请求
        request_id = str(next(self.request_counter))

        self._add_request(
            request_id=request_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            params=sampling_params,
        )
```

### 2.3 创建 EngineCoreRequest

```python
# vllm/v1/engine/llm_engine.py

def add_request(self, request_id, prompt, params, ...):
    # 构建 EngineCoreRequest
    engine_request = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=params,
        arrival_time=time.time(),
        eos_token_id=self.tokenizer.eos_token_id,
    )

    # 发送到 EngineCore
    self.engine_core.add_request(engine_request)
```

### 2.4 加入 Waiting 队列

```python
# vllm/v1/core/sched/scheduler.py

def add_request(self, request: EngineCoreRequest) -> None:
    # 1. 创建内部 Request 对象
    internal_request = Request(
        request_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        sampling_params=request.sampling_params,
    )

    # 2. 计算 block hashes（用于前缀缓存）
    if self.enable_caching:
        internal_request.block_hashes = compute_block_hashes(
            internal_request.prompt_token_ids,
            self.block_size,
        )

    # 3. 加入 waiting 队列
    internal_request.status = RequestStatus.WAITING
    self.waiting.append_request(internal_request)

    # 4. 记录到请求字典
    self.requests[request.request_id] = internal_request
```

### 2.5 提交阶段时序图

```mermaid
sequenceDiagram
    participant User as 用户
    participant LLM as LLM
    participant Tokenizer as Tokenizer
    participant Engine as LLMEngine
    participant Core as EngineCore
    participant Sched as Scheduler

    User->>LLM: generate(prompts, params)
    LLM->>Tokenizer: encode(prompt)
    Tokenizer-->>LLM: token_ids

    LLM->>Engine: add_request(id, tokens, params)
    Engine->>Engine: 创建 EngineCoreRequest
    Engine->>Core: add_request(request)

    Core->>Sched: add_request(request)
    Sched->>Sched: 创建 internal Request
    Sched->>Sched: 计算 block_hashes
    Sched->>Sched: waiting.append(request)

    Note over Sched: 请求进入 WAITING 状态
```

---

## 3. 阶段 2：调度

### 3.1 查找前缀缓存

```python
# vllm/v1/core/sched/scheduler.py :: schedule()

# 从 waiting 队列取出请求
request = self.waiting.peek_request()

# 查找前缀缓存
new_computed_blocks, num_cached_tokens = (
    self.kv_cache_manager.get_computed_blocks(request)
)

# num_cached_tokens 表示可以跳过的 token 数
# 例如：prompt 有 100 tokens，前 64 个已缓存
# 则只需要计算后 36 个
```

### 3.2 分配 KV Cache

```python
# 计算需要处理的 token 数
num_new_tokens = request.num_tokens - num_cached_tokens

# 分配 KV Cache slots
new_blocks = self.kv_cache_manager.allocate_slots(
    request,
    num_new_tokens,
    num_new_computed_tokens=num_cached_tokens,
    new_computed_blocks=new_computed_blocks,
)

if new_blocks is None:
    # 内存不足，请求继续等待
    return

# 分配成功
```

### 3.3 移入 Running 队列

```python
# 从 waiting 移除
request = self.waiting.pop_request()

# 加入 running
self.running.append(request)

# 更新状态
request.status = RequestStatus.RUNNING
request.num_computed_tokens = num_cached_tokens
```

### 3.4 调度阶段示意图

```mermaid
flowchart TD
    subgraph Scheduler.schedule
        W[waiting 队列] --> Peek[peek_request]
        Peek --> Cache[get_computed_blocks]
        Cache --> Alloc[allocate_slots]

        Alloc --> Check{分配成功?}
        Check -->|是| Move[移入 running]
        Check -->|否| Wait[继续等待]

        Move --> SO[构建 SchedulerOutput]
    end

    subgraph SchedulerOutput
        SO --> Reqs[scheduled_new_reqs]
        SO --> Blocks[req_to_new_blocks]
        SO --> Tokens[num_scheduled_tokens]
    end
```

---

## 4. 阶段 3：模型执行

### 4.1 准备输入

```python
# vllm/v1/worker/gpu_model_runner.py

def execute_model(self, scheduler_output: SchedulerOutput):
    # 1. 准备 input_ids
    input_ids = self._prepare_input_ids(scheduler_output)

    # 2. 准备 positions
    positions = self._prepare_positions(scheduler_output)

    # 3. 准备 attention metadata
    attn_metadata = self._prepare_attention_metadata(scheduler_output)

    # 4. 更新 block table
    self._update_block_table(scheduler_output)
```

### 4.2 模型前向传播

```python
    # 5. 前向传播
    with torch.inference_mode():
        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            kv_caches=self.kv_caches,
            attn_metadata=attn_metadata,
        )

    # 6. 计算 logits
    logits = self.model.compute_logits(hidden_states)

    return ModelRunnerOutput(logits=logits, ...)
```

### 4.3 采样

```python
# vllm/v1/executor/abstract.py

def sample_tokens(self, model_output: ModelRunnerOutput) -> SamplerOutput:
    # 构建采样元数据
    sampling_metadata = self._prepare_sampling_metadata()

    # 采样
    sampler_output = self.sampler(
        model_output.logits,
        sampling_metadata,
    )

    return sampler_output
```

### 4.4 执行阶段时序图

```mermaid
sequenceDiagram
    participant Core as EngineCore
    participant Exec as Executor
    participant Worker as Worker
    participant Runner as ModelRunner
    participant Model as Model
    participant Sampler as Sampler

    Core->>Exec: execute_model(scheduler_output)
    Exec->>Worker: execute_model()
    Worker->>Runner: execute_model()

    Runner->>Runner: _prepare_inputs()
    Runner->>Model: forward(input_ids, positions, kv_caches)

    Note over Model: Embedding → Transformer Layers → Norm

    Model-->>Runner: hidden_states
    Runner->>Model: compute_logits(hidden_states)
    Model-->>Runner: logits

    Runner-->>Worker: ModelRunnerOutput
    Worker-->>Exec: output

    Core->>Exec: sample_tokens()
    Exec->>Sampler: forward(logits, metadata)

    Note over Sampler: Temperature → Top-k/p → Sample

    Sampler-->>Exec: SamplerOutput
    Exec-->>Core: sampled_tokens
```

---

## 5. 阶段 4：状态更新

### 5.1 追加 Token

```python
# vllm/v1/core/sched/scheduler.py

def update_from_output(self, model_output, sampler_output, scheduler_output):
    for req_id, output in sampler_output.items():
        request = self.requests[req_id]

        # 获取新生成的 token
        new_token_ids = output.sampled_token_ids.tolist()

        # 追加到请求
        request.append_output_token_ids(new_token_ids)

        # 更新 computed_tokens
        request.num_computed_tokens += 1
```

### 5.2 检查停止条件

```python
        # 检查是否完成
        finish_reason, stop_str = check_stop(request, self.max_model_len)

        if finish_reason is not None:
            # 请求完成
            self._finish_request(request, finish_reason)
            finished_outputs.append(...)
        else:
            # 继续生成
            outputs.append(...)
```

### 5.3 完成请求处理

```python
def _finish_request(self, request: Request, reason: FinishReason):
    # 1. 释放 KV Cache
    self.kv_cache_manager.free(request)

    # 2. 从 running 移除
    self.running.remove(request)

    # 3. 更新状态
    request.status = RequestStatus.FINISHED

    # 4. 记录完成
    self.finished_req_ids.add(request.request_id)
```

---

## 6. 阶段 5：返回结果

### 6.1 Detokenize

```python
# vllm/v1/engine/llm_engine.py

def _process_outputs(self, engine_outputs: EngineCoreOutputs):
    results = []

    for output in engine_outputs.outputs:
        request = self.requests[output.request_id]

        # 增量解码
        new_text = self.detokenizer.decode(
            request,
            output.new_token_ids,
        )

        # 更新请求的输出文本
        request.output_text += new_text

        results.append(...)

    return results
```

### 6.2 构建 RequestOutput

```python
def _make_request_output(self, request: Request, finished: bool):
    return RequestOutput(
        request_id=request.request_id,
        prompt=request.prompt,
        prompt_token_ids=request.prompt_token_ids,
        outputs=[
            CompletionOutput(
                index=0,
                text=request.output_text,
                token_ids=request.output_token_ids,
                finish_reason=request.finish_reason,
                logprobs=request.logprobs,
            )
        ],
        finished=finished,
    )
```

### 6.3 返回用户

```python
# vllm/entrypoints/llm.py

def _run_engine(self, use_tqdm: bool):
    outputs = []

    while self.llm_engine.has_unfinished_requests():
        step_outputs = self.llm_engine.step()

        for output in step_outputs:
            if output.finished:
                outputs.append(output)

    return sorted(outputs, key=lambda x: int(x.request_id))
```

---

## 7. 完整生命周期时序图

```mermaid
sequenceDiagram
    participant User as 用户
    participant LLM as LLM
    participant Engine as LLMEngine
    participant Core as EngineCore
    participant Sched as Scheduler
    participant KVM as KVCacheManager
    participant Exec as Executor
    participant Model as Model

    rect rgb(230, 245, 230)
        Note over User,Model: 1. 提交阶段
        User->>LLM: generate(prompt, params)
        LLM->>Engine: add_request()
        Engine->>Core: add_request()
        Core->>Sched: add_request()
        Note over Sched: status = WAITING
    end

    loop 每个 step
        rect rgb(255, 245, 230)
            Note over User,Model: 2. 调度阶段
            Core->>Sched: schedule()
            Sched->>KVM: get_computed_blocks()
            KVM-->>Sched: cached_blocks, num_cached

            Sched->>KVM: allocate_slots()
            KVM-->>Sched: new_blocks

            Note over Sched: status = RUNNING
            Sched-->>Core: SchedulerOutput
        end

        rect rgb(245, 230, 230)
            Note over User,Model: 3. 执行阶段
            Core->>Exec: execute_model()
            Exec->>Model: forward()
            Model-->>Exec: logits

            Exec->>Exec: sample()
            Exec-->>Core: SamplerOutput
        end

        rect rgb(230, 230, 245)
            Note over User,Model: 4. 更新阶段
            Core->>Sched: update_from_output()
            Sched->>Sched: append_token()
            Sched->>Sched: check_stop()

            alt 完成
                Sched->>KVM: free()
                Note over Sched: status = FINISHED
            end
        end
    end

    rect rgb(245, 245, 230)
        Note over User,Model: 5. 返回阶段
        Core-->>Engine: EngineCoreOutputs
        Engine->>Engine: detokenize()
        Engine-->>LLM: RequestOutput
        LLM-->>User: outputs
    end
```

---

## 8. 状态转换汇总

```mermaid
stateDiagram-v2
    [*] --> WAITING: add_request()

    WAITING --> RUNNING: schedule() 成功
    WAITING --> WAITING_FOR_FSM: 需要 FSM 编译
    WAITING --> WAITING_FOR_REMOTE_KVS: 等待远程 KV

    WAITING_FOR_FSM --> WAITING: FSM 就绪
    WAITING_FOR_REMOTE_KVS --> WAITING: KV 就绪

    RUNNING --> RUNNING: step() 继续生成
    RUNNING --> PREEMPTED: 内存不足被抢占
    RUNNING --> FINISHED_STOPPED: EOS 或停止字符串
    RUNNING --> FINISHED_LENGTH: 达到 max_tokens
    RUNNING --> FINISHED_ABORTED: 用户取消

    PREEMPTED --> WAITING: 重新排队

    FINISHED_STOPPED --> [*]: 释放资源
    FINISHED_LENGTH --> [*]: 释放资源
    FINISHED_ABORTED --> [*]: 释放资源
```

---

## 9. 关键数据结构流转

```
用户输入
    ↓
prompt: str
    ↓ Tokenize
prompt_token_ids: list[int]
    ↓ 创建请求
EngineCoreRequest
    ↓ 调度器内部
Request (internal)
    ↓ 调度
SchedulerOutput
    ↓ 执行
ModelRunnerOutput (logits)
    ↓ 采样
SamplerOutput (token_ids)
    ↓ 更新
EngineCoreOutput
    ↓ Detokenize
RequestOutput
    ↓
用户输出
```

---

## 10. 小结

本章我们完整跟踪了一个请求的生命周期：

1. **提交阶段**：
   - Tokenize → 创建请求 → 加入 waiting 队列

2. **调度阶段**：
   - 查找缓存 → 分配 KV Cache → 移入 running

3. **执行阶段**：
   - 准备输入 → 前向传播 → 采样

4. **更新阶段**：
   - 追加 token → 检查停止 → 更新状态

5. **返回阶段**：
   - Detokenize → 构建输出 → 返回用户

通过这个完整的流程分析，我们可以看到 vLLM 的各个组件是如何协同工作的，以及为什么它能够实现高效的 LLM 推理。

---

## 导航

- 上一篇：[输出处理流程](05-output-processing.md)
- 下一篇：[投机解码](../05-advanced-topics/01-speculative-decoding.md)
- [返回目录](../README.md)
