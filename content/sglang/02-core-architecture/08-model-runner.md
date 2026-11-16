---
title: "ModelRunner"
weight: 5
---

## 从一个调度批次找到设备输入

先确认 Scheduler 选择的前向模式、请求数和本轮新增 token 数，再看 ModelRunner 怎样准备本轮输入与缓存位置。CUDA Graph 在新版区分 decode 与 prefill runner，下面的通用前向和捕获步骤按教学控制流阅读。 [0.5.23 ModelRunner](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/model_executor/model_runner.py)

## ModelRunner 职责

ModelRunner 是 SGLang 的模型执行核心，负责：

```mermaid
graph TB
    subgraph responsibilities["核心职责"]
        R1["模型加载"]
        R2["内存池管理"]
        R3["注意力后端初始化"]
        R4["CUDA Graph 管理"]
        R5["前向计算执行"]
    end

    subgraph interactions["交互对象"]
        SCHED["Scheduler"]
        MODEL["LLM Model"]
        CACHE["KV Cache"]
        GRAPH["CUDA Graph"]
    end

    SCHED --> R5
    R1 --> MODEL
    R2 --> CACHE
    R4 --> GRAPH
    R5 --> MODEL
```

---

## 初始化流程

### 初始化链

**关键文件**：`python/sglang/srt/model_executor/model_runner.py`

```mermaid
graph TB
    subgraph init["初始化流程"]
        I1["init_torch_distributed"]
        I2["load_model"]
        I3["init_memory_pool"]
        I4["init_attention_backend"]
        I5["init_device_graphs"]

        I1 --> I2 --> I3 --> I4 --> I5
    end
```

### 核心属性

```python
class ModelRunner:
    def __init__(self, model_config, server_args, gpu_id, tp_rank, ...):
        # 分布式参数
        self.tp_rank = tp_rank
        self.tp_size = tp_size
        self.pp_rank = pp_rank
        self.pp_size = pp_size
        self.dp_size = dp_size

        # 模型和设备
        self.model: nn.Module
        self.device: str
        self.gpu_id: int

        # 内存池
        self.req_to_token_pool: ReqToTokenPool
        self.token_to_kv_pool: KVCache

        # 注意力后端
        self.attn_backend: AttentionBackend

        # CUDA Graph
        self.graph_runner: CudaGraphRunner
```

---

## 模型加载

### 加载流程

```mermaid
graph TB
    subgraph load["load_model"]
        CHECK["内存检查"]
        CONFIG["准备配置"]
        LOADER["获取加载器"]
        LOAD["加载权重"]
        POST["后处理"]

        CHECK --> CONFIG --> LOADER --> LOAD --> POST
    end
```

### 加载实现

```python
def load_model(self):
    # 1. 内存检查和准备
    available_memory = get_available_gpu_memory(self.gpu_id)

    # 2. 准备配置
    self.load_config = LoadConfig(
        load_format=self.server_args.load_format,
        download_dir=self.server_args.download_dir,
    )

    # 3. 获取模型加载器
    self.loader = get_model_loader(
        load_config=self.load_config,
        model_config=self.model_config,
    )

    # 4. 加载模型
    with self.memory_saver_adapter.region(GPU_MEMORY_TYPE_WEIGHTS):
        self.model = self.loader.load_model(
            model_config=self.model_config,
            device_config=DeviceConfig(self.device, self.gpu_id),
        )

    # 5. 后处理
    # - 加载 KV 缓存缩放因子（FP8）
    # - 提取滑动窗口大小
    # - 预展开 RoPE 缓存
    self._post_load_model()
```

---

## ForwardBatch 数据结构

### 结构定义

**关键文件**：`python/sglang/srt/model_executor/forward_batch_info.py`

```python
@dataclass
class ForwardBatch:
    # 基础信息
    forward_mode: ForwardMode       # 前向模式
    batch_size: int                 # 批次大小
    input_ids: torch.Tensor         # 输入 token IDs [num_tokens]
    req_pool_indices: torch.Tensor  # 请求池索引 [batch_size]
    seq_lens: torch.Tensor          # 序列长度 [batch_size]
    out_cache_loc: torch.Tensor     # KV 缓存位置 [num_tokens]
    seq_lens_sum: int               # 总 token 数

    # 位置信息
    positions: torch.Tensor         # token 位置 [num_tokens]
    extend_seq_lens: Optional[torch.Tensor]    # 扩展长度
    extend_prefix_lens: Optional[torch.Tensor] # 前缀长度
    extend_num_tokens: Optional[int]           # 扩展 token 数

    # 内存池引用
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: KVCache
    attn_backend: AttentionBackend

    # 采样信息
    sampling_info: SamplingBatchInfo
```

### ForwardMode 类型

```python
class ForwardMode(IntEnum):
    EXTEND = 1          # Prefill：扩展序列
    DECODE = 2          # Decode：解码单个 token
    MIXED = 3           # 混合模式
    IDLE = 4            # 空闲（数据并行）
    TARGET_VERIFY = 5   # 推测解码验证
    DRAFT_EXTEND = 6    # Draft 模型扩展
```

### 构建流程

{{< trace >}}
title: 从两条请求组装一次设备输入
actors: ["调度批", "元数据", "ModelRunner"]
steps:
  - from: 0
    to: 1
    label: "选出 A 与 B"
    detail: "A、B 各有一个待执行位置。"
  - from: 1
    to: 1
    label: "保留请求映射"
    detail: "每条输入都带着自己的位置、序列长度和 KV 索引。"
  - from: 1
    to: 2
    label: "构造 ForwardBatch"
    detail: "runner 得到与前向模式相符的输入张量和元数据。"
  - from: 2
    to: 2
    label: "执行并分回"
    detail: "模型一次执行两条输入，结果按原先的请求映射写回。"
{{< /trace >}}

```python
@classmethod
def init_new(cls, batch: ModelWorkerBatch, model_runner: ModelRunner):
    ret = cls(
        forward_mode=batch.forward_mode,
        batch_size=len(batch.seq_lens),
        input_ids=batch.input_ids,
        req_pool_indices=batch.req_pool_indices,
        seq_lens=batch.seq_lens,
        out_cache_loc=batch.out_cache_loc,
        # ...
    )

    # 计算位置
    if ret.forward_mode.is_decode():
        ret.positions = clamp_position(batch.seq_lens)
    else:
        # Prefill 模式
        positions, ret.extend_start_loc = compute_position(
            model_runner.server_args.attention_backend,
            ret.extend_prefix_lens,
            ret.extend_seq_lens,
            ret.extend_num_tokens,
        )
        ret.positions = positions

    # LoRA 准备
    if model_runner.server_args.enable_lora:
        model_runner.lora_manager.prepare_lora_batch(ret)

    return ret
```

---

## 前向计算流程

### 主前向方法

```mermaid
graph TB
    subgraph forward["forward"]
        CHECK{"可用 CUDA Graph?"}
        REPLAY["graph_runner.replay"]
        DECODE["forward_decode"]
        EXTEND["forward_extend"]
        IDLE["forward_idle"]
        OUTPUT["ModelRunnerOutput"]

        CHECK -->|"是"| REPLAY
        CHECK -->|"否 - Decode"| DECODE
        CHECK -->|"否 - Extend"| EXTEND
        CHECK -->|"否 - Idle"| IDLE
        REPLAY --> OUTPUT
        DECODE --> OUTPUT
        EXTEND --> OUTPUT
        IDLE --> OUTPUT
    end
```

### 实现代码

```python
def _forward_raw(self, forward_batch: ForwardBatch, ...):
    # 1. 检查是否可用 CUDA Graph
    can_run_graph = (
        forward_batch.forward_mode.is_cuda_graph()
        and self.graph_runner
        and self.graph_runner.can_run(forward_batch)
    )

    # 2. 如果可用，直接重放 CUDA Graph
    if can_run_graph:
        ret = self.graph_runner.replay(
            forward_batch,
            skip_attn_backend_init=skip_attn_backend_init,
        )
        return ModelRunnerOutput(
            logits_output=ret,
            can_run_graph=can_run_graph
        )

    # 3. 否则执行动态前向
    if forward_batch.forward_mode.is_decode():
        ret = self.forward_decode(forward_batch, ...)
    elif forward_batch.forward_mode.is_extend():
        ret = self.forward_extend(forward_batch, ...)
    elif forward_batch.forward_mode.is_idle():
        ret = self.forward_idle(forward_batch, ...)

    return ModelRunnerOutput(
        logits_output=ret,
        can_run_graph=can_run_graph
    )
```

### Decode 前向

```python
def forward_decode(self, forward_batch: ForwardBatch, ...):
    # 初始化注意力后端元数据
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)

    # 调用模型前向
    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )
```

### Extend 前向（Prefill）

```python
def forward_extend(self, forward_batch: ForwardBatch, ...):
    # 尝试使用分段 CUDA Graph
    if (self.piecewise_cuda_graph_runner is not None
        and self.piecewise_cuda_graph_runner.can_run(forward_batch)):
        return self.piecewise_cuda_graph_runner.replay(forward_batch)

    # 标准 Prefill 前向
    if not skip_attn_backend_init:
        self.attn_backend.init_forward_metadata(forward_batch)

    return self.model.forward(
        forward_batch.input_ids,
        forward_batch.positions,
        forward_batch,
    )
```

---

## CUDA Graph 优化

### 为什么使用 CUDA Graph

| 问题 | CUDA Graph 解决方案 |
|------|---------------------|
| Kernel Launch 开销 | 预先录制，一次提交 |
| CPU-GPU 同步 | 消除中间同步点 |
| 小批次低效 | 批量执行优化 |

### CudaGraphRunner 结构

**关键文件**：`python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py`

```python
# 教学示意，实际解码图由 DecodeCudaGraphRunner 组织。
class GraphReplayExample:
    def __init__(self, model_runner: ModelRunner):
        # 批次大小列表
        self.capture_bs: List[int]

        # 捕获的图
        self.graphs: Dict[int, torch.cuda.CUDAGraph]

        # 输入缓冲区
        self.buffers: GraphInputBuffers

        # 输出缓冲区
        self.output_buffers: Dict[int, LogitsProcessorOutput]
```

### 捕获流程

```mermaid
graph TB
    subgraph capture["CUDA Graph 捕获"]
        PREPARE["准备缓冲区"]
        WARMUP["预热运行"]
        RECORD["录制图"]
        STORE["存储图"]

        PREPARE --> WARMUP --> RECORD --> STORE
    end
```

```python
def capture(self) -> None:
    with freeze_gc(self.model_runner.server_args.enable_cudagraph_gc):
        # 反向遍历批次大小（内存重用）
        for bs in reversed(self.capture_bs):
            with patch_model(self.model_runner.model, bs in self.compile_bs):
                graph, output_buffers = self.capture_one_batch_size(bs, forward)
                self.graphs[bs] = graph
                self.output_buffers[bs] = output_buffers
```

### 单批次捕获

```python
def capture_one_batch_size(self, bs: int, forward: Callable):
    # 1. 准备输入
    num_tokens = bs * self.num_tokens_per_bs
    input_ids = self.buffers.input_ids[:num_tokens]
    seq_lens = self.buffers.seq_lens[:bs]
    positions = self.buffers.positions[:num_tokens]

    # 2. 构建 ForwardBatch
    forward_batch = ForwardBatch(
        forward_mode=self.capture_forward_mode,
        batch_size=bs,
        input_ids=input_ids,
        # ...
    )

    # 3. 初始化注意力后端
    self.attn_backend.init_forward_metadata_capture_cuda_graph(
        bs, num_tokens, req_pool_indices, seq_lens, ...
    )

    # 4. 定义运行函数
    def run_once():
        return forward(input_ids, positions, forward_batch)

    # 5. 预热（两次运行）
    for _ in range(2):
        self.device_module.synchronize()
        run_once()

    # 6. 捕获 CUDA Graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, pool=self.pool):
        out = run_once()

    return graph, out
```

### 重放流程

{{< trace >}}
title: 已有图怎样执行一组新输入
actors: ["调度器", "ModelRunner", "设备图"]
steps:
  - from: 0
    to: 1
    label: "送入批次"
    detail: "本轮有两个请求，各提交一个新位置。"
  - from: 1
    to: 1
    label: "匹配图形状"
    detail: "runner 根据模式和批大小选择已捕获的图，并准备对应缓冲。"
  - from: 1
    to: 2
    label: "重放"
    detail: "设备读取这次写入缓冲的 token、长度和位置，运行录制好的计算。"
  - from: 2
    to: 1
    label: "取回结果"
    detail: "runner 把这次的输出关联到实际请求，padding 位置不会提交给客户端。"
{{< /trace >}}

```python
def replay(self, forward_batch: ForwardBatch, ...):
    # 1. 准备重放
    if not skip_attn_backend_init:
        self.replay_prepare(forward_batch)

    # 2. 复制输入到缓冲区
    self.buffers.input_ids[:self.raw_num_token].copy_(forward_batch.input_ids)
    self.buffers.positions[:self.raw_num_token].copy_(forward_batch.positions)

    # 3. 重放 CUDA Graph
    self.graphs[self.bs].replay()
    output = self.output_buffers[self.bs]

    # 4. 提取实际输出
    next_token_logits = output.next_token_logits[:self.raw_num_token]

    return LogitsProcessorOutput(
        next_token_logits=next_token_logits,
        # ...
    )
```

---

## 输入缓冲区

### GraphInputBuffers 结构

```python
@dataclass
class GraphInputBuffers:
    input_ids: torch.Tensor          # [max_num_token]
    req_pool_indices: torch.Tensor   # [max_bs]
    seq_lens: torch.Tensor           # [max_bs]
    seq_lens_cpu: torch.Tensor       # CPU 副本
    out_cache_loc: torch.Tensor      # [max_num_token]
    positions: torch.Tensor          # [max_num_token]

    # 全局 token 数（MLP 同步）
    global_num_tokens_gpu: torch.Tensor

    # Pipeline 并行代理张量
    pp_proxy_tensors: Dict[str, torch.Tensor]
```

### 缓冲区填充

```python
def populate_from_forward_batch(
    self,
    forward_batch: ForwardBatch,
    raw_bs: int,
    raw_num_token: int,
    bs: int,  # 填充后的批次大小
):
    # 1. 复制原始数据
    self.input_ids[:raw_num_token].copy_(forward_batch.input_ids)
    self.req_pool_indices[:raw_bs].copy_(forward_batch.req_pool_indices)
    self.seq_lens[:raw_bs].copy_(forward_batch.seq_lens)

    # 2. 填充到目标大小
    if bs > raw_bs:
        self.seq_lens[raw_bs:bs].fill_(seq_len_fill_value)
        self.req_pool_indices[raw_bs:bs].fill_(0)

    return self.seq_lens[:bs].cpu()
```

---

## 内存池管理

### 内存池初始化

```python
def init_memory_pool(self, total_gpu_memory: int):
    # 计算最大 token 数
    self.max_total_num_tokens = self.profile_max_num_token(total_gpu_memory)

    # 请求到 token 的映射池
    self.req_to_token_pool = ReqToTokenPool(
        size=self.server_args.max_running_requests,
        num_token=self.max_total_num_tokens,
    )

    # KV 缓存池
    self.token_to_kv_pool = BaseTokenToKVPoolAllocator.create(
        allocator_type=self.server_args.kv_cache_allocator,
        num_token=self.max_total_num_tokens,
        dtype=self.kv_cache_dtype,
        head_num=self.model_config.num_kv_heads,
        head_dim=self.model_config.head_dim,
        layer_num=self.model_config.num_hidden_layers,
        device=self.device,
    )
```

### 内存布局

```mermaid
graph TB
    subgraph memory["内存布局"]
        WEIGHTS["模型权重"]
        KV["KV Cache 池"]
        TEMP["临时张量"]
        GRAPH["CUDA Graph 缓冲"]

        GPU["GPU 内存"] --> WEIGHTS
        GPU --> KV
        GPU --> TEMP
        GPU --> GRAPH
    end
```

---

## 注意力后端

### 支持的后端

| 后端 | 特点 |
|------|------|
| FlashInfer | 高性能，支持 PagedAttention |
| FlashAttention3 | 新版 FlashAttention |
| Triton | 纯 Python 实现 |
| torch_native | PyTorch 原生 |

### 后端初始化

```python
def init_attention_backend(self):
    backend_name = self.server_args.attention_backend

    if backend_name == "flashinfer":
        self.attn_backend = FlashInferAttnBackend(
            self.model_config,
            self.server_args,
            self.max_total_num_tokens,
        )
    elif backend_name == "fa3":
        self.attn_backend = FlashAttention3Backend(...)
    elif backend_name == "triton":
        self.attn_backend = TritonAttnBackend(...)
```

---

## 数据流总结

```mermaid
graph LR
    subgraph input["输入"]
        SB["ScheduleBatch"]
    end

    subgraph worker["Worker"]
        MWB["ModelWorkerBatch"]
    end

    subgraph runner["ModelRunner"]
        FB["ForwardBatch"]
        MODEL["Model.forward"]
        SAMPLE["Sample"]
    end

    subgraph output["输出"]
        OUT["next_token_ids"]
    end

    SB --> MWB
    MWB --> FB
    FB --> MODEL
    MODEL --> SAMPLE
    SAMPLE --> OUT
```

---

## 性能优化要点

### 1. CUDA Graph

- 预捕获常见批次大小
- 消除 Kernel Launch 开销
- 内存重用（大批次先捕获）

### 2. 注意力后端

- FlashInfer: PagedAttention 优化
- FlashAttention3: 最新优化
- 自动选择最佳后端

### 3. 内存管理

- 预分配内存池
- 避免动态分配
- KV Cache 复用

### 4. 并行优化

- TP/PP/DP 支持
- MLP 同步优化
- 通信与计算重叠

---

## 模型执行源码索引

| 文件 | 职责 |
|------|------|
| `model_runner.py` | 核心执行器 |
| `forward_batch_info.py` | ForwardBatch 定义 |
| `python/sglang/srt/model_executor/runner/decode_cuda_graph_runner.py` | CUDA Graph 管理 |
| `input_buffers.py` | 缓冲区管理 |
