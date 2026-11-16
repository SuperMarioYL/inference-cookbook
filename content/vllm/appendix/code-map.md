---
title: "vLLM 0.33.0 源码索引"
weight: 2
---

按“一次在线请求走到哪里”读源码。先确定组件职责，再看具体后端；同一个 `Executor` 接口可以选择不同的 worker 进程组织方式。

| 追踪问题 | 固定版本入口 |
|---|---|
| 谁接收 HTTP 请求 | [api_server.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/api_server.py) |
| 谁创建异步引擎 | [async_llm.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/async_llm.py) |
| 前端如何访问 core | [core_client.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/core_client.py) |
| 谁组织一轮执行 | [core.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/core.py) |
| 谁决定本轮 token | [scheduler.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/sched/scheduler.py) |
| 在哪里核对本地缓存命中 | [kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_manager.py) |
| 谁决定物理块可否重用 | [block_pool.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/block_pool.py) |
| 模型执行如何跨进程分发 | [abstract.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/executor/abstract.py) |
| CUDA worker 如何选择并驱动 runner | [gpu_worker.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_worker.py) |
| 设备 attention 实现如何选择 | [selector.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/selector.py) |
| 如何收集增量输出 | [output_processor.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/output_processor.py) |

源码中的 V1 指引擎架构。发行版本、GPU 模型执行器版本和 attention backend 是不同维度，定位问题时应一并记录启动日志和运行配置。

## 按模块展开目录

### 入口点（Entry Points）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/entrypoints/llm.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/llm.py) | Python API 入口 |
| [vllm/entrypoints/cli/main.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/cli/main.py) | CLI 入口 |
| [vllm/entrypoints/openai/api_server.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/entrypoints/openai/api_server.py) | OpenAI API 服务 |
| [vllm/engine/arg_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/engine/arg_utils.py) | 参数解析 |

### V1 引擎（Engine）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/engine/llm_engine.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/llm_engine.py) | LLM 引擎入口 |
| [vllm/v1/engine/core.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/core.py) | 引擎核心 |
| [vllm/v1/engine/async_llm.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/async_llm.py) | 异步引擎 |

### 核心调度（Core Scheduling）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/core/sched/scheduler.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/sched/scheduler.py) | 调度器 |
| [vllm/v1/core/sched/request_queue.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/sched/request_queue.py) | 请求队列 |
| [vllm/v1/core/kv_cache_manager.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_manager.py) | KV Cache 管理 |
| [vllm/v1/core/block_pool.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/block_pool.py) | 块池管理 |
| [vllm/v1/core/kv_cache_utils.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/core/kv_cache_utils.py) | KV Cache 工具 |

### 请求处理（Request）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/request.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/request.py) | 请求数据结构 |
| [vllm/sampling_params.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/sampling_params.py) | 采样参数 |
| [vllm/outputs.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/outputs.py) | 输出数据结构 |

### Worker 执行（Worker）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/worker/gpu_worker.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_worker.py) | GPU Worker |
| [vllm/v1/worker/gpu_model_runner.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_model_runner.py) | 模型执行 |
| [vllm/v1/worker/gpu_input_batch.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/worker/gpu_input_batch.py) | 输入批处理 |

### Executor 执行器

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/executor/abstract.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/executor/abstract.py) | 执行器基类 |
| [vllm/v1/executor/uniproc_executor.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/executor/uniproc_executor.py) | 单进程执行器 |
| [vllm/v1/executor/multiproc_executor.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/executor/multiproc_executor.py) | 多进程执行器 |

### 注意力机制（Attention）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/attention/ops/paged_attn.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/ops/paged_attn.py) | PagedAttention 接口 |
| [vllm/v1/attention/backends/flash_attn.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/backends/flash_attn.py) | Flash Attention 后端 |
| [vllm/v1/attention/backends/triton_attn.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/attention/backends/triton_attn.py) | Triton Attention 后端 |

### 采样（Sampling）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/sample/sampler.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/sample/sampler.py) | 采样器 |
| [vllm/v1/sample/metadata.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/sample/metadata.py) | 采样元数据 |
| [vllm/v1/sample/ops/penalties.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/sample/ops/penalties.py) | 惩罚项计算 |

### 投机解码（Speculative Decoding）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/v1/spec_decode/eagle.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/spec_decode/eagle.py) | EAGLE 基类 |
| [vllm/v1/spec_decode/draft_model.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/spec_decode/draft_model.py) | Draft Model |

### 模型实现（Models）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/model_executor/models/llama.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/llama.py) | LLaMA 模型 |
| [vllm/model_executor/models/qwen2.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/qwen2.py) | Qwen2 模型 |
| [vllm/model_executor/models/mixtral.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/mixtral.py) | Mixtral MoE |
| [vllm/model_executor/models/deepseek_v2.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/models/deepseek_v2.py) | DeepSeek V2 |

### 量化（Quantization）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/model_executor/layers/quantization/__init__.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/quantization/__init__.py) | 量化入口 |
| [vllm/model_executor/layers/quantization/base_config.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/quantization/base_config.py) | 量化基类 |
| [vllm/model_executor/layers/quantization/fp8.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/model_executor/layers/quantization/fp8.py) | FP8 量化 |

### 分布式通信（Distributed）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/distributed/parallel_state.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/parallel_state.py) | 并行状态管理 |
| [vllm/distributed/communication_op.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/communication_op.py) | 通信操作 |
| [vllm/distributed/device_communicators/pynccl.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/device_communicators/pynccl.py) | NCCL 通信 |
| [vllm/distributed/device_communicators/custom_all_reduce.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/distributed/device_communicators/custom_all_reduce.py) | 自定义 AllReduce |

### 配置（Config）

| 固定版本文件 | 阅读用途 |
|---|---|
| [vllm/config/vllm.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/vllm.py) | 总配置 |
| [vllm/config/model.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/model.py) | 模型配置 |
| [vllm/config/parallel.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/parallel.py) | 并行配置 |
| [vllm/config/scheduler.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/scheduler.py) | 调度器配置 |
| [vllm/config/cache.py](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/config/cache.py) | 缓存配置 |
