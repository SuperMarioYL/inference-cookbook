---
title: "模型前向传播"
weight: 3
---

## 先区分两套 GPU model runner

0.33.0 的 worker 在 _make_model_runner 中选择传统 GPUModelRunner 或 V2。V2 把持久状态与本次执行的输入分开维护；本章的张量示例讲相同计算依赖，实际输入布局和采样路径要按选中的 runner 阅读。 [Model Runner V2 设计](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/model_runner_v2.md)

以 Llama 为例，输入 token 先映射为向量，依次通过注意力和前馈层。最后的位置表示经输出层映射为词表上的 logits。

---

## 1. 模型架构概览

### 1.1 Llama 模型结构

```mermaid
graph TD
    subgraph llama_model["LlamaForCausalLM"]
        Input[input_ids] --> Embed[Embedding]
        Embed --> Layers[Transformer Layers × N]
        Layers --> Norm[RMSNorm]
        Norm --> LMHead[LM Head]
        LMHead --> Logits[logits]
    end

    subgraph transformer_layer["Transformer Layer"]
        H[hidden_states] --> LN1[RMSNorm]
        LN1 --> Attn[Self-Attention]
        Attn --> Add1[+]
        H --> Add1
        Add1 --> LN2[RMSNorm]
        LN2 --> MLP[MLP]
        MLP --> Add2[+]
        Add1 --> Add2
        Add2 --> Out[output]
    end
```

### 1.2 核心组件对应关系

| 组件 | vLLM 类 | 功能 |
|-----|--------|------|
| Embedding | `VocabParallelEmbedding` | token 到向量的映射 |
| Transformer Layer | `LlamaDecoderLayer` | 主要计算单元 |
| Self-Attention | `LlamaAttention` | 注意力计算 |
| MLP | `LlamaMLP` | 前馈网络 |
| LayerNorm | `RMSNorm` | 归一化 |
| LM Head | `ParallelLMHead` | 输出词表概率 |

---

## 2. vLLM 中的 Llama 实现

### 2.1 LlamaForCausalLM 类

```python
# vllm/model_executor/models/llama.py

class LlamaForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle):
    """用于因果语言建模的 Llama 模型"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        # 主模型（Transformer 层）
        self.model = LlamaModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model"))

        # 输出层（LM Head）
        if config.tie_word_embeddings:
            self.lm_head = self.model.embed_tokens  # 权重共享
        else:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
            )

        # Logits 处理器
        self.logits_processor = LogitsProcessor(config.vocab_size)

        # 采样器
        self.sampler = get_sampler()

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: list[torch.Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        """模型前向传播"""

        # 1. 通过 Transformer 层
        hidden_states = self.model(input_ids, positions, intermediate_tensors)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        """计算 logits"""
        logits = self.logits_processor(
            self.lm_head,
            hidden_states,
            sampling_metadata,
        )
        return logits

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> SamplerOutput:
        """采样下一个 token"""
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens
```

### 2.2 LlamaModel 类

```python
# vllm/model_executor/models/llama.py

@support_torch_compile(shape_invariants=llama_model_invariants)
class LlamaModel(nn.Module):
    """Llama 的 Transformer 模型"""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Embedding 层
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )

        # Transformer 层
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: LlamaDecoderLayer(vllm_config=vllm_config, prefix=prefix),
            prefix=f"{prefix}.layers",
        )

        # 最终归一化
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:

        # 1. Embedding
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_tokens(input_ids)
            residual = None
        else:
            # 流水线并行：从前一阶段获取中间结果
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        # 2. Transformer 层
        for layer in self.layers[self.start_layer:self.end_layer]:
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
            )

        # 3. 最终归一化（仅最后一个 PP 阶段）
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({
                "hidden_states": hidden_states,
                "residual": residual
            })

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states
```

---

## 3. Transformer 层详解

### 3.1 LlamaDecoderLayer

```python
# vllm/model_executor/models/llama.py

class LlamaDecoderLayer(nn.Module):
    """单个 Transformer 解码器层"""

    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config

        # Self-Attention
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            cache_config=vllm_config.cache_config,
            prefix=f"{prefix}.self_attn",
        )

        # MLP
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            prefix=f"{prefix}.mlp",
        )

        # 归一化层
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        层前向传播

        使用 Pre-LN 结构（LayerNorm 在子层之前）
        """
        # 1. 第一个归一化 + 残差处理
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # 2. Self-Attention
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # 3. 第二个归一化 + 残差处理
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        # 4. MLP
        hidden_states = self.mlp(hidden_states)

        return hidden_states, residual
```

### 3.2 前向传播数据流

```mermaid
flowchart TD
    subgraph decoder_layer["LlamaDecoderLayer Forward"]
        Input["输入<br/>(hidden_states, residual)"]

        Input --> Check{residual is None?}
        Check -->|是| Init["residual = hidden_states"]
        Check -->|否| Fuse1["hidden_states, residual =<br/>input_layernorm(hidden_states, residual)"]
        Init --> LN1["hidden_states = input_layernorm(hidden_states)"]

        LN1 --> Attn["hidden_states = self_attn(positions, hidden_states)"]
        Fuse1 --> Attn

        Attn --> Fuse2["hidden_states, residual =<br/>post_attention_layernorm(hidden_states, residual)"]

        Fuse2 --> MLP["hidden_states = mlp(hidden_states)"]

        MLP --> Output["输出<br/>(hidden_states, residual)"]
    end

    style Attn fill:#e1f5fe
    style MLP fill:#fff3e0
```

---

## 4. Self-Attention 详解

### 4.1 LlamaAttention 类

```python
# vllm/model_executor/models/llama.py

class LlamaAttention(nn.Module):
    """Llama 的多头注意力层"""

    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        cache_config: CacheConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()

        # 张量并行配置
        tp_size = get_tensor_model_parallel_world_size()
        self.num_heads = num_heads // tp_size
        self.num_kv_heads = max(1, num_kv_heads // tp_size)
        self.head_dim = hidden_size // num_heads
        self.scaling = self.head_dim ** -0.5

        # Q、K、V 投影（合并为一个线性层）
        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=num_heads,
            total_num_kv_heads=num_kv_heads,
            prefix=f"{prefix}.qkv_proj",
        )

        # 输出投影
        self.o_proj = RowParallelLinear(
            input_size=num_heads * self.head_dim,
            output_size=hidden_size,
            prefix=f"{prefix}.o_proj",
        )

        # 旋转位置编码
        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=config.max_position_embeddings,
        )

        # 注意力后端
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        注意力前向传播

        Args:
            positions: 位置 IDs [num_tokens]
            hidden_states: 隐藏状态 [num_tokens, hidden_size]

        Returns:
            输出 [num_tokens, hidden_size]
        """
        # 1. QKV 投影
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # 2. 旋转位置编码
        q, k = self.rotary_emb(positions, q, k)

        # 3. 注意力计算（使用 vLLM 的优化后端）
        attn_output = self.attn(q, k, v)

        # 4. 输出投影
        output, _ = self.o_proj(attn_output)

        return output
```

### 4.2 注意力计算流程

```mermaid
flowchart LR
    subgraph self_attn["Self-Attention"]
        H["hidden_states<br/>[tokens, hidden]"]
        QKV["QKV Projection"]
        Split["Split"]
        Q["Q"]
        K["K"]
        V["V"]
        RoPE["RoPE"]
        Attn["Attention<br/>(with KV Cache)"]
        O["O Projection"]
        Out["output"]

        H --> QKV
        QKV --> Split
        Split --> Q
        Split --> K
        Split --> V

        Q --> RoPE
        K --> RoPE

        RoPE --> Attn
        V --> Attn

        Attn --> O
        O --> Out
    end
```

### 4.3 GQA (Grouped-Query Attention)

```
标准 MHA (Multi-Head Attention):
Q heads: [h1, h2, h3, h4, h5, h6, h7, h8]
K heads: [h1, h2, h3, h4, h5, h6, h7, h8]
V heads: [h1, h2, h3, h4, h5, h6, h7, h8]

GQA (num_kv_heads=2):
Q heads: [h1, h2, h3, h4, h5, h6, h7, h8]
K heads: [k1,   k1,   k1,   k1, | k2,   k2,   k2,   k2]
V heads: [v1,   v1,   v1,   v1, | v2,   v2,   v2,   v2]

每 4 个 Q head 共享 1 个 KV head，减少 KV Cache 内存
```

---

## 5. MLP 详解

### 5.1 LlamaMLP 类

```python
# vllm/model_executor/models/llama.py

class LlamaMLP(nn.Module):
    """Llama 的 MLP（SwiGLU 激活）"""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        prefix: str = "",
    ):
        super().__init__()

        # gate 和 up 投影（合并）
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,  # gate 和 up 各一个
            prefix=f"{prefix}.gate_up_proj",
        )

        # down 投影
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            prefix=f"{prefix}.down_proj",
        )

        # SiLU 激活 + 门控
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        MLP 前向传播

        SwiGLU: down(SiLU(gate(x)) * up(x))
        """
        # 1. gate 和 up 投影（合并执行）
        gate_up, _ = self.gate_up_proj(x)

        # 2. SiLU 激活 + 门控相乘
        x = self.act_fn(gate_up)

        # 3. down 投影
        x, _ = self.down_proj(x)

        return x
```

### 5.2 SwiGLU 计算流程

```mermaid
flowchart LR
    subgraph mlp_swiglu["MLP (SwiGLU)"]
        X["x<br/>[tokens, hidden]"]
        GU["gate_up_proj"]
        Gate["gate"]
        Up["up"]
        SiLU["SiLU(gate)"]
        Mul["×"]
        Down["down_proj"]
        Out["output"]

        X --> GU
        GU --> Gate
        GU --> Up
        Gate --> SiLU
        SiLU --> Mul
        Up --> Mul
        Mul --> Down
        Down --> Out
    end
```

---

## 6. KV Cache 集成

### 6.1 Attention 层的 KV Cache 使用

```python
# vllm/model_executor/layers/attention/attention.py

class Attention(nn.Module):
    """vLLM 的注意力层，集成 KV Cache"""

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            query: [num_tokens, num_heads * head_dim]
            key: [num_tokens, num_kv_heads * head_dim]
            value: [num_tokens, num_kv_heads * head_dim]

        内部会：
        1. 将新的 K、V 写入 KV Cache
        2. 从 KV Cache 读取完整的 K、V
        3. 计算注意力
        """
        return self.impl.forward(
            query,
            key,
            value,
            kv_cache=self.kv_cache,
            attn_metadata=self.attn_metadata,
        )
```

### 6.2 PagedAttention 后端

{{< trace >}}
title: 块表如何参与一次注意力前向
actors: ["注意力层", "后端", "KV 缓存"]
steps:
  - from: 0
    to: 1
    label: "提交 Q、K、V"
    detail: "当前 token 的投影与批次的注意力元数据一起传给后端。"
  - from: 1
    to: 2
    label: "定位写入槽"
    detail: "slot mapping 把新 K/V 放到当前请求的物理位置。"
  - from: 2
    to: 1
    label: "按块表读取"
    detail: "读取逻辑历史所对应的物理块，块号可以不连续。"
  - from: 1
    to: 1
    label: "汇总注意力"
    detail: "后端在模型允许的上下文中求权重并汇总值向量。"
  - from: 1
    to: 0
    label: "送回输出"
    detail: "本层输出回到模型后续计算。"
{{< /trace >}}

---

## 7. 张量并行实现

### 7.1 线性层的张量并行

```python
# vllm/model_executor/layers/linear.py

class ColumnParallelLinear(nn.Module):
    """列并行线性层（输出切分）"""

    def __init__(self, input_size: int, output_size: int, ...):
        tp_size = get_tensor_model_parallel_world_size()
        # 每个 GPU 只有 output_size / tp_size 的输出
        self.output_size_per_partition = output_size // tp_size
        self.weight = Parameter(torch.empty(
            self.output_size_per_partition,
            input_size,
        ))

class RowParallelLinear(nn.Module):
    """行并行线性层（输入切分）"""

    def __init__(self, input_size: int, output_size: int, ...):
        tp_size = get_tensor_model_parallel_world_size()
        # 每个 GPU 只处理 input_size / tp_size 的输入
        self.input_size_per_partition = input_size // tp_size
        self.weight = Parameter(torch.empty(
            output_size,
            self.input_size_per_partition,
        ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = F.linear(x, self.weight)
        if self.reduce_results:
            # AllReduce 收集所有 GPU 的结果
            output = tensor_model_parallel_all_reduce(output)
        return output
```

### 7.2 QKV 投影的张量并行

```mermaid
graph TD
    subgraph gpu0["GPU 0"]
        Q0["Q[:, :dim/2]"]
        K0["K[:, :dim/4]"]
        V0["V[:, :dim/4]"]
    end

    subgraph gpu1["GPU 1"]
        Q1["Q[:, dim/2:]"]
        K1["K[:, dim/4:]"]
        V1["V[:, dim/4:]"]
    end

    Input["hidden_states"] --> gpu0
    Input --> gpu1

    Q0 --> Attn0["Attention 0"]
    K0 --> Attn0
    V0 --> Attn0

    Q1 --> Attn1["Attention 1"]
    K1 --> Attn1
    V1 --> Attn1

    Attn0 --> AllReduce
    Attn1 --> AllReduce
    AllReduce --> Output["output"]
```

---

## 8. 完整前向传播流程

```mermaid
flowchart TD
    subgraph execute_model["execute_model"]
        Input["SchedulerOutput"]
        Prep["_prepare_inputs()"]
        FWD["model.forward()"]
        Logits["compute_logits()"]
        Sample["sample()"]
        Output["ModelRunnerOutput"]

        Input --> Prep
        Prep --> FWD
        FWD --> Logits
        Logits --> Sample
        Sample --> Output
    end

    subgraph model_forward["model.forward()"]
        Embed["embed_tokens(input_ids)"]
        Layers["for layer in layers:<br/>  hidden, residual = layer(...)"]
        Norm["norm(hidden, residual)"]

        Embed --> Layers
        Layers --> Norm
    end

    subgraph layer_forward["layer.forward()"]
        LN1["input_layernorm"]
        Attn["self_attn(Q,K,V + KV Cache)"]
        LN2["post_attention_layernorm"]
        MLP["mlp(SwiGLU)"]

        LN1 --> Attn
        Attn --> LN2
        LN2 --> MLP
    end

    FWD --> Embed
    Norm --> Logits
```

---

## 9. 代码位置速查

| 组件 | 文件 | 关键类/函数 |
|-----|------|------------|
| Llama 模型 | `vllm/model_executor/models/llama.py` | `LlamaForCausalLM` |
| Transformer 层 | `vllm/model_executor/models/llama.py` | `LlamaDecoderLayer` |
| Self-Attention | `vllm/model_executor/models/llama.py` | `LlamaAttention` |
| MLP | `vllm/model_executor/models/llama.py` | `LlamaMLP` |
| Attention 后端 | `vllm/model_executor/layers/attention/attention.py` | `Attention` |
| 旋转位置编码 | `vllm/model_executor/layers/rotary_embedding/__init__.py` | `get_rope()` |
| 并行线性层 | `vllm/model_executor/layers/linear.py` | `*ParallelLinear` |
| RMSNorm | `vllm/model_executor/layers/layernorm.py` | `RMSNorm` |

---

## 导航

- 上一篇：[Executor 与 Worker](02-executor-worker.md)
- 下一篇：[采样过程分析](04-sampling.md)
- [返回目录](/vllm/00-overview/overview/)
