---
title: "张量并行"
weight: 1
---

## 把一个层切到多个 rank 后如何验证

TP 把同一层的张量工作分布到多个 rank，并在层内通信。只增加卡数不会自动消除通信。先核对各 rank 的模型分片、设备可见性和 collective，再将单请求延迟与可支持的批量分别对比。 [0.5.23 并行参数](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)

## 张量并行原理

### 为什么需要张量并行

```mermaid
graph TB
    subgraph problem["单 GPU 限制"]
        MODEL["70B 模型<br/>140GB FP16"]
        GPU["单 GPU<br/>80GB"]

        MODEL -.->|"无法装载"| GPU
    end
```

**解决方案**：将模型分割到多个 GPU 上。

### TP vs DP vs PP

```mermaid
graph TB
    subgraph parallel["并行方式"]
        TP["张量并行 (TP)<br/>切分模型层内"]
        DP["数据并行 (DP)<br/>复制模型"]
        PP["流水线并行 (PP)<br/>切分模型层间"]
    end
```

| 类型 | 切分方式 | 通信模式 | 适用场景 |
|------|----------|----------|----------|
| TP | 层内切分 | All-Reduce | 单机多卡 |
| DP | 复制模型 | All-Gather | 大批次 |
| PP | 层间切分 | Point-to-Point | 超大模型 |

---

## TP 实现原理

### 线性层切分

**列并行** (Column Parallel)：

```mermaid
graph TB
    subgraph col["列并行"]
        X["输入 X"]
        W1["W[:,0:n/2]<br/>GPU 0"]
        W2["W[:,n/2:n]<br/>GPU 1"]
        Y1["Y1"]
        Y2["Y2"]
        Y["[Y1, Y2]"]

        X --> W1 --> Y1
        X --> W2 --> Y2
        Y1 --> Y
        Y2 --> Y
    end
```

```python
# 列并行：输出维度切分
# GPU 0: Y1 = X @ W[:, :n//2]
# GPU 1: Y2 = X @ W[:, n//2:]
# 结果：Y = [Y1, Y2] (AllGather)
```

**行并行** (Row Parallel)：

{{< stateboard >}}
title: 行并行的两个局部和
rows: ["rank 0", "rank 1", "完整输出"]
steps:
  - title: "切开输入维度"
    detail: "演算 y=x·w，x=[2,3]、w=[4,5]。两卡各持有一个输入维度的权重。"
    cells: [["2×4"], ["3×5"], ["·"]]
    focus: [[0, 0], [1, 0]]
  - title: "各自计算"
    detail: "rank 0 得到 8，rank 1 得到 15，每个结果只覆盖本地维度。"
    cells: [["8"], ["15"], ["·"]]
    focus: [[0, 0], [1, 0]]
  - title: "组内求和"
    detail: "all-reduce 把局部贡献相加，两个 rank 均获得 23。"
    cells: [["23"], ["23"], ["23"]]
    focus: [[0, 0], [1, 0], [2, 0]]
{{< /stateboard >}}

```python
# 行并行：输入维度切分
# GPU 0: Y1 = X1 @ W[:n//2, :]
# GPU 1: Y2 = X2 @ W[n//2:, :]
# 结果：Y = Y1 + Y2 (AllReduce)
```

### 注意力层并行

```mermaid
graph TB
    subgraph attn_tp["注意力 TP"]
        INPUT["输入"]

        Q0["Q Heads 0-3<br/>GPU 0"]
        Q1["Q Heads 4-7<br/>GPU 1"]

        K0["KV Heads 0-1<br/>GPU 0"]
        K1["KV Heads 2-3<br/>GPU 1"]

        ATTN0["Attention 0"]
        ATTN1["Attention 1"]

        OUT["AllReduce"]

        INPUT --> Q0 --> ATTN0
        INPUT --> Q1 --> ATTN1
        K0 --> ATTN0
        K1 --> ATTN1
        ATTN0 --> OUT
        ATTN1 --> OUT
    end
```

---

## SGLang 实现

### 配置

```bash
# 2 GPU 张量并行
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 2

# 4 GPU 张量并行
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4
```

### 初始化

```python
def init_torch_distributed(self):
    """初始化分布式环境"""
    # 设置设备
    torch.cuda.set_device(self.gpu_id)

    # 初始化进程组
    init_distributed_environment(
        world_size=self.tp_size,
        rank=self.tp_rank,
        backend="nccl",
    )

    # 创建 TP 通信组
    self.tp_group = create_tensor_parallel_group(self.tp_size)
```

### 权重切分

```python
def load_weights_tp(self, weights):
    """加载 TP 切分的权重"""
    for name, param in weights.items():
        if "q_proj" in name or "k_proj" in name or "v_proj" in name:
            # QKV 列并行
            param = split_tensor_along_dim(
                param, dim=0, rank=self.tp_rank, world_size=self.tp_size
            )
        elif "o_proj" in name:
            # Output 行并行
            param = split_tensor_along_dim(
                param, dim=1, rank=self.tp_rank, world_size=self.tp_size
            )
        elif "gate_proj" in name or "up_proj" in name:
            # MLP 列并行
            param = split_tensor_along_dim(
                param, dim=0, rank=self.tp_rank, world_size=self.tp_size
            )
        elif "down_proj" in name:
            # MLP 行并行
            param = split_tensor_along_dim(
                param, dim=1, rank=self.tp_rank, world_size=self.tp_size
            )

        self.model.load_state_dict({name: param}, strict=False)
```

### 前向计算

```python
class TPLinear(nn.Module):
    """张量并行线性层"""

    def __init__(self, in_features, out_features, parallel_mode, tp_group):
        self.parallel_mode = parallel_mode  # "column" or "row"
        self.tp_group = tp_group

    def forward(self, x):
        if self.parallel_mode == "column":
            # 列并行：各自计算，最后 AllGather
            out = F.linear(x, self.weight)
            out = all_gather(out, self.tp_group)
        else:
            # 行并行：各自计算，最后 AllReduce
            out = F.linear(x, self.weight)
            out = all_reduce(out, self.tp_group)
        return out
```

---

## 通信优化

### 通信与计算重叠

```mermaid
sequenceDiagram
    participant GPU0
    participant GPU1
    participant NCCL

    Note over GPU0,NCCL: 重叠通信

    GPU0->>GPU0: 计算 Layer N
    GPU1->>GPU1: 计算 Layer N

    par 并行执行
        GPU0->>NCCL: AllReduce Layer N
        GPU1->>NCCL: AllReduce Layer N
        GPU0->>GPU0: 计算 Layer N+1
        GPU1->>GPU1: 计算 Layer N+1
    end
```

```python
# 使用异步通信
handle = torch.distributed.all_reduce(
    tensor, op=ReduceOp.SUM, group=tp_group, async_op=True
)
# 继续计算
result = compute_next_layer(...)
# 等待通信完成
handle.wait()
```

### 减少通信量

```python
# 融合 AllReduce
# 多个小张量合并为一个大张量通信
def fused_all_reduce(tensors, group):
    # 打包
    flat = torch.cat([t.flatten() for t in tensors])
    # 单次 AllReduce
    torch.distributed.all_reduce(flat, group=group)
    # 解包
    offset = 0
    for t in tensors:
        t.copy_(flat[offset:offset + t.numel()].view_as(t))
        offset += t.numel()
```

---

## KV Cache 管理

### TP 下的 KV Cache

```python
class TPKVCache:
    """张量并行 KV Cache"""

    def __init__(self, num_heads, head_dim, tp_rank, tp_size):
        # 每个 GPU 只存储部分 heads
        self.local_num_heads = num_heads // tp_size
        self.head_start = tp_rank * self.local_num_heads

        # 本地 KV Cache
        self.k_cache = torch.zeros(
            (num_layers, max_tokens, self.local_num_heads, head_dim)
        )
        self.v_cache = torch.zeros(
            (num_layers, max_tokens, self.local_num_heads, head_dim)
        )
```

---

## 性能分析

### TP 开销

```
单次 AllReduce 延迟:
- NVLink: ~5-10μs
- PCIe: ~50-100μs

通信量:
- 每层: 2 * hidden_size * batch_size * seq_len * dtype_size
- 例: 2 * 8192 * 32 * 1 * 2 = 1MB
```

### 扩展效率

| TP Size | 理想加速 | 实际加速 | 效率 |
|---------|----------|----------|------|
| 2 | 2x | 1.9x | 95% |
| 4 | 4x | 3.6x | 90% |
| 8 | 8x | 6.4x | 80% |

### 选择 TP Size

```mermaid
graph TB
    START["选择 TP Size"]

    Q1{"模型大小?"}
    Q2{"GPU 连接?"}

    TP1["TP=1"]
    TP2["TP=2"]
    TP4["TP=4"]
    TP8["TP=8"]

    START --> Q1
    Q1 -->|"<15GB"| TP1
    Q1 -->|"15-40GB"| Q2
    Q1 -->|">40GB"| TP4

    Q2 -->|"NVLink"| TP2
    Q2 -->|"PCIe"| TP1
```

---

## 配置建议

### 启动参数

```bash
# 基本 TP
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4

# 指定 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4

# 多节点（使用 torchrun）
torchrun --nproc_per_node=8 --nnodes=2 \
    -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-405B-Instruct \
    --tensor-parallel-size 16
```

### 模型与 TP Size 匹配

| 模型 | 推荐 TP Size | GPU 需求 |
|------|-------------|----------|
| 7-8B | 1 | 1x 24GB |
| 13B | 1-2 | 1-2x 24GB |
| 34B | 2-4 | 2-4x 24GB |
| 70B | 4-8 | 4-8x 40GB |
| 405B | 8-16 | 8-16x 80GB |

---

## 张量并行的启动参数

| 参数 | 作用 |
|------|------|
| `--tensor-parallel-size` | TP 大小 |
| `CUDA_VISIBLE_DEVICES` | 指定 GPU |
