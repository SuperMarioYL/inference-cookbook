---
title: "Chunked Prefill"
weight: 2
---

## 分块改变哪段工作占用当前轮次

一条提示还有 160 个 token 待算，若每轮最多分给它 64 个位置，三块依次是 64、64、32。其他请求也占用预算时，还可以再拆小。比较分块前后，观察长请求的首 token 延迟和已有请求的解码间隔。 [分块与批次参数](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)

## 长上下文挑战

### 问题场景

当 Prefill 请求包含大量 tokens 时：

```mermaid
graph TB
    subgraph problem["长 Prefill 问题"]
        LONG["长 Prefill 请求<br/>10K tokens"]
        DECODE["Decode 请求<br/>等待中..."]
        GPU["GPU"]

        LONG -->|"独占"| GPU
        DECODE -.->|"等待"| GPU
    end
```

**问题**：
- 长 Prefill 独占 GPU
- Decode 请求被饿死
- TTFT 延迟高
- ITL 不稳定

### 延迟分析

```
传统方式:
- 长 Prefill: 1000ms
- 等待的 Decode: +1000ms 延迟

用户体验:
- 正在生成的回答突然卡顿
- 新请求等待时间长
```

---

## Chunked Prefill 原理

### 一个长提示和一个正在生成的请求

P 有 160 个尚未计算的位置，D 已在逐步生成。图示一种允许 D 在两个分块之间推进的调度顺序。

{{< mechanism "sglang-prefill-turns" >}}

若还有 10,240 个提示位置，每块最多 2,048 个位置且设备空间充足，单个请求分五轮可以处理完。插入解码请求时，五个预填充轮次在时间轴上可以相隔更远。

## SGLang 实现

### 请求分块

**关键文件**：`python/sglang/srt/managers/scheduler.py`

```python
class PrefillAdder:
    def add_one_req(self, req: Req) -> AddReqResult:
        # 检查是否需要分块
        if req.extend_input_len > self.chunked_prefill_size:
            # 只处理 chunk_size 个 tokens
            req.is_chunked = req.extend_input_len - self.chunked_prefill_size
            req.extend_input_len = self.chunked_prefill_size

        # 扣减预算
        self.rem_chunk_tokens -= req.extend_input_len

        return AddReqResult.CONTINUE

    def add_chunked_req(self, req: Req) -> Optional[Req]:
        """处理分块请求的后续部分"""
        # 计算这一轮可以处理的 tokens
        num_tokens = min(req.is_chunked, self.rem_chunk_tokens)

        req.extend_input_len = num_tokens
        req.is_chunked -= num_tokens

        if req.is_chunked > 0:
            return req  # 还有剩余，下一轮继续
        else:
            return None  # 分块完成
```

### 分块状态管理

```python
class Req:
    def __init__(self, ...):
        # 分块状态
        self.is_chunked: int = 0        # 剩余待处理的 tokens
        self.chunk_offset: int = 0       # 当前处理的偏移

    def init_next_round_input(self, tree_cache):
        """初始化下一轮的输入"""
        if self.is_chunked > 0:
            # 分块模式：从上次结束位置继续
            start = self.chunk_offset
            end = start + self.extend_input_len
            self.input_ids = self.origin_input_ids[start:end]
            self.chunk_offset = end
        else:
            # 正常模式
            self.input_ids = self.origin_input_ids[self.prefix_len:]
```

### 调度器处理

```python
def _get_new_batch_prefill_raw(self):
    adder = PrefillAdder(
        chunked_prefill_size=self.chunked_prefill_size,
        # ...
    )

    # 1. 优先处理正在进行的分块请求
    if self.chunked_req is not None:
        self.chunked_req.init_next_round_input()
        self.chunked_req = adder.add_chunked_req(self.chunked_req)

    # 2. 添加新请求
    for req in self.waiting_queue:
        res = adder.add_one_req(req)
        if res != AddReqResult.CONTINUE:
            break

    # 3. 记录新的分块请求
    for req in adder.can_run_list:
        if req.is_chunked > 0:
            self.chunked_req = req
            break
```

---

## 混合模式调度

### Mixed ForwardMode

当批次同时包含 Prefill 和 Decode 请求时：

```mermaid
graph TB
    subgraph mixed["混合模式"]
        P1["Prefill Chunk<br/>1K tokens"]
        D1["Decode Req 1<br/>1 token"]
        D2["Decode Req 2<br/>1 token"]
        D3["Decode Req 3<br/>1 token"]

        BATCH["混合批次"]
        P1 --> BATCH
        D1 --> BATCH
        D2 --> BATCH
        D3 --> BATCH

        BATCH --> GPU["单次 GPU 前向"]
    end
```

### 实现细节

```python
class ForwardMode(IntEnum):
    EXTEND = 1    # 纯 Prefill
    DECODE = 2    # 纯 Decode
    MIXED = 3     # 混合模式

def get_forward_mode(batch: ScheduleBatch) -> ForwardMode:
    has_extend = any(req.extend_input_len > 1 for req in batch.reqs)
    has_decode = any(req.extend_input_len == 1 for req in batch.reqs)

    if has_extend and has_decode:
        return ForwardMode.MIXED
    elif has_extend:
        return ForwardMode.EXTEND
    else:
        return ForwardMode.DECODE
```

### 注意力计算调整

```python
def mixed_attention_forward(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    seq_lens: List[int],
    extend_lens: List[int],
):
    """混合模式注意力计算"""
    # 区分 Prefill 和 Decode tokens
    prefill_mask = extend_lens > 1
    decode_mask = extend_lens == 1

    # Prefill tokens: 自注意力
    # Decode tokens: 与所有历史 KV 做注意力

    # 使用 FlashInfer 等优化后端处理
    # ...
```

---

## 调度优化

### Token 预算管理

```python
class PrefillAdder:
    def __init__(self, ...):
        # 总 token 预算
        self.rem_total_tokens = available_tokens

        # Prefill token 预算（限制单批次 Prefill）
        self.rem_input_tokens = max_prefill_tokens

        # 分块预算（单个分块大小）
        self.rem_chunk_tokens = chunked_prefill_size

    def budget_state(self) -> AddReqResult:
        if self.rem_total_tokens <= 0:
            return AddReqResult.NO_TOKEN
        if self.rem_input_tokens <= 0:
            return AddReqResult.NO_TOKEN
        if self.rem_chunk_tokens <= 0:
            return AddReqResult.NO_TOKEN
        return AddReqResult.CONTINUE
```

### 平衡策略

```mermaid
graph TB
    subgraph balance["平衡 Prefill 和 Decode"]
        PREFILL["Prefill 请求"]
        DECODE["Decode 请求"]

        CHECK{"Decode 饿死?"}
        LIMIT["限制 Prefill 预算"]
        NORMAL["正常调度"]

        CHECK -->|"是"| LIMIT
        CHECK -->|"否"| NORMAL

        PREFILL --> CHECK
        DECODE --> CHECK
    end
```

```python
def get_prefill_budget(self) -> int:
    """动态计算 Prefill 预算"""
    # 基础预算
    budget = self.max_prefill_tokens

    # 如果有 Decode 请求等待，减少预算
    if self.running_batch.batch_size() > 0:
        decode_wait_time = self._get_decode_wait_time()
        if decode_wait_time > self.decode_max_wait:
            budget = min(budget, self.chunked_prefill_size)

    return budget
```

---

## 配置参数

### 服务器参数

```python
@dataclass
class ServerArgs:
    # 分块 Prefill
    chunked_prefill_size: int = 8192  # 单个分块大小
    max_prefill_tokens: int = 16384   # 单批次最大 Prefill tokens

    # -1 表示禁用分块
    # chunked_prefill_size: int = -1
```

### 启动命令

```bash
# 启用 Chunked Prefill（默认）
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --chunked-prefill-size 8192

# 禁用 Chunked Prefill
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --chunked-prefill-size -1
```

---

## 性能影响

### 延迟对比

```
场景: 10K token Prefill + 并发 Decode

无 Chunked Prefill:
- Prefill TTFT: 500ms
- Decode ITL: 50ms (正常) -> 500ms (被阻塞)

有 Chunked Prefill (2K chunks):
- Prefill TTFT: 600ms (略增)
- Decode ITL: 50ms -> 60ms (轻微增加)
```

### 吞吐量对比

| 场景 | 无 Chunked | 有 Chunked | 差异 |
|------|-----------|-----------|------|
| 短请求为主 | 100% | 98% | -2% |
| 长请求为主 | 100% | 95% | -5% |
| 混合负载 | 100% | 110% | +10% |

**结论**：
- 单纯 Prefill 略有开销
- 混合负载时吞吐量提升
- ITL 稳定性大幅改善

---

## 实践建议

### 1. 分块大小选择

```python
# 短上下文服务
chunked_prefill_size = 4096

# 长上下文服务
chunked_prefill_size = 8192

# 延迟敏感服务
chunked_prefill_size = 2048
```

### 2. 监控指标

```python
# 监控分块状态
metrics = {
    "chunked_reqs_count": len([r for r in reqs if r.is_chunked > 0]),
    "avg_chunk_rounds": avg_chunk_rounds,
    "decode_wait_time_p99": decode_wait_p99,
}
```

### 3. 调优流程

```mermaid
graph TB
    MONITOR["监控 ITL"]
    CHECK{"ITL 稳定?"}
    INCREASE["增大分块"]
    DECREASE["减小分块"]
    DONE["完成"]

    MONITOR --> CHECK
    CHECK -->|"不稳定"| DECREASE
    CHECK -->|"稳定"| INCREASE
    DECREASE --> MONITOR
    INCREASE --> MONITOR
```

---
