---
title: "连续批处理"
weight: 1
---

## 从两轮观察同一条运行请求

A 再生成一个 token 就结束，B 还会继续。两者一同执行后，引擎回收 A 的位置，新请求可以进入下一轮。0.5.23 的普通循环从 `ingest_requests` 走到 `get_next_batch_to_run` 和 `process_batch_result`。 [0.5.23 调度循环](https://github.com/sgl-project/sglang/blob/v0.5.23/python/sglang/srt/managers/scheduler.py)

## 传统静态批处理

### 工作方式

传统批处理等待所有请求完成后才处理下一批：

```mermaid
graph TB
    subgraph static["静态批处理"]
        B1["Batch 1: [R1, R2, R3]"]
        WAIT1["等待所有完成"]
        B2["Batch 2: [R4, R5, R6]"]
        WAIT2["等待所有完成"]

        B1 --> WAIT1 --> B2 --> WAIT2
    end
```

### 问题分析

```mermaid
sequenceDiagram
    participant R1 as 请求 1 (短)
    participant R2 as 请求 2 (中)
    participant R3 as 请求 3 (长)
    participant GPU

    Note over R1,GPU: 静态批处理
    R1->>GPU: 开始
    R2->>GPU: 开始
    R3->>GPU: 开始

    R1-->>GPU: 完成 (等待)
    Note over R1: 空闲等待

    R2-->>GPU: 完成 (等待)
    Note over R1,R2: 空闲等待

    R3-->>GPU: 完成
    Note over R1,R3: 批次结束
```

**问题**：
- 短请求等待长请求
- GPU 利用率低
- 延迟不可预测
- 吞吐量受限

---

## 连续批处理原理

### 核心思想

请求完成后立即释放资源，新请求可以加入：

```mermaid
graph TB
    subgraph continuous["连续批处理"]
        T1["时刻 1: [R1, R2, R3]"]
        T2["时刻 2: [R1, R2, R4]<br/>R3 完成，R4 加入"]
        T3["时刻 3: [R2, R4, R5]<br/>R1 完成，R5 加入"]

        T1 --> T2 --> T3
    end
```

### 动态调整

{{< stateboard >}}
title: 三个执行位置怎样换人
rows: ["槽 0", "槽 1", "槽 2"]
steps:
  - title: "首轮"
    detail: "R1、R2、R3 进入同一批，R4、R5 在等待。"
    cells: [["R1"], ["R2"], ["R3"]]
    focus: [[0, 0], [1, 0], [2, 0]]
  - title: "R1 先结束"
    detail: "下一次选批时 R4 占用腾出的执行位置，R2 与 R3 继续。"
    cells: [["R4"], ["R2"], ["R3"]]
    focus: [[0, 0]]
  - title: "R2 到达停止条件"
    detail: "下一轮把 R5 加入，另外两个请求沿用各自的历史 KV。"
    cells: [["R4"], ["R5"], ["R3"]]
    focus: [[1, 0]]
{{< /stateboard >}}

---

## Prefill vs Decode

### 两个阶段的特点

```mermaid
graph LR
    subgraph prefill["Prefill 阶段"]
        P_IN["完整输入序列"]
        P_COMP["并行计算"]
        P_KV["生成 KV Cache"]
        P_OUT["第一个 Token"]

        P_IN --> P_COMP --> P_KV --> P_OUT
    end

    subgraph decode["Decode 阶段"]
        D_IN["单个 Token"]
        D_LOAD["加载 KV Cache"]
        D_COMP["注意力计算"]
        D_OUT["下一个 Token"]

        D_IN --> D_LOAD --> D_COMP --> D_OUT
    end
```

### 性能特征对比

| 特征 | Prefill | Decode |
|------|---------|--------|
| 计算类型 | 计算密集 | 内存带宽密集 |
| 每 Token 计算 | 高（全序列注意力） | 低（单 Token） |
| GPU 利用率 | 高 | 低（需批量化） |
| 批次大小影响 | 线性增长 | 亚线性增长 |

### 计算量分析

**Prefill**：
```
计算量 ∝ seq_len² × hidden_size
```

**Decode**：
```
计算量 ∝ seq_len × hidden_size
内存访问 ∝ seq_len × kv_cache_size
```

---

## SGLang 连续批处理实现

### 批次管理

**关键文件**：`python/sglang/srt/managers/scheduler.py`

```python
class Scheduler:
    def __init__(self, ...):
        # 等待队列
        self.waiting_queue: List[Req] = []

        # 运行批次
        self.running_batch: ScheduleBatch = ScheduleBatch(
            reqs=[], batch_is_full=False
        )

    def event_loop_normal(self):
        while True:
            # 1. 接收新请求
            recv_reqs = self.recv_requests()
            self.process_input_requests(recv_reqs)

            # 2. 获取下一个批次
            batch = self.get_next_batch_to_run()

            # 3. 执行批次
            if batch:
                result = self.run_batch(batch)
                self.process_batch_result(batch, result)
```

### 批次组装

```python
def get_next_batch_to_run(self) -> Optional[ScheduleBatch]:
    # 1. 合并已完成 Prefill 的请求到 running_batch
    if self.last_batch and self.last_batch.forward_mode.is_extend():
        self.running_batch.merge_batch(self.last_batch)

    # 2. 尝试获取新的 Prefill 批次
    new_batch = self.get_new_batch_prefill()

    # 3. 选择执行
    if new_batch is not None:
        return new_batch  # 优先执行 Prefill
    elif not self.running_batch.is_empty():
        # 更新并执行 Decode
        self.running_batch = self.update_running_batch(self.running_batch)
        return self.running_batch

    return None
```

### 请求状态流转

```mermaid
stateDiagram-v2
    [*] --> Waiting: 新请求到达
    Waiting --> Prefilling: 加入 Prefill 批次
    Prefilling --> Decoding: Prefill 完成
    Decoding --> Decoding: 继续生成
    Decoding --> Finished: 生成完成
    Finished --> [*]
```

---

## 动态批次更新

### 请求完成处理

```python
def process_batch_result_decode(self, batch, result):
    next_token_ids = result.next_token_ids.tolist()

    finished_reqs = []
    for req, next_token_id in zip(batch.reqs, next_token_ids):
        # 添加新 token
        req.output_ids.append(next_token_id)

        # 检查完成条件
        req.check_finished()

        if req.finished():
            finished_reqs.append(req)
            # 释放 KV Cache
            release_kv_cache(req, self.tree_cache)

    # 从运行批次中移除已完成的请求
    self.running_batch.remove_reqs(finished_reqs)
```

### 新请求加入

```python
def update_running_batch(self, batch: ScheduleBatch) -> ScheduleBatch:
    """更新运行批次，移除已完成请求"""
    # 移除已完成的请求
    remaining_reqs = [req for req in batch.reqs if not req.finished()]

    if not remaining_reqs:
        return ScheduleBatch(reqs=[], batch_is_full=False)

    # 更新批次
    batch.reqs = remaining_reqs
    batch.batch_is_full = len(remaining_reqs) >= self.max_running_requests

    return batch
```

---

## 批次大小优化

### 动态批次大小

```mermaid
graph TB
    subgraph dynamic["动态批次大小"]
        CHECK["检查资源"]
        CALC["计算可用空间"]
        LIMIT["应用限制"]
        SIZE["确定批次大小"]

        CHECK --> CALC --> LIMIT --> SIZE
    end
```

```python
def get_num_allocatable_reqs(self, running_bs: int) -> int:
    """计算可分配的请求数"""
    # 获取可用 token 空间
    available_tokens = self.token_to_kv_pool.available_size()

    # 减去当前运行批次需要的空间
    available_tokens -= running_bs * self.new_token_ratio

    # 计算可容纳的请求数
    max_new_reqs = available_tokens // self.avg_tokens_per_req

    # 应用最大限制
    return min(max_new_reqs, self.max_running_requests - running_bs)
```

### 批次大小限制

```python
# 配置参数
max_running_requests: int  # 最大并发请求
max_prefill_tokens: int    # 最大 Prefill token 数
max_batch_size: int        # 最大批次大小
```

---

## 吞吐量优化

### 影响因素

```mermaid
graph TB
    subgraph factors["吞吐量影响因素"]
        BS["批次大小"]
        MEM["内存利用"]
        CACHE["缓存命中"]
        SCHED["调度效率"]

        BS --> THROUGHPUT["吞吐量"]
        MEM --> THROUGHPUT
        CACHE --> THROUGHPUT
        SCHED --> THROUGHPUT
    end
```

### 优化策略

**1. 最大化批次大小**
```python
# 尽可能填满批次
while len(can_run_list) < max_batch_size:
    if not waiting_queue:
        break
    req = waiting_queue.pop(0)
    can_run_list.append(req)
```

**2. 平衡 Prefill 和 Decode**
```python
# 限制 Prefill token 数，避免饿死 Decode
if prefill_tokens > max_prefill_tokens:
    break
```

**3. 前缀缓存复用**
```python
# 优先调度有缓存命中的请求
waiting_queue.sort(key=lambda r: -r.prefix_len)
```

---

## 性能对比

### 静态 vs 连续批处理

```
场景: 100 个请求，输出长度 100-1000 tokens

静态批处理:
- 平均延迟: 高（等待最长请求）
- 吞吐量: 低（资源浪费）
- GPU 利用率: ~30-50%

连续批处理:
- 平均延迟: 低（即完即释放）
- 吞吐量: 高（资源复用）
- GPU 利用率: ~70-90%
```

### 吞吐量提升

| 场景 | 静态批处理 | 连续批处理 | 提升 |
|------|-----------|-----------|------|
| 均匀长度 | 100 tok/s | 150 tok/s | 1.5x |
| 混合长度 | 60 tok/s | 180 tok/s | 3x |
| 高并发 | 80 tok/s | 200 tok/s | 2.5x |

---

## 实现细节

### ScheduleBatch 数据结构

```python
@dataclass
class ScheduleBatch:
    reqs: List[Req]           # 请求列表
    batch_is_full: bool       # 批次是否已满

    def batch_size(self) -> int:
        return len(self.reqs)

    def is_empty(self) -> bool:
        return len(self.reqs) == 0

    def merge_batch(self, other: "ScheduleBatch"):
        """合并另一个批次"""
        self.reqs.extend(other.reqs)
        self.batch_is_full = len(self.reqs) >= self.max_size

    def remove_reqs(self, to_remove: List[Req]):
        """移除请求"""
        remove_set = set(to_remove)
        self.reqs = [r for r in self.reqs if r not in remove_set]
```

### ForwardMode 转换

```python
def get_forward_mode(self) -> ForwardMode:
    """根据批次状态确定前向模式"""
    has_prefill = any(req.is_prefill for req in self.reqs)
    has_decode = any(not req.is_prefill for req in self.reqs)

    if has_prefill and has_decode:
        return ForwardMode.MIXED
    elif has_prefill:
        return ForwardMode.EXTEND
    else:
        return ForwardMode.DECODE
```

---
