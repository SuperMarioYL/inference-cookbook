---
title: "GSA 算法详解"
weight: 4
---


> **前置要求**: [UcmSparseBase 接口](./02-base-interface.md)

---

## 概述

GSA（Gather-Scatter Attention）按选中的位置收集 KV 块。预取是否来得及，取决于块何时选出、数据放在哪里，以及计算端何时要用到它们。

---

v0.9.1 保留了 [GSA 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/gsa.py)，但 [Sparse 工厂](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/factory.py) 中 `GSA` 的注册处于注释状态。以下用这份源码解释检索与预取。按本版特性表复现实验时，应选择表中列出的设备路径 [GSAOnDevice](./05-gsa-on-device.md)。

## 1. 算法原理

### 1.1 核心思想

```mermaid
graph TB
    subgraph gsa_idea["GSA 核心思想"]
        Prefetch["预取引擎<br/>Prefetch Engine"]
        Predict["预测相关 Blocks"]
        Load["提前加载到 GPU"]
        Compute["零等待注意力计算"]

        Prefetch --> Predict
        Predict --> Load
        Load --> Compute
    end
```

GSA 的核心创新在于**预测性预取**：在模型执行前，预测哪些 KV Block 将被使用，提前加载到 GPU 内存。

### 1.2 与 ESA 的区别

| 维度 | ESA | GSA |
|------|-----|-----|
| 检索时机 | 运行时动态检索 | 预取引擎提前预测 |
| 延迟特性 | 检索引入延迟 | 零等待（预取完成时） |
| 内存使用 | 按需加载 | 预取缓冲区 |
| 适用场景 | 通用场景 | 对延迟敏感的场景 |
### 1.3 算法流程
{{< trace >}}
title: "GSA 怎样把候选预取到 worker"
actors: ["调度侧", "预取队列", "存储", "worker"]
steps:
  - from: 0
    to: 1
    label: "提交计划"
    detail: "调度侧根据本步请求和历史状态构造待预取的块。"
  - from: 1
    to: 2
    label: "取回候选"
    detail: "预取任务把所需的数据从存储取回。"
  - from: 2
    to: 3
    label: "供数"
    detail: "worker 按准备好的块表和完成状态取得数据。"
  - from: 3
    to: 3
    label: "执行本层"
    detail: "本层根据实际可用的候选执行注意力，后续步骤继续更新状态。"
{{< /trace >}}

---

## 2. 核心组件

### 2.1 GSA 类结构

[v0.9.1 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/gsa.py)

```python
class GSA(UcmSparseBase):
    """Gather-Scatter Attention 实现"""

    def __init__(self, role: UcmSparseRole, config: dict):
        super().__init__(role, config)
        # 算法参数
        self.sparse_ratio = config.get('sparse_ratio', 0.3)
        self.local_window_sz = config.get('local_window_sz', 2)
        self.init_window_sz = config.get('init_window_sz', 1)
        self.min_blocks = config.get('min_blocks', 4)
        # 请求状态管理
        self._request_stats: Dict[str, GSAReqStat] = {}

        # Worker 侧组件
        if role == UcmSparseRole.WORKER:
            self.prefetch_engine = PrefetchEngine(config)
            self.sparse_attention_kernel = GSASparseKernel(config)
```

### 2.2 请求状态

[v0.9.1 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/gsa.py)

```python
@dataclass
class GSAReqStat:
    """GSA 请求状态追踪"""
    request_id: str
    index_in_batch: int
    # Token 计数
    num_scheduled_tokens: int
    num_computed_tokens: int
    # Block 信息
    vllm_block_ids: List[int]      # vLLM 分配的 Block ID
    ucm_block_hashes: List[str]    # UCM 内容哈希
    # 选择结果
    selected_blocks: List[int]     # 选中的 Block 索引

    # 预取状态
    prefetch_task: Optional[Task] = None
    prefetch_ready: bool = False
```
---
## 3. Prefetch Engine

### 3.1 架构设计

[v0.9.1 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/prefetch/prefetch_engine.py)

```mermaid
graph TB
    subgraph prefetch_engine["Prefetch Engine 架构"]
        subgraph input["输入"]
            Plans["预取计划队列"]
        end

        subgraph core["核心组件"]
            Scheduler["优先级调度器"]
            Workers["Worker 池"]
            Cache["预取缓存"]
        end

        subgraph output["输出"]
            GPU["GPU 内存"]
            Notify["完成通知"]
        end

        Plans --> Scheduler
        Scheduler --> Workers
        Workers --> Cache
        Cache --> GPU
        Workers --> Notify
    end
```

### 3.2 实现

```python
class PrefetchEngine:
    """GSA 预取引擎"""

    def __init__(self, config: dict):
        self.num_workers = config.get('prefetch_workers', 4)
        self.max_pending = config.get('max_pending_prefetch', 64)

        # 线程池
        self.executor = ThreadPoolExecutor(max_workers=self.num_workers)

        # 任务队列（优先级队列）
        self.task_queue = PriorityQueue()

        # 状态追踪
        self.pending_tasks: Dict[str, PrefetchTask] = {}
        self.completed_tasks: Dict[str, torch.Tensor] = {}

        # 启动调度线程
        self.scheduler_thread = Thread(target=self._scheduler_loop)
        self.scheduler_thread.start()

    def submit_prefetch(
        self,
        request_id: str,
        block_hashes: List[str],
        priority: int = 0
    ) -> PrefetchHandle:
        """提交预取任务"""
        task = PrefetchTask(
            request_id=request_id,
            block_hashes=block_hashes,
            priority=priority,
            submit_time=time.time()
        )

        self.task_queue.put((priority, task))
        self.pending_tasks[request_id] = task

        return PrefetchHandle(request_id, self)

    def wait_for_completion(
        self,
        request_id: str,
        timeout: float = None
    ) -> bool:
        """等待预取完成"""
        task = self.pending_tasks.get(request_id)
        if task is None:
            return True  # 已完成

        return task.event.wait(timeout)

    def get_prefetched_kv(self, request_id: str) -> Optional[torch.Tensor]:
        """获取预取的 KV"""
        return self.completed_tasks.get(request_id)

    def _scheduler_loop(self):
        """调度循环"""
        while not self.shutdown:
            try:
                priority, task = self.task_queue.get(timeout=0.1)
                self.executor.submit(self._do_prefetch, task)
            except Empty:
                continue

    def _do_prefetch(self, task: PrefetchTask):
        """执行预取"""
        try:
            # 从 Store 加载
            kv_data = self._load_blocks(task.block_hashes)

            # 保存到完成列表
            self.completed_tasks[task.request_id] = kv_data

            # 通知完成
            task.event.set()

        except Exception as e:
            task.error = e
            task.event.set()
```

### 3.3 优先级策略

```python
class PrefetchPriority:
    """预取优先级策略"""

    URGENT = 0      # 即将执行的请求
    HIGH = 1        # 下一批次的请求
    NORMAL = 2      # 普通请求
    LOW = 3         # 投机性预取

    @staticmethod
    def compute_priority(
        request: Request,
        current_step: int,
        queue_position: int
    ) -> int:
        """计算请求的预取优先级"""
        # 即将执行
        if queue_position == 0:
            return PrefetchPriority.URGENT

        # 下一批次
        if queue_position <= 4:
            return PrefetchPriority.HIGH

        # 普通请求
        if queue_position <= 16:
            return PrefetchPriority.NORMAL

        return PrefetchPriority.LOW
```

---
## 4. 稀疏注意力计算
### 4.1 Gather-Scatter 模式
{{< stateboard >}}
title: "按索引收集不连续的历史数据"
rows: ["原位置", "候选索引", "紧凑数据"]
steps:
  - title: "原始布局"
    detail: "四个 KV 块保存在原来的逻辑顺序中。"
    cells:
      - ["K0", "K1", "K2", "K3"]
      - ["·", "·", "·", "·"]
      - ["·", "·", "·", "·"]
    focus: [[0, 0], [0, 1], [0, 2], [0, 3]]
  - title: "选出位置"
    detail: "用 1、3 举例；索引引用原块表中的两项。"
    cells:
      - ["K0", "K1", "K2", "K3"]
      - ["1", "3", "·", "·"]
      - ["·", "·", "·", "·"]
    focus: [[1, 0], [1, 1]]
  - title: "收集候选"
    detail: "按选择次序得到 K1、K3，相关注意力计算使用这个较小集合。"
    cells:
      - ["K0", "K1", "K2", "K3"]
      - ["1", "3", "·", "·"]
      - ["K1", "K3", "·", "·"]
    focus: [[2, 0], [2, 1]]
{{< /stateboard >}}

### 4.2 Kernel 实现

```python
class GSASparseKernel:
    """GSA 稀疏注意力 Kernel"""

    def __init__(self, config: dict):
        self.head_dim = config.get('head_dim', 128)
        self.num_heads = config.get('num_heads', 32)
        self.block_size = config.get('block_size', 16)
    def forward(
        self,
        query: torch.Tensor,           # [batch, heads, seq, dim]
        key_cache: torch.Tensor,       # [batch, blocks, block_size, heads, dim]
        value_cache: torch.Tensor,     # [batch, blocks, block_size, heads, dim]
        selected_indices: torch.Tensor, # [batch, num_selected]
        attention_mask: torch.Tensor   # [batch, seq, total_kv_len]
    ) -> torch.Tensor:
        """执行稀疏注意力"""
        batch_size = query.shape[0]
        num_selected = selected_indices.shape[1]
        # Gather: 收集选中的 KV
        selected_k = self._gather_kv(key_cache, selected_indices)
        selected_v = self._gather_kv(value_cache, selected_indices)
        # Compute: 稀疏注意力
        # selected_k: [batch, num_selected * block_size, heads, dim]
        scores = torch.einsum('bshd,bkhd->bshk', query, selected_k)
        scores = scores / math.sqrt(self.head_dim)

        # 应用 mask
        sparse_mask = self._build_sparse_mask(
            selected_indices, attention_mask
        )
        scores = scores.masked_fill(~sparse_mask, float('-inf'))

        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        # 加权求和
        output = torch.einsum('bshk,bkhd->bshd', attn_weights, selected_v)

        return output

    def _gather_kv(
        self,
        kv_cache: torch.Tensor,
        indices: torch.Tensor
    ) -> torch.Tensor:
        """收集选中的 KV"""
        batch_size = kv_cache.shape[0]
        num_selected = indices.shape[1]
        # 使用 index_select 收集
        gathered = []
        for b in range(batch_size):
            selected = kv_cache[b, indices[b]]  # [num_selected, block_size, heads, dim]
            gathered.append(selected)
        return torch.stack(gathered)  # [batch, num_selected, block_size, heads, dim]
```

---

## 5. Scheduler 侧流程

### 5.1 构建稀疏元数据

```python
class GSA(UcmSparseBase):
    def build_sparse_meta(
        self,
        requests: List[Request],
        scheduler_output: SchedulerOutput
    ) -> SparseMeta:
        """构建稀疏元数据（Scheduler 侧）"""
        meta = SparseMeta()
        for req in requests:
            # 获取或创建请求状态
            stat = self._get_or_create_stat(req)
            # 更新 Token 计数
            stat.num_scheduled_tokens = req.num_scheduled_tokens
            stat.num_computed_tokens = req.num_computed_tokens
            # 生成 Block 哈希
            stat.ucm_block_hashes = self._generate_block_hashes(req)

            # 执行 Block 选择
            selected = self._select_blocks(stat)
            stat.selected_blocks = selected

            # 添加到元数据
            meta.add_request(
                request_id=req.request_id,
                selected_blocks=selected,
                block_hashes=stat.ucm_block_hashes
            )
            # 提交预取（如果需要）
            if self._should_prefetch(stat):
                self._submit_prefetch(stat)
        return meta
    def _select_blocks(self, stat: GSAReqStat) -> List[int]:
        """选择要计算的 Blocks"""
        total_blocks = len(stat.ucm_block_hashes)
        # 计算需要选择的数量
        k = max(
            self.min_blocks,
            int(total_blocks * self.sparse_ratio)
        )
        k = min(k, total_blocks)

        # 始终包含的 Blocks
        sink_blocks = list(range(self.init_window_sz))
        local_blocks = list(range(
            max(0, total_blocks - self.local_window_sz),
            total_blocks
        ))
        # 计算剩余需要选择的数量
        required_blocks = set(sink_blocks + local_blocks)
        k_remaining = k - len(required_blocks)
        if k_remaining > 0:
            # 使用评分选择额外的 Blocks
            scores = self._compute_block_scores(stat)
            # 排除已选择的
            for idx in required_blocks:
                scores[idx] = float('-inf')
            # Top-K 选择
            _, top_indices = torch.topk(scores, k_remaining)
            selected = top_indices.tolist()
        else:
            selected = []
        return sorted(set(sink_blocks + selected + local_blocks))
```

### 5.2 Block 评分

```python
def _compute_block_scores(self, stat: GSAReqStat) -> torch.Tensor:
    """计算 Block 评分"""
    num_blocks = len(stat.ucm_block_hashes)
    scores = torch.zeros(num_blocks)

    # 基于历史访问模式
    if stat.request_id in self._access_history:
        history = self._access_history[stat.request_id]
        for idx, count in history.items():
            if idx < num_blocks:
                scores[idx] += count * 0.5
    # 基于位置（最近的 Block 更重要）
    position_weight = torch.linspace(0.1, 1.0, num_blocks)
    scores += position_weight
    return scores
```

---

## 6. Worker 侧流程

### 6.1 注意力计算钩子

```python
class GSA(UcmSparseBase):
    def attention_begin(
        self,
        layer_idx: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        metadata: AttentionMetadata
    ) -> Tuple[torch.Tensor, ...]:
        """注意力计算前处理（Worker 侧）"""
        # 等待预取完成
        for request_id in self._active_requests:
            stat = self._request_stats.get(request_id)
            if stat and stat.prefetch_task:
                self.prefetch_engine.wait_for_completion(request_id)

        # 获取选中的 Block 索引
        selected_indices = self._get_selected_indices(metadata)
        # 存储供 Kernel 使用
        self._current_selected_indices = selected_indices

        return query, key, value

    def attention_finished(
        self,
        layer_idx: int,
        output: torch.Tensor
    ) -> torch.Tensor:
        """注意力计算后处理"""
        # 清理当前状态
        self._current_selected_indices = None
        return output
```

---

## 7. 预取在关键路径上的作用

假设执行第 t 步时，旧候选已经就绪，另一个任务正为后续步预取。应记录前台 attention 等待旧候选多久、后台预取何时结束、下一步用到了多少已经加载的数据。减少的等待与额外读入的字节可能有所抵消；实际效果由候选准确性、带宽与重叠决定。

以下时序解释代码保留的机制。v0.9.1 默认工厂未注册普通 GSA，使用该实现要先核对对应历史引擎环境。当前支持矩阵列出的部署入口是 [GSAOnDevice](./05-gsa-on-device.md)。[预取实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/prefetch/prefetch_engine.py)

### 8.2 延迟对比
```mermaid
graph LR
    subgraph standard["标准注意力"]
        ST1["加载 KV"] --> ST2["计算注意力"]
        ST2 --> ST3["输出"]
    end

    subgraph gsa["GSA"]
        GSA1["预取 KV<br/>（异步）"]
        GSA2["计算注意力"]
        GSA3["输出"]
        GSA1 -.-> |"并行"| GSA2
        GSA2 --> GSA3
    end
```

## 8. 在固定源码中检索调用链

```bash
rg -n 'class (GSAReqStat|TopkCal|GSA)|def (attention_begin|attention_finished|build_sparse_meta)' ucm/sparse/gsa/gsa.py
rg -n 'deal_async_prefetch' ucm/sparse/gsa/prefetch/prefetch_engine.py
```

用完整注意力作为输出质量基线，再比较相同 token、相同并发的延迟。上文数据结构和流程片段用于解释数据依赖；实际类型与接口以 [GSA 固定版本文件](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/gsa/gsa.py)为准。
