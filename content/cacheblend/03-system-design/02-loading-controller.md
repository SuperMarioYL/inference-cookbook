---
title: "Loading Controller"
weight: 2
---

给定一个模型、一个存储设备，加载一层 KV 和重算这一层各花多长时间？加载控制器用这两个时间估计可以重叠多少工作。

---

## 3.2.1 核心洞察

> **基本洞察**：如果选择性 KV 重计算的延迟比将 KV 加载到 GPU 内存的时间更快，那么正确地流水线化选择性 KV 重计算和 KV 加载可以使 KV 重计算的额外延迟可忽略不计。

## 3.2.2 重计算比例估算

Loading Controller 使用两个延迟估算器来找到理想的重计算比例：

**重计算延迟估算器**：
$$T_{recompute}(r\%, LLM, L) = r\% \times Prefill(LLM, L)$$

其中 $Prefill(LLM, L)$ 是离线预测量的。

**加载延迟估算器**：
$$T_{load}(LLM, L, storage\_device) = \frac{PerTokenKVSize(LLM) \times L}{Throughput(storage\_device)}$$

**理想重计算比例计算**：
1. 选择 $r\%$ 使得 $T_{recompute}(r\%, LLM, L) = T_{load}(LLM, L, storage\_device)$
2. 取 $\max(r\%, r^*\%)$，其中 $r^*\%$ 是经验上质量损失可忽略的最小重计算比例（实践中约 15%）

```mermaid
graph LR
    subgraph "重计算比例选择"
        A["存储设备速度"] --> B["计算 T_load"]
        C["模型参数"] --> D["计算 T_recompute"]
        B --> E{"T_recompute ≈ T_load?"}
        D --> E
        E -->|"是"| F["使用该比例"]
        E -->|"否"| G["调整比例"]
        G --> E
        F --> H["达到任务所需的质量阈值"]
    end
```

## 3.2.3 存储设备选择

对于固定的重计算比例（如 15%），Loading Controller 还可以帮助选择最佳存储设备：

**问题**：如果只做固定选择性重计算比例（如 15%）的 KV 重计算，如何选择合适的存储设备来存储 KV，使其不会增加额外延迟？

**解决方案**：
1. 使用存储成本估算器估算每个设备存储 KV 的成本
2. 估算所有设备的重计算和加载延迟
3. 找出 $T_{recompute} \geq T_{load}$ 的最便宜存储设备

```mermaid
graph TB
    subgraph "存储设备选择"
        A["GPU HBM<br>最快/最贵"] --> D{r=15% 时}
        B["CPU RAM<br>快/中等"] --> D
        C["SSD<br>慢/便宜"] --> D
        E["HDD<br>最慢/最便宜"] --> D

        D -->|"选择最便宜且<br>T_load ≤ T_recompute"| F["最优存储"]
    end
```

## 3.2.4 论文中两组层耗时的对照

**Llama-7B，4K 上下文**：
- 重计算 15% token：3 ms/层
- 从 NVME SSD 加载一层 KV Cache：16 ms
- **结论**：理想稳态下，16 ms 的读取可覆盖 3 ms 的重算；启动和排空仍可能增加首 token 等待

**Llama-70B，4K 上下文**：
- 重计算 15% token：7 ms/层
- 从 NVME SSD 加载一层 KV Cache：4 ms
- **结论**：理想稳态由 7 ms 的计算限制，4 ms 的读取只能覆盖部分计算

---

**上一步**: [整体架构](./01-architecture.md)

**下一步**: [KV Cache Store 与 Fusor](./03-kv-cache-store.md)

上述 3、16、7、4 ms 是论文用来说明加载控制器的层级测量值。比较本机时请重测同一模型的每层加载和计算时间。[论文第 5 节](https://arxiv.org/html/2405.16444v3)
