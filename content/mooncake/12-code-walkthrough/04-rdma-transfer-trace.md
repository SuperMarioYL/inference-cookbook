---
title: "RDMA 数据传输详细流程"
weight: 4
---

一笔 RDMA 读依次经过内存注册、QP 建连、提交和 CQ 完成。两端的地址、长度和访问键可以把同一条请求在各阶段连起来。

---

## 1. RDMA 传输流程概览

{{< trace >}}
title: "一次 RDMA Read 分成准备与读回两段"
actors: ["读方进程", "本地 RNIC", "远端 RNIC", "远端缓冲"]
steps:
  - from: 0
    to: 3
    label: "取得远端描述"
    detail: "通过控制通道协商远端地址、长度与访问键。两端先建立适用的队列对和内存注册。"
  - from: 0
    to: 1
    label: "提交读取"
    detail: "读方提交包含本地目的地和远端源范围的工作请求。"
  - from: 1
    to: 2
    label: "网络读请求"
    detail: "本地 RNIC 与远端 RNIC 交换所需数据。"
  - from: 2
    to: 3
    label: "读取源范围"
    detail: "远端 RNIC 在已授权的范围内访问注册缓冲。"
  - from: 1
    to: 0
    label: "消费完成事件"
    detail: "读方根据 CQ 完成结果确认本地目标数据就绪。"
{{< /trace >}}

---

## 2. 关键 RDMA 操作

### 2.1 内存注册 (Memory Registration)

注册内存看 RdmaTransport::registerLocalMemory 和 RdmaContext::registerMemoryRegion。[v0.3.14 实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-transfer-engine/src/transport/rdma_transport/rdma_context.cpp)

```cpp
// 下例用 verbs 说明操作顺序，是教学伪代码。
struct RegisteredMemory {
    void* addr;           // 内存地址
    size_t length;        // 内存长度
    ibv_mr* mr;          // Memory Region 句柄
    uint32_t lkey;       // Local Key (本地访问)
    uint32_t rkey;       // Remote Key (远程访问)
};

RegisteredMemory* register_memory(void* addr, size_t length) {
    /**
     * 向 RDMA NIC 注册内存区域,使其可被远程访问
     *
     * 关键参数:
     * - IBV_ACCESS_LOCAL_WRITE: 允许本地写
     * - IBV_ACCESS_REMOTE_READ: 允许远程读
     * - IBV_ACCESS_REMOTE_WRITE: 允许远程写
     */

    // 1. 分配 Memory Region
    ibv_mr* mr = ibv_reg_mr(
        pd,                          // Protection Domain
        addr,                        // 内存起始地址
        length,                      // 内存长度
        IBV_ACCESS_LOCAL_WRITE |     // 访问权限
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE
    );

    if (!mr) {
        throw std::runtime_error("Failed to register memory");
    }

    // 2. 创建注册记录
    auto reg = new RegisteredMemory{
        .addr = addr,
        .length = length,
        .mr = mr,
        .lkey = mr->lkey,
        .rkey = mr->rkey
    };

    LOG_DEBUG("Registered memory: addr=%p, length=%zu, rkey=0x%x",
              addr, length, reg->rkey);

    return reg;
}
```

**GPU Direct RDMA 支持**:

```cpp
RegisteredMemory* register_gpu_memory(void* gpu_ptr, size_t length) {
    /**
     * 注册 GPU 显存,支持 GPU Direct RDMA
     * 需要:
     * - NVIDIA GPUDirect RDMA 驱动
     * - CUDA-aware RDMA 网卡
     */

    // 检查是否支持 GPU Direct
    if (!is_gpu_direct_supported()) {
        throw std::runtime_error("GPU Direct RDMA not supported");
    }

    // 使用特殊标志注册 GPU 内存
    ibv_mr* mr = ibv_reg_mr(
        pd,
        gpu_ptr,
        length,
        IBV_ACCESS_LOCAL_WRITE |
        IBV_ACCESS_REMOTE_READ |
        IBV_ACCESS_REMOTE_WRITE
    );

    return new RegisteredMemory{...};
}
```

**调试断点建议**:
- `ibv_reg_mr` 调用后,检查返回值
- 验证 `rkey` 是否有效

---

### 2.2 QP 创建与连接

主动与被动握手看 RdmaEndPoint::setupConnectionsByActive / setupConnectionsByPassive。[v0.3.14 实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-transfer-engine/src/transport/rdma_transport/rdma_endpoint.cpp)

```cpp
// 下例用 verbs 说明操作顺序，是教学伪代码。
struct QPPair {
    ibv_qp* qp;              // Queue Pair 句柄
    ibv_cq* send_cq;         // Send Completion Queue
    ibv_cq* recv_cq;         // Receive Completion Queue
    uint32_t qp_num;         // QP 编号
    std::string remote_addr; // 对端地址
};

QPPair* create_qp_pair(const std::string& remote_addr) {
    /**
     * 创建 QP 并连接到远端节点
     *
     * QP 状态转换: RESET → INIT → RTR → RTS
     */

    // 1. 创建 Completion Queues
    ibv_cq* send_cq = ibv_create_cq(
        ctx,              // Context
        128,              // CQ 深度
        nullptr,          // User context
        nullptr,          // Channel
        0                 // Comp vector
    );

    ibv_cq* recv_cq = ibv_create_cq(ctx, 128, nullptr, nullptr, 0);

    // 2. 创建 Queue Pair
    ibv_qp_init_attr qp_attr = {
        .send_cq = send_cq,
        .recv_cq = recv_cq,
        .cap = {
            .max_send_wr = 128,      // 最大发送 WR 数
            .max_recv_wr = 128,      // 最大接收 WR 数
            .max_send_sge = 4,       // 最大发送 SGE 数
            .max_recv_sge = 4,       // 最大接收 SGE 数
        },
        .qp_type = IBV_QPT_RC,       // Reliable Connection
    };

    ibv_qp* qp = ibv_create_qp(pd, &qp_attr);

    // 3. 修改 QP 状态: RESET → INIT
    modify_qp_to_init(qp);

    // 4. 交换 QP 信息 (通过 TCP 控制通道)
    QPInfo local_info = {
        .qp_num = qp->qp_num,
        .lid = get_local_lid(),
        .gid = get_local_gid()
    };

    QPInfo remote_info = exchange_qp_info(remote_addr, local_info);

    // 5. 修改 QP 状态: INIT → RTR (Ready to Receive)
    modify_qp_to_rtr(qp, remote_info);

    // 6. 修改 QP 状态: RTR → RTS (Ready to Send)
    modify_qp_to_rts(qp);

    LOG_INFO("QP connected: local_qp_num=%u, remote_qp_num=%u",
             qp->qp_num, remote_info.qp_num);

    return new QPPair{qp, send_cq, recv_cq, qp->qp_num, remote_addr};
}
```

**QP 状态转换细节**:

```cpp
void modify_qp_to_rtr(ibv_qp* qp, const QPInfo& remote_info) {
    /**
     * 修改 QP 到 RTR 状态,允许接收数据
     */
    ibv_qp_attr attr = {
        .qp_state = IBV_QPS_RTR,
        .path_mtu = IBV_MTU_4096,           // 最大传输单元
        .dest_qp_num = remote_info.qp_num,  // 对端 QP 编号
        .rq_psn = 0,                        // 接收包序列号
        .max_dest_rd_atomic = 16,           // 最大 RDMA Read 请求数
        .min_rnr_timer = 12,                // RNR NAK 超时
        .ah_attr = {
            .dlid = remote_info.lid,
            .sl = 0,
            .src_path_bits = 0,
            .is_global = 1,
            .grh = {
                .dgid = remote_info.gid,
                .flow_label = 0,
                .sgid_index = 0,
                .hop_limit = 64,
                .traffic_class = 0
            }
        }
    };

    int ret = ibv_modify_qp(qp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU |
        IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
        IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER
    );

    if (ret) {
        throw std::runtime_error("Failed to modify QP to RTR");
    }
}
```

**调试断点建议**:
- `ibv_create_qp` 后,检查 QP 句柄
- `exchange_qp_info` 后,验证远端信息
- QP 状态转换后,查看 `qp->state`

---

### 2.3 RDMA Read 操作

slice 提交看 RdmaTransport::submitTransfer 和 RdmaEndPoint::submitPostSend。[v0.3.14 实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-transfer-engine/src/transport/rdma_transport/rdma_transport.cpp)

```cpp
// 下例用 verbs 说明操作顺序，是教学伪代码。
int rdma_read(
    QPPair* qp,
    void* local_addr,        // 本地缓冲区
    size_t length,           // 读取长度
    uint32_t local_key,      // 本地 MR 的 lkey
    uint64_t remote_addr,    // 远端内存地址
    uint32_t remote_key      // 远端 MR 的 rkey
) {
    /**
     * 执行 RDMA Read,从远端读取数据到本地
     *
     * 优点:
     * - 单边操作,不需要对端 CPU 参与
     * - 适合 Prefill/Decode 节点主动拉取 KVCache
     */

    // 1. 准备 Scatter-Gather Entry
    ibv_sge sge = {
        .addr = (uint64_t)local_addr,
        .length = (uint32_t)length,
        .lkey = local_key
    };

    // 2. 准备 Work Request
    ibv_send_wr wr = {
        .wr_id = next_wr_id++,           // Work Request ID (用于匹配完成事件)
        .next = nullptr,
        .sg_list = &sge,
        .num_sge = 1,
        .opcode = IBV_WR_RDMA_READ,      // RDMA Read 操作
        .send_flags = IBV_SEND_SIGNALED, // 请求完成信号
        .wr = {
            .rdma = {
                .remote_addr = remote_addr,
                .rkey = remote_key
            }
        }
    };

    // 3. Post Send
    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp->qp, &wr, &bad_wr);

    if (ret) {
        LOG_ERROR("ibv_post_send failed: %s", strerror(ret));
        return -1;
    }

    LOG_DEBUG("Posted RDMA Read: local=%p, remote=0x%lx, length=%zu",
              local_addr, remote_addr, length);

    return 0;
}
```

**批量传输优化**:

```cpp
int rdma_read_batch(
    QPPair* qp,
    const std::vector<TransferTask>& tasks
) {
    /**
     * 批量提交多个 RDMA Read 请求
     * 优化:
     * - 减少系统调用次数
     * - 提升网卡利用率
     */

    std::vector<ibv_sge> sges(tasks.size());
    std::vector<ibv_send_wr> wrs(tasks.size());

    // 1. 准备所有 WR
    for (size_t i = 0; i < tasks.size(); i++) {
        sges[i] = {
            .addr = (uint64_t)tasks[i].local_addr,
            .length = tasks[i].length,
            .lkey = tasks[i].local_key
        };

        wrs[i] = {
            .wr_id = next_wr_id++,
            .next = (i < tasks.size() - 1) ? &wrs[i+1] : nullptr, // 链接
            .sg_list = &sges[i],
            .num_sge = 1,
            .opcode = IBV_WR_RDMA_READ,
            .send_flags = IBV_SEND_SIGNALED,
            .wr = {
                .rdma = {
                    .remote_addr = tasks[i].remote_addr,
                    .rkey = tasks[i].remote_key
                }
            }
        };
    }

    // 2. 一次性提交所有 WR
    ibv_send_wr* bad_wr = nullptr;
    int ret = ibv_post_send(qp->qp, &wrs[0], &bad_wr);

    LOG_DEBUG("Posted %zu RDMA Read requests", tasks.size());

    return ret;
}
```

---

### 2.4 完成队列轮询 (CQ Polling)

数据完成路径看 worker pool 处理 CQ 的代码。[v0.3.14 实现](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/mooncake-transfer-engine/src/transport/rdma_transport/worker_pool.cpp)

```cpp
// 下例用 verbs 说明操作顺序，是教学伪代码。
struct CompletionEvent {
    uint64_t wr_id;          // Work Request ID
    ibv_wc_status status;    // 完成状态
    uint32_t byte_len;       // 传输字节数
};

std::vector<CompletionEvent> poll_completions(ibv_cq* cq, int max_events) {
    /**
     * 轮询完成队列,获取传输完成事件
     *
     * 轮询策略:
     * - Busy Polling: 低延迟,高 CPU 占用
     * - Event-driven: 低 CPU 占用,延迟稍高
     */

    std::vector<ibv_wc> wcs(max_events);
    std::vector<CompletionEvent> events;

    // 1. 轮询 CQ
    int n = ibv_poll_cq(cq, max_events, wcs.data());

    if (n < 0) {
        LOG_ERROR("ibv_poll_cq failed");
        return events;
    }

    // 2. 处理完成事件
    for (int i = 0; i < n; i++) {
        if (wcs[i].status != IBV_WC_SUCCESS) {
            LOG_ERROR("WR %lu failed with status %d (%s)",
                      wcs[i].wr_id, wcs[i].status,
                      ibv_wc_status_str(wcs[i].status));
            continue;
        }

        events.push_back({
            .wr_id = wcs[i].wr_id,
            .status = wcs[i].status,
            .byte_len = wcs[i].byte_len
        });

        LOG_DEBUG("Completion: wr_id=%lu, byte_len=%u",
                  wcs[i].wr_id, wcs[i].byte_len);
    }

    return events;
}
```

**Event-driven 模式**:

```cpp
void wait_for_completion(ibv_cq* cq) {
    /**
     * 使用 Completion Channel 等待完成事件
     * 优点: 节省 CPU
     * 缺点: 延迟增加 ~10us
     */

    // 1. 请求通知
    ibv_req_notify_cq(cq, 0);

    // 2. 等待事件
    ibv_cq* ev_cq;
    void* ev_ctx;
    ibv_get_cq_event(comp_channel, &ev_cq, &ev_ctx);

    // 3. 确认事件
    ibv_ack_cq_events(ev_cq, 1);

    // 4. 轮询 CQ
    auto events = poll_completions(cq, 128);

    LOG_DEBUG("Got %zu completion events", events.size());
}
```

---

## 3. Transfer Engine 集成

### 3.1 TransferBatch 执行流程

```cpp
// 下例用 verbs 说明操作顺序，是教学伪代码。
class TransferBatch {
public:
    void add_task(const std::string& segment_name,
                  const BatchTransferMetadata& metadata) {
        /**
         * 添加传输任务到批次
         */
        tasks_.push_back({segment_name, metadata});
    }

    std::vector<TransferResult> execute() {
        /**
         * 执行批量传输
         *
         * 流程:
         * 1. 准备本地缓冲区
         * 2. 提交 RDMA Read 请求
         * 3. 轮询完成事件
         * 4. 返回结果
         */

        std::vector<TransferResult> results;

        // 1. 分配接收缓冲区
        for (auto& task : tasks_) {
            void* buffer = allocate_buffer(task.metadata.data_size);
            task.local_buffer = buffer;
        }

        // 2. 批量提交 RDMA Read
        std::vector<TransferTask> rdma_tasks;
        for (auto& task : tasks_) {
            rdma_tasks.push_back({
                .local_addr = task.local_buffer,
                .length = task.metadata.data_size,
                .remote_addr = task.metadata.remote_addr,
                .remote_key = task.metadata.remote_key
            });
        }

        rdma_read_batch(qp_, rdma_tasks);

        // 3. 等待所有传输完成
        size_t completed = 0;
        while (completed < tasks_.size()) {
            auto events = poll_completions(qp_->send_cq, 128);
            completed += events.size();
        }

        // 4. 构造结果
        for (auto& task : tasks_) {
            results.push_back({
                .segment_name = task.segment_name,
                .data = task.local_buffer,
                .size = task.metadata.data_size,
                .status = TransferStatus::SUCCESS
            });
        }

        LOG_INFO("TransferBatch completed: %zu tasks", tasks_.size());

        return results;
    }

private:
    QPPair* qp_;
    std::vector<Task> tasks_;
};
```

---

## 4. 性能数据与调优

### 4.1 怎样测量这一层

固定网卡型号、链路、NUMA、消息大小、并发和方向。用小消息看延迟，用足够多的在途字节观察能否填满链路，分别记录提交时刻和 CQ 完成时刻。把网络线速换成应用可见的字节速率还要考虑协议开销；两者也都不能直接换算为模型的首 token 延迟。[本版 TE benchmark 与测试参数](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/performance/mooncake/tebench.md)

### 4.2 调优技巧

#### 4.2.1 批处理优化

```cpp
// 不推荐: 逐个传输
for (auto& task : tasks) {
    rdma_read(qp, task.local_addr, task.length, ...);
    wait_completion(qp->send_cq);  // 每次等待
}

// 推荐: 批量传输
rdma_read_batch(qp, tasks);
wait_all_completions(qp->send_cq, tasks.size());
```

**收益**: 延迟降低 60%,吞吐量提升 3x

---

#### 4.2.2 内存池预注册

```cpp
class MemoryPool {
public:
    void* allocate(size_t size) {
        // 从预注册的内存池分配
        void* ptr = pool_.allocate(size);

        // 已注册,无需再次调用 ibv_reg_mr
        return ptr;
    }

private:
    PreRegisteredPool pool_;  // 预注册 1GB 内存池
};
```

**收益**: 避免动态注册开销 (~100μs)

---

#### 4.2.3 多 QP 并发

```cpp
// 为每个远端节点创建多个 QP
std::vector<QPPair*> qps;
for (int i = 0; i < 4; i++) {
    qps.push_back(create_qp_pair(remote_addr));
}

// 负载均衡分配任务
int qp_idx = task_id % qps.size();
rdma_read(qps[qp_idx], ...);
```

**收益**: 吞吐量提升 2-3x (避免单 QP 瓶颈)

---

## 5. 调试技巧

### 5.1 检查 RDMA 设备

```bash
# 列出所有 RDMA 设备
ibv_devices

# 查看设备详情
ibv_devinfo mlx5_0

# 检查端口状态
ibstat mlx5_0
```

**期望输出**:

```
State: Active
Physical state: LinkUp
Rate: 100 Gbps
Link layer: InfiniBand
```

---

### 5.2 监控 RDMA 性能

```bash
# 实时监控 RDMA 流量
watch -n 1 'cat /sys/class/infiniband/mlx5_0/ports/1/counters/*'

# 查看关键计数器
cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_rcv_data
cat /sys/class/infiniband/mlx5_0/ports/1/counters/port_xmit_data
```

---

### 5.3 调试 QP 连接问题

```cpp
// 添加详细日志
void debug_qp_state(ibv_qp* qp) {
    ibv_qp_attr attr;
    ibv_qp_init_attr init_attr;

    ibv_query_qp(qp, &attr, IBV_QP_STATE, &init_attr);

    LOG_DEBUG("QP state: %s", qp_state_to_string(attr.qp_state));
    LOG_DEBUG("QP num: %u", qp->qp_num);
    LOG_DEBUG("Dest QP num: %u", attr.dest_qp_num);
}
```

**常见问题**:

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| QP 无法连接 | GID/LID 不匹配 | 检查 `exchange_qp_info` 逻辑 |
| RDMA Read 失败 | rkey 无效 | 验证远端 MR 注册是否成功 |
| 性能低 | MTU 过小 | 设置 `IBV_MTU_4096` |

---

## 6. 常见错误排查

### 6.1 错误: "Remote Operation Error"

**可能原因**:
- 远端 MR 未注册或已释放
- rkey 不匹配

**排查**:

```cpp
// 在远端节点验证 MR
LOG_DEBUG("Remote MR: addr=%p, rkey=0x%x", mr->addr, mr->rkey);

// 在本地节点验证参数
LOG_DEBUG("RDMA Read params: remote_addr=0x%lx, rkey=0x%x",
          remote_addr, remote_key);
```

---

### 6.2 错误: "Transport Retry Counter Exceeded"

**可能原因**:
- 网络丢包
- 远端节点故障

**排查**:

```bash
# 检查网络丢包
ethtool -S "$NETDEV" | grep drop

# 检查远端节点连通性
rping -c -a "$REMOTE_IP" -v -C 10
```

---

## 7. 相关阅读

- [04-transfer-engine/02-rdma-transport.md](../04-transfer-engine/02-rdma-transport.md) - RDMA 传输实现
- [批量传输实现](../04-transfer-engine/03-data-transfer-impl.md) - 批量传输优化
- [性能调优](../09-performance/03-tuning-guide.md) - 性能优化技术

---

**注**: 实际 RDMA 代码位于 `mooncake-transfer-engine` 子模块,本文基于典型 RDMA 编程范式推导。
