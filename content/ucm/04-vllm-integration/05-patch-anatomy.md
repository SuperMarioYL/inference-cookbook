---
title: "从 0.30.0 的混合状态问题读补丁"
weight: 5
---

v0.9.1 发布记录里有两种不同的 GLM5.3 问题。GLM5.3-W8A8C8 在部分数据上的超时，涉及 scheduler 发起同步 KV 加载的标志；GLM5.3-Flash-W8A8 在 A3 上的 POSIX 命中挂起，则涉及共享 MLA 与每 rank 私有状态的加载分发。[发布说明及修复 PR](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

## 调度到执行的关口

对于同步命中，scheduler 把外部 load 排入本步，模型 forward 等待所需结果。在 tag 对应的 `ucm/integration/vllm/patch/v0300/vllm_ascend` 中沿调度路径找调用点，将状态位的设置和消费放在同一次请求里对照。[v0.30.0 平台补丁](https://github.com/ModelEngine-Group/unified-cache-management/tree/v0.9.1/ucm/integration/vllm/patch/v0300/vllm_ascend)

## 同一逻辑块里的两类内容

共享 MLA 缓存可以按共享策略传递；每个 rank 各自持有的 Mamba/KDA 状态需要对应 rank 的内容。将两者合成一次具有相同分发策略的任务会丢失这个区别。排查时记录缓存布局、各 rank 期待的目标张量和后端实际建立的 load 任务，再检查所有等待是否都有对应完成路径。[Connector 布局与任务构造](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py)

## 案例一：一个布尔标志出现在哪一步

`patch_balance_sync_kv_load` 包装 Ascend `BalanceScheduler.schedule`，先调用原始方法，再检查返回的 `SchedulerOutput`。如果 `has_sync_kv_loads` 已经为真，就保留原结果。否则 `_has_ucm_sync_load` 查看 UCM 请求是否有要加载的块且 `load_async` 为假；若外层是多 connector 的 metadata，则递归查看子项。只有发现这类同步 load 才把输出标志设为真。[具体补丁](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/v0300/vllm_ascend/balance_sync_kv_load_patch.py)

|调度产物|包装后的行为|
|---|---|
|已有同步标志|直接使用原结果|
|UCM 要读块且为同步加载，含嵌套子项|标识本步含同步 load，使执行端能在 forward 前启动加载|
|只有异步 load，或没有 UCM 待读块|保持该步已有的调度语义|

## 案例二：一个请求为什么需要两个加载任务

HLA 直接加载路径先准备计划，然后将共享 MLA 的键段和 rank 私有状态的键段分开提交。对每段调用 `RankConsistencyManager.submit_load` 时传入各自的 `distribute_backend_load`，等待时通过与该任务关联的 Store 调用 `wait_load`。这意味着“是否分摊后端读取”是具体传输任务的属性；把配置存在全局会让本应私有读取的数据误用共享策略。[HLA 实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/hla_connector.py) · [任务与 Store 的绑定](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/rank_consistency.py)

## 混合层的 forward 也要接上等待与保存

标准全注意力算子的内置 hook 未必经过 GDN/KDA 线性层。v0.30.0 的 UCM 适配对有可用自定义算子边界的模型在 op 内挂接；仅能包住模块 forward 的路径则有 eager 执行条件。读一个模型时分别确认它的层类型、实际挂钩点和编译模式。把所有模型统一解释成“在每层 forward 上包一层 Python”会遗漏编译后 hook 的执行边界。[按层 hook 说明与实现](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/v0300/mamba_layer_hooks_patch.py)
