---
title: "从代码看 UCM 的集成、缓存键和异步接口"
weight: 1
---

UCM 把外部缓存接入推理引擎时，有三个不同的责任边界：引擎决定本步如何调度，connector 计算缓存匹配并把工作交给 worker，存储后端操作块数据。下面的取舍都可以在 v0.9.1 的接口和测试中定位。

## 哪些行为通过接口接入，哪些需要版本适配

vLLM V1 通过指定的 `UCMConnector` 调用匹配、分配后的状态更新、元数据构建与传输回调。`ENABLE_UCM_PATCH` 提供额外启动适配，使特定版本上的 scheduler、worker 或特殊布局走正确的钩子。其他引擎使用各自的入口，例如 SGLang 的 HiCache backend。升级后先确认发布组合，再核对该引擎实际采用的适配分支。[vLLM 调用顺序](../04-vllm-integration/01-connector-interface.md) · [按版本分派的补丁](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/apply_patch.py)

## 相同文字要满足哪些条件才能共享 KV

`build_kv_hash_meta` 包含模型路径末级名、TP 大小、模型 dtype、rank、推测解码参数、相应量化标记、DCP 大小以及布局实现摘要。`RequestHasher(meta)` 把这些命名空间字节纳入 MD5 摘要。普通请求为每个**完整块**计算下一段链值，输入包含上一块摘要、本块的 token 元组和引擎提供的额外语义键。末尾不够一块的 token 不产生普通前缀块键。[命名空间构造](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py) · [请求哈希](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/request_hasher.py)

举例来说，两条请求起始文本一致，但第二块对应的图像不同。第一块之前的公共文本仍可匹配；图像所在块及其后的链应当分开。LoRA、cache salt、prompt embeddings 等也可能改变计算语义。上游 helper 不能表达所需语义时，UCM 会报告哈希错误。同名目录也不能替读者验证两套权重真的相同，部署时要管理模型身份和缓存生命周期。[跨请求和布局隔离测试](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/test/test_ucm_request_hasher.py)

## 一次成功查询为什么还要等待

`lookup` 回答一批块是否存在，`lookup_on_prefix` 描述连续可用的前缀范围。`load` / `dump` 提交传输任务，`check` / `wait` 用于确认任务状态或完成。worker 要在目的数据可消费时才推进对应计算，写方也要在使用源缓冲的操作完成后才复用空间。把这些时刻分开，存储和计算才有重叠的余地。[Store V1 接口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/ucmstore_v1.py)

## Pipeline 后端如何装配

`UcmPipelineStoreBuilder` 以完整组合名注册构造函数。`Cache|Posix` 和 `Mooncake|Posix` 都能提供查询与读取，但它们需要不同的网络、文件和缓冲资源。可用组合由注册表和 wheel 内的后端共同决定。[管线装配](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/store/pipeline/connector.py)

## 稀疏计算的额外约束

`UcmSparseBase` 提供请求、批次和注意力的生命周期钩子；工厂决定当前可创建哪些方法。它们与普通前缀缓存分别列入模型支持矩阵。启用某个后端后，需要通过匹配负载的输出质量对照评估选择性计算的影响。[稀疏工厂](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/factory.py) · [支持表](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)
