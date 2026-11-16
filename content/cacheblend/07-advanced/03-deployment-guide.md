---
title: "部署 CacheBlend 或复现论文原型"
weight: 3
---

LMCache `v0.5.7.post1` 把 KV 服务放在独立进程中。服务端维护已存文本的索引和 KV，推理引擎负责在模型前向中选择并重算 token。上线前需要把两端一起接通。[服务端文档](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/kv_cache_optimizations/cacheblend.rst)

## 启动多进程服务

在满足 LMCache 的 CUDA、Python 与 PyTorch wheel 要求的 Linux 环境安装固定发行版。以下命令给一级缓存分配 20 GB，机器要预留相应容量。`blend` 引擎要求传输模式为 `lmcache_driven`（默认值）或 `auto`。[安装矩阵](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/getting_started/installation.rst)

```bash
python -m pip install 'lmcache==0.5.7.post1'
lmcache server --l1-size-gb 20 --eviction-policy LRU --engine-type blend
```

在 Kubernetes 接入时，上游的 `CacheBlendEngine` 部署服务端，webhook 为选定的 vLLM Pod 注入 `lmcache-cacheblend` 插件和 `CBKVConnector` 配置。插件镜像需要部署者有权拉取。普通 `LMCacheMPConnector` 的前缀命中只能验证常规缓存通路。先检查引擎就绪、推理 Pod 获得对应的连接配置，再让同一篇手册先出现在开头、随后移到另一段检索材料后面，观察非前缀匹配及任务答案。[Operator 的部署清单、插件镜像要求与检查步骤](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/mp/operator.rst)

### 在已有 Operator 的集群接入 vLLM

准备一台支持 CUDA IPC 的 GPU 节点，安装带 webhook 的 LMCache Operator 与 cert-manager。Blend 服务和推理 Pod 需要访问同一块 GPU 和主机共享内存。选用集群已配置好权限的命名空间，将两个清单的 `metadata.namespace` 设为同一个值。

1. 取固定版本的[引擎清单](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/operator/config/samples/lmcache_v1alpha1_cacheblendengine.yaml)和[推理 Pod 清单](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/operator/config/samples/vllm_cacheblend_deployment.yaml)。将引擎、vLLM 和 `spec.injection.payloadImage` 换成相互兼容且可拉取的镜像标签。私有插件的拉取凭据需要在推理 Pod 所在的命名空间可用。
2. 引擎清单中的 `server.chunkSize` 使用本版匹配器要求的 256。推理 Pod 模板带上标签 `lmcache.ai/cacheblend-inject: "true"` 和注解 `lmcache.ai/cacheblend-engine: my-cacheblend`。使用镜像的默认入口，只在 `args` 中给模型与服务参数，连接器参数由 webhook 补入。用真实权重比较答案时，删除上游示例用于空载测试的 `--load-format dummy`。
3. 先创建引擎，出现连接 ConfigMap 后再创建推理 Pod。在清单所在目录执行下列命令。`NS` 是两份清单中填写的命名空间。

```bash
kubectl apply -f lmcache_v1alpha1_cacheblendengine.yaml
kubectl -n "$NS" get configmap my-cacheblend-connection
kubectl apply -f vllm_cacheblend_deployment.yaml
kubectl -n "$NS" rollout status deployment/vllm-cacheblend
kubectl -n "$NS" get pod -l app=vllm-cacheblend -o yaml | \
  grep -E 'cb-plugin|kv-transfer-config|cacheblend-injected|cacheblend-skip-reason'
```

查看已经启动的 vLLM Pod，确认 CacheBlend 插件与 connector 配置均已注入。随后发送“手册 M + 问题”和“检索材料 R + 手册 M + 新问题”；M 要长到足以跨过完整的匹配块。对照引擎的非前缀匹配、重算记录与生成结果。分别记录两次请求的首 token 延迟、命中区间和答案。[两次输入的构造](../01-introduction/02-problem-definition.md)

## 复现论文中的单进程流程

原型示例使用仓库自带的 `vllm_blend`。原作者要求 CUDA 12.1+，建议 NVIDIA GPU 至少 40 GB 显存。作者仓库的 CUDA 依赖固定 `torch==2.2.1`、`xformers==0.0.25`；复现时可选独立 CPython 3.10 或 3.11 环境，并准备 Mistral-7B-Instruct-v0.2 权重。[作者说明](https://github.com/YaoJiayi/CacheBlend/tree/55ad02675939f783a38d579393527d218a7fd581) · [原型 CUDA 依赖](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/requirements-cuda.txt)

```bash
git clone https://github.com/YaoJiayi/CacheBlend.git
cd CacheBlend
git checkout 55ad02675939f783a38d579393527d218a7fd581
python -m pip install -e ./vllm_blend
python -m pip install -r requirements.txt
python example/blend.py
```

作者示例先为各文档收集 KV，再运行选择性融合和常规 prefill，便于对照生成结果。脚本打印的 TTFT 是 `first_token_time - first_scheduled_time`，从请求首次被调度计时。在线端到端 TTFT 还要计入进入引擎前后和首次调度前的等待。要复现任务指标，使用同目录下针对 MuSiQue、SAMSum 和 2WikiMQA 的脚本。LMCache 自己保留的进程内示例另有 vLLM worker 改动和注册时序要求。[作者示例](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/example/blend.py) · [LMCache 进程内示例](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/examples/inprocess/blend/README.md)

## 一次非前缀命中在服务里经历什么

在 v0.5.7.post1 的 MP 服务里，存储端提交完成后才登记相应 chunk 的指纹，以便查找侧只匹配可读取的数据。一次 unified lookup 同时考虑连续前缀和可独立迁移的区间。需要从下一层加载时，首次调用提交任务，后续调用轮询；尚未就绪可返回 `None`。连续前缀走到第一个缺口便停止，非前缀匹配则要求该 chunk 涉及的 rank 和读取组都有对应对象。缺少一份时会剔除整个候选区间。[查找、读取与生命周期](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/design/v1/multiprocess/modules/blend.md)

找到的区间仍需回填。`cb_retrieve_pre_computed` 从 L1 读入临时区域，对位移的 K 做旋转，再按 token scatter 到当前请求分配的 paged KV。按 token 写使匹配区间跨页、或与需要重算的 token 共用一页时仍能定位。无法形成完整计划时，该请求退回完整重算。跟踪同一请求的非前缀匹配和最终 scatter，就能看到哪些位置实际写回。[读取计划实现](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/lmcache/v1/multiprocess/modules/blend/retrieve.py)

原型示例中的 `check_layers`、`recomp_ratio` 可用来研究重算与质量的关系。MP 场景还涉及 L1 缓存容量、文档命中和数据回填。先用完全相同的输入验证基础前缀，然后实验文档换位，并为冷/热各保留同一组完整重算质量基线。[配置入口](../appendix/B-config-reference.md) · [实验方法](../appendix/E-benchmarks.md)
