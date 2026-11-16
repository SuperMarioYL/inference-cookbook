---
title: "按推理引擎、设备和 Python ABI 部署"
weight: 1
---

先在 v0.9.1 的发行矩阵选择与已有引擎匹配的**整行组合**。例如 CUDA 12.9、vLLM 0.30.0、CPython 3.12 的后端 extra 是 `cu129`；CANN 9.2.1 的 A2 和 A3 分别使用 `cann921-a2` 和 `cann921-a3`。Python 模块名为 `ucm`，安装用的分发包是 `uc-manager`。[官方运行时、wheel 和镜像清单](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

|推理环境|配置入口|应核对的条件|
|---|---|---|
|CUDA + vLLM|[快速开始](../01-overview/03-quick-start.md)|CUDA、Python ABI、设备和模型前缀缓存支持|
|vLLM-Ascend|[Ascend 指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_vllm_ascend.md)|CANN、芯片系列、vLLM-Ascend 与 UCM wheel|
|SGLang|[SGLang 指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_sglang.md)|对应的 SGLang runtime 以及它的缓存接入方式|
|MindIE LLM|[MindIE 指南](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_mindie_llm.md)|MindIE/CANN 环境和引擎自己的配置入口|

在已经安装对应 vLLM、CUDA 12.9、CPython 3.12 的环境中，可以直接安装该组合的 wheel。也可以使用同一行发布的预装镜像，两者各有自己的 Python 环境。

```bash
python -m pip install 'uc-manager[cu129]==0.9.1'
python -m pip check
python -c 'import ucm; print(ucm.__file__)'
```

UCM 配置文件需要在 scheduler 和 worker 的容器中都能通过同一配置路径读取；缓存目录则需要满足所选后端的可访问范围和权限。先验证固定前缀确实写出并读回，再加多机或启用稀疏算子。`v0.9.1` 发布说明还列出 vLLM-Ascend 0.30.0 中特定混合 GDN 模型、MTP 和异步调度组合的已知限制。[特性与模型矩阵](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md) · [发布说明](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

## 换引擎时接入点怎样变化

在 SGLang，v0.9.1 示例走 HiCache 的 dynamic storage backend，通过 `--hicache-storage-backend-extra-config` 交给引擎 UCM 后端所需的配置。配置由 SGLang 的启动命令读取。[SGLang 实际启动参数](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_sglang.md)

在 MindIE LLM，安装对应集成后，要在 MindIE 服务配置的 `BackendConfig.kvPoolConfig` 中选择 `unifiedcache`，将 `configPath` 指向 UCM 的 JSON。先检查安装的 MindIE 模块已接入，再从服务日志查看 `unifiedcache` 后端是否初始化，随后发送重复前缀观察外部读回。[MindIE 的配置和验证](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_mindie_llm.md)

## 从单机扩成多节点

多实例接入同一存储时，对齐模型与并行布局、tokenizer、后端可见性和运行镜像。先在 A 实例写入，等保存完成，再让 B 实例读取。若是 prefill/decode 分离，另行检查路由、实例注册和传输时序。上游将集中式、分布式以及大规模 EP 部署分开说明。[P/D 方案](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/pd-disaggregation/index.md)
