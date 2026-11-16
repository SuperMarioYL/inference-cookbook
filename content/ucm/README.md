---
title: "UCM 阅读指南"
---

本栏的工程示例对应 `UCM v0.9.1`。选择安装包时同时核对推理引擎、CUDA 或 CANN 版本、Python ABI 和芯片。前缀缓存、稀疏注意力、ReRoPE、CacheBlend 各有自己的支持范围。[发布组合](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1) · [特性支持表](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)

|想解决的问题|阅读入口|
|---|---|
|如何确认同一长前缀从存储中读回|[快速开始](./01-overview/03-quick-start.md)和[指标](./05-shared-infrastructure/02-metrics-observability.md)|
|匹配、分配目标块和实际搬运发生在哪里|[Connector 接口](./04-vllm-integration/01-connector-interface.md)、[UCM 的分派逻辑](./04-vllm-integration/02-ucm-connector.md)|
|外部 KV 放在主机、磁盘还是共享后端|[存储管线](./02-storage/05-pipeline-composition.md)和[后端配置](./06-engineering-practice/02-configuration-reference.md)|
|怎样确认稀疏模块能在选定的模型上运行|[稀疏注意力](./03-sparse-attention/_index.md)及[部署指南](./06-engineering-practice/01-deployment-guide.md)|
