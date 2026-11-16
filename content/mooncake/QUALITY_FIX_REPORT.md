---
title: "Mooncake 版本说明与勘误"
---

本栏讲解 `v0.3.14` 的公开组件，并保留 FAST ’25 论文中的系统设计。论文实验和开源软件可以回答不同的问题。[论文](https://www.usenix.org/conference/fast25/presentation/qin) · [固定版本](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14)

- Conductor 已有开源实现，维护设备、主机和磁盘层的 KV 前缀索引，供上层路由器选择实例。[实现与接口](./02-architecture/03-conductor-scheduler.md)
- vLLM V1 的 `MooncakeConnector` 用于 P/D 直传，`MooncakeStoreConnector` 用于共享池。前者需要代理传递交接元数据；后者需要可写入和读取的 Store。[集成指南](./08-integration/01-vllm-integration.md)
- Python 单机 Store 示例使用 `P2PHANDSHAKE` 和 TCP，Master 默认 RPC 端口为 50051。进程内本地副本读回和远端网络读回是两个不同的实验。[快速开始](./00-beginner-guide/03-quick-start-tutorial.md)
- 配置和源码参考跟随具体子项目。TENT 代码在 `mooncake-transfer-engine/tent`，PyTorch process group 适配位于 `mooncake-pg`。[TENT](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/tent/overview.md) · [PG](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/mooncake-backend-pg.md)
