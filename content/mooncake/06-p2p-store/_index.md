---
title: "P2P Store"
linkTitle: "P2P Store"
weight: 6
description: >
  点对点对象存储与 Checkpoint 支持
---

P2P Store 的 Go 接口可用于模型权重等大对象的点到点分发。训练端发布内容，推理端通过元数据查找来源并取回数据。它的入口和部署条件与按 key 存储 KV 的 Mooncake Store 分属不同组件。[P2P Store 示例与构建选项](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/p2p-store.md)
