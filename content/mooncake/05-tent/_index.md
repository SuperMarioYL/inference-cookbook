---
title: "TENT 传输运行时"
linkTitle: "TENT 传输运行时"
weight: 5
description: >
  下一代拓扑感知传输引擎 - Transfer Engine NEXT
---

同样一笔读请求，在本机 GPU 互联和跨主机 NIC 上可能走不同的通路。TENT 为可达的路径选择 transport，再用切片和运行时反馈分配工作。[选路](./01-design-overview.md) · [v0.3.14 新增能力](https://github.com/kvcache-ai/Mooncake/releases/tag/v0.3.14)
