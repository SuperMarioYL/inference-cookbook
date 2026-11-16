---
title: "Mooncake Store"
linkTitle: "Mooncake Store"
weight: 3
description: >
  分布式 KVCache 存储引擎的核心实现原理
---

写入方为 key 取得副本位置，把数据写完，再告知 Master。读取方查到可读副本之后，从实际持有者获取数据。用[单机实验](../00-beginner-guide/03-quick-start-tutorial.md)建立基线，再沿[Master 实现](./02-master-service.md)区分对象缺失与副本尚未就绪。
