---
title: "代码走读"
linkTitle: "代码走读"
weight: 4
description: >
  跟踪关键代码路径，调试和理解 PyTorch 内部执行流程
---

从一次反向、算子派发、保存状态或自定义求导出发，区分调用者看到的接口与框架内部的生命周期。加载旧 checkpoint 时重点检查保存的是 state dict 还是完整 Python 对象。
