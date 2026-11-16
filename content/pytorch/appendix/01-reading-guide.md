---
title: "阅读指南"
weight: 1
---

## 从一个公开接口追到实现

| 从哪里起步 | 按问题往下读 |
|---|---|
| 张量和算子 | 先观察[张量的 stride 与共享存储](/pytorch/01-introduction/01-tensor-basics-tutorial/)，再读 [TensorImpl](/pytorch/02-fundamentals/01-tensorimpl-deep-dive/) 和 [Dispatcher](/pytorch/03-core-modules/07-dispatcher-core/)。 |
| 模型训练 | 从[模块如何管理参数](/pytorch/03-core-modules/01-module-core-mechanism/)进入[反向计算图](/pytorch/02-fundamentals/05-computational-graph/)，最后看[优化器何时更新参数](/pytorch/01-introduction/04-optimizer-tutorial/)。 |
| 编译推理 | 从[三次相同函数调用](/pytorch/05-advanced-topics/01-compiler-stack-overview/)开始，再看 [FX 图](/pytorch/05-advanced-topics/02-fx-tutorial/)、[Dynamo 的 guard](/pytorch/05-advanced-topics/06-guard-mechanism/) 和 [Inductor](/pytorch/05-advanced-topics/07-inductor-overview/)。 |
| 分布式与显存 | 先读[进程组和集合通信](/pytorch/05-advanced-topics/09-distributed-overview/)，再比较 [DDP、FSDP1 与 FSDP2](/pytorch/05-advanced-topics/10-ddp-fsdp/)。单卡显存问题可直接读[分配器](/pytorch/05-advanced-topics/11-cuda-memory-tutorial/)。 |

需要部署导出模型时，`torch.export.export` 会生成带输入约束的 `ExportedProgram`。运行时遇到不支持的 Python 代码，`torch.compile` 可以选择 graph break，而导出需要满足导出接口对整图的要求。[2.15 导出文档](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/export.html)
