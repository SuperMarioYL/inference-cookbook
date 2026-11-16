---
title: "从发布记录跟踪 UCM 的演进"
weight: 3
---

判断新引擎版本能否接入，从两份资料开始。发布说明给出已打包的运行时、wheel 和具体修复，模型支持矩阵给出前缀缓存与增强特性的适用范围。前者回答“能装哪套环境”，后者回答“这一功能适用于哪种模型、哪一版引擎”。[v0.9.1 发布说明](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1) · [支持矩阵](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)

`v0.9.1` 针对 v0.30.0 引擎补充了 MiniMax-M3.1-Flash 在 H100 上的验证，并修复两类 GLM5.3 加载问题。要评估下一次升级，先重做冷启动、同前缀外部命中和输出正确性，再用实际模型验证多 rank 或混合布局。

跟踪稀疏或融合功能时，另查对应版本行。例如上游表将 GsaOnDevice 和 ReRoPE 列在 v0.11.0，将 UCM 的 CacheBlend 列在 vLLM 0.9.2。选择某个新镜像部署前缀缓存，并不会自动扩大这些特性的验证范围。
