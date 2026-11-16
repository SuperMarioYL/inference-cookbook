---
title: "UCM 怎样融合移位文档的 KV"
weight: 7
---

两条提示中都含有同一份手册，但第二条在手册前插入了检索段落。UCM 的 Blend connector 先查询连续前缀，再按 chunk 查找候选 KV。位置发生变化的块要校正 RoPE，缺失位置、新后缀及选定的高偏差位置进入计算集合。[算法与适配说明](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/sparse-attention/cacheblend.md)

## 校正的是哪份张量

`block_wise_rope_forward` 的输入包括 K cache、引擎中的 block ID、每块的位置差和 cos/sin cache。它按 block ID 找到目标块，按位移旋转 K；V 的上下文变化仍需要相应重算。上层 `Blend` 与 `BlendMetaData` 负责将请求中需要计算的位置传给模型。[块级旋转](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/blend/blockwise_rope.py) · [模型侧融合](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/sparse/blend/blend.py)

## 从示例看数据与质量

运行 `examples/offline_inference_blend.py` 前，要准备脚本指定的模型、数据集和缓存目录。该 tag 的支持矩阵将此实现列在 vLLM 0.9.2；算法页给出的 kernel 限于 CUDA。将原顺序、重排顺序与完整重算分别作为对照，观察实际复用的 chunk 和答案质量。[示例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/offline_inference_blend.py) · [支持表](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)

同名算法在 LMCache 还有另一条实现路径，使用独立服务和推理插件。两条实现共享研究问题，配置入口分别由各自项目维护。[CacheBlend 工程指南](/cacheblend/07-advanced/03-deployment-guide/)
