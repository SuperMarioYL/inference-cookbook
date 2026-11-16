---
title: "位置、偏差和融合"
linkTitle: "算法详解"
weight: 5
description: "RoPE 恢复、HKVD 选择、KV 融合算法"
---

跟踪同一块 K/V 的三个时刻：收集原始 K，用新位置编码；在检查层按 V 差值挑出高偏差位置；在后续层写回这些位置的 K 和 V。原型中的实现入口分别是模型与注意力后端。[模型](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py) · [后端](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py)
