---
title: "把参数交给正确的组件"
weight: 2
---

|配置位置|参数|作用|
|---|---|---|
|LMCache 多进程服务|`--engine-type blend`|启用非前缀匹配和读取模块|
|LMCache 多进程服务|`--l1-size-gb`|用于一级缓存的容量，单位 GB|
|LMCache 多进程服务|`--supported-transfer-mode`|Blend 支持 `lmcache_driven` 或 `auto`|
|论文原型|`check_layers`、`recomp_ratio`|模型在哪里评估偏差，以及从候选位置中重算多大比例|
|论文原型|`suffix_len`|输入末尾新后缀的 token 数，这些位置加入重算集合|

研究原型通过 `cache_fuse_metadata` 在模型内部传递控制状态。LMCache 的进程内示例有自己的配置，例如 `LMCACHE_ENABLE_BLENDING`。多进程部署用服务参数和推理侧插件接入。[原型检查与筛选](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py) · [LMCache 当前配置](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/kv_cache_optimizations/cacheblend.rst) · [历史进程内配置](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/legacy/kv_cache_optimizations/blending.rst)
