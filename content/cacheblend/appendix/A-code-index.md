---
title: "原型和 LMCache 的源码入口"
weight: 1
---

|希望追踪的行为|入口|
|---|---|
|原型收集旧 KV、组合文档并执行对照|[example/blend.py](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/example/blend.py)|
|原型逐层切换检查与重算状态|[llama.py](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py)|
|原型筛选位置并更新缓存|[xformers.py](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py)|
|多进程模块的查找、登记和读取协议|[LMCache Blend 模块说明](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/design/v1/multiprocess/modules/blend.md)|
|当前服务的位置匹配|[matcher.py](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/lmcache/v1/multiprocess/modules/blend/matcher.py)|
|把匹配结果组装成读取计划|[retrieve.py](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/lmcache/v1/multiprocess/modules/blend/retrieve.py)|

原型里的 `old_kvs` 和 `cache_fuse_metadata` 由模型与注意力后端直接共享。多进程服务把文本匹配和缓存读取改为进程间协议，推理端的重算还需要相应插件。[部署指南](../07-advanced/03-deployment-guide.md)
