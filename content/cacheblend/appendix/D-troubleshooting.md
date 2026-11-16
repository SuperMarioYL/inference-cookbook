---
title: "查找、读取和重算分别怎样排查"
weight: 4
---

## 文本重复却没有匹配

把两条提示经过聊天模板和 tokenizer 处理后的 token 列表并列检查。空格、特殊 token、分隔符、边界和上下文长度都可能改变实际匹配区间。先用完全相同的长前缀验证普通缓存，再让其中一个文档移动位置。

## 前缀命中正常，文档换序后没有效果

独立服务要选择 `--engine-type blend`；推理端也要具备对应插件，在请求中使用正确的 connector。服务接受普通 KV 查询只验证了缓存连接。Kubernetes 路径可分别检查 `CacheBlendEngine` 与被注入的 vLLM Pod。[部署与注入步骤](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/source/mp/operator.rst)

## 已命中，答案或 TTFT 却退化

固定采样和同一任务，先与完整 prefill 比较答案，再逐项记录匹配范围、重算位置、位移和外部读取耗时。重算过少可能遗漏跨文档依赖，读取时间长也可能抵消节省的计算。在作者原型中关注 `imp_indices` 和后缀，在当前 LMCache 中按匹配、retrieve 与推理插件分开定位。[原型源码](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py) · [当前服务端](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/docs/design/v1/multiprocess/modules/blend.md)
