---
title: "旧 K 在新位置怎样应用 RoPE"
weight: 1
---

手册 M 单独出现时从位置 0 开始，放到长度为 32 个 token 的材料后面时，手册第一个 token 的目标位置变成 32。要使用它的旧 K，首先要知道旧 K 保存在哪个阶段：是已经旋转到旧位置，还是投影之后、尚未加入 RoPE。

## 原型保存的是旋转前的 K

在 `LlamaAttention.forward` 中，`qkv_proj` 后拆出 Q、K、V；`collect` 打开时将 `k.clone()` 和 `v.clone()` 存入 `hack_kv`。随后才对当前 Q/K 调用 `rotary_emb(positions, q, k)`。所以此处保存的原始 K 可以在重用时直接按组合后的 `org_pos` 旋转。原型对旧 K 调用相同的双输入 API，提供 `fake_q`，只保留 K 的输出。[作者源码](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/model_executor/models/llama.py)

{{< mechanism "cacheblend-rope-angle" >}}

若一个实现保存的是已旋转 K，标准固定频率 RoPE 可写成 `R(new) × R(old)⁻¹ × K_old`，即施加相对位移；直接对已经旋转过的 K 再施加完整新位置旋转会重复计入旧位置。LMCache 多进程服务按缓存布局处理旋转信息，有独立的登记和回填逻辑。[LMCache 固定版本 RoPE 实现](https://github.com/LMCache/LMCache/blob/v0.5.7.post1/lmcache/v1/multiprocess/modules/blend/rope.py)

位置对齐只解决“词出现在什么位置”。手册前面多了一份材料时，“这个词看到了什么”也变了，这部分由[高偏差 token 的选择与重算](./02-hkvd-selection.md)处理。
