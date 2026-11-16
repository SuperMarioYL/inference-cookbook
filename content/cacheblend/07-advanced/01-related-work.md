---
title: "相关工作"
weight: 1
---

把问题和手册交换位置后，完整重算、前缀缓存和按文档融合分别还能复用哪些数据？下面按复用范围和新上文的处理方式比较。

---

## KV Cache 复用方案对比

| 方案 | 复用范围 | Cross-Attention | 质量 | 速度 |
|------|----------|-----------------|------|------|
| **Prefix Caching** | 仅前缀 | 完整 | 高 | 中等 |
| **PromptCache** | 所有块 | 忽略 | 低 | 快 |
| **RAGCache** | 仅前缀 | 完整 | 高 | 中等 |
| **CacheBlend** | 所有块 | 部分恢复 | 高 | 快 |

---

## 与 PromptCache 的对比

**PromptCache** 的主要问题：
1. 使用 buffer 来维护位置准确性，但需要多次预计算每个 chunk
2. 完全忽略 Cross-Attention，导致质量下降

**CacheBlend** 的改进：
1. 使用 RoPE 位置恢复，无需多次预计算
2. 通过选择性重计算恢复 Cross-Attention

---

## 与 RAGCache 的对比

**RAGCache** 的局限：
- 仅支持 Prefix Caching
- 当有多个非前缀 chunk 时加速有限

**CacheBlend** 的优势：
- 支持所有位置的 chunk 复用
- 在多 chunk 场景下加速更明显

---

## 与上下文压缩方法的互补性

CacheBlend 与以下技术互补：

1. **Prompt 压缩** (LLMLingua)：缩短 prompt 长度，CacheBlend 可处理不同 chunk 长度
2. **KV Cache 压缩** (H2O, ScissorHands)：减少 KV Cache 大小，CacheBlend 可存储和加载更少的 KV

---

## 参考文献

1. CacheBlend 论文: https://arxiv.org/abs/2405.16444
2. vLLM: https://github.com/vllm-project/vllm
3. SGLang: https://github.com/sgl-project/sglang
4. PromptCache: https://arxiv.org/abs/2311.04934
5. RAGCache: https://arxiv.org/abs/2404.12457
