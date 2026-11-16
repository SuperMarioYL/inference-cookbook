---
title: "同一份手册换了位置，KV 还能用吗？"
weight: 2
---

第一次请求是“手册 M + 问题一”，第二次是“手册 M + 问题二”。两次请求从开头读到手册末尾，看到的是同一串 token。普通前缀缓存能复用这段连续前缀中所有已缓存、满足块边界的部分；手册本身可以占很多块。

现在把检索材料 R 插到手册之前。普通前缀匹配从首个不同 token 起中断。读到 M 时，每个词可以注意到前面的 R。原先单独为 M 算出的 KV 缺少这种影响；如果位置编码依赖位置，还要对齐它在新序列中的位置。

{{< mechanism "cacheblend-reorder" >}}

## CacheBlend 怎样利用这份手册

整段重新 prefill 能算出当前输入的完整结果。直接把旧 KV 搬到新位置省下计算，但会保留旧上文带来的误差。CacheBlend 走第三条路径，先找到 M 的缓存，在检查层估计各位置偏差，再为选中的 token 和新问题做计算，把结果与复用部分合并。新问题本身也要读取它之前的上下文。

这里的“段落”是应用提供的文档，“token”是模型的输入单位，“缓存块”是引擎分配、索引 KV 的单位。段落边界与缓存块边界可以落在不同位置。实验时记录分词后的长度；聊天模板、分隔符乃至前导空格都可能移动匹配边界。[CacheBlend 论文](https://arxiv.org/abs/2405.16444v3) · [原型的筛选与合并](https://github.com/YaoJiayi/CacheBlend/blob/55ad02675939f783a38d579393527d218a7fd581/vllm_blend/vllm/attention/backends/xformers.py)

## 用两个 token 看出上文的影响

把手册 M 的两个 token 简化成标量 1 和 2，把检索材料 R 简化成 3。只用一层因果自注意力，Q、K、V 的投影都取恒等映射。这样第二层若也用恒等投影，拿到的 K 和 V 就是第一层的输出。下面的例子可直接在 PyTorch CPU 上运行。

```python
import torch

def layer(x):
    scores = x @ x.T
    visible = torch.ones_like(scores, dtype=torch.bool).tril()
    return scores.masked_fill(~visible, -torch.inf).softmax(-1) @ x

manual = torch.tensor([[1.], [2.]], dtype=torch.float64)
retrieved = torch.tensor([[3.]], dtype=torch.float64)
old = layer(manual)
new = layer(torch.cat((retrieved, manual)))[1:]
print("旧上文", old[:, 0].tolist())
print("新上文", new[:, 0].tolist())
assert not torch.allclose(old, new)
```

单独处理 M 得到约 `[1.0000, 1.8808]`；在开头加入 R 后，M 两个位置变成了 `[2.7616, 2.8509]`。即使不使用位置编码，旧 KV 到第二层也已经出现偏差。真实模型里的偏差分布决定了选择性重算能保留多少收益。

把材料 R 去掉，手册又回到输入开头，此时可以用普通前缀匹配。[UCM 的两次请求实验](/ucm/01-overview/03-quick-start/) 演示怎样把这样的前缀从外部存储读回。
