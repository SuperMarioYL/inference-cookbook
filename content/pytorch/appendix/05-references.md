---
title: "参考资源"
weight: 5
---

| 查什么 | 入口 |
|---|---|
| 安装要求、2.15 新功能和兼容变化 | [发布说明](https://pytorch.org/blog/pytorch-2-15-release-blog/)与[源码构建说明](https://github.com/pytorch/pytorch/blob/v2.15.0/README.md#from-source) |
| 精确定位函数、类型或模板 | [v2.15.0 源码树](https://github.com/pytorch/pytorch/tree/v2.15.0)与[本站源码地图](/pytorch/appendix/04-code-map/) |
| 查询公开接口的参数、返回值、dtype 与设备范围 | [2.15 API 文档](https://docs.pytorch.org/docs/2.15/) |
| 查编译边界、动态形状和调试日志 | [torch.compiler](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler.html) |
| 按设备核对混合精度与多卡行为 | [AMP](https://docs.pytorch.org/docs/2.15/amp.html)、[进程组](https://docs.pytorch.org/docs/2.15/distributed.html)、[FSDP2](https://docs.pytorch.org/docs/2.15/distributed.fsdp.fully_shard.html) |

查内部实现时，在 tag 中搜索符号名；编译生成的注册文件通常位于构建目录。接口是否受支持、默认值和限制以对应版本的公开文档为准。
