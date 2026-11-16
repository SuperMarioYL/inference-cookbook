---
title: "项目结构"
weight: 1
---

作者原型从 `example/blend.py` 发起两次前向。顺着模型执行器往下走，`llama.py` 负责选层，`xformers.py` 负责选位置和写回。

## 目录结构概览

```
CacheBlend/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖 (rouge_score)
├── example/                           # 示例脚本
│   ├── blend.py                       # 基础示例
│   ├── blend_musique.py               # MusiQue 数据集评估
│   ├── blend_samsum.py                # SAMSum 数据集评估
│   ├── blend_wikimqa.py               # WikiMQA 数据集评估
│   └── utils.py                       # 工具函数
├── inputs/                            # 输入数据
│   ├── 1-10.json                      # 测试样本
│   ├── musique_s.json                 # MusiQue 数据集
│   ├── samsum.json                    # SAMSum 数据集
│   └── wikimqa_s.json                 # WikiMQA 数据集
└── vllm_blend/                        # 修改后的 vLLM
    ├── vllm/                          # 核心代码
    │   ├── attention/                 # 注意力机制
    │   │   ├── backends/
    │   │   │   └── xformers.py        # CacheBlend 核心
    │   │   └── layer.py               # 注意力层接口
    │   ├── model_executor/
    │   │   └── models/
    │   │       └── llama.py           # CacheBlend 核心
    │   ├── engine/                    # 推理引擎
    │   ├── core/                      # 核心模块
    │   └── worker/                    # Worker 进程
    ├── csrc/                          # CUDA 核心代码
    └── setup.py                       # 安装脚本
```

## 核心文件清单

| 文件 | 行数 | 核心功能 |
|------|------|----------|
| `llama.py` | ~523 | LlamaModel、LlamaAttention 实现，状态机管理 |
| `xformers.py` | ~571 | HKVD 选择、KV 融合、注意力计算 |
| `blend.py` | ~108 | 使用示例，展示完整工作流程 |
| `layer.py` | ~50 | 注意力层接口，参数传递 |

## 依赖关系图

```mermaid
graph TB
    subgraph "代码依赖关系"
        Example["example/blend.py"] --> LLM["vllm.LLM"]
        LLM --> Engine["LLMEngine"]
        Engine --> Executor["ModelExecutor"]
        Executor --> Runner["ModelRunner"]
        Runner --> Model["LlamaForCausalLM"]
        Model --> LlamaModel["LlamaModel"]
        LlamaModel --> LlamaAttn["LlamaAttention"]
        LlamaAttn --> Attention["Attention Layer"]
        Attention --> XFormers["XFormersImpl"]
    end

    style LlamaModel fill:#ff9999
    style LlamaAttn fill:#ff9999
    style XFormers fill:#ff9999
```

---

**下一步**: [核心数据结构](./02-data-structures.md)
