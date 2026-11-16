---
title: "代码地图"
weight: 4
---

## 概述

从手里的问题选入口。算子走错设备时从 Dispatcher 查起，张量之间意外共享数据时先看 Storage 和 stride。

## 源码仓库结构

```
pytorch/
├── torch/                      # Python 前端
│   ├── __init__.py            # torch 命名空间入口
│   ├── nn/                    # 神经网络模块
│   ├── optim/                 # 优化器
│   ├── autograd/              # 自动微分
│   ├── utils/                 # 工具函数
│   ├── distributed/           # 分布式训练
│   ├── _dynamo/               # TorchDynamo 编译器
│   ├── fx/                    # FX 图系统
│   ├── _inductor/             # TorchInductor 后端
│   └── cuda/                  # CUDA 接口
├── aten/                      # ATen（A Tensor Library）
│   ├── src/ATen/              # ATen 核心实现
│   │   ├── core/              # 核心数据结构
│   │   ├── native/            # CPU/CUDA 算子实现
│   │   ├── templates/         # 代码生成模板
│   │   └── Tensor.h           # Tensor 类定义
├── c10/                       # Core 10 基础库
│   ├── core/                  # 核心工具
│   ├── cuda/                  # CUDA 工具
│   └── util/                  # 通用工具
├── torch/csrc/                # C++ 扩展和绑定
│   ├── autograd/              # Autograd C++ 实现
│   ├── jit/                   # TorchScript JIT
│   ├── distributed/           # 分布式 C++ 实现
│   └── api/                   # C++ Frontend API
└── test/                      # 测试代码
```

## 核心模块代码路径

### 1. Tensor 和 Storage

| 组件 | 路径 | 说明 |
|------|------|------|
| Tensor 定义 | `aten/src/ATen/Tensor.h` | Tensor 类的公共接口 |
| TensorImpl | `c10/core/TensorImpl.h` | Tensor 的底层实现 |
| Storage | `c10/core/Storage.h` | 数据存储抽象 |
| Allocator | `c10/core/Allocator.h` | 内存分配器接口 |
| CUDA Allocator | `c10/cuda/CUDACachingAllocator.cpp` | CUDA 缓存分配器 |
| Dtype | `c10/core/ScalarType.h` | 数据类型定义 |
| Device | `c10/core/Device.h` | 设备抽象 |

**关键文件详解**：

```cpp
// c10/core/TensorImpl.h - Tensor 的核心实现
class TensorImpl {
  Storage storage_;              // 底层数据存储
  DispatchKeySet key_set_;       // 分发键集合
  IntArrayRef sizes_;            // 形状
  IntArrayRef strides_;          // 步长
  int64_t numel_;                // 元素总数
  caffe2::TypeMeta dtype_;       // 数据类型
  // ...
};
```

### 2. Autograd（自动微分）

| 组件 | 路径 | 说明 |
|------|------|------|
| Python API | `torch/autograd/__init__.py` | Autograd Python 接口 |
| Variable | `torch/csrc/autograd/variable.h` | 自动求导 Tensor 封装 |
| Function | `torch/csrc/autograd/function.h` | 反向传播函数基类 |
| Engine | `torch/csrc/autograd/engine.h` | 反向传播执行引擎 |
| Saved Tensors | `torch/csrc/autograd/saved_variable.h` | 保存的中间张量 |
| Grad Mode | `torch/csrc/autograd/grad_mode.h` | 梯度模式控制 |

**计算图构建流程**：
```
torch/autograd/grad_mode.py        # 梯度开关
→ torch/csrc/autograd/variable.cpp # Variable 创建
→ torch/csrc/autograd/function.cpp # GradFn 绑定
```

**反向传播流程**：
```
torch.Tensor.backward()
→ torch/csrc/autograd/python_engine.cpp::backward()
→ torch/csrc/autograd/engine.cpp::execute()
→ torch/csrc/autograd/function.cpp::apply()
```

### 3. nn.Module

| 组件 | 路径 | 说明 |
|------|------|------|
| Module 基类 | `torch/nn/modules/module.py` | nn.Module 实现 |
| Parameter | `torch/nn/parameter.py` | 参数封装 |
| Functional | `torch/nn/functional.py` | 无状态函数API |
| 卷积层 | `torch/nn/modules/conv.py` | Conv1d/Conv2d/Conv3d |
| 线性层 | `torch/nn/modules/linear.py` | Linear |
| 激活函数 | `torch/nn/modules/activation.py` | ReLU/Sigmoid 等 |
| 归一化 | `torch/nn/modules/normalization.py` | BatchNorm/LayerNorm |

**模块注册流程**：
```python
# torch/nn/modules/module.py
class Module:
    def __setattr__(self, name, value):
        # 自动注册 Parameter 和子 Module
        if isinstance(value, Parameter):
            self._parameters[name] = value
        elif isinstance(value, Module):
            self._modules[name] = value
```

### 4. Optimizer

| 组件 | 路径 | 说明 |
|------|------|------|
| Optimizer 基类 | `torch/optim/optimizer.py` | 优化器基类 |
| SGD | `torch/optim/sgd.py` | 随机梯度下降 |
| Adam | `torch/optim/adam.py` | Adam 优化器 |
| AdamW | `torch/optim/adamw.py` | Adam with weight decay |
| LR Scheduler | `torch/optim/lr_scheduler.py` | 学习率调度器 |

### 5. Dispatch（算子分发）

| 组件 | 路径 | 说明 |
|------|------|------|
| DispatchKey | `c10/core/DispatchKey.h` | 分发键定义 |
| Dispatcher | `c10/core/impl/LocalDispatchKeySet.h` | 分发器核心 |
| Operator Registration | `aten/src/ATen/core/op_registration/` | 算子注册 |
| Boxing | `aten/src/ATen/core/boxing/` | 装箱机制 |
| Native Functions | `aten/src/ATen/native/` | 原生算子实现 |

**定位稠密张量的 add**：在 `aten/src/ATen/native/native_functions.yaml` 搜 `add.Tensor`，查看它的 `structured_delegate: add.out`。沿 `add.out` 的 ufunc 配置找到 `aten/src/ATen/native/ufunc/add.h` 和对应构建生成物。稀疏与其他布局可以走不同的 dispatch 注册。[算子描述](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/native_functions.yaml) · [add ufunc](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/ufunc/add.h)

### 6. 编译器栈

#### TorchDynamo

| 组件 | 路径 | 说明 |
|------|------|------|
| 主入口 | `torch/_dynamo/eval_frame.py` | 字节码拦截入口 |
| 字节码分析 | `torch/_dynamo/symbolic_convert.py` | 字节码符号执行 |
| Guard 系统 | `torch/_dynamo/guards.py` | 保护条件管理 |
| 图生成 | `torch/_dynamo/output_graph.py` | 输出 FX 图 |

#### FX

| 组件 | 路径 | 说明 |
|------|------|------|
| Graph | `torch/fx/graph.py` | 图数据结构 |
| Node | `torch/fx/node.py` | 节点定义 |
| GraphModule | `torch/fx/graph_module.py` | 可执行图模块 |
| Tracer | `torch/fx/proxy.py` | 图追踪器 |
| Passes | `torch/fx/passes/` | 图变换 Pass |

#### TorchInductor

| 组件 | 路径 | 说明 |
|------|------|------|
| 编译入口 | `torch/_inductor/compile_fx.py` | 编译 FX 图 |
| IR Lowering | `torch/_inductor/lowering.py` | IR 降级 |
| Triton Codegen | `torch/_inductor/codegen/triton.py` | Triton 代码生成 |
| C++ Codegen | `torch/_inductor/codegen/cpp.py` | C++ 代码生成 |
| Scheduler | `torch/_inductor/scheduler.py` | 调度和融合 |

### 7. 分布式训练

| 组件 | 路径 | 说明 |
|------|------|------|
| DDP | `torch/nn/parallel/distributed.py` | 分布式数据并行 |
| FSDP | `torch/distributed/fsdp/fully_sharded_data_parallel.py` | 全分片数据并行 |
| c10d | `torch/distributed/distributed_c10d.py` | 分布式通信库 |
| ProcessGroup | `torch/csrc/distributed/c10d/ProcessGroup.hpp` | 进程组抽象 |
| NCCL Backend | `torch/csrc/distributed/c10d/nccl2/` | 2.15 默认 NCCL 后端；ProcessGroupNCCL.cpp 为旧实现 |

### 8. CUDA

| 组件 | 路径 | 说明 |
|------|------|------|
| CUDA API | `torch/cuda/__init__.py` | CUDA Python 接口 |
| Caching Allocator | `c10/cuda/CUDACachingAllocator.cpp` | CUDA 缓存分配器 |
| Stream | `c10/cuda/CUDAStream.h` | CUDA 流管理 |
| Event | `c10/cuda/CUDAEvent.h` | CUDA 事件 |
| AMP | `torch/amp/autocast_mode.py` | 自动混合精度 |

## 快速定位技巧

### 1. 查找 Python API 实现

```bash
# 例如：查找 torch.nn.functional.relu
cd pytorch
grep -r "def relu" torch/nn/functional.py
```

### 2. 查找 C++ 算子实现

```bash
# 例如：查找 add 算子的 CPU 实现
grep -r "add_out" aten/src/ATen/native/BinaryOps.cpp

# 查找 CUDA 实现
grep -r "add_kernel" aten/src/ATen/native/cuda/
```

### 3. 查找类定义

```bash
# 例如：查找 TensorImpl 类
find . -name "*.h" | xargs grep "class TensorImpl"
```

### 4. 查找 Pybind11 绑定

```bash
# 例如：查找 backward 函数的 Python 绑定
grep -r "def.*backward" torch/csrc/autograd/python_engine.cpp
```

## 代码生成文件

PyTorch 使用代码生成来自动生成大量重复代码：

| 生成器 | 输入 | 输出 | 说明 |
|--------|------|------|------|
| gen.py | `aten/src/ATen/native/native_functions.yaml` | `aten/src/ATen/ops/` | 算子声明和分发代码 |
| autograd_gen.py | `tools/autograd/derivatives.yaml` | `torch/csrc/autograd/generated/` | 反向传播函数 |

修改算子定义或导数公式后，按[源码构建说明](/pytorch/00-overview/02-build-and-dev-env/)重新运行构建，CMake 会触发对应的生成步骤。

## 调试技巧

### 1. 打印 Tensor 元信息

```cpp
// 在 C++ 代码中
std::cout << self.sizes() << std::endl;        // 形状
std::cout << self.strides() << std::endl;      // 步长
std::cout << self.dtype() << std::endl;        // 数据类型
std::cout << self.device() << std::endl;       // 设备
std::cout << self.key_set() << std::endl;      // DispatchKeySet
```

### 2. 追踪算子调用栈

```python
# Python 侧
import torch
torch._C._set_print_stack_traces_on_fatal_signal(True)
```

### 3. 查看生成的 Dynamo 图

```python
import torch._dynamo as dynamo

@torch.compile
def model(x):
    return x * 2

# 查看生成的图
dynamo.reset()
torch._dynamo.config.verbose = True
model(torch.randn(10))
```

## 推荐阅读顺序

1. **入门**：先读 Python 层代码（`torch/nn/`, `torch/optim/`）
2. **进阶**：理解 C++ 绑定（`torch/csrc/`）
3. **深入**：研究核心实现（`c10/`, `aten/`）
4. **专家**：分析编译器栈（`torch/_dynamo/`, `torch/_inductor/`）

## 参考资源

- [PyTorch 源码仓库](https://github.com/pytorch/pytorch)
- [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)
- [PyTorch C++ API 文档](https://pytorch.org/cppdocs/)
