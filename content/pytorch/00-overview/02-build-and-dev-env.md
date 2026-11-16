---
title: "开发环境搭建"
weight: 2
---

## 安装运行环境

只运行本栏 Python 示例可以使用 wheel。PyTorch 2.15 发布版提供 Python 3.11–3.15 支持；设备、操作系统与 Python ABI 一起决定可用 wheel。下面的例子面向有对应 wheel 的 Apple Silicon，计算显式放在 CPU 上。Linux CUDA 或 ROCm 环境应在官方安装页选择相应索引。[2.15 环境说明](https://pytorch.org/blog/pytorch-2-15-release-blog/) · [安装选择器](https://pytorch.org/get-started/locally/)

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install 'torch==2.15.0'
```

```python
import torch

x = torch.tensor([1.0, 2.0], device="cpu", requires_grad=True)
x.square().sum().backward()
print(torch.__version__, x.grad.tolist())  # 2.15.0；[2.0, 4.0]
print(torch.version.cuda, torch.version.hip)
```

## 从固定 tag 构建

修改 C++ 调度、原生内核或绑定时需要编译源码。2.15.0 使用 C++20，Linux GCC 至少为 11.3，顶层 CMake 至少为 3.27。开发环境安装在虚拟环境里，`pip` 需要支持 `--group`。以下命令展示 macOS/Unix shell 中的源码准备流程。[README](https://github.com/pytorch/pytorch/blob/v2.15.0/README.md#from-source) · [CMake 入口](https://github.com/pytorch/pytorch/blob/v2.15.0/CMakeLists.txt)

```bash
git clone --branch v2.15.0 --recursive https://github.com/pytorch/pytorch.git
cd pytorch
python -m pip install --upgrade pip
python -m pip install --group dev
```

Linux 下按上游要求安装 MKL 构建依赖（`python -m pip install mkl-static mkl-include`），使用虚拟环境时将其前缀加入 `CMAKE_PREFIX_PATH`；macOS 使用已配置的 Xcode C++ 工具链。下面仅构建 CPU 相关路径，可用 `MAX_JOBS` 控制编译并发。CUDA 开发还需与目标平台匹配的工具链，官方 2.15 发布的默认 CUDA wheel 采用 13.2，这与本机源码构建的选择是两个设置。

```bash
USE_CUDA=0 MAX_JOBS=4 python -m pip install --no-build-isolation -v -e .
```

```mermaid
flowchart LR
    P["pip editable 安装"] --> S["pyproject.toml / scikit-build-core"]
    S --> C["CMake 配置"]
    C --> G["torchgen 生成注册和绑定源码"]
    G --> B["编译、链接并安装二进制库"]
```

`pyproject.toml` 声明了 PEP 517 backend 和从 `MAX_JOBS` 到构建并发的映射。需要调试符号时在第一次配置前设置 `DEBUG=1` 或 `REL_WITH_DEB_INFO=1`；已经生成的 CMake cache 会影响后续配置。纯 Python 修改可直接从 editable 安装中观察；原生文件改动需要重新构建。[固定构建配置](https://github.com/pytorch/pytorch/blob/v2.15.0/pyproject.toml)

## 确认解释器加载了预期构建

离开源码目录后运行 `python -c 'import torch; print(torch.__file__, torch.__version__, torch.version.git_version)'`。若错误发生在加载扩展，先核对实际包路径；若由某个算子触发，保留最小输入并设置 `TORCH_SHOW_CPP_STACKTRACES=1` 获取原生调用栈。编译器行为可通过 `TORCH_LOGS="graph_breaks,recompiles"` 查看。[开发者环境变量](https://docs.pytorch.org/docs/2.15/torch_environment_variables.html) · [编译器诊断](/pytorch/05-advanced-topics/01-compiler-stack-overview/)

## 原生断点和生成源码

调试时先找到本机实际加载的共享库，再使用有符号的构建运行：

```bash
lldb -- python reproduce.py
```

在 lldb 中输入 `breakpoint set -n at::native::empty_cpu` 后运行 `run`。命中与否取决于复现脚本是否真的创建 CPU 张量、是否选中相应符号。定位一次算子注册时可以先在 `aten/src/ATen/native/native_functions.yaml` 查 schema，再找对应的 backend dispatch。CMake 调用 `torchgen/gen.py` 生成 C++ 注册与声明；生成结果用于对照断点，维护者修改 YAML、模板或手写实现，再重新构建。

| 正在追踪的行为 | 权威输入 |
|---|---|
| 算子的参数、返回值和设备分发 | [native_functions.yaml](https://github.com/pytorch/pytorch/blob/v2.15.0/aten/src/ATen/native/native_functions.yaml) |
| 反向导数公式、生成的 autograd Function | [derivatives.yaml](https://github.com/pytorch/pytorch/blob/v2.15.0/tools/autograd/derivatives.yaml) |
| 生成哪些声明和注册文件 | [torchgen](https://github.com/pytorch/pytorch/tree/v2.15.0/torchgen) |

调试 `torch.compile` 时先用 `TORCH_LOGS="graph_breaks,recompiles"` 检查捕获。若需要多帧编译记录，可以设置 `TORCH_TRACE` 指向一个空目录，再用 `tlparse` 读取；trace 中可能包含模型代码，要按对应环境保管。[编译诊断](https://docs.pytorch.org/docs/2.15/user_guide/torch_compiler/torch.compiler_troubleshooting.html)
