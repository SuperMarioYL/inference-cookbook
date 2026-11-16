---
title: "用 SGLang 0.5.23 启动文本服务"
weight: 2
---

下面用 Qwen3-0.6B 验证单 GPU 的 HTTP 入口。0.5.23 的 NVIDIA 发行环境使用 CUDA 13，Python 安装要求 3.11 或更高。请先确认 Linux 驱动兼容、GPU 对环境可见且模型可下载。使用其他设备时，查对应平台的安装页。

## 在 NVIDIA CUDA 环境启动

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install --prerelease=allow 'sglang==0.5.23'
sglang serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 30000
```

特定 CUDA 构建需要使用官方安装页列出的 wheel 索引。等待终端打印服务就绪信息，在同一台机器的另一终端执行以下命令。

```bash
curl -fsS http://127.0.0.1:30000/v1/models
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"一句话解释前缀缓存。"}],"max_tokens":64}'
```

查看返回的模型标识和 `choices[0].message`。若模型列表不可达，先查看启动进程是否仍在下载、加载或报驱动错误；若只有聊天调用失败，核对请求模型名和消息格式。测试结束可在服务终端按 Ctrl+C。

## 在 Apple Silicon 上使用 MLX 与 Metal

文本服务可通过 MLX 在 Apple GPU 上执行。这里按 0.5.23 的源码标签安装文本运行时。需要 Apple Silicon、macOS 14 或更新版本、Python 3.12 和 uv。这个例子使用 MLX 标准算子，不编译可选的 Metal 或多模态 Rust 扩展。[官方 Apple Metal 说明](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/hardware-platforms/apple_metal.mdx) · [该标签的 Darwin 依赖](https://github.com/sgl-project/sglang/blob/v0.5.23/python/pyproject_other.toml)

在一个新目录中获取源码，再切换到同一版本自带的非 CUDA 包配置：

```bash
git clone --depth 1 --branch v0.5.23 https://github.com/sgl-project/sglang.git
cd sglang
uv venv --python 3.12 .venv
source .venv/bin/activate
cp python/pyproject_other.toml python/pyproject.toml
SGLANG_BUILD_RUST_EXTS=none uv pip install -e 'python[srt_mps]'
```

将下面的 `MODEL_PATH` 设为本机下载完整的模型目录，包含权重、配置与 tokenizer 文件。有网络时也可以传入模型 ID，让运行时从 Hugging Face 获取文件。示例限制上下文和并发，便于首次验证。

```bash
MODEL_PATH=/path/to/Qwen3-0.6B
SGLANG_USE_MLX=1 sglang serve \
  --model-path "$MODEL_PATH" --served-model-name Qwen/Qwen3-0.6B \
  --cuda-graph-backend-decode disabled --cuda-graph-backend-prefill disabled \
  --context-length 1024 --max-total-tokens 2048 --max-running-requests 2 \
  --host 127.0.0.1 --port 30000
```

等日志出现 `Initializing MlxModelRunner` 并报告 HTTP 服务就绪，再在另一终端验证：

```bash
curl -fsS http://127.0.0.1:30000/v1/models
curl -fsS http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"What is two plus three? Reply with only the digit."}],"temperature":0,"max_tokens":32,"chat_template_kwargs":{"enable_thinking":false}}'
```

在 M1 Max、macOS 27.2、Python 3.12.13、PyTorch 2.13.0 与 MLX 0.32.4 上，以本地 Qwen3-0.6B 权重实测。模型列表给出了指定名称和 1024 的上下文限制，聊天接口返回了非空 `choices[0].message.content`、正数 `usage.completion_tokens` 和 `finish_reason: stop`。完成后按 Ctrl+C 退出服务。

## 调整显存前先看启动日志

`--mem-fraction-static` 决定静态分配所使用的显存比例，包括权重和 KV 缓存池。此版会根据可检测到的显存和预留空间计算默认值，显存探测失败时才回退到 0.88。调低该比例给执行留下空间，同时也会降低可容纳的 KV token 数。并发限制与提示词长度同样会影响能否接收下一条请求。

## 使用 Python 调用两种接口

### 方法一：OpenAI 兼容 API

SGLang 兼容 OpenAI API 格式：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:30000/v1",
    api_key="EMPTY"  # 此处服务未配置鉴权
)

# Chat Completion
response = client.chat.completions.create(
    model="Qwen/Qwen3-0.6B",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ],
    max_tokens=100,
    temperature=0.7
)

print(response.choices[0].message.content)
```

### 方法二：原生 API

```python
import requests

response = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": "What is the capital of France?",
        "sampling_params": {
            "max_new_tokens": 100,
            "temperature": 0.7
        }
    }
)

response.raise_for_status()
print(response.json()["text"])
```

## 离线批量生成

### 使用 sgl.Engine

```python
import sglang as sgl

def main():
    # 创建离线引擎
    engine = sgl.Engine(model_path="Qwen/Qwen3-0.6B")

    # 批量推理
    prompts = [
        "What is AI?",
        "Explain quantum computing.",
        "What is the meaning of life?"
    ]

    outputs = engine.generate(prompts, sampling_params={"max_new_tokens": 100})

    for prompt, output in zip(prompts, outputs):
        print(f"Prompt: {prompt}")
        print(f"Output: {output['text']}\n")

    # 关闭引擎
    engine.shutdown()


if __name__ == "__main__":
    main()
```

在脚本内运行离线引擎时，采用 Python 的 main 入口保护，避免多进程重复执行顶层初始化。先停掉占用同一设备的 HTTP 服务，再运行这个示例。

[安装与硬件条件](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/install.mdx) · [HTTP 快速开始](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/get-started/quickstart.mdx) · [参数默认行为](https://github.com/sgl-project/sglang/blob/v0.5.23/docs/docs/advanced_features/server_arguments.mdx)
