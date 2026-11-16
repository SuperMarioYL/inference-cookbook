---
title: "运行 vLLM 服务与离线推理"
weight: 1
---

以可访问 Hugging Face 模型的 Linux NVIDIA 环境为例，先用同一个模型打通 HTTP 与离线推理。0.33.0 官方 CUDA wheel 与容器使用 CUDA 13.0；机器需要兼容的驱动、NVIDIA Container Toolkit，以及足以载入模型和 KV 缓存的可见显存。

## 启动并验证 HTTP 服务

```bash
docker run --rm --gpus all --ipc=host -p 127.0.0.1:8000:8000 \
  vllm/vllm-openai:v0.33.0 Qwen/Qwen3-0.6B
```

首次启动包含模型下载、加载和执行环境初始化。在另一个终端列出服务公布的模型，再发一个短请求。请求中的模型名应与 `/v1/models` 返回值一致。

```bash
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"什么是 KV 缓存？"}],"max_tokens":64}'
```

非流式响应可从 `choices[0].message` 读取。服务在 GPU 上加载完成才具备推理能力；在容器日志中区分模型下载失败、驱动不兼容和显存不足。

## 在 ARM CPU 上复现同一个 HTTP 请求

有 Linux/arm64 Docker 环境时，也可以使用同版本 CPU 镜像。先用 Hugging Face CLI 下载 Qwen3-0.6B。下面固定了权重版本，容器只读挂载模型，KV 池设置为 96 MiB，适合验证短请求。[ARM CPU 安装说明](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/getting_started/installation/cpu.arm.inc.md)

```bash
hf download Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca --local-dir ./qwen
docker run --rm --platform linux/arm64 --shm-size=1g -p 127.0.0.1:8000:8000 \
  -e VLLM_CPU_OMP_THREADS_BIND=nobind -v "$PWD/qwen:/model:ro" \
  vllm/vllm-openai-cpu:v0.33.0-arm64 /model --served-model-name Qwen/Qwen3-0.6B \
  --dtype float32 --enforce-eager --max-model-len 256 --max-num-seqs 1 \
  --max-num-batched-tokens 256 --kv-cache-memory-bytes 100663296
```

打开另一个终端，使用前面的模型列表和聊天端点。实测在请求体加上 `"temperature": 0, "chat_template_kwargs": {"enable_thinking": false}` 后，模型列表返回 `Qwen/Qwen3-0.6B`，聊天响应记录了 18 个输入 token 和 64 个输出 token。达到本例的输出上限时，`finish_reason` 为 `length`。

## 在 Python 中离线生成

在已安装 `vllm==0.33.0` 的独立环境运行。若复用同一块 GPU，先停止上面的服务以释放它预留的显存。

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-0.6B")
results = llm.generate(["KV cache stores"], SamplingParams(max_tokens=16))
print(results[0].outputs[0].text)
```

`LLM.generate` 接收原始提示词；HTTP 示例走聊天模板。两条路径最终都进入 V1 引擎。在线进程通过 `AsyncLLM` 和 core client 向 EngineCore 投递请求；离线调用把一个或多个提示词交给 `LLM`，调用方接收每个提示词的输出对象。

## 服务端如何向客户端推送增量

{{< trace >}}
title: 聊天请求 R7 的首个增量
actors: ["客户端", "API 进程", "引擎核心"]
steps:
  - from: 0
    to: 1
    label: "提交 R7"
    detail: "客户端在聊天请求中设置 stream=true。"
  - from: 1
    to: 1
    label: "套用模板"
    detail: "前端将消息渲染为模型输入，并保留 R7 的输出状态。"
  - from: 1
    to: 2
    label: "请求入队"
    detail: "异步引擎通过 core client 把 R7 交给引擎核心。"
  - from: 2
    to: 2
    label: "执行本轮"
    detail: "调度器给 R7 分配 token 预算，worker 完成模型计算。"
  - from: 2
    to: 1
    label: "返回新 ID"
    detail: "前端将属于 R7 的 token ID 解码为可发出的增量。"
  - from: 1
    to: 0
    label: "首段文本"
    detail: "客户端从这条请求的 SSE 流里读到文本。"
{{< /trace >}}

## 用一个请求对齐两种入口

给 HTTP 端和离线端相同可见文本时，应先确认聊天模板、特殊 token 和采样参数。离线 `generate` 收到原始字符串，而 chat endpoint 会先把消息列表渲染成 prompt；若要比较结果，离线端应使用相同聊天模板或相应 chat 接口。固定 temperature、输出上限和随机种子后，再看输入 token ID 与输出 token ID。

## 源码断点顺序

从固定版本的 API server 建立引擎客户端处开始，进入 `vllm/v1/engine/async_llm.py` 的请求入口，再从 `core_client.py` 过进程边界。在 EngineCore 中查看调度记录；HTTP 处理进程里的普通 Python 断点通常捕捉不到设备 worker 的执行。

[快速开始](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/getting_started/quickstart.md) · [进程边界](https://github.com/vllm-project/vllm/blob/v0.33.0/docs/design/arch_overview.md) · [AsyncLLM](https://github.com/vllm-project/vllm/blob/v0.33.0/vllm/v1/engine/async_llm.py)
