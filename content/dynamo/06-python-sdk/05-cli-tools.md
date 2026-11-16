---
title: "在本机启动 Dynamo frontend 和 worker"
weight: 5
---

在可使用 GPU 的 Linux 主机上准备 Dynamo **1.6.2** 对应后端环境，以下选择官方 CUDA 13 vLLM 镜像所带的 Python 模块。模型可访问、显存足以加载模型，两个终端要在同一容器或能访问同一发现目录的环境里工作。打开另一台机器或另一个没有共享目录的容器，会让文件发现看到不同的 worker 集合。

## 建立两个进程共用的容器环境

下面使用已发布的 CUDA 13 vLLM 镜像。第一个主机终端启动有名字的容器，第二个主机终端通过 `docker exec` 进入同一容器。容器将默认 HTTP 端口只映射到主机回环地址。

```bash
docker run --rm -it --name cookbook-dynamo --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 --entrypoint bash \
  nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.2
```

```bash
docker exec -it cookbook-dynamo bash
```

## 验证从前端到推理引擎的一次调用

先启动 frontend 校验请求和接收响应。

```bash
python3 -m dynamo.frontend --discovery-backend file
```

在第二个终端启动 vLLM worker。单 worker 同时完成预填充与解码。

```bash
python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file
```

等模型注册后，在第三个终端发请求。HTTP 进程存活与模型可以生成是不同的观察点，因此这里同时检查健康端点和模型输出。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","messages":[{"role":"user","content":"什么是预填充？"}],"max_tokens":64}'
```

如果模型列表还没有该模型，先查 worker 初始化日志，再核对发现后端、命名空间及文件可见性。两终端均可用 Ctrl+C 结束。P/D 分离还需要两个角色及后端传输配置，可对照此版官方双 GPU 脚本。

## 健康检查与真实请求各验证什么

在 1.6.2 中，`/health` 可在 200 响应中列出空的 endpoints 或 instances。`/v1/models/<model>/ready` 再检查指定模型的 worker 拓扑与推理入口。要检查 tokenizer 和后端一起生成的结果，给该模型发送一条 chat completion。 [1.6.2 健康检查语义](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/reference/observability/health-checks.mdx)

[官方快速开始](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/cli/getting-started/quickstart.mdx) · [vLLM 聚合启动脚本](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/agg.sh) · [vLLM 分离启动脚本](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh)
