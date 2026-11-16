---
title: "vLLM Worker 的启动、注册和执行"
weight: 4
---

Dynamo 1.6.2 的 vLLM worker 以 `python3 -m dynamo.vllm` 启动。它将后端模型引擎接入运行时，为其他组件提供可调用的推理端点。这里跟踪官方 CUDA 13 镜像的普通生成 worker，其中 vLLM 版本为 0.31.0。

## 顺着 worker 入口读一遍启动路径

| 在源码中找 | 解决的问题 |
|---|---|
| `worker` 的配置解析和模型准备 | 把命令行选项转为 Dynamo 与 vLLM 各自的设置，在需要时取得模型。 |
| `create_runtime` | 以所选发现、请求和事件后端建立运行时。headless 分支会绕过普通端点注册流程。 |
| `WorkerFactory` 及传入的初始化函数 | 不同角色需要启动不同类型的 handler、引擎与事件发布能力。 |
| `register_vllm_model` | 将模型和可服务的 endpoint 对外公布，供 frontend 与路由器查找。 |

{{< trace >}}
title: vLLM worker 何时出现在模型列表
actors: ["启动进程", "模型引擎", "发现服务", "Frontend"]
steps:
  - from: 0
    to: 1
    label: "加载权重"
    detail: "worker 按指定模型初始化设备与推理引擎。"
  - from: 1
    to: 0
    label: "引擎就绪"
    detail: "角色和 handler 可用后，worker 准备注册端点与模型。"
  - from: 0
    to: 2
    label: "发布模型"
    detail: "注册信息包含 frontend 查找服务所需的模型身份和端点。"
  - from: 2
    to: 3
    label: "发现更新"
    detail: "Frontend 接收到可用成员，模型列表可展示该模型。"
  - from: 3
    to: 0
    label: "转发验证请求"
    detail: "真正的聊天请求到达 worker，执行生成并返回结果。"
{{< /trace >}}

例如在聚合部署中，一个 worker 收到请求后同时推进 prefill 和 decode。将启动参数换为分离部署时，P 要发布可供交接的 KV，D 要取得对应状态后继续解码；复现两种模式应使用对应启动脚本，记录各自设备和端口。

## 区分已启动与已经能生成

启动可以停在权重下载、初始化设备或模型注册。先查 worker 自身日志和退出状态，再查询 frontend 的模型列表，最后发一条真实生成请求。模型尚未注册时先查 namespace 与发现，模型已注册但没有输出时，再比较 handler 收到请求、设备执行及结果送回各步。

[worker 启动与注册](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/components/src/dynamo/vllm/main.py) · [请求处理器](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/components/src/dynamo/vllm/handlers.py) · [聚合部署](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/agg.sh) · [分离部署](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/backends/vllm/launch/disagg.sh)
