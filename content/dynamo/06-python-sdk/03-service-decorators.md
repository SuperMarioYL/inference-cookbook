---
title: "用 Python 注册流式 endpoint"
weight: 3
---

Dynamo 1.6.2 的自定义 worker 可以通过 `dynamo.runtime` 暴露异步处理函数。一个 endpoint 的全名由 namespace、component、endpoint 三段组成。下面是单 worker 的字符串回显，适合看懂注册过程，无需加载模型权重。

```python
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_endpoint, dynamo_worker

@dynamo_endpoint(str, str)
async def echo(message: str):
    yield message

@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("cookbook.echo.generate")
    await endpoint.serve_endpoint(echo)

if __name__ == "__main__":
    asyncio.run(worker())
```

装饰器描述 handler 的输入与输出，并给 worker 准备 runtime。真正让客户端能找到服务的是创建同名 endpoint 并在 worker 生命周期内执行 `serve_endpoint`。客户端要使用相同命名空间和发现配置。上游完整示例还提供独立 client 和部署说明。

## 用两个终端验证端点协议

将上面的服务代码保存为 `echo_worker.py`，与下一节客户端放到同一个已安装 1.6.2 的环境。运行时选择本机文件发现，两个进程使用相同账户、工作环境和命名空间。先在 Linux 环境安装对应版本，再在第一终端运行服务端。

```bash
python3 -m pip install ai-dynamo-runtime==1.6.2
DYN_DISCOVERY_BACKEND=file python3 echo_worker.py
```

在相同的 Linux 环境里准备客户端文件 `echo_client.py`，确保客户端同样指定文件发现。下面的客户端只执行一次调用，并检查返回流的完整内容。

```python
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def client(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("cookbook.echo.generate")
    consumer = await endpoint.client()
    await asyncio.wait_for(consumer.wait_for_instances(), timeout=18)
    stream = await consumer.generate("cookbook")
    result = [message.data() async for message in stream]
    assert result == ["cookbook"], result
    print(result)


if __name__ == "__main__":
    asyncio.run(client())
```

```bash
DYN_DISCOVERY_BACKEND=file python3 echo_client.py
```

在 Linux 容器的 `ai-dynamo-runtime==1.6.2` 环境中，以上端点通过文件发现和默认 TCP 请求通道返回了 `['cookbook']`。模型权重和 GPU 不参与这个实验。如果把客户端端点名改为 `cookbook.echo.other`，等待实例会超时，可据此区分命名与输出类型的问题。更多运行时状态见[客户端生命周期](/dynamo/06-python-sdk/04-dynamo-service-class/)。

[上游服务端](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/custom_backend/hello_world/hello_world.py) · [上游客户端](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/custom_backend/hello_world/client.py)
