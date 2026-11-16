---
title: "Python 客户端、端点与服务生命周期"
weight: 4
---

自定义服务的运行边界可以用一对同名 endpoint 观察。在一个 worker 里创建 `runtime.endpoint("cookbook.echo.generate")` 并服务 handler；在另一进程创建同名 endpoint，通过 client 等到可用实例，再发起生成调用。两个进程各自拥有 runtime，成员变化由发现后端通知。

将下面保存为 `echo_client.py`，与上一节启动的字符串服务配套。示例只发一次请求，并给服务发现设置 20 秒上限，方便在本机观察地址配错时的结果。

```python
import asyncio

from dynamo.runtime import DistributedRuntime, dynamo_worker


@dynamo_worker()
async def read_echo(runtime: DistributedRuntime):
    endpoint = runtime.endpoint("cookbook.echo.generate")
    client = await endpoint.client()
    await asyncio.wait_for(client.wait_for_instances(), timeout=20)
    responses = await client.generate("hello")
    messages = [item.data() async for item in responses]
    assert messages == ["hello"], messages
    print(messages)


if __name__ == "__main__":
    asyncio.run(read_echo())
```

```bash
DYN_DISCOVERY_BACKEND=file python3 echo_client.py
```

“等待实例”解决服务发现和注册时间差。若一直没有返回，核对两端 endpoint 全名、发现配置与 worker 启动日志。如果已能找到实例但生成流失败，应转查 handler 输入输出类型、请求传输与取消。上游 1.6.2 的 hello-world 示例展示了完整的启动与关闭边界。

[服务端生命周期](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/custom_backend/hello_world/hello_world.py) · [客户端消费流](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/examples/custom_backend/hello_world/client.py) · [运行时端点](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/lib/runtime/src/component/endpoint.rs)
