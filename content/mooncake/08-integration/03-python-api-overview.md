---
title: "Python API 的对象、地址与生命周期"
weight: 3
---

同一份手册可以被两种方式传递：只有名字 `manual-v1` 时，读者向 Store 查询对象；已经知道双方内存地址时，应用可以直接让 Transfer Engine 搬字节。下面用 Mooncake v0.3.14 默认的 C++ Store Python 绑定说明这些选择。`MOONCAKE_STORE_BACKEND` 在导入 `mooncake.store` 时生效，含 Store-RS 的构建另有支持范围。[Python Store](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/mooncake-store.md)

| 入口 | 调用方需要提供 | 成功能证明什么 |
|---|---|---|
| `MooncakeDistributedStore` | Master 地址、传输配置、key、对象或缓冲 | 可按该 key 找到副本并取到字节 |
| `TransferEngine` | 注册的源/目标地址、字节长度、对端名称 | 指定的一次搬运已完成；命名、通知、内存何时释放由调用方管理 |
| Conductor | 模型、租户、分块与请求 token | 返回各实例及各层的前缀匹配估计，查询不会分配 vLLM 目标块 |
| EP 后端 | 专家并行的分派和归并数据 | 对应通信操作按其同步契约完成。[EP 接口](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/ep-backend.md) |

## 先用 bytes 建立参照，再使用已注册缓冲

[部署章节](/mooncake/10-deployment/01-deployment-guide/)提供 Master 的启动方式。例中的进程主动贡献 128 MiB，便于在只有 Master 和此进程时运行。对象内容采用固定字节，再把返回的数据和实际读取的长度分别断言。每次运行使用独立 key，避免将上次的旧对象认作本次成功写入。

```python
import ctypes
import uuid
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
key = f"manual-{uuid.uuid4().hex}"
value = b"shared-manual" * 1024
size = len(value)
try:
    status = store.setup(
        local_hostname="127.0.0.1", metadata_server="P2PHANDSHAKE",
        global_segment_size=128 * 1024**2, local_buffer_size=64 * 1024**2,
        protocol="tcp", rdma_devices="", master_server_addr="127.0.0.1:50051",
    )
    assert status == 0, status
    assert store.put(key, value) == 0
    assert store.get(key) == value

    # ctypes 对象拥有内存；直到 unregister_buffer 完成才允许销毁。
    dest = ctypes.create_string_buffer(size)
    addr = ctypes.addressof(dest)
    assert store.register_buffer(addr, size) == 0
    try:
        read = store.get_into(key, addr, size)
        assert read == size, read
        assert dest.raw == value
        print(key, read)
    finally:
        store.unregister_buffer(addr)
finally:
    store.close()
```

`put` 和 `register_buffer` 的成功值为 0，`get_into` 成功时返回读取的字节数，本例为 13,312。`is_exist` 返回 1、0、负数分别表示存在、不存在、查询出错；不能直接用 `bool(status)` 判断命中。目标缓冲要能容纳整个对象。若源本来就在长生命周期的注册区，可用 `put_from(key, addr, size)` 避免先构造一个 Python `bytes`，仍然要等操作完成再释放注册。[缓冲注册及读写契约](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/mooncake-store.md)

## 沿裸传输接口检查生命周期

假设 D 进程拥有目标缓冲，P 进程拥有已有数据。两边分别初始化引擎、注册自己的内存，D 经应用的控制通道发出**可达的引擎端点、目标地址、可写长度**。目标指针仅在 D 进程的地址空间有意义。P 把它作为远端地址交给传输引擎，由 D 端按该进程的地址空间解释。

{{< trace >}}
title: "两个独立进程怎样交接一次远端写入"
actors: ["目的进程 D", "控制通道", "源进程 P", "传输引擎"]
steps:
  - from: 0
    to: 1
    label: "发布缓冲"
    detail: "D 注册目标缓冲并保持引用，把可达端点、地址和长度发到控制通道。"
  - from: 1
    to: 2
    label: "获得目的描述"
    detail: "P 取得 D 的传输参数并注册自己的源缓冲。"
  - from: 2
    to: 3
    label: "写入远端"
    detail: "P 把远端地址和长度交给传输引擎。"
  - from: 3
    to: 2
    label: "返回完成"
    detail: "P 通过同步返回值或异步状态确认这份范围已经传输。"
  - from: 2
    to: 0
    label: "交给 D 消费"
    detail: "应用协议通知 D 可以读取，双方按完成条件回收自己的内存。"
{{< /trace >}}

同步 `transfer_sync_write` 以 0 表示成功。异步 `transfer_submit_write` 返回批次 ID；`transfer_check_status` 的 0 表示进行中，1 才是成功，负值表示失败或超时。超时后先保留相关缓冲区，等传输停止访问后再复用地址。需要区分“失败但已经停止访问”和“尚不确定”时，查看 scatter ticket 的 `drained` 与 `drain` 契约。具体到 vLLM，P 的指针数组、目标地址与释放通知由 connector 守护。[Transfer Engine 接口](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/transfer-engine.md) · [P 端跟踪](/mooncake/12-code-walkthrough/01-prefill-request-trace/)

### 不经过 Store，实际传一次 13 字节

保存为 `transfer_pair.py`，使用相同的 Linux 非 CUDA 环境执行 `python transfer_pair.py`。父进程是 D，子进程是 P。演示使用子进程参数传递端点和地址，用子进程退出通知发送已完成；这两个进程都不用连接前面的 Store Master。

```python
import ctypes
import subprocess
import sys
from mooncake.engine import TransferEngine

payload = b"shared-manual"
engine = TransferEngine()
host = "127.0.0.1"
assert engine.initialize(host, "P2PHANDSHAKE", "tcp", "") == 0
buffer = ctypes.create_string_buffer(len(payload))
address = ctypes.addressof(buffer)
assert engine.register_memory(address, len(payload)) == 0

if len(sys.argv) == 1:
    # 目的进程把自己的端点与地址交给子进程。
    endpoint = f"{host}:{engine.get_rpc_port()}"
    subprocess.run([sys.executable, __file__, endpoint, str(address)], check=True)
    assert buffer.raw == payload
    print("received", buffer.raw)
else:
    endpoint, peer_address = sys.argv[1:]
    buffer.raw = payload
    result = engine.transfer_sync_write(endpoint, address, int(peer_address), len(payload))
    assert result == 0, result

assert engine.unregister_memory(address) == 0
```

在 Linux amd64、Python 3.12.11、Mooncake 非 CUDA wheel 0.3.14 上，父进程读回 `b'shared-manual'`。替换为跨机运行时，需要应用自己的控制通道和对端可达的端点；替换成 GPU 内存，还需匹配设备运行库并保持底层存储及计算活动的生命周期。[上游两进程传输示例](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/design/transfer-engine/index.md)

## 前缀路由怎样使用查询结果

两个实例都声称覆盖 32 个 token 时，一个可以命中设备层，另一个可能要从主机或磁盘读回。检查返回状态和 `hits`，再读具体实例的 `rank_matches`、`longest_matched`，不同层的覆盖范围不能简单相加。模型、租户、cache salt 及分块方式需与发布者匹配。路由选中实例以后，实际能读出的对象与字节仍由数据路径确认。[Conductor 的字段与返回语义](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/conductor.md)
