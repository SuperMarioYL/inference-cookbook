---
title: "在单机写入并读回 Mooncake Store"
weight: 3
---

这个练习在一台 Linux 主机上用 TCP 操作 Store，能独立验证 Master、客户端和传输链路。准备 Python 3.10+ 和足够的可用主存。以下示例的非 CUDA wheel 适合 Ubuntu；CUDA、Ascend 和其他设备有对应发行包。[v0.3.14 安装前提](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/getting_started/quick-start.md)

```bash
sudo apt-get install -y libcurl4 libibverbs1 rdma-core librdmacm1 libnuma1 liburing2
python3 -m venv .venv
. .venv/bin/activate
python -m pip install 'mooncake-transfer-engine-non-cuda==0.3.14'
mooncake_master
```

Master 默认在 50051 端口提供 RPC。保持它运行，另开终端，激活同一个虚拟环境并运行以下 Python 程序。`global_segment_size` 是本进程贡献给共享池的容量；`local_buffer_size` 是本地传输缓冲区。这个例子用握手交换元数据。

```python
from mooncake.store import MooncakeDistributedStore

store = MooncakeDistributedStore()
try:
    status = store.setup(
        local_hostname="localhost",
        metadata_server="P2PHANDSHAKE",
        global_segment_size=512 * 1024**2,
        local_buffer_size=128 * 1024**2,
        protocol="tcp",
        rdma_devices="",
        master_server_addr="127.0.0.1:50051",
    )
    assert status == 0, status
    value = b"chapter-one"
    assert store.put("cookbook/manual", value) == 0
    assert store.get("cookbook/manual") == value
    print("Store read-back passed")
finally:
    store.close()
```

如果 setup 失败，先检查 Master 地址和本机运行库。如果写入失败，检查是否有在线客户端贡献了可分配空间；读取时对照写入的 key 和字节。成功读回后再接入引擎，就能区分 Store 本身的问题和前缀匹配问题。[Python API](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/api-reference/python/mooncake-store.md) · [部署指南](../10-deployment/01-deployment-guide.md)

本例已在 Linux amd64 容器（Debian Bookworm、Python 3.12.11、非 CUDA wheel 0.3.14）读回原始字节，打印 `Store read-back passed`。Master 与客户端握手成功时，同进程本地副本可通过内存拷贝完成读回。要测网络数据传输，应把写入方与读取方放到不同进程或主机，并确认目标副本实际位于远端。

## 再验证一次跨进程读取

在同一 Linux amd64 环境，用 `MC_STORE_MEMCPY=0` 启动两个独立客户端进程并选择 TCP。写进程贡献 512 MiB，读进程把 `global_segment_size` 设为 0，只分配自己的传输缓冲。读进程能完整读回写者保存的 **13,312 字节**。

复现时保持上面的 Master 运行，将下面的脚本保存为 `shared_read.py`，执行 `MC_STORE_MEMCPY=0 python shared_read.py`。写者等待子进程退出再关闭 Store，保证作为副本持有者的进程仍在运行。

```python
import subprocess
import sys
from mooncake.store import MooncakeDistributedStore

reader = sys.argv[1:] == ["read"]
store = MooncakeDistributedStore()
try:
    status = store.setup(
        local_hostname="127.0.0.1", metadata_server="P2PHANDSHAKE",
        global_segment_size=0 if reader else 512 * 1024**2,
        local_buffer_size=128 * 1024**2, protocol="tcp", rdma_devices="",
        master_server_addr="127.0.0.1:50051",
    )
    assert status == 0, status
    key, payload = "cookbook/shared-manual", b"shared-manual" * 1024
    if reader:
        assert store.get(key) == payload
        print(f"read {len(payload)} bytes from the writer")
    else:
        assert store.put(key, payload) == 0
        subprocess.run([sys.executable, __file__, "read"], check=True)
finally:
    store.close()
```
