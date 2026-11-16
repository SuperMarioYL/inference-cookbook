---
title: "用两次请求验证 vLLM 外部前缀缓存"
weight: 3
---

下面选用发布矩阵中的 **vLLM 0.30.0、CUDA 12.9、CPython 3.12、UCM 0.9.1**。准备支持该 CUDA 运行时的 Linux GPU 主机、本地模型目录及可写缓存目录。官方已经把匹配的后端 wheel 装进镜像，避免另行猜测 PyTorch/CUDA 组合。[运行时与 wheel 对照](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

## 启动带外部存储的引擎

把以下 YAML 存为 `ucm.yaml`。示例使用挂载进容器的 `/cache` 作为 POSIX 存储，容量配置应小于磁盘可用空间。其他参数按同版本发布示例核对。[官方配置样例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/examples/ucm_config_example.yaml)

```yaml
ucm_connectors:
  - ucm_connector_name: UcmPipelineStore
    ucm_connector_config:
      store_pipeline: "Cache|Posix"
      storage_backends: /cache
      cache_buffer_capacity_gb: 2
      posix_capacity_gb: 8
      posix_io_engine: psync
```

`MODEL_DIR` 设置为已下载、单卡可加载且支持前缀缓存的模型绝对路径，例如 Qwen2.5 指令模型。`/v1/completions` 使用固定模型名 `cookbook`。此处关闭引擎本地前缀命中，便于观察第二次请求从 UCM 读回的块数。

```bash
mkdir -p ./ucm-data
docker run --rm --gpus all --ipc=host -p 7800:7800 \
  -e ENABLE_UCM_PATCH=1 -v "${MODEL_DIR:?set MODEL_DIR}:/model:ro" \
  -v "$PWD/ucm.yaml:/workspace/ucm.yaml:ro" -v "$PWD/ucm-data:/cache" \
  --entrypoint vllm ghcr.io/modelengine-group/vllm-openai:v0.30.0-cu129-ucm-0.9.1 \
  serve /model --served-model-name cookbook --port 7800 --enforce-eager \
  --max-model-len 4096 --block-size 128 --no-enable-prefix-caching \
  --kv-transfer-config '{"kv_connector":"UCMConnector","kv_connector_module_path":"ucm.integration.vllm.ucm_connector","kv_role":"kv_both","kv_connector_extra_config":{"UCM_CONFIG_FILE":"/workspace/ucm.yaml"}}'
```

## 发送同一份手册、两个问题

在另一终端运行。重复的手册应超过若干完整块，才能排除“文字看起来相同，但开头不足一个完整缓存块”的情况。若第一次请求的异步保存尚未结束，等写入完成后再运行第二次。

```python
import json
from urllib.request import Request, urlopen

manual = "设备每周检查电源、温度和日志；异常情况记入维护记录。\n" * 56
for question in ("每周检查哪些项目？", "异常应该记在哪里？"):
    payload = json.dumps({"model": "cookbook", "prompt": manual + question,
                          "max_tokens": 24, "temperature": 0}).encode()
    req = Request("http://localhost:7800/v1/completions", data=payload,
                  headers={"Content-Type": "application/json"})
    with urlopen(req) as response:
        print(json.load(response)["choices"][0]["text"])
```

先确认引擎确实加载了 connector，然后对照两次请求的 HBM 命中与外部命中。启用相应指标采集时，比较 `ucm:gpu_hbm_hit_tokens_total`、`ucm:ucm_hit_tokens_total` 和 POSIX 读取计数的增量。第一轮观察写入完成，第二轮观察外部命中和读取字节的增量。线上保留本地前缀缓存时，两层可以同时贡献命中。[指标说明](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md) · [上游 vLLM 示例](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/getting-started/quickstart_vllm.md)
