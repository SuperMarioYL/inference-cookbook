---
title: "UCM 的启动补丁如何选择版本"
weight: 4
---

启动 `vllm serve` 之前设置 `ENABLE_UCM_PATCH=1`。当 Python 载入 UCM 安装的启动钩子时，UCM 读取 vLLM 和 vLLM-Ascend 的版本，选择适用补丁并将集成逻辑装入相应进程。不同版本有不同的 scheduler、worker 或缓存布局入口，故障排查应看实际分支。[启动钩子](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/boot_patch.py) · [版本分派](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/apply_patch.py)

## 跟踪当前镜像的版本

```bash
python -m pip show vllm uc-manager
python -c 'import ucm; print(ucm.__file__)'
```

在与镜像一致的 tag 中查看 `get_supported_versions`，再找对应版本目录。例如 vLLM 0.30.0 的目录是 `v0300`。版本识别会处理 build、rc 或 post 后缀；具体适配需要时也读取完整版本。切换镜像后重新核对引擎、平台、ABI 和 UCM 版本，再用外部命中请求验证。[发布组合](https://github.com/ModelEngine-Group/unified-cache-management/releases/tag/v0.9.1)

启用稀疏算子还需要平台、模型及对应特性的匹配条件。`ENABLE_SPARSE=1` 的作用与启用通用补丁不同。[增强特性的版本范围](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)

## 首次导入 vLLM 时如何装入补丁

安装包包含 `ucm_patch.pth`，解释器启动时会尝试安装对应引擎的导入钩子。vLLM 路径的 `install_hook` 把 `VllmImportTrigger` 加到 `sys.meta_path`，已存在同类 finder 就跳过。第一次导入顶层 `vllm` 时，finder 临时绕开自己找到原模块的 spec，并用 `VllmPatchLoader` 包装原 loader。这决定了补丁是在原始模块完成加载后应用，而非在它尚未定义类时替换方法。[启动入口](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm_patch.pth) · [finder 和 loader](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/patch/boot_patch.py)

{{< trace >}}
title: "首次 import vLLM 时补丁在哪里生效"
actors: ["Python", "finder", "原 loader", "UCM 适配"]
steps:
  - from: 0
    to: 1
    label: "安装 finder"
    detail: "解释器启动时经 .pth 装入 import hook。"
  - from: 0
    to: 1
    label: "首次 import"
    detail: "finder 识别顶层 vllm，并找到原模块的加载方式。"
  - from: 1
    to: 2
    label: "执行原 loader"
    detail: "先让 vLLM 完成模块定义。"
  - from: 2
    to: 3
    label: "选择补丁"
    detail: "包装后的加载流程检查是否启用、版本及平台，再安装匹配的钩子。"
{{< /trace >}}

## 一个进程生效，另一个为什么没有生效

API 进程的 Python 解释器、worker 的 Python 解释器和镜像中的文件都影响是否能加载同一个 .pth。诊断时分别在两端核对 `sys.executable`、`ucm.__file__` 和 `vllm.__version__`。同一进程在导入后再修改环境变量也可能错过模块读取开关的时刻，应先设置再启动服务。

检查 0.30.0 的 balance scheduler 时，可分三步：确认进入了平台分支，确认实际导入的是被包装的模块，再用构造了同步或异步 UCM metadata 的请求对照 `has_sync_kv_loads`。只有同步路径需要补齐标志；已经异步排队的任务按原时序进行。[补丁行为](./05-patch-anatomy.md)
