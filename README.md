# Inference Cookbook

中文 LLM 推理手册。可以从一个算子开始，跟到引擎输出第一个 token；也可以从一块已经算好的 KV 开始，看另一条请求怎样找到、传输和复用它。[在线阅读](https://inference.lei6393.com/)

| 阅读路线 | 栏目 |
| --- | --- |
| 从算子到推理引擎 | [PyTorch](content/pytorch/00-overview/01-architecture-overview.md) · [FlashAttention](content/flashattention/00-overview/overview.md) · [vLLM](content/vllm/00-overview/overview.md) · [SGLang](content/sglang/00-overview/overview.md) |
| 当 KV 跨设备移动 | [NVIDIA Dynamo](content/dynamo/00-overview/overview.md) · [Mooncake](content/mooncake/00-beginner-guide/01-what-is-mooncake.md) · [UCM](content/ucm/01-overview/01-introduction.md) · [CacheBlend](content/cacheblend/00-overview/overview.md) |

## 在本地阅读

安装 Node.js 20.11 或更新版本，以及 Hugo Extended。CI 使用 Hugo 0.147.0。

```bash
git clone https://github.com/SuperMarioYL/inference-cookbook.git
cd inference-cookbook
npm ci
npm run dev
```

打开 [本地站点](http://localhost:1313/)。`npm run build` 会清理旧的构建产物，再将静态站点写入 `public/`。`python3 scripts/check-site-links.py public` 检查站内页面和标题锚点。主分支通过 GitHub Actions 发布到 Cloudflare Pages。

## 已运行的例子

| 环境 | 操作与结果 |
| --- | --- |
| [vLLM 0.33.0，Linux ARM CPU](content/vllm/04-code-walkthrough/01-entry-points.md) | Qwen3-0.6B 成功加载。查询模型列表并向聊天端点发送请求，响应包含 18 个输入 token、64 个输出 token。 |
| [PyTorch 2.15，CPU](content/pytorch/05-advanced-topics/01-compiler-stack-overview.md) | 输入形状依次为 4、4、8，累计捕获图数为 1、1、2。另用两个 Gloo 进程验证了 DDP 梯度规约和 AdamW 从检查点续训。 |
| [SGLang 0.5.23，Apple M1 Max](content/sglang/01-getting-started/02-quick-start.md) | MLX 后端加载 Qwen3-0.6B，模型列表和聊天端点均返回 HTTP 200。 |
| [Dynamo 1.6.2，Linux](content/dynamo/06-python-sdk/03-service-decorators.md) | 两个 Python 进程通过文件发现和 TCP 请求通道完成回显。 |
| [Mooncake 0.3.14，Linux](content/mooncake/00-beginner-guide/03-quick-start-tutorial.md) | 单机 Store 写入后读回原值。关闭本地 memcpy 后，另一个进程经 TCP 读回 13,312 字节。 |
| [UCM 0.9.1 POSIX 后端，Linux ARM CPU](content/ucm/02-storage/03-posix-backend.md) | 写入两个 4 KiB 块后，连续前缀最后命中的下标从 −1 变成 1。重新加载的数据逐字节一致。 |
| [FlashAttention 算法参考](content/flashattention/02-core-algorithm/01-online-softmax.md) | 两块数据的在线 softmax 得到 2.375，与四个位置一次归一化的结果相同。 |
| [CacheBlend 的上下文示例，CPU](content/cacheblend/01-introduction/02-problem-definition.md) | 在手册前插入一段材料后，同一组 token 的第二层输入发生变化。 |

## 修改章节与配图

各项目的文章放在对应的 `content/` 子目录。代码路径和配置项链接到文章采用的上游版本。运行示例注明模型、设备和预期输出。

组件依赖、类关系和目录图用 Mermaid。要讲一次消息交接，在 `{{< trace >}}` 中写参与者和各步消息；要比较同一组位置上的值，在 `{{< stateboard >}}` 中写每一步的行、格子和当前关注的位置。`{{< timeline >}}` 用带单位的横轴显示排队、计算和数据复制何时重叠。这三种图随正文构建为原生 SVG。

矩阵分块、stride、页表寻址等需要按实际空间布局绘制。SVG 源文件放在 `assets/images/mechanisms/`，正文用 `{{< mechanism "fa-tiles" >}}` 引用。每个 `data-frame` 分组的 `title` 和 `desc` 对应一步图注。可参考[微批流水线](content/sglang/06-distributed/22-pipeline-parallelism.md)和[在线 softmax](content/flashattention/02-core-algorithm/01-online-softmax.md)。

MIT License
