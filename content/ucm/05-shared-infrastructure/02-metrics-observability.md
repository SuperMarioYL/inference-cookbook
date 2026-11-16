---
title: "怎样从指标区分 HBM、UCM 和存储读取"
weight: 2
---

一次长前缀请求可能先命中 HBM，余下部分再去 UCM 查。统计时给结果配上它对应的请求范围。若总输入 1,024 个 token，HBM 覆盖 512，UCM 再覆盖 256，外部贡献占全部输入 25%，占剩余候选 50%。前一个比例衡量对全部输入的贡献，后一个衡量外部查询的命中率。

|v0.9.1 指标|可回答的问题|
|---|---|
|`ucm:total_prefix_query_tokens_total`|connector 看到了多少前缀查询 token|
|`ucm:gpu_hbm_hit_tokens_total`|多少 token 在进入外部查找前已命中设备|
|`ucm:ucm_hit_tokens_total`|UCM 又命中了多少 token|
|`ucm:load_bytes_total`|成功加载了多少字节|
|`ucm:save_bytes_total`|有多少字节已成功提交给保存路径|
|`ucm:save_completion_wait_duration`|确认异步保存完成时，实际阻塞了多久|

先对齐一次冷请求和一条共享前缀的热请求，用同一时间窗口的计数增量判断来源。若外部查询命中增加而 `load_bytes_total` 没有相应变化，沿请求 ID 查 worker 是否提交了 load。`Cache|Posix` 的缓存与后端读取份额由实际 shard 计数拆分，混用 token、block 和 shard 会使比率失去意义。[指标与分层计算](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics_list.md)

## 在服务实例定位采集层

部署开启对应指标后，先在实际暴露的 metrics 端点检索 `ucm:` 系列，再确认自己访问的是当前进程或正确实例。把调度侧的查询计数与 worker、后端的加载数据对齐；仪表盘累计值还要扣掉其他请求与 rank 的影响。[采集与展示](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/metrics/metrics.md) · [Connector 请求时序](../04-vllm-integration/01-connector-interface.md)

## 什么时刻抓取端点

vLLM 处理请求时从各 DP 的 scheduler 和 worker 汇集 connector 统计，再写入 API Server 的 Prometheus registry。请求 `/metrics` 只读取该 registry；空闲时在后端发生的事件可能要等下次服务请求才出现在端点。测试可先开启 `enable_metrics: true`、记录初值、完成冷/热请求，再抓取一个完整观测窗口。

```bash
curl -fsS http://localhost:7800/metrics | grep -E '^ucm:(gpu_hbm_hit_tokens_total|ucm_hit_tokens_total|load_bytes_total|save_bytes_total)'
```

对有多个 worker 的聚合要留意 `engine`、`worker_rank` 和模型标签。观察累计值应使用相同时间段的增量；平均加载量可以按字节与实际操作次数计算，命中率则按 token 或 block 单独计算。
