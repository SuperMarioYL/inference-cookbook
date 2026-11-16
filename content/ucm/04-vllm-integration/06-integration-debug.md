---
title: "用一次请求定位集成问题"
weight: 6
---

准备两条相同长前缀的请求。给请求、worker rank、模型和 UCM YAML 各记一个可核对的标识。先让第一次请求把数据存入预期后端，再看第二次的匹配、目标块和加载。

|停在哪里|去哪里看|要查的状态|
|---|---|---|
|启动时没有创建 UCMConnector|环境与版本补丁|`ENABLE_UCM_PATCH` 是否在启动进程中；worker 实际导入的模块路径|
|查不到预期前缀|调度侧匹配|token 序列、完整块数、本地先命中的 token 和实际后端 namespace|
|查到但没有回填|元数据与 worker|`update_state_after_alloc` 后记录的目标块、`build_connector_meta` 中的加载任务|
|提交了加载，模型还在等待|存储任务和目的张量|回调完成状态、各 rank 的 layout、是否有尚未满足的逐层等待|

在匹配版本的源码中按符号定位，行号会随修复变动。若只看调度端，容易把“已发起”误判成“可消费”。

```bash
rg -n 'def (get_num_new_matched_tokens|update_state_after_alloc|build_connector_meta|start_load_kv|wait_for_layer_load|get_finished)' ucm/integration/vllm
rg -n 'def (lookup_on_prefix|load_data|dump_data|wait|check)' ucm/store
```

把同一问题在完整重算和外部回填两种路径下各运行一次，同时检查答案与指标，可以区分一般推理错误和回填引入的问题。[Connector 源码](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py) · [已知的加载问题](./05-patch-anatomy.md)
