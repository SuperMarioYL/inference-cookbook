---
title: "同一批 KV 块的两条路径"
weight: 2
---

预填充实例 P 读完整段提示后，解码实例 D 要拿到提示对应的 KV 才能继续生成。选直传时，P 和 D 的 connector 协调源块、目标块和传输完成。选共享存储时，P 先发布可读的缓存对象，D 再按匹配的前缀查找并取回。

{{< mechanism "mooncake-transfer" >}}

|观察点|P/D 直传|经 Store 读取|
|---|---|---|
|谁建立关联|路由器传递请求交接信息，两端 connector 关联同一传输|引擎按前缀找到 Store 中对应的对象|
|数据在哪里|P 的源缓冲区、D 已分配的目标块|Store 中可读副本、D 已分配的目标块|
|何时可消费|目标块传输完成并满足框架的 ready 条件|匹配、读取和回填完成|
|适用需求|交接同一个请求|跨实例共享、卸载和后续前缀复用|

Store Master 用于管理副本位置。直传时，路由器把 prefill 的 bootstrap 地址和 transfer ID 发给 decode。用同一个请求 ID 查看两端搬运的块数或字节数，就能区分成功交接和 decode 自行重算。[vLLM 直传约定](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/disagg-prefill-decode.md) · [存储集成](https://github.com/kvcache-ai/Mooncake/blob/v0.3.14/docs/source/deployment/integrations/vllm/kv-cache-storage.md)
