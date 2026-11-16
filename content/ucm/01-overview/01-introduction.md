---
title: "从两次请求看 UCM 如何复用前缀"
weight: 1
---

第一条请求以同一份手册开头，后面的问题是“怎样安装”。引擎计算手册时产生 KV，UCM 可以把符合保存条件的完整块写到存储。第二条请求仍以这份手册开头，末尾改问“怎样卸载”。假设设备缓存已释放，手册对应的两个块还在外部存储。引擎给它们分配目标块，UCM 读回 KV，随后只需继续计算新后缀和未覆盖的边界 token。

{{< mechanism "ucm-refill" >}}

把这个流程跑通需要三方配合。调度器知道请求已有多少本地前缀，connector 负责查询外部存储和组织任务，worker 知道本次分配的块在设备上的位置。存储后端决定键是否可读取及由哪一层返回数据。[固定版本 Connector](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/ucm/integration/vllm/ucm_connector.py)

相同文本还须对应同一分词结果、模型及匹配的缓存布局。输入改动出现在前缀中间时，后续普通前缀块也会失配。只改末尾问题时，可以比较外部命中数、回填耗时和首 token 延迟。[快速开始](./03-quick-start.md) · [支持矩阵](https://github.com/ModelEngine-Group/unified-cache-management/blob/v0.9.1/docs/source/user-guide/support-matrix/support_matrix.md)
