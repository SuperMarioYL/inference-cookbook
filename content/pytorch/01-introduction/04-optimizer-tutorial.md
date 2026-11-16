---
title: "Optimizer 教程"
weight: 4
---

训练一个线性层时，可以先打印 `param.grad`，再调用 `optimizer.step()`，观察梯度怎样改变参数。

## 1. 什么是优化器

在深度学习中，训练模型的核心过程可以归纳为一句话：**通过调整模型参数来最小化损失函数**。优化器（Optimizer）就是负责执行这一"调整参数"步骤的组件。

数学上，给定损失函数 $L(\theta)$ 和参数 $\theta$，优化器根据梯度 $\nabla_\theta L$ 来更新参数：

$$\theta_{t+1} = \theta_t - \eta \cdot g(\nabla_\theta L, \text{state})$$

其中 $\eta$ 是学习率，$g(\cdot)$ 是优化器特有的更新规则。PyTorch 在 `torch.optim` 模块中提供了丰富的优化器实现。

## 2. SGD - 最基础的优化器

随机梯度下降（Stochastic Gradient Descent）是最经典的优化算法。其更新规则为：

$$\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta L$$

### 2.1 基本用法

```python
import torch
import torch.nn as nn

# 创建一个简单的模型
model = nn.Linear(10, 2)

# 创建 SGD 优化器，传入模型参数和学习率
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

### 2.2 带动量的 SGD

动量（momentum）机制可以加速收敛并减少震荡：

$$v_t = \mu \cdot v_{t-1} + \nabla_\theta L$$
$$\theta_{t+1} = \theta_t - \eta \cdot v_t$$

```python
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,       # 动量系数
    weight_decay=1e-4,   # L2 正则化
    nesterov=True,       # 使用 Nesterov 动量
)
```

对应的源码位于 `torch/optim/sgd.py`，其中 `SGD` 类继承自 `Optimizer` 基类。

## 3. Adam - 自适应学习率优化器

Adam 结合了动量和自适应学习率两个思想，是目前最常用的优化器之一：

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-3,              # 学习率
    betas=(0.9, 0.999),   # 一阶和二阶矩的衰减系数
    eps=1e-8,             # 数值稳定性常数
    weight_decay=0,       # 权重衰减
)
```

Adam 的核心优势在于：它为每个参数维护独立的自适应学习率，对于稀疏梯度和非平稳目标函数都有良好表现。

### 3.1 AdamW - 解耦权重衰减

AdamW 是 Adam 的改进版本，将权重衰减从梯度更新中解耦出来：

```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-3,
    weight_decay=0.01,  # 解耦的权重衰减，默认 0.01
)
```

在源码中（`torch/optim/adamw.py`），`AdamW` 实际上继承自 `Adam`，区别仅在于构造时设置 `decoupled_weight_decay=True`。

## 4. 参数组 - 为不同层设置不同超参数

实际训练中，经常需要为模型的不同部分设置不同的学习率。例如在微调预训练模型时，通常给预训练层设置较小的学习率，而给新增层设置较大的学习率。

### 4.1 基本语法

```python
model = torchvision.models.resnet18(pretrained=True)

# 用列表传入多个参数组，每个参数组是一个字典
optimizer = torch.optim.Adam([
    # 参数组 1：预训练的特征提取层，学习率较低
    {"params": model.layer4.parameters(), "lr": 1e-4},
    # 参数组 2：最后的全连接层，学习率较高
    {"params": model.fc.parameters(), "lr": 1e-3},
], lr=1e-5)  # 全局默认学习率（未在组内指定时使用）
```

### 4.2 参数组的数据结构

每个参数组在内部是一个字典，包含以下字段：

| 字段 | 说明 |
|------|------|
| `params` | 该组包含的参数张量列表 |
| `lr` | 学习率 |
| `weight_decay` | 权重衰减系数 |
| `momentum` | 动量（SGD）/ `betas`（Adam） |
| 其他 | 优化器特有的超参数 |

访问方式：

```python
# 查看所有参数组
for i, group in enumerate(optimizer.param_groups):
    print(f"Group {i}: lr={group['lr']}, weight_decay={group['weight_decay']}")

# 动态修改某个组的学习率
optimizer.param_groups[0]["lr"] = 5e-5
```

## 5. 学习率调度器

学习率调度器（LR Scheduler）在训练过程中自动调整学习率，是提升模型性能的重要手段。

### 5.1 常用调度器

```python
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

# StepLR：每 30 个 epoch 将学习率乘以 0.1
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

# CosineAnnealingLR：余弦退火
scheduler = CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-6)

# ReduceLROnPlateau：当验证指标不再提升时降低学习率
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=10)
```

### 5.2 调度器的使用位置

调度器的 `step()` 必须在优化器的 `step()` **之后**调用：

```python
for epoch in range(num_epochs):
    train(...)
    val_loss = validate(...)

    optimizer.step()           # 先更新参数
    scheduler.step()           # 再更新学习率（StepLR/CosineAnnealingLR）
    # scheduler.step(val_loss) # ReduceLROnPlateau 需要传入指标
```

### 5.3 Warmup 策略

Warmup 是在训练初期逐步增大学习率的策略，可以通过 `LinearLR` 和 `SequentialLR` 组合实现：

```python
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# 前 10 个 epoch 线性 warmup
warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=10)
# 后续使用余弦退火
cosine = CosineAnnealingLR(optimizer, T_max=90)
# 组合两个调度器
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[10])
```

## 6. 标准训练循环

下面是一个完整的 PyTorch 训练循环，展示了优化器与调度器的标准使用模式。

```python
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# 1. 初始化
model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=100)

# 2. 训练循环
for epoch in range(num_epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        # Step 1: 清零梯度
        optimizer.zero_grad()

        # Step 2: 前向传播
        output = model(batch_x)

        # Step 3: 计算损失
        loss = criterion(output, batch_y)

        # Step 4: 反向传播（计算梯度）
        loss.backward()

        # Step 5: 更新参数
        optimizer.step()

    # 每个 epoch 结束后调整学习率
    scheduler.step()

    # 打印当前学习率
    print(f"Epoch {epoch}, LR: {scheduler.get_last_lr()}")
```

### 6.1 训练循环状态图

{{< stateboard >}}
title: "一步 SGD 更新一个权重"
rows: ["权重 w", "损失 L", "梯度"]
steps:
  - title: "清除旧梯度"
    detail: "令 w=2，学习率为 0.1，采用没有动量和权重衰减的 SGD。zero_grad(set_to_none=True) 重置梯度。"
    cells: [["2"], ["·"], ["None"]]
    focus: [[2, 0]]
  - title: "前向计算"
    detail: "L=w²，代入 w=2 得到损失 4。"
    cells: [["2"], ["4"], ["None"]]
    focus: [[1, 0]]
  - title: "反向计算"
    detail: "对 L 求导得 2w=4，梯度写入 w.grad；此时权重仍为 2。"
    cells: [["2"], ["4"], ["4"]]
    focus: [[2, 0]]
  - title: "更新参数"
    detail: "optimizer.step() 计算 2−0.1×4=1.6；已有梯度在下一次清零前仍留在参数上。"
    cells: [["1.6"], ["4"], ["4"]]
    focus: [[0, 0]]
{{< /stateboard >}}

### 6.2 各步骤详解

| 步骤 | 方法 | 作用 |
|------|------|------|
| 清零梯度 | `optimizer.zero_grad()` | 将所有参数的 `.grad` 置为 `None`（默认）或清零 |
| 前向传播 | `model(input)` | 计算模型输出，构建计算图 |
| 计算损失 | `criterion(output, target)` | 计算标量损失值 |
| 反向传播 | `loss.backward()` | 沿计算图反向传播，计算每个参数的梯度 |
| 参数更新 | `optimizer.step()` | 根据梯度和优化策略更新参数 |
| 学习率调整 | `scheduler.step()` | 按预定策略调整下一轮的学习率 |

**为什么需要 `zero_grad()`？** PyTorch 的梯度默认是**累加**的。如果不清零，上一次 `backward()` 计算出的梯度会和本次的梯度叠加，导致参数更新方向错误。

## 7. 模型保存与恢复

优化器和调度器都有自己的状态，在保存/恢复训练检查点时需要一并处理：

```python
# 保存检查点
torch.save({
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),
    "loss": loss,
}, "checkpoint.pt")

# 恢复检查点
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])

# 注意顺序：先恢复 scheduler，再恢复 optimizer
scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
```

恢复顺序很重要：由于 `LRScheduler` 初始化时会覆写 optimizer 的 `param_group["lr"]`，应先初始化 scheduler 再调用 `optimizer.load_state_dict()`，这样加载的学习率不会被覆盖。

## 8. 实用技巧总结

1. **选择优化器**：大多数场景下推荐 `AdamW`；对于需要精细调参的场景可以尝试 `SGD + momentum`。
2. **学习率调度**：`CosineAnnealingLR` + Warmup 是一个通用且表现良好的组合。
3. **梯度裁剪**：在 `optimizer.step()` 之前使用 `torch.nn.utils.clip_grad_norm_()` 防止梯度爆炸。
4. **混合精度训练**：配合 `torch.amp.GradScaler` 使用，可以在保持精度的同时加速训练。
5. **性能优化**：在 GPU 上训练时，Adam 会自动选择 `foreach` 或 `fused` 实现，无需手动配置。

## 9. 各文件源码位置速查

| 组件 | 源码路径 |
|------|----------|
| Optimizer 基类 | `torch/optim/optimizer.py` |
| SGD | `torch/optim/sgd.py` |
| Adam | `torch/optim/adam.py` |
| AdamW | `torch/optim/adamw.py` |
| LR Scheduler | `torch/optim/lr_scheduler.py` |

后续文章将深入分析这些源码的实现细节。
