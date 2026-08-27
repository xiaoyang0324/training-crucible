# 已知精度问题模式库 (Known Precision Patterns)

> 本文档收录训练中常见的精度问题模式，每个模式包含症状描述、根因分析、解决方案和源码引用。
> 用于 Step 4 (Hypothesize) 快速匹配候选根因。

---

## 1. Loss NaN 模式

### 1.1 FP16 数值溢出 → Loss NaN

**症状描述：**
- loss 突然变为 NaN，后续步持续 NaN
- 多发生在训练初期或 loss scale 调整后
- 常伴随 grad norm 异常增大

**根因分析：**
- FP16 表示范围有限 (max=65504)，大梯度或大 loss 值导致上溢
- 动态 loss scale 调整不及时，scale 过大导致反向传播值溢出

**解决方案：**
1. 启用动态 loss scaling（PyTorch `GradScaler`）
2. 切换 BF16 训练（BF16 表示范围与 FP32 相同）
3. 降低初始 loss scale，让动态调整有更多空间

**源码引用：**
- Megatron-LM: `megatron/core/transformer/transformer_config.py` — `fp16`, `bf16`, `loss_scale` 配置
- Megatron-LM: `megatron/core/distributed/distributed_data_parallel.py` — DDP 中 loss scale 管理
- PyTorch: `torch.cuda.amp.GradScaler` — 动态 loss scaling 实现

---

### 1.2 除零错误 → Loss NaN

**症状描述：**
- loss 或 grad norm 计算中出现 NaN
- 可能只在特定 batch 出现（如全 padding batch）

**根因分析：**
- 梯度归一化时分母为零（grad norm = 0）
- 某些算子的除法未加 epsilon 保护

**解决方案：**
1. 在 grad norm 计算中添加 epsilon：`norm = sqrt(sum(grad^2) + eps)`
2. 检查数据 pipeline，过滤全 padding batch
3. 检查自定义 loss 函数中的除法操作

**源码引用：**
- Megatron-LM: `megatron/core/distributed/distributed_data_parallel.py` — grad norm 计算
- torchtitan: `torchtitan/components/` — loss 计算相关

---

### 1.3 log(0) in Cross-Entropy → Loss NaN

**症状描述：**
- CE loss 计算中出现 NaN
- 多发生在训练初期，模型输出极端 logits

**根因分析：**
- softmax 输出接近 0 的位置取 log 得到 -inf
- 无 label smoothing 时，one-hot 标签放大该问题

**解决方案：**
1. 启用 label smoothing（如 `label_smoothing=0.1`）
2. 在 log 计算中添加 epsilon：`log(max(x, eps))`
3. 使用 fused CE kernel（如 `torch.nn.CrossEntropyLoss` 内部已处理）

**源码引用：**
- Megatron-LM: `megatron/core/tensor_parallel/cross_entropy.py` — 分布式 CE 实现
- PyTorch: `torch.nn.CrossEntropyLoss` — 数值稳定的 CE 实现

---

## 2. Loss Spike 模式

### 2.1 脏数据 → Loss Spike

**症状描述：**
- loss 在单步突增 3x 以上，后续步恢复
- 多发生在切换数据 shard 时

**根因分析：**
- 数据中包含异常样本（超长序列、乱码、错误 label）
- 数据 pipeline 解析错误

**解决方案：**
1. 数据清洗：过滤异常长度样本、检查 label 合法性
2. 添加 loss spike 检测：超过阈值自动跳过该步
3. 记录异常步的数据 index，回溯数据源

**源码引用：**
- Megatron-LM: `megatron/core/datasets/` — 数据加载与预处理
- torchtitan: `torchtitan/components/dataloader.py` — dataloader 实现

---

### 2.2 学习率过大 → Loss Spike

**症状描述：**
- loss 突增后持续不降或震荡
- 多发生在训练初期或恢复训练时

**根因分析：**
- warmup 步数不足，LR 过早达到峰值
- 恢复训练时 LR schedule 未正确恢复

**解决方案：**
1. 增加 warmup 步数（通常为总步数的 1-5%）
2. 检查 LR schedule 恢复逻辑
3. 使用 slant LR schedule 替代阶跃式

**源码引用：**
- Megatron-LM: `megatron/core/optimizer/lr_scheduler.py` — LR schedule 实现（warmup + decay）
- torchtitan: `torchtitan/components/` — optimizer 配置

---

## 3. Grad Norm 模式

### 3.1 梯度爆炸 → Grad Norm 爆炸

**症状描述：**
- grad norm 突增 >10x 正常值
- 可能伴随 loss NaN 或 spike

**根因分析：**
- 深层网络梯度累积
- loss scale 过大（FP16 训练）
- 残差连接异常

**解决方案：**
1. 启用梯度裁剪（`clip_grad=1.0`）
2. 降低 loss scale（FP16 训练）
3. 检查残差连接实现

**源码引用：**
- Megatron-LM: `megatron/core/distributed/distributed_data_parallel.py` — `clip_grad_norm` 实现
- Megatron-LM: `megatron/core/transformer/transformer_block.py` — 残差连接实现
- torchtitan: `torchtitan/distributed/` — 分布式梯度处理

---

### 3.2 梯度消失 → Grad Norm 趋零

**症状描述：**
- grad norm 持续下降趋近 0
- loss 几乎不下降

**根因分析：**
- 深层网络梯度逐层衰减
- 残差连接实现错误（缩放因子异常）
- 激活函数饱和（如 sigmoid/tanh）

**解决方案：**
1. 检查残差连接缩放因子
2. 调整初始化策略
3. 使用 ReLU/GELU 替代饱和激活

**源码引用：**
- Megatron-LM: `megatron/core/transformer/transformer_block.py` — 残差连接与初始化

---

## 4. RL 特有模式

### 4.1 Reward Hacking → 精度回退

**症状描述：**
- reward 持续上升但真实精度下降
- 模型输出出现重复模式或异常格式

**根因分析：**
- 策略找到 reward model 的漏洞而非真正优化目标
- KL 约束不足，策略偏离太远

**解决方案：**
1. 增强 KL 约束（增大 KL 系数）
2. 添加 reference model 的 KL 惩罚
3. 检查 reward model 设计

**源码引用：**
- miles: `miles/true_on_policy/` — on-policy 训练中的 KL 约束实现
- slime: `slime/rollout/` — rollout 与 reward 计算

---

### 4.2 Policy Collapse → 输出单一化

**症状描述：**
- 策略 entropy 趋近 0
- 模型对所有输入输出相同/相似内容

**根因分析：**
- clip range 过小，策略更新受限
- 缺乏 entropy 正则化
- advantage 估计偏差

**解决方案：**
1. 调整 clip range（PPO 中通常 0.1-0.3）
2. 添加 entropy bonus
3. 检查 advantage 计算（GAE 参数）

**源码引用：**
- miles: `miles/true_on_policy/` — PPO/GRPO 策略更新实现
- slime: `slime/rollout/` — rollout 数据与策略更新

---

## 5. 速查表

| 模式 | 症状 | 根因 | 方案 | 紧急度 |
|------|------|------|------|--------|
| FP16 溢出 | Loss NaN | 数值上溢 | loss scaling / BF16 | 🔴 |
| 除零 | Loss NaN | 分母为零 | epsilon 保护 | 🔴 |
| log(0) | Loss NaN | CE 数值不稳定 | label smoothing | 🔴 |
| 脏数据 | Loss Spike | 异常数据 | 数据清洗 | 🟠 |
| LR 过大 | Loss Spike | warmup 不足 | 增加 warmup | 🟠 |
| 梯度爆炸 | Grad Norm ↑ | 累积/scale 过大 | grad clipping | 🔴 |
| 梯度消失 | Grad Norm ↓ | 残差/初始化 | 检查实现 | 🟡 |
| Reward Hacking | reward↑/精度↓ | KL 不足 | 增强 KL 约束 | 🟠 |
| Policy Collapse | entropy→0 | clip 过小 | 调 clip/entropy | 🟠 |

---

## 引用

- `precision/SKILL.md` — 5 步诊断工作流
- `precision/references/failure-taxonomy.md` — 精度故障分类体系
- `tickets/` — 历史案例库
