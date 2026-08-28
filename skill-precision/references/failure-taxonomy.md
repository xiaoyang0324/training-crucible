# 精度故障分类 (Precision Failure Taxonomy)

> 本文档提供精度故障的多维分类体系，用于 Step 2 (Classify) 和 Step 3 (Localize) 快速定位问题类型。
> 每个故障类型均标注了**代码位置**与**相关配置参数**。

---

## 1. 按症状分类

### 1.1 Loss NaN

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **数值溢出** | FP16 上溢 (max=65504) | 大 loss scale、大梯度 |
| **除零** | 分母为零 (grad norm=0) | 空 batch、全 padding |
| **log(0)** | CE loss 中 log(0) | 无 label smoothing、极端 logits |
| **Inf 传播** | 中间值 Inf 向后传播 | 未做数值裁剪 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| loss 计算入口 | `pretrain_gpt.py:222 loss_func()` | 检查 output_tensor 是否含 NaN/Inf |
| loss 聚合 | `megatron/training/training.py:3105 losses_reduced` | 聚合多 micro-batch loss |
| vocab parallel CE | `megatron/core/tensor_parallel/cross_entropy.py:13 VocabParallelCrossEntropy` | 分布式 CE 实现，`:20 calculate_logits_max()` 做 FP32 max 减法 |
| FP16 配置检查 | `megatron/training/arguments.py:3063 --loss-scale` | 静态/动态 loss scale 配置 |
| grad norm 检查 | `megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` → `:410 clip_grad_by_total_norm_fp32()` | 除零可能发生在 norm 计算 |
| torchtitan CE | `torchtitan/components/loss.py:282 CrossEntropyLoss` | 检查 CE 数值稳定性 |

---

### 1.2 Loss Spike

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **LR 过大** | 学习率突增 / warmup 不足 | 自定义 schedule、恢复训练 |
| **数据异常** | 脏数据 / 错误 label | 新数据 shard、数据 pipeline bug |
| **梯度突变** | 某层梯度异常 | 新数据分布、权重损坏 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| LR schedule 配置 | `megatron/training/config/training_config.py:165 SchedulerConfig` → `:190 lr_warmup_iters` | warmup 步数是否足够 |
| LR CLI 参数 | `megatron/training/arguments.py:2992 --lr`, `:2996 --warmup` | 用户配置的 LR/warmup |
| attention 精度 | `megatron/core/transformer/attention.py:1279 Attention.forward()` | 检查 attention 输出是否异常 |
| MLP 精度 | `megatron/core/transformer/mlp.py:257 MLP.forward()` | 检查 MLP 输出是否异常 |
| 激活值检查 | `megatron/core/transformer/transformer_layer.py:802 forward()` → `:812` / `:814` | 逐层打印 min/max/mean |
| 数据加载 | `megatron/core/datasets/` — 数据预处理 | 检查异常步的数据 shard |

---

### 1.3 Loss Divergence

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **LR 过大** | 学习率始终过高 | 配置错误 |
| **Loss scale 不足** | 动态 loss scale 频繁下调 | FP16 训练 |
| **模型损坏** | 权重出现 NaN/Inf | 上游数值问题未处理 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| loss scale 动态调整 | `megatron/training/arguments.py:3067 --initial-loss-scale`, `:3069 --min-loss-scale` | 动态 scale 上下界 |
| FP16 精度保护 | `megatron/core/transformer/transformer_config.py:482 apply_query_key_layer_scaling` | 启用 QK 缩放 |
| softmax 精度 | `megatron/core/transformer/transformer_config.py:486 attention_softmax_in_fp32` | 确保 softmax 在 FP32 |
| optimizer 状态 | `megatron/core/optimizer/distrib_optimizer.py:3177 step_with_ready_grads()` | 检查 optimizer m/v 状态 |

---

### 1.4 Grad Norm Explosion

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **梯度爆炸** | 深层网络梯度累积 | 无 grad clip、残差连接异常 |
| **Loss scale 过大** | 动态 loss scale 过高 | FP16 训练初期 |
| **并行同步错误** | allreduce 异常 | 分布式配置错误 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| grad clip 配置 | `megatron/training/arguments.py:2618 --clip-grad` (default=1.0) | 全局 L2 grad clipping |
| grad clip 实现 | `megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` → `:410 clip_grad_by_total_norm_fp32()` | clip 执行入口 |
| torchtitan grad clip | `torchtitan/distributed/utils.py:570 clip_grad_norm_()` | 支持 PP/EP 的分布式 clip |
| DDP backward hook | `megatron/core/distributed/distributed_data_parallel.py:553 _make_backward_post_hook()` | 梯度 all-reduce / reduce-scatter |
| async comm 精度 | `megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication` | TP async 梯度累积 |
| grad norm 日志 | `megatron/training/training.py:3177 optimizer.step()` → `:3213 grad_norm` | 每步 grad norm 聚合 |

---

### 1.5 Train-Infer Mismatch

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **模式差异** | train/eval 模式未切换 | Dropout / BatchNorm |
| **权重同步延迟** | 推理使用旧权重 | RL rollout 权重同步 |
| **数值精度差异** | 训练 FP16 / 推理 FP32 | 精度敏感模型 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| 训练模式切换 | `megatron/training/training.py:3105` 训练循环 | train/eval 模式切换逻辑 |
| logit dtype 配置 | `megatron/training/arguments.py:3083 --output-logit-dtype` | lm-head 输出精度 (bf16/fp32) |
| RL 权重同步 | `slime/backends/megatron_utils/loss.py:513 get_log_probs_and_entropy()` | rollout 权重同步后 log-prob 计算 |
| FP8 反量化 | `megatron/core/fp8_utils.py:303 dequantize_fp8_tensor()` | FP8 → 高精度对比 |

---

### 1.6 Accuracy Regression

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **数据分布变化** | 训练数据分布偏移 | 新数据集、数据混合比例变化 |
| **超参漂移** | 配置被意外修改 | 多人协作、配置管理不当 |
| **框架版本差异** | 算子实现变化 | 框架升级 |

**代码位置与排查路径：**

| 排查点 | 代码位置 | 说明 |
|--------|---------|------|
| FP8 recipe 配置 | `megatron/core/transformer/transformer_config.py:595 fp8_recipe` | delayed/tensorwise/mxfp8/blockwise |
| FP8 margin | `megatron/core/transformer/transformer_config.py:614 fp8_margin` | scaling factor margin |
| BF16 matmul 保护 | `megatron/core/transformer/transformer_config.py:490 disable_bf16_reduced_precision_matmul` | 防止 BF16 降精度累积 |
| 精度感知优化器 | `megatron/training/arguments.py:3679 --main-grads-dtype`, `:3681 --main-params-dtype` | main params/grads 精度配置 |

---

## 2. 按训练阶段分类

### 2.1 预训练 (Pre-training)

| 常见问题 | 典型表现 | 高发阶段 | 关键代码位置 |
|---------|---------|---------|------------|
| FP16 溢出 | Loss NaN | 训练初期 | `transformer_config.py:482 apply_query_key_layer_scaling` |
| 数据加载异常 | Loss Spike | 切换数据 shard | `megatron/core/datasets/` 数据预处理 |
| 学习率 warmup 不足 | Loss Spike / Divergence | 训练前几百步 | `training_config.py:190 lr_warmup_iters` |
| 梯度爆炸 | Grad Norm 爆炸 | 深层模型 (>100 层) | `optimizer.py:374 clip_grad_norm()` |

### 2.2 后训练 (Post-training: SFT/DPO/RLHF)

| 常见问题 | 典型表现 | 高发阶段 | 关键代码位置 |
|---------|---------|---------|------------|
| 数据质量差 | Loss Spike | 新数据混入 | `megatron/core/datasets/` |
| 过拟合 | Train loss ↓ / Eval loss ↑ | 训练后期 | `training.py:3105` eval 阶段 |
| DPO 数值不稳定 | Loss NaN | 极端偏好对 | `miles/backends/training_utils/loss_hub/losses.py` |
| RLHF reward 异常 | 精度回退 | reward model 偏差 | `slime/backends/megatron_utils/loss.py:513` |

### 2.3 强化学习 (RL: GRPO/PPO)

| 常见问题 | 典型表现 | 高发阶段 | 关键代码位置 |
|---------|---------|---------|------------|
| Reward hacking | reward ↑ / 真实精度 ↓ | 训练中期 | `miles/backends/training_utils/loss_hub/math_utils.py:139 compute_approx_kl()` |
| Policy collapse | entropy → 0, 输出单一 | 训练后期 | `miles/backends/training_utils/loss_hub/logit_processors.py:184 get_log_probs_and_entropy()` |
| KL 约束失效 | 策略偏离过大 | KL 系数过小 | `math_utils.py:139 compute_approx_kl()` |
| Rollout 数据偏差 | 训练-推理分布不一致 | 权重同步延迟 | `slime/backends/megatron_utils/loss.py:513` |

---

## 3. 按层级分类

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        精度故障层级模型 (含代码位置)                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Layer 6: Loss 计算层                                                           │
│    ── loss 函数、label smoothing、loss scale                                     │
│    ── pretrain_gpt.py:222 loss_func()                                           │
│    ── megatron/training/training.py:3105 losses_reduced                         │
│    ── torchtitan/components/loss.py:282 CrossEntropyLoss                        │
│                                                                                 │
│  Layer 5: 权重层                                                                │
│    ── 权重初始化、权重更新、权重同步                                              │
│    ── megatron/core/optimizer/distrib_optimizer.py:3177 step_with_ready_grads() │
│    ── megatron/core/tensor_parallel/layers.py:634 (async comm)                  │
│                                                                                 │
│  Layer 4: 激活层                                                                │
│    ── 激活函数、softmax、layer norm、残差连接                                    │
│    ── megatron/core/transformer/attention.py:1279 Attention.forward()          │
│    ── megatron/core/transformer/mlp.py:257 MLP.forward()                        │
│    ── megatron/core/transformer/transformer_block.py:755 final_layernorm        │
│                                                                                 │
│  Layer 3: 梯度层                                                                │
│    ── 反向传播、梯度裁剪、梯度同步 (allreduce)                                    │
│    ── megatron/core/optimizer/optimizer.py:374 clip_grad_norm()                │
│    ── megatron/core/distributed/distributed_data_parallel.py:553 backward hook  │
│    ── torchtitan/distributed/utils.py:570 clip_grad_norm_()                     │
│                                                                                 │
│  Layer 2: 优化器层                                                              │
│    ── Adam/LR schedule、状态 (m/v)、权重衰减                                     │
│    ── megatron/training/config/training_config.py:165 SchedulerConfig           │
│    ── megatron/training/arguments.py:3679 --main-grads-dtype                    │
│                                                                                 │
│  Layer 1: 数据层                                                                │
│    ── 数据加载、预处理、tokenization、label                                      │
│    ── megatron/core/datasets/ 数据预处理                                        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**层级排查原则：自下而上排查，自上而下验证。**
- 排查：先查数据层（最常见），再逐层向上
- 验证：修复后从 loss 层向下确认每层正常

---

## 4. 症状→根因→方案 速查表

| 症状 | 最可能根因 | 首选方案 | 关键代码 | 引用 |
|------|-----------|---------|---------|------|
| **Loss NaN (FP16)** | 数值溢出 | 启用/增大 loss scale，或切 BF16 | `arguments.py:3063 --loss-scale` | `known-patterns.md` §1.1 |
| **Loss NaN (除零)** | grad norm=0 或空 batch | 添加 epsilon、检查数据 | `optimizer.py:374 clip_grad_norm()` | `known-patterns.md` §1.1 |
| **Loss NaN (log(0))** | CE loss 中 log(0) | 启用 label smoothing | `cross_entropy.py:13 VocabParallelCrossEntropy` | `known-patterns.md` §1.1 |
| **Loss Spike (脏数据)** | 异常数据批次 | 数据清洗、跳过异常 batch | `megatron/core/datasets/` | `known-patterns.md` §2.1 |
| **Loss Spike (LR)** | 学习率过大 | 增加 warmup、降低 peak LR | `training_config.py:190 lr_warmup_iters` | `known-patterns.md` §2.2 |
| **Loss Divergence** | LR 过大 / loss scale 不足 | 调整 LR schedule、检查 loss scale | `arguments.py:3063 --loss-scale` | `known-patterns.md` §1 |
| **Grad Norm 爆炸** | 梯度累积 / loss scale 过大 | 启用 grad clipping、降低 loss scale | `arguments.py:2618 --clip-grad` | `known-patterns.md` §3.1 |
| **Grad Norm 消失** | 残差连接异常 / 深层网络 | 检查残差缩放、调整初始化 | `transformer_block.py:755 final_layernorm` | `known-patterns.md` §3.2 |
| **Train-Infer Mismatch** | 模式未切换 / 权重同步延迟 | 检查 train/eval 模式、确认权重同步 | `training.py:3105` | `known-patterns.md` |
| **Accuracy Regression** | 数据分布变化 / 超参漂移 | 对比数据分布、回滚配置 | `transformer_config.py:595 fp8_recipe` | `known-patterns.md` |
| **RL: Reward Hacking** | reward 被策略利用 | 增强 KL 约束、检查 reward 设计 | `math_utils.py:139 compute_approx_kl()` | `known-patterns.md` §4.1 |
| **RL: Policy Collapse** | 策略熵降至 0 | 调整 clip range、添加 entropy bonus | `logit_processors.py:184 get_log_probs_and_entropy()` | `known-patterns.md` §4.2 |

---

## 引用

- `skill-precision/SKILL.md` — 5 步诊断工作流（含代码位置映射）
- `skill-precision/references/known-patterns.md` — 已知精度问题模式库（含源码引用与调用链）
- `skill-tickets/` — 历史案例库（按 `type: precision` 检索）
