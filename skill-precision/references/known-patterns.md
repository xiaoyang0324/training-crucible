# 已知精度问题模式库 (Known Precision Patterns)

> 本文档收录训练中常见的精度问题模式，每个模式包含症状描述、根因分析、解决方案、**代码定位**和**源码调用链**。
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

**代码定位：**
- 症状首现位置：`pretrain_gpt.py:222 loss_func()` — output_tensor 出现 NaN
- 配置检查点：`megatron/core/transformer/transformer_config.py:482 apply_query_key_layer_scaling`
- 配置检查点：`megatron/training/arguments.py:3063 --loss-scale`

**源码调用链：**
```
pretrain_gpt.py:222 loss_func()
  → output_tensor 含 NaN
    ← megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication.forward()
      ← FP16 matmul 上溢
        ← megatron/core/transformer/transformer_layer.py:802 forward()
          ← megatron/core/transformer/attention.py:1279 Attention.forward()
          ← megatron/core/transformer/mlp.py:257 MLP.forward()
```

**修复方案与代码修改位置：**
1. 启用动态 loss scaling — `megatron/training/arguments.py:3063 --loss-scale` (设为 None 启用动态)
2. 启用 QK layer scaling — `megatron/core/transformer/transformer_config.py:482 apply_query_key_layer_scaling=True`
3. 切换 BF16 训练 — `megatron/training/arguments.py` 中 `--bf16` 配置
4. 降低初始 loss scale — `megatron/training/arguments.py:3067 --initial-loss-scale`

---

### 1.2 除零错误 → Loss NaN

**症状描述：**
- loss 或 grad norm 计算中出现 NaN
- 可能只在特定 batch 出现（如全 padding batch）

**根因分析：**
- 梯度归一化时分母为零（grad norm = 0）
- 某些算子的除法未加 epsilon 保护

**代码定位：**
- 症状首现位置：`megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` → `:410 clip_grad_by_total_norm_fp32()`
- 配置检查点：`megatron/training/arguments.py:2618 --clip-grad`

**源码调用链：**
```
megatron/core/optimizer/optimizer.py:374 clip_grad_norm()
  → :410 clip_grad_by_total_norm_fp32()
    → grad_norm = sqrt(sum(grad^2))  ← 此处若 grad 全零则 norm=0
      → 后续除以 norm 产生 NaN
```

**修复方案与代码修改位置：**
1. 在 grad norm 计算中添加 epsilon — `megatron/core/optimizer/clip_grads.py:66` (clip_grad_by_total_norm_fp32 实现)
2. 检查数据 pipeline — `megatron/core/datasets/` 过滤全 padding batch
3. 检查自定义 loss 函数中的除法操作 — `pretrain_gpt.py:222 loss_func()`

---

### 1.3 log(0) in Cross-Entropy → Loss NaN

**症状描述：**
- CE loss 计算中出现 NaN
- 多发生在训练初期，模型输出极端 logits

**根因分析：**
- softmax 输出接近 0 的位置取 log 得到 -inf
- 无 label smoothing 时，one-hot 标签放大该问题

**代码定位：**
- 症状首现位置：`megatron/core/tensor_parallel/cross_entropy.py:13 VocabParallelCrossEntropy`
- 数值稳定处理：`cross_entropy.py:20 calculate_logits_max()` — FP32 max 减法

**源码调用链：**
```
pretrain_gpt.py:222 loss_func()
  → megatron/core/tensor_parallel/cross_entropy.py:13 VocabParallelCrossEntropy
    → :20 calculate_logits_max()  ← FP32 max 减法 (数值稳定)
      → :32 calculate_predicted_logits()  ← 减去 logits_max 后求 log-sum-exp
        → log(softmax) 在极端 logits 下仍可能 -inf
```

**修复方案与代码修改位置：**
1. 启用 label smoothing — `megatron/training/arguments.py` 中 `--label-smoothing`
2. 在 log 计算中添加 epsilon — `cross_entropy.py` 中 log-sum-exp 实现
3. 使用 fused CE kernel — PyTorch `torch.nn.CrossEntropyLoss` 内部已处理

---

## 2. Loss Spike 模式

### 2.1 脏数据 → Loss Spike

**症状描述：**
- loss 在单步突增 3x 以上，后续步恢复
- 多发生在切换数据 shard 时

**根因分析：**
- 数据中包含异常样本（超长序列、乱码、错误 label）
- 数据 pipeline 解析错误

**代码定位：**
- 症状首现位置：`pretrain_gpt.py:222 loss_func()` — 单步 output_tensor 突增
- 数据加载入口：`megatron/core/datasets/` — 数据预处理与 shard 加载

**源码调用链：**
```
megatron/training/training.py:3105 losses_reduced
  ← forward_backward_func() 调用 forward_step_func()
    ← pretrain_gpt.py forward_step() 从 data_iterator 取 batch
      ← megatron/core/datasets/ 数据预处理
        ← 异常数据 shard → 异常 label/序列 → loss 突增
```

**修复方案与代码修改位置：**
1. 数据清洗 — `megatron/core/datasets/` 过滤异常长度样本
2. 添加 loss spike 检测 — `megatron/training/training.py:3105` 附近添加阈值检测
3. 记录异常步数据 index — `megatron/training/training.py` 训练循环

---

### 2.2 学习率过大 → Loss Spike

**症状描述：**
- loss 突增后持续不降或震荡
- 多发生在训练初期或恢复训练时

**根因分析：**
- warmup 步数不足，LR 过早达到峰值
- 恢复训练时 LR schedule 未正确恢复

**代码定位：**
- 症状首现位置：`megatron/training/training.py:3177 optimizer.step()` — 异常步 grad_norm 突增
- 配置检查点：`megatron/training/config/training_config.py:190 lr_warmup_iters`
- 配置检查点：`megatron/training/arguments.py:2992 --lr`, `:2996 --warmup`

**源码调用链：**
```
megatron/training/training.py:3177 optimizer.step()
  ← LR 过大 → 权重更新过大 → loss 突增
    ← megatron/training/config/training_config.py:165 SchedulerConfig
      → :190 lr_warmup_iters (warmup 步数)
      → :187 lr_warmup_fraction (warmup 比例)
      → :169 lr_decay_style (衰减策略)
```

**修复方案与代码修改位置：**
1. 增加 warmup 步数 — `megatron/training/config/training_config.py:190 lr_warmup_iters`
2. 检查 LR schedule 恢复逻辑 — `megatron/training/training.py:2665 setup_model_and_optimizer()`
3. 使用 slant LR schedule 替代阶跃式 — `training_config.py:169 lr_decay_style`

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

**代码定位：**
- 症状首现位置：`megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` — 返回异常大 norm
- 配置检查点：`megatron/training/arguments.py:2618 --clip-grad`
- 配置检查点：`megatron/training/arguments.py:3063 --loss-scale`

**源码调用链：**
```
megatron/core/optimizer/optimizer.py:830
  → if clip_grad > 0.0:
    → :831 grad_norm = self.clip_grad_norm(self.config.clip_grad)
      → :374 clip_grad_norm()
        → :410 clip_grad_by_total_norm_fp32()
          ← 梯度累积 / loss scale 过大 → norm 爆炸
            ← megatron/core/distributed/distributed_data_parallel.py:553 _make_backward_post_hook()
              ← megatron/core/tensor_parallel/layers.py:634 (async comm)
```

**修复方案与代码修改位置：**
1. 启用梯度裁剪 — `megatron/training/arguments.py:2618 --clip-grad=1.0`
2. 降低 loss scale — `megatron/training/arguments.py:3063 --loss-scale`
3. 检查残差连接 — `megatron/core/transformer/transformer_block.py:755 final_layernorm`
4. torchtitan grad clip — `torchtitan/distributed/utils.py:570 clip_grad_norm_()`

---

### 3.2 梯度消失 → Grad Norm 趋零

**症状描述：**
- grad norm 持续下降趋近 0
- loss 几乎不下降

**根因分析：**
- 深层网络梯度逐层衰减
- 残差连接实现错误（缩放因子异常）
- 激活函数饱和（如 sigmoid/tanh）

**代码定位：**
- 症状首现位置：`megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` — 返回趋零 norm
- 残差连接检查：`megatron/core/transformer/transformer_block.py:755 final_layernorm`
- 激活函数检查：`megatron/core/transformer/mlp.py:257 MLP.forward()`

**源码调用链：**
```
megatron/core/optimizer/optimizer.py:374 clip_grad_norm()
  ← 梯度逐层衰减
    ← megatron/core/transformer/transformer_layer.py:802 forward()
      ← :812 _forward_attention() → attention.py:1279
      ← :814 _forward_mlp() → mlp.py:257
        ← 激活函数饱和 / 残差缩放异常
```

**修复方案与代码修改位置：**
1. 检查残差连接缩放因子 — `megatron/core/transformer/transformer_block.py:755`
2. 调整初始化策略 — `megatron/core/transformer/` 权重初始化
3. 使用 ReLU/GELU 替代饱和激活 — `megatron/core/transformer/mlp.py:257`

---

## 4. RL 特有模式

### 4.1 Reward Hacking → 精度回退

**症状描述：**
- reward 持续上升但真实精度下降
- 模型输出出现重复模式或异常格式

**根因分析：**
- 策略找到 reward model 的漏洞而非真正优化目标
- KL 约束不足，策略偏离太远

**代码定位：**
- 症状首现位置：`miles/backends/training_utils/loss_hub/math_utils.py:139 compute_approx_kl()` — KL 值异常小
- 配置检查点：`miles/backends/training_utils/loss_hub/losses.py` (LossFunction protocol)

**源码调用链：**
```
miles/backends/training_utils/loss_hub/losses.py (LossFunction)
  → compute_policy_loss()
    → math_utils.py:139 compute_approx_kl()
      ← KL 约束不足 → 策略偏离 → reward hacking
        ← logit_processors.py:184 get_log_probs_and_entropy()
          ← 策略 log-prob 与 base 差异过大
```

**修复方案与代码修改位置：**
1. 增强 KL 约束 — `math_utils.py:139 compute_approx_kl()` 中调整 KL 系数
2. 添加 reference model 的 KL 惩罚 — `miles/backends/training_utils/loss_hub/losses.py`
3. 检查 reward model 设计 — `slime/backends/megatron_utils/loss.py:513`

---

### 4.2 Policy Collapse → 输出单一化

**症状描述：**
- 策略 entropy 趋近 0
- 模型对所有输入输出相同/相似内容

**根因分析：**
- clip range 过小，策略更新受限
- 缺乏 entropy 正则化
- advantage 估计偏差

**代码定位：**
- 症状首现位置：`miles/backends/training_utils/loss_hub/logit_processors.py:184 get_log_probs_and_entropy()` — entropy 输出趋零
- 配置检查点：`miles/backends/training_utils/loss_hub/losses.py` (PPO eps / entropy bonus)

**源码调用链：**
```
miles/backends/training_utils/loss_hub/losses.py (LossFunction)
  → compute_policy_loss()
    ← clip range 过小 → 策略更新受限
      ← logit_processors.py:184 get_log_probs_and_entropy()
        ← entropy → 0 → 输出单一化
```

**修复方案与代码修改位置：**
1. 调整 clip range — `miles/backends/training_utils/loss_hub/losses.py` 中 PPO epsilon
2. 添加 entropy bonus — `miles/backends/training_utils/loss_hub/losses.py`
3. 检查 advantage 计算 — `miles/backends/training_utils/loss_hub/advantages.py`

---

## 5. FP8 特有模式

### 5.1 FP8 量化误差 → 精度回退

**症状描述：**
- 启用 FP8 后 loss 曲线与 BF16 基线出现偏差
- 训练后期精度回退

**根因分析：**
- FP8 表示精度有限 (e4m3/e5m2)，量化误差累积
- scaling factor 计算不准确
- 首尾层 FP8 精度不足

**代码定位：**
- 症状首现位置：`megatron/core/fp8_utils.py:122 is_float8tensor()` — 检查 FP8 tensor 类型
- 配置检查点：`megatron/core/transformer/transformer_config.py:588 fp8`, `:595 fp8_recipe`, `:614 fp8_margin`

**源码调用链：**
```
megatron/core/transformer/transformer_layer.py:802 forward()
  → _forward_attention() / _forward_mlp()
    → megatron/core/tensor_parallel/layers.py:634 (FP8 GEMM)
      → megatron/core/fp8_utils.py:303 dequantize_fp8_tensor()
        ← 量化误差累积
          ← transformer_config.py:595 fp8_recipe (delayed/tensorwise/mxfp8/blockwise)
```

**修复方案与代码修改位置：**
1. 切换 FP8 recipe — `megatron/core/transformer/transformer_config.py:595 fp8_recipe`
2. 调整 FP8 margin — `megatron/core/transformer/transformer_config.py:614 fp8_margin`
3. 首尾层退 BF16 — `megatron/core/transformer/transformer_config.py:650 first_last_layers_bf16`
4. torchtitan FP8 recipe — `torchtitan/components/quantization/float8.py:58 recipe_name`

---

### 5.2 FP8 异步通信精度损失

**症状描述：**
- TP 下 FP8 训练 loss 曲线与单卡不一致
- 大模型下精度回退更明显

**根因分析：**
- async all-reduce 中 FP8 梯度累积精度损失
- 通信与计算重叠导致舍入误差

**代码定位：**
- 症状首现位置：`megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication`
- 配置检查点：`megatron/training/arguments.py:3061 --grad-reduce-in-bf16`

**源码调用链：**
```
megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication.forward()
  → allreduce_dgrad=True → async all-reduce
    ← FP8/BF16 梯度通信精度损失
      ← megatron/core/distributed/distributed_data_parallel.py:553 _make_backward_post_hook()
```

**修复方案与代码修改位置：**
1. 梯度通信切 BF16 — `megatron/training/arguments.py:3061 --grad-reduce-in-bf16`
2. 梯度累积切 FP32 — `megatron/training/arguments.py:3077 --accumulate-allreduce-grads-in-fp32`
3. 精度感知优化器 — `megatron/training/arguments.py:3679 --main-grads-dtype`

---

## 6. 速查表

| 模式 | 症状 | 根因 | 方案 | 关键代码位置 | 紧急度 |
|------|------|------|------|------------|--------|
| FP16 溢出 | Loss NaN | 数值上溢 | loss scaling / BF16 | `transformer_config.py:482` | 🔴 |
| 除零 | Loss NaN | 分母为零 | epsilon 保护 | `optimizer.py:374` | 🔴 |
| log(0) | Loss NaN | CE 数值不稳定 | label smoothing | `cross_entropy.py:13` | 🔴 |
| 脏数据 | Loss Spike | 异常数据 | 数据清洗 | `megatron/core/datasets/` | 🟠 |
| LR 过大 | Loss Spike | warmup 不足 | 增加 warmup | `training_config.py:190` | 🟠 |
| 梯度爆炸 | Grad Norm ↑ | 累积/scale 过大 | grad clipping | `arguments.py:2618` | 🔴 |
| 梯度消失 | Grad Norm ↓ | 残差/初始化 | 检查实现 | `transformer_block.py:755` | 🟡 |
| Reward Hacking | reward↑/精度↓ | KL 不足 | 增强 KL 约束 | `math_utils.py:139` | 🟠 |
| Policy Collapse | entropy→0 | clip 过小 | 调 clip/entropy | `logit_processors.py:184` | 🟠 |
| FP8 量化误差 | 精度回退 | 量化误差累积 | 调 recipe/margin | `transformer_config.py:595` | 🟠 |
| FP8 async comm | TP 精度不一致 | 通信精度损失 | grad reduce BF16 | `layers.py:634` | 🟠 |

---

## 引用

- `skill-precision/SKILL.md` — 5 步诊断工作流（含代码位置映射）
- `skill-precision/references/failure-taxonomy.md` — 精度故障分类体系（含代码位置）
- `skill-tickets/` — 历史案例库
