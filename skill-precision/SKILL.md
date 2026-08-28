---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → skill-precision/SKILL.md
description: >
  精度诊断专家——5 步诊断工作流，覆盖 loss NaN/spike/divergence、梯度爆炸、
  train-infer 不一致、精度回退等精度问题。
  触发条件：用户报告 loss 异常、梯度异常、训练不收敛、推理结果不一致、
  精度回退、数值溢出、NaN、Inf 等精度相关问题。
---

# 精度诊断专家 — 5 步诊断工作流

## The Iron Law

```
诊断精度问题前，必须先拿到四样东西：
  1. 症状描述（loss 曲线 / 报错信息 / 异常步数）
  2. 训练阶段（预训练 / 后训练 / RL）
  3. 环境信息（模型规模、并行配置、硬件、框架版本）
  4. 变更历史（问题出现前改了什么：数据 / 配置 / 代码）

没有这些不做根因判断。绝不凭经验跳过定位步骤。
```

---

## 触发条件

| 症状 | 典型表现 | 紧急度 |
|------|---------|--------|
| **Loss NaN** | loss 突然变为 NaN，后续步持续 NaN | 🔴 紧急 |
| **Loss Spike** | loss 突增后恢复或持续升高 | 🔴 紧急 |
| **Loss Divergence** | loss 单调上升不收敛 | 🟠 高 |
| **Grad Norm 爆炸** | grad norm 突增 >10x 正常值 | 🔴 紧急 |
| **Train-Infer Mismatch** | 训练 loss 正常但推理结果异常 | 🟠 高 |
| **Accuracy Regression** | 同配置下精度相比基线下降 | 🟡 中 |

---

## 5 步诊断工作流总览（含代码位置）

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    精度诊断 5 步工作流 (Code-Level)                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Step 1: 捕获            Step 2: 分类           Step 3: 定位                      │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────────────────────┐  │
│  │ pretrain_gpt │       │ fp8_utils.py │       │ transformer_layer.py:802     │  │
│  │   :222       │       │   :122       │       │   forward()                  │  │
│  │ loss_func()  │──────▶│ is_float8    │──────▶│     ├─ :812 _forward_attn()  │  │
│  │              │       │   tensor()   │       │     └─ :814 _forward_mlp()   │  │
│  │ training.py  │       │              │       │ attention.py:1279            │  │
│  │   :3105      │       │ fp8_utils.py │       │   Attention.forward()        │  │
│  │ losses_red.  │       │   :303       │       │ mlp.py:257                   │  │
│  │              │       │ dequantize   │       │   MLP.forward()              │  │
│  └──────────────┘       └──────────────┘       └──────────────────────────────┘  │
│         │                      │                          │                      │
│         ▼                      ▼                          ▼                      │
│  ┌──────────────┐       ┌──────────────┐       ┌──────────────────────────────┐  │
│  │ arguments.py │       │ transformer  │       │ layers.py:634                │  │
│  │   :3063      │       │   _config.py │       │ LinearWithGradAccumulation    │  │
│  │ --loss-scale │       │   :588 fp8   │       │   AndAsyncCommunication      │  │
│  │   :2618      │       │   :595       │       │ layers.py:634 (async comm)   │  │
│  │ --clip-grad  │       │   fp8_recipe │       │                              │  │
│  └──────────────┘       └──────────────┘       └──────────────────────────────┘  │
│                                                                                  │
│  Step 4: 假设            Step 5: 解决                                                 │
│  ┌──────────────┐       ┌──────────────────────────────────────────────────┐      │
│  │ known-       │       │ arguments.py (配置调整入口)                       │      │
│  │ patterns.md  │       │   :2618 --clip-grad                              │      │
│  │              │──────▶│   :2992 --lr / :2996 --warmup                     │      │
│  │ failure-     │       │   :3063 --loss-scale / :3067 --initial-loss-scale│      │
│  │ taxonomy.md  │       │ transformer_config.py                            │      │
│  │              │       │   :482 --apply-query-key-layer-scaling (fp16)    │      │
│  └──────────────┘       └──────────────────────────────────────────────────┘      │
│                                                                                  │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 1: 捕获 (Capture)

**目标:** 收集错误消息、训练曲线快照 (loss/grad_norm/lr)、环境信息 (模型/并行/硬件/框架)、变更历史。不做根因判断。

**向用户确认的问题：**
- 问题首次出现的步数？之前跑了多少步正常？
- 是偶发（单步）还是持续（后续步都异常）？
- 问题出现前做了什么变更？（数据 / 配置 / 代码 / 硬件）
- 是否在多卡上都出现？还是只有个别 rank？

**需要收集的日志/指标：**
- 完整错误消息 / traceback（如有）
- 训练曲线：loss, grad norm, learning rate, 最近 100 步
- 环境信息：模型规模、并行配置 (TP/PP/DP/CP/EP)、硬件、框架 commit
- 数据批次信息：异常步使用的数据 shard / index

**代码位置映射 (Code Location Mapping):**

| 排查目标 | 代码位置 | 说明 |
|---------|---------|------|
| loss 计算入口 | `pretrain_gpt.py:222 loss_func()` | 计算 micro-batch loss，含 loss_mask 和 output_tensor |
| loss 聚合 (多 micro-batch) | `megatron/training/training.py:3105 losses_reduced` | `forward_backward_func()` 返回的 per-step losses |
| 精度配置读取 | `megatron/core/transformer/transformer_config.py:588 fp8`, `:595 fp8_recipe` | FP8 模式与 recipe 配置 |
| loss scale 配置 | `megatron/training/arguments.py:3063 --loss-scale`, `:3067 --initial-loss-scale` | 静态/动态 loss scaling 参数 |
| 训练环境初始化 | `megatron/training/training.py:2665 setup_model_and_optimizer()` | 模型/优化器/DDP 初始化入口 |
| grad norm 日志输出 | `megatron/training/training.py:3177 optimizer.step()` → `:3213 grad_norm` | 每步 grad norm 聚合与日志 |
| torchtitan loss 入口 | `torchtitan/components/loss.py:282 CrossEntropyLoss` | 预训练 CE loss 实现 |
| torchtitan 配置加载 | `torchtitan/distributed/utils.py:570 clip_grad_norm_()` | 分布式 grad clipping 入口 |

---

## Step 2: 分类 (Classify)

**目标:** 归类症状 (NaN/Spike/Divergence/GradExplosion/TrainInferMismatch/AccuracyRegression)，判断首次出现步数 & 可复现性。

**症状判定规则：**

| 观察到的现象 | 分类 |
|-------------|------|
| loss = NaN 或 Inf | Loss NaN |
| loss 突增 >3x 后恢复 | Loss Spike (瞬时) |
| loss 突增后持续不降 | Loss Spike (持续) |
| loss 单调上升 | Loss Divergence |
| grad norm 突增 >10x | Grad Norm Explosion |
| 训练 loss 正常但推理结果异常 | Train-Infer Mismatch |
| 同配置精度低于基线 | Accuracy Regression |

**关键判断：**
- 异常是**首次出现**还是**一直存在**？
- 异常是**确定性复现**还是**随机出现**？
- 异常前是否有**配置/数据/代码变更**？

**代码位置映射 (Code Location Mapping):**

| 排查目标 | 代码位置 | 说明 |
|---------|---------|------|
| FP8 tensor 类型检查 | `megatron/core/fp8_utils.py:122 is_float8tensor()` | 判断 tensor 是否为 TE Float8Tensor (支持 TE1.x/2.x) |
| FP8 反量化 | `megatron/core/fp8_utils.py:303 dequantize_fp8_tensor()` | FP8 → 高精度回退，用于精度对比 |
| FP8 recipe 与 margin | `megatron/core/transformer/transformer_config.py:595 fp8_recipe`, `:614 fp8_margin` | 检查 scaling factor 计算配置 |
| attention softmax 精度 | `megatron/core/transformer/transformer_config.py:486 attention_softmax_in_fp32` | 是否强制 softmax 在 FP32 执行 |
| QK layer scaling (FP16) | `megatron/core/transformer/transformer_config.py:482 apply_query_key_layer_scaling` | FP16 下 Q*K^T 按 1/layer_num 缩放 |
| BF16 降精度 matmul 保护 | `megatron/core/transformer/transformer_config.py:490 disable_bf16_reduced_precision_matmul` | 防止 BF16 matmul 累积精度损失 |
| async 通信精度 | `megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication` | TP 下 async all-reduce 的梯度精度 |
| torchtitan FP8 转换 | `torchtitan/components/quantization/float8.py:53 Float8LinearConverter` | Float8 Linear 层转换入口 |
| RL log-prob 精度 | `miles/backends/training_utils/loss_hub/logit_processors.py:184 get_log_probs_and_entropy()` | RL 策略 log-prob 计算入口 |
| RL KL 散度精度 | `miles/backends/training_utils/loss_hub/math_utils.py:139 compute_approx_kl()` | KL 散度 k1/k2/k3/low_var_kl 估计 |

---

## Step 3: 定位 (Localize)

**目标:** 逐层排查：数据 → 优化器 → 梯度 → 激活 → 权重 → loss 计算，对比正常/异常步的中间值。

**逐层排查顺序：**

```
数据层 ──► 优化器层 ──► 梯度层 ──► 激活层 ──► 权重层 ──► loss 层
  │           │           │          │          │          │
  │           │           │          │          │          └─ loss 计算溢出?
  │           │           │          │          └─ 权重含 NaN/Inf?
  │           │           │          └─ 激活值异常 (ReLU/Softmax)?
  │           │           └─ 梯度含 NaN/Inf? 梯度同步是否正确?
  │           └─ lr 是否正确? 状态是否损坏?
  └─ 数据是否含 NaN/异常值? label 是否正确?
```

**定位方法：**
- 对比正常步和异常步的中间激活值 / 梯度
- 逐层打印 tensor 的 min/max/mean，定位首次出现 NaN 的层
- 检查数据 pipeline：异常步的数据是否正常加载

**代码位置映射 (Code Location Mapping):**

| 排查目标 | 代码位置 | 说明 |
|---------|---------|------|
| Transformer 层入口 | `megatron/core/transformer/transformer_layer.py:802 forward()` | 单 TransformerLayer forward 总入口 |
| attention 子层 | `megatron/core/transformer/transformer_layer.py:812 _forward_attention()` | 调用 self-attention / cross-attention |
| MLP 子层 | `megatron/core/transformer/transformer_layer.py:814 _forward_mlp()` | 调用 feed-forward MLP |
| Attention forward | `megatron/core/transformer/attention.py:1279 Attention.forward()` | QKV projection + softmax + output |
| MLP forward | `megatron/core/transformer/mlp.py:257 MLP.forward()` | fc1 → activation → fc2 |
| selective recompute (精度影响) | `megatron/core/transformer/transformer_layer.py:474-476` | `recompute_input_layernorm` / `recompute_pre_mlp_layernorm` / `recompute_mlp` |
| 残差连接 | `megatron/core/transformer/transformer_block.py:755 final_layernorm` | block 输出残差 + final LN |
| DDP backward hook (梯度精度) | `megatron/core/distributed/distributed_data_parallel.py:553 _make_backward_post_hook()` | 梯度 all-reduce / reduce-scatter 触发点 |
| grad norm 计算 | `megatron/core/optimizer/optimizer.py:374 clip_grad_norm()` → `:410 clip_grad_by_total_norm_fp32()` | 全局 L2 norm 计算 |
| grad clip 调用 | `megatron/core/optimizer/optimizer.py:830` | `if clip_grad > 0: grad_norm = clip_grad_norm()` |
| torchtitan grad clip | `torchtitan/distributed/utils.py:570 clip_grad_norm_()` | 支持 PP/EP 的分布式 grad clipping |
| optimizer step 精度 | `megatron/core/optimizer/distrib_optimizer.py:3177 step_with_ready_grads()` | 分布式优化器 step + param all-gather |
| FP8 首尾层 BF16 | `megatron/core/transformer/transformer_config.py:650 first_last_layers_bf16` | 首尾层退 BF16 的精度保护 |

---

## Step 4: 假设 (Hypothesize)

**目标:** 匹配 `skill-precision/references/known-patterns.md` 已知模式，检索 `skill-tickets/` 类似案例，按概率排序列出候选根因。

**常见候选根因：**
- 数值精度不足（FP16 溢出 → 需 loss scaling / BF16）
- 数据异常（脏数据 / 错误 label → 需数据清洗）
- 学习率过大（→ 需 warmup / 降低 LR）
- 梯度同步错误（→ 检查 allreduce / grad norm 计算）
- 算子数值不稳定（→ 检查 softmax / layer norm 实现）

**代码位置映射 (Code Location Mapping):**

| 排查目标 | 代码位置 | 说明 |
|---------|---------|------|
| 已知模式匹配 | `skill-precision/references/known-patterns.md` | 9 大精度模式 + 代码定位 |
| 故障分类体系 | `skill-precision/references/failure-taxonomy.md` | 按症状/阶段/层级三维分类 |
| LR schedule 配置 | `megatron/training/config/training_config.py:165 SchedulerConfig` → `:190 lr_warmup_iters`, `:187 lr_warmup_fraction` | warmup 步数与衰减策略 |
| LR schedule 入口 | `megatron/training/arguments.py:2992 --lr`, `:2996 --warmup`, `:2999 --min-lr` | CLI LR 相关参数 |
| mixed precision 参数 | `megatron/training/arguments.py:3058 _add_mixed_precision_args()` | loss-scale / grad-reduce / softmax 等参数组 |
| TransformerEngine 参数 | `megatron/training/arguments.py:2011 _add_transformer_engine_args()` | fp8-param-gather / te-precision-config-file |
| 精度感知优化器 | `megatron/training/arguments.py:3679 --main-grads-dtype`, `:3681 --main-params-dtype`, `:3683 --exp-avg-dtype` | FP8/BF16 main params + optimizer states |
| RL loss 入口 | `miles/backends/training_utils/loss_hub/losses.py` (LossFunction protocol) | RL 策略 loss 计算总入口 |
| RL loss 分发 | `slime/backends/megatron_utils/loss.py:513 get_log_probs_and_entropy()` | slime 的 log-prob 计算 |
| 历史案例检索 | `skill-tickets/` 中 `type: precision` 标签 | 按症状标签过滤 |

---

## Step 5: 解决与归档 (Resolve & Archive)

**目标:** 提出修复方案 (配置调整/代码patch/流程变更)，验证指标恢复正常，归档到 `skill-tickets/`。

**修复方案类型：**
- **配置调整**：loss scale、LR、batch size、grad clip
- **代码 patch**：修复算子实现、添加数值保护
- **流程变更**：数据清洗流程、checkpoint 回滚

**验证标准：**
- 修复后连续跑 N 步（N ≥ 100），loss / grad norm 恢复正常
- 对比修复前后指标曲线

**归档要求：**
- 按 `skill-tickets/TEMPLATE.md` 格式创建 ticket
- 填写完整：Symptom / Environment / Analysis / Root Cause / Resolution / Verification
- 引用真实源码路径作为 `source_refs`

**代码位置映射 (Code Location Mapping):**

| 修复操作 | 代码位置 | 说明 |
|---------|---------|------|
| grad clipping 参数 | `megatron/training/arguments.py:2618 --clip-grad` (default=1.0) | 全局 L2 grad clipping 阈值 |
| loss scale 参数 | `megatron/training/arguments.py:3063 --loss-scale`, `:3067 --initial-loss-scale`, `:3069 --min-loss-scale`, `:3071 --loss-scale-window` | 动态/静态 loss scaling 全参数 |
| FP16 交叉熵 | `megatron/training/arguments.py:3080 --fp16-lm-cross-entropy` | FP16 下 lm-head CE 精度控制 |
| FP8 精度配置 | `megatron/core/transformer/transformer_config.py:588 fp8`, `:595 fp8_recipe`, `:614 fp8_margin` | FP8 模式/recipe/margin 调整 |
| QK scaling (FP16) | `megatron/core/transformer/transformer_config.py:482 apply_query_key_layer_scaling` | 启用 QK 缩放提升 FP16 稳定性 |
| BF16 matmul 保护 | `megatron/core/transformer/transformer_config.py:490 disable_bf16_reduced_precision_matmul` | 禁用 BF16 降精度累积 |
| FP8 首尾层 BF16 | `megatron/core/transformer/transformer_config.py:650 first_last_layers_bf16`, `:653 num_layers_at_start_in_bf16` | 首尾层退 BF16 保护 |
| output logit dtype | `megatron/training/arguments.py:3083 --output-logit-dtype` | lm-head 输出 logit 精度 (bf16/fp32) |
| main grads dtype | `megatron/training/arguments.py:3679 --main-grads-dtype` | 精度感知优化器 main grads 精度 |
| torchtitan FP8 recipe | `torchtitan/components/quantization/float8.py:58 recipe_name` | rowwise / rowwise_with_gw_hp |
| 数值验证脚本 | `torchtitan/scripts/loss_compare.py` | 对比修复前后 loss/grad_norm 曲线 |

---

## 引用

| 文件 | 内容 |
|------|------|
| `skill-precision/references/failure-taxonomy.md` | 精度故障分类（按症状 / 阶段 / 层级）含代码位置 |
| `skill-precision/references/known-patterns.md` | 已知精度问题模式库（含代码定位与调用链） |
| `skill-tickets/TEMPLATE.md` | 问题归档模板 |
| `skill-references/source-repo-map.md` | 源码仓路由（用于定位代码引用） |
