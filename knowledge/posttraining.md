# 后训练 (Post-training) 知识专家

## 1. 概述

后训练在预训练模型基础上，使用高质量标注数据进一步对齐模型行为，使其满足人类偏好和任务需求。

- **预训练 vs 后训练对比**：

| 维度 | 预训练 | 后训练 |
|------|--------|--------|
| 数据 | 海量无标注语料 | 高质量标注数据 |
| 数据量 | 万亿级 token | 万~百万级样本 |
| 目标 | 语言建模能力 | 对齐人类偏好 |
| 学习率 | 1e-4 ~ 3e-4 | 1e-6 ~ 1e-5 |
| 训练成本 | 占 90%+ | 相对较小 |

- **后训练三阶段**：SFT → RM 训练 → RLHF/DPO

## 2. SFT (Supervised Fine-Tuning)

### 2.1 数据格式

SFT 数据为 (instruction, response) 对，常见格式：

```json
{
  "instruction": "解释量子纠缠",
  "output": "量子纠缠是...",
  "input": ""  // 可选上下文
}
```

### 2.2 Loss Masking

SFT 核心技巧：只在 response 部分计算 loss，不惩罚 instruction/prompt 部分。

```
Tokens:  [SYS][USER][PROMPT][RESPONSE][EOS]
Mask:    [0 ][0  ][0    ][1      ][1 ]
Loss:    ────────────────────────────────
                Only compute on response
```

> 源码参考：Megatron-LM `megatron/core/post_training/` — 后训练数据管道与 loss 计算

### 2.3 训练要点

- **Epoch 控制**：通常 1-3 epoch，过多导致过拟合和灾难性遗忘。
- **数据质量 > 数量**：1000 条高质量数据 > 10000 条噪声数据。
- **学习率**：比预训练小 1-2 数量级，避免破坏预训练知识。

## 3. DPO (Direct Preference Optimization)

### 3.1 原理

DPO 绕过显式 reward model，直接从偏好数据优化策略。

```
Loss = -E[ log σ( β × ( log π_θ(y_w|x)/π_ref(y_w|x)
                        - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]

其中：
  y_w = preferred response (chosen)
  y_l = rejected response
  π_ref = reference model (冻结的 SFT 模型)
  β = KL 约束系数
```

### 3.2 关键要素

- **Reference Model**：冻结的 SFT 模型，提供 KL 锚点防止策略偏离。
- **偏好对 (chosen, rejected)**：需要成对的偏好标注。
- **β 系数**：控制偏离 SFT 模型的程度，β 越大越保守。

### 3.3 DPO vs RLHF

| 维度 | DPO | RLHF |
|------|-----|------|
| Reward Model | 不需要 | 需要单独训练 |
| 训练稳定性 | 较稳定 | 需调 KL 系数 |
| 计算成本 | 较低 | 较高（需采样） |
| 理论等价性 | 特定条件下等价 | 更通用 |

## 4. RLHF (Reinforcement Learning from Human Feedback)

### 4.1 流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SFT Model  │────▶│ Reward Model│────▶│  PPO/GRPO   │
│  (初始策略)  │     │ (偏好学习)   │     │ (策略优化)   │
└─────────────┘     └─────────────┘     └─────────────┘
       │                   │                   │
       │                   ▼                   │
       │            score(response)            │
       │                   │                   │
       └───── KL 约束 ─────┴───────────────────┘
```

### 4.2 Reward Model

- **训练数据**：偏好对 (chosen, rejected)
- **Loss**：`L = -log(σ(r(x, y_w) - r(x, y_l)))`
- **奖励塑形**：`reward = r_model - β × KL(π_θ || π_ref)`

### 4.3 KL 约束

防止策略过度优化 reward hacking：

```
final_reward = reward_model_score - β × KL_divergence
```

- β 越大，策略越接近 SFT 模型
- β 过小会导致 reward hacking（生成怪异文本获取高奖励）

## 5. 关键配置参数表

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `learning_rate` | 学习率 | 1e-6 ~ 1e-5 | 过大破坏预训练知识 |
| `epoch` | 训练轮数 | 1-3 (SFT) | 过多过拟合 |
| `max_length` | 最大序列长度 | 2049-8192 | 显存与效果权衡 |
| `warmup_ratio` | 预热比例 | 0.03-0.1 | 训练稳定性 |
| `dpo_beta` | DPO KL 系数 | 0.1-0.5 | 偏离 SFT 程度 |
| `kl_coeff` | RLHF KL 系数 | 0.01-0.2 | 策略保守性 |
| `reward_baseline` | 奖励基线 | running mean | 减少方差 |
| `data_mixing_ratio` | 数据混合比例 | 视任务 | 多任务平衡 |

## 6. 常见误区

### ❌ 误区 1："SFT epoch 越多越好"
**正解**：SFT 极易过拟合。超过 3 epoch 通常导致：
- 灾难性遗忘（丢失预训练知识）
- 模式坍塌（输出单一化）
- 泛化性下降
建议：1-2 epoch 为佳，配合 early stopping。

### ❌ 误区 2："DPO 完全替代 RLHF"
**正解**：DPO 在离线偏好数据上表现好，但：
- 无法利用在线采样探索
- 对新分布数据适应性差
- 工业界（如 OpenAI）仍大量使用 RLHF/PPO

### ❌ 误区 3："Reward Model 精度越高越好"
**正解**：Reward model 过度拟合会导致：
- 策略学会 reward hacking
- 生成高奖励但低质量文本
需要定期用人工评估校准。

### ❌ 误区 4："Loss Masking 可以忽略"
**正解**：不做 loss masking 会导致：
- 模型学习预测 prompt 内容
- 指令跟随能力下降
- 训练信号噪声增大

### ❌ 误区 5："后训练可以修复预训练的所有缺陷"
**正解**：后训练能力上限由预训练决定。预训练缺乏的知识/能力，后训练难以弥补。数据质量是根本。

## 7. 源码文件索引表

| 文件路径 | 功能描述 |
|----------|----------|
| `Megatron-LM/megatron/core/post_training/` | 后训练数据管道与训练逻辑 |
| `Megatron-LM/megatron/core/post_training/modelopt/` | 模型优化后训练（量化感知） |
| `Megatron-LM/megatron/core/config.py` | 训练配置定义 |
| `miles/miles/rollout/sft_rollout.py` | SFT 数据 rollout 生成 |
| `slime/slime/rollout/sft_rollout.py` | SFT rollout 实现 |
| `miles/miles/true_on_policy/` | 在线策略训练框架 |
| `torchtitan/torchtitan/experiments/rl/losses/` | RL 损失函数实现 |
