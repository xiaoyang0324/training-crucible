# 强化学习训练 (RL Training) 知识专家

## 1. 概述

LLM 强化学习训练通过奖励信号优化模型策略，是 GPT-4、Claude、DeepSeek-R1 等模型的核心对齐技术。

- **LLM RL 特点**：
  - 动作空间巨大（词表大小 10万+）
  - 奖励稀疏（序列级别奖励）
  - 采样成本高（每次 rollout 需完整生成）
  - 训练不稳定（需精细的 KL 约束）

- **GRPO vs PPO 对比**：

| 维度 | PPO | GRPO |
|------|-----|------|
| Value Model | 需要（Critic） | 不需要 |
| Advantage 计算 | GAE (Generalized AE) | Group 内相对奖励 |
| 内存占用 | 高（多一个 Critic） | 低 |
| 稳定性 | 依赖 Critic 质量 | 依赖 group size |
| 代表实现 | OpenAI InstructGPT | DeepSeek-R1 |

## 2. GRPO (Group Relative Policy Optimization)

### 2.1 原理

GRPO 的核心思想：对同一 prompt 采样一组回答，用组内相对奖励作为 baseline，无需独立 value model。

```
对于 prompt x，采样 G 个回答 {o_1, o_2, ..., o_G}

reward_i = reward_model(x, o_i)
baseline = mean(rewards)           # 组内均值
advantage_i = (reward_i - baseline) / std(rewards)  # 标准化

L_GRPO = -E[ min( r_t × A, clip(r_t, 1-ε, 1+ε) × A ) ]
其中 r_t = π_θ(o_i|x) / π_old(o_i|x)
```

### 2.2 Group Sampling

```
                    ┌─── o_1 (reward=0.8) ─── advantage=+0.6
                    │
Prompt x ──采样G=4──┼─── o_2 (reward=0.3) ─── advantage=-0.2
                    │
                    ├─── o_3 (reward=0.5) ─── advantage=0.0
                    │
                    └─── o_4 (reward=0.1) ─── advantage=-0.4
                    
baseline = (0.8+0.3+0.5+0.1)/4 = 0.425
```

> 源码参考：
> - miles `miles/true_on_policy/` — 在线策略训练核心（GRPO 实现）
> - miles `miles/true_on_policy/config.py` — GRPO 配置
> - slime `slime/rollout/` — rollout 生成与 GRPO 训练
> - torchtitan `torchtitan/experiments/rl/` — RL 训练实验框架
> - torchtitan `torchtitan/experiments/rl/losses/` — GRPO/PPO 损失函数

### 2.3 GRPO 优势

- **无需 Value Model**：节省约 50% 显存（Critic 与 Actor 等大）
- **实现简单**：组内统计即可得到 baseline
- **适合推理任务**：数学/代码有明确正确性奖励

## 3. PPO (Proximal Policy Optimization)

### 3.1 Clipped Surrogate Objective

```
概率比：r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)

L_CLIP = E[ min( r_t × Â_t, clip(r_t, 1-ε, 1+ε) × Â_t ) ]

        ┌──────────────────────────────┐
        │  ___                         │
        │ /   \___                     │
        │/       \________             │
        │                 \_______     │
        └──────────────────────────────┘
        1-ε    1    1+ε
        概率比 r_t
```

### 3.2 Advantage Estimation (GAE)

```
Â_t = Σ(γλ)^l × δ_{t+l}
其中 δ_t = r_t + γ×V(s_{t+1}) - V(s_t)

γ = 折扣因子 (通常 1.0)
λ = GAE 参数 (0.9-0.95)
```

### 3.3 PPO 组件

```
┌─────────────────────────────────────────────────────┐
│                    PPO Training Loop                 │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐       │
│  │  Actor   │    │  Critic  │    │  Reward  │       │
│  │ (Policy) │    │ (Value)  │    │  Model   │       │
│  └────┬─────┘    └────┬─────┘    └────┬─────┘       │
│       │               │               │              │
│       ▼               ▼               ▼              │
│  ┌─────────────────────────────────────────────┐    │
│  │         GAE Advantage Estimation            │    │
│  └─────────────────────────────────────────────┘    │
│       │                                              │
│       ▼                                              │
│  ┌─────────────────────────────────────────────┐    │
│  │      Clipped Surrogate Update               │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

## 4. Rollout 生成

### 4.1 SGLang 集成

RL 训练中 rollout 需要高效推理引擎生成回答，SGLang 是主流选择。

> 源码参考：
> - miles `miles/rollout/sglang_rollout.py` — SGLang rollout 集成
> - slime `slime/rollout/sglang_rollout.py` — SGLang rollout 实现
> - slime `slime/rollout/sglang_streaming_rollout.py` — 流式 rollout

### 4.2 Async Rollout (异步生成)

训练与推理解耦，避免互相等待：

```
┌─────────────┐         ┌─────────────┐
│  Training   │◀─weight──│   Rollout   │
│   Engine    │─delta──▶│   Engine    │
│  (Megatron) │         │   (SGLang)  │
└─────────────┘         └─────────────┘
     │                        │
     │  并行执行               │
     ▼                        ▼
  Step N                  Step N+1
  (用旧权重训练)            (用新权重采样)
```

> 源码参考：
> - miles `miles/rollout/fully_async_rollout.py` — 完全异步 rollout
> - miles `miles/rollout/fully_async_data_buffer.py` — 异步数据缓冲
> - slime `slime/rollout/fully_async_rollout.py` — 异步 rollout 实现

### 4.3 Agent Rollout

对于工具调用/多步推理场景，rollout 需支持 agent 交互：

> 源码参考：
> - slime `slime/agent/` — Agent 训练框架
> - slime `slime/agent/trajectory.py` — 轨迹收集
> - slime `slime/agent/sandbox.py` — 沙箱执行环境

## 5. 训推一体 (Training-Inference Integration)

### 5.1 Weight Sync

训练引擎（Megatron）与推理引擎（SGLang）之间的权重同步：

```
┌─────────────────┐     Weight Sync      ┌─────────────────┐
│  Megatron       │ ──────────────────▶  │  SGLang         │
│  (Training)     │  delta weight / full │  (Inference)    │
│                 │  NCCL broadcast      │                 │
└─────────────────┘                      └─────────────────┘
```

### 5.2 Delta Weight

只同步变化的权重，减少通信量：

> 源码参考：
> - miles `miles/backends/megatron_utils/` — Megatron 后端工具
> - miles `miles/backends/sglang_utils/` — SGLang 后端工具
> - slime `slime/backends/megatron_utils/` — Megatron 集成
> - slime `slime/backends/sglang_utils/` — SGLang 集成

## 6. 关键配置参数表

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `group_size` | GRPO 采样组大小 | 4-64 | 越大 baseline 越稳 |
| `kl_coeff` | KL 惩罚系数 | 0.001-0.1 | 策略保守性 |
| `clip_range` | PPO clip ε | 0.1-0.3 | 更新步长限制 |
| `gamma` | 折扣因子 | 1.0 | 远期奖励权重 |
| `lam` | GAE λ 参数 | 0.9-0.95 | 偏差-方差权衡 |
| `num_rollout` | 每次训练 rollout 数 | 32-1024 | 样本效率 |
| `temperature` | 采样温度 | 0.7-1.0 | 探索程度 |
| `top_p` | nucleus sampling | 0.9-1.0 | 多样性控制 |
| `ppo_epochs` | 每批数据更新轮数 | 1-4 | 样本复用 |
| `max_new_tokens` | 最大生成长度 | 512-8192 | 任务相关 |

## 7. 常见误区

### ❌ 误区 1："GRPO 完全不需要 value model"
**正解**：GRPO 用组内统计替代 value model，但：
- Group size 过小时 baseline 噪声大
- 对 reward 分布敏感（全正/全负奖励时失效）
- 实际实现中仍有隐式的 value 估计（如 KL 惩罚项）

### ❌ 误区 2："KL 系数越小越好（允许更大更新）"
**正解**：KL 系数过小导致：
- 策略偏离预训练分布过远
- 生成质量崩溃（重复、无意义文本）
- Reward hacking
建议从 0.01 开始，观察生成质量调整。

### ❌ 误区 3："异步 rollout 总是优于同步"
**正解**：异步 rollout 引入 off-policy 问题：
- 数据来自旧策略，影响梯度估计
- 需要 importance sampling 修正
- 在 KL 约束足够强时影响可忽略

### ❌ 误区 4："PPO 的 clip 保证训练稳定"
**正解**：Clip 只限制单步更新幅度，不解决：
- Reward model 偏差
- 价值函数估计误差
- 探索不足问题
需配合 entropy bonus、KL 约束等。

### ❌ 误区 5："RL 训练可以无限提升模型能力"
**正解**：RL 有收益递减：
- 初期提升明显（学会格式、推理链）
- 后期提升缓慢（受限于预训练知识）
- 过度 RL 可能导致能力失衡

## 8. 源码文件索引表

| 文件路径 | 功能描述 |
|----------|----------|
| `miles/miles/true_on_policy/` | 在线策略训练核心框架 |
| `miles/miles/true_on_policy/config.py` | GRPO/PPO 训练配置 |
| `miles/miles/true_on_policy/contracts.py` | 训练接口定义 |
| `miles/miles/rollout/sglang_rollout.py` | SGLang rollout 集成 |
| `miles/miles/rollout/fully_async_rollout.py` | 完全异步 rollout |
| `miles/miles/rollout/fully_async_data_buffer.py` | 异步数据缓冲 |
| `miles/miles/backends/megatron_utils/` | Megatron 训练后端 |
| `miles/miles/backends/sglang_utils/` | SGLang 推理后端 |
| `slime/slime/rollout/sglang_rollout.py` | SGLang rollout 实现 |
| `slime/slime/rollout/sglang_streaming_rollout.py` | 流式 rollout |
| `slime/slime/rollout/fully_async_rollout.py` | 异步 rollout |
| `slime/slime/agent/` | Agent 训练框架 |
| `slime/slime/agent/trajectory.py` | 轨迹收集 |
| `slime/slime/backends/megatron_utils/` | Megatron 集成 |
| `slime/slime/backends/sglang_utils/` | SGLang 集成 |
| `torchtitan/torchtitan/experiments/rl/` | RL 训练实验框架 |
| `torchtitan/torchtitan/experiments/rl/losses/` | GRPO/PPO 损失函数 |
| `torchtitan/torchtitan/experiments/rl/rollout/` | Rollout 实现 |
| `torchtitan/torchtitan/experiments/rl/actors/` | Actor 实现 |
