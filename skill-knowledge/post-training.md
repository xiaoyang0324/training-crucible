# 后训练 (Post-training) — 代码级深度分析

## 0. 后训练流程全景图

后训练在预训练模型基础上，通过三种主要路径对齐模型行为：**SFT**（监督微调）、**DPO**（直接偏好优化）、**RLHF**（基于人类反馈的强化学习，以 GRPO 为主流算法）。完整流程如下：

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          后训练流程全景图                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Pretrained Model                                                               │
│       │                                                                         │
│       ├──────────────────────────────┬──────────────────────────────┐           │
│       ▼                              ▼                              ▼           │
│  SFT Data                      DPO Data                    RLHF Data            │
│  (prompt, response)            (chosen, rejected)          (prompts only)       │
│       │                              │                              │           │
│       ▼                              ▼                              ▼           │
│  SFT Training                  DPO Training               Rollout Generation   │
│  Cross-entropy +               Preference                 InferenceEngine       │
│  loss masking                  likelihood                        │           │
│       │                              │                              ▼           │
│       ▼                              ▼                        Reward Function     │
│  SFT Checkpoint                DPO Checkpoint              Rule-based / RM      │
│       │                              │                              │           │
│       │                              │                              ▼           │
│       │                              │                   GRPO/PPO Update       │
│       │                              │                   Advantage + KL         │
│       │                              │                              │           │
│       ▼                              ▼                              ▼           │
│  ┌─────────────────────────────────────────────────────────────────────┐        │
│  │                       Aligned Model (后训练产出)                      │        │
│  └─────────────────────────────────────────────────────────────────────┘        │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**三条路径对比**：

| 路径 | 数据形式 | 损失函数 | 是否需要 RM | 本研究栈支持 |
|------|---------|---------|------------|------------|
| SFT | (prompt, response) 对 | Cross-entropy + loss masking | 否 | Megatron-LM, torchtitan |
| DPO | (chosen, rejected) 偏好对 | Preference likelihood | 否 | 无（TRL/OpenRLHF） |
| RLHF/GRPO | 在线采样 + verifiable reward | PPO/GRPO clip loss | 否（规则奖励） | Megatron-LM, torchtitan, slime, miles |

> **关键结论**：GRPO 已取代 PPO 成为主流 RLHF 算法（DeepSeek-V3、Qwen3 路线）。DPO 目前主流实现集中在 HuggingFace TRL / OpenRLHF，不在本研究栈中。

---

## 0. 后训练总览与仓库定位

后训练在预训练模型基础上，使用高质量标注数据对齐模型行为。四个仓库的定位差异显著，面试时必须能清晰区分：

```
                     SFT        DPO        RLHF/GRPO/PPO
Megatron-LM       ✓ 完整      ✗ 无        ✓ 原生 GRPO (megatron/rl/)
torchtitan        ✓ 完整      ✗ 无        △ experiments only (experiments/rl/)
slime             ✗ 无        ✗ 无        ✓ 完整 PPO/GRPO/CISPO/GSPO
miles             ✗ 无        ✗ 无        ✓ 完整 PPO/GRPO/GSPO + Reward Hub
```

> **关键结论**：Megatron-LM 与 torchtitan 均 **无 DPO 实现**。DPO 目前主流实现集中在 HuggingFace TRL / OpenRLHF，不在本研究栈中。GRPO 已取代 PPO 成为主流 RLHF 算法（DeepSeek-V3、Qwen3 路线）。

| 维度 | 预训练 | 后训练 |
|------|--------|--------|
| 数据 | 万亿级 token 无标注语料 | 万~百万级高质量标注样本 |
| 学习率 | 1e-4 ~ 3e-4 | 1e-6 ~ 1e-5（小 1-2 数量级） |
| 目标 | 语言建模能力 | 对齐人类偏好 / 指令跟随 |
| 损失掩码 | 全 token 计算 | 仅 response 部分（loss masking） |

---

## 1. SFT（监督微调）

### 1.1 概念原理

SFT 是最基础的后训练形式：在 (prompt, response) 对上以交叉熵损失微调预训练模型。**核心技巧是 loss masking**——仅在 assistant response  token 上计算损失，prompt/system 部分通过 `IGNORE_INDEX = -100` 掩码掉，避免模型学习"预测用户问题"。

### 1.2 各仓库 SFT 实现代码位置

#### Megatron-LM（生产级，两套 SFT 数据集）

| 文件 | 功能 |
|------|------|
| `megatron/training/datasets/sft_dataset.py:51` | `SFTDataset` — 基于 jsonl messages 格式的核心数据集 |
| `megatron/training/datasets/sft_dataset.py:17` | `SFTLowLevelDataset` — HuggingFace `datasets` 加载 jsonl |
| `megatron/training/datasets/varlen_dataset.py:3` | `VarlenDataset` — 变长多源 SFT 数据集（HF Hub/parquet/jsonl） |
| `megatron/core/tokenizers/text/libraries/sft_tokenizer.py:46` | `SFTTokenizer` — 对话分词 + 自动 loss masking |
| `examples/post_training/modelopt/finetune.py:64` | `SFTDataset`（ModelOpt 版）— 支持 HF 数据集 + sequence packing |
| `megatron/post_training/loss_func.py:39` | SFT 损失函数（含 KD 蒸馏扩展） |

#### torchtitan（研究级，Grain 数据管道）

| 文件 | 功能 |
|------|------|
| `torchtitan/hf_datasets/text_datasets.py:61` | `ChatProcessor` — 对话分词 + prompt 掩码 |
| `torchtitan/hf_datasets/text_datasets.py:32` | `TextProcessor` — 纯文本预训练式分词 |
| `torchtitan/components/loss.py:32` | `cross_entropy_loss` + `IGNORE_INDEX = -100` |
| `torchtitan/components/loss.py:326` | `compute_logprobs` — 对数概率计算（RL 复用） |
| `torchtitan/components/data/collators.py:41` | `TextCollator` — 变长序列拼接 |
| `torchtitan/components/data/packing.py:144` | FirstFitPack + IGNORE_INDEX 填充掩码 |

### 1.3 数据格式与 DataLoader

**Megatron-LM SFT 数据格式**（jsonl messages）：

```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "解释量子纠缠"},
  {"role": "assistant", "content": "量子纠缠是..."}
]}
```

`SFTLowLevelDataset.__getitem__` 返回 `messages` 列表（`sft_dataset.py:47`），由 `SFTTokenizer.tokenize_conversation`（`sft_tokenizer.py:130`）分词并生成 target：system/user token 设为 `IGNORE_INDEX = -100`，仅 assistant token 保留真实标签。

**loss masking 核心逻辑**（`sft_tokenizer.py:164-205`）：

```python
target = tokens.copy()
for turn_idx, turn in enumerate(conversation):
    role = turn["role"].lower()
    if role in ("system", "user", "tool"):
        target[idx : idx + turn_len] = IGNORE_INDEX      # 掩码 prompt
    elif role == "assistant":
        target[idx : idx + assistant_prefix_len] = IGNORE_INDEX  # 掩码前缀
```

**torchtitan ChatProcessor**（`text_datasets.py:100-150`）采用**前缀重分词**策略精确定位 prompt/response 边界：先 tokenize 完整对话，再单独 tokenize user 消息（`add_generation_prompt=True`），用长度差确定掩码位置，避免 BPE 合并导致的边界错位。

**sequence packing**：两者均支持将多条样本拼接到固定长度（`cu_seqlens` 记录边界）。Megatron-LM 在 `sft_dataset.py:107-150` 实现；torchtitan 在 `packing.py:80-110` 使用 `FirstFitPackIterDataset`。

### 1.4 损失函数与训练循环

**Megatron-LM SFT 损失**（`post_training/loss_func.py:39-72`）：

`_mask_loss` 是核心掩码损失计算（`loss_func.py:13-36`）：

```python
# megatron/post_training/loss_func.py:13
def _mask_loss(output_tensor, loss_mask):
    """Apply mask to the unreduced loss tensor."""
    args = get_args()

    if isinstance(output_tensor, tuple):
        output_tensor, tp_reduce, is_sequence_parallel = output_tensor
    else:
        tp_reduce, is_sequence_parallel = False, False

    if is_sequence_parallel:
        # Sequence-parallel tensor derived from intermediate activation - need to split loss mask.
        idx = parallel_state.get_tensor_model_parallel_rank()
        loss_mask = torch.tensor_split(loss_mask, args.tensor_model_parallel_size, dim=1)[idx]

    losses = output_tensor.view(-1).float()
    loss_mask = loss_mask.reshape(-1).float()
    loss = torch.sum(losses * loss_mask)

    if tp_reduce or is_sequence_parallel:
        # Losses on parallel tensors require extra all-reduce to sync across MP ranks.
        torch.distributed.all_reduce(loss, group=parallel_state.get_tensor_model_parallel_group())

    return loss
```

`loss_func` 主函数（`loss_func.py:39-72`）在标准 LM 损失基础上支持 KD 蒸馏扩展：

```python
# megatron/post_training/loss_func.py:39
def loss_func(loss_mask: torch.Tensor, output_tensor: torch.Tensor, model: GPTModel):
    """Loss function (with KD Loss support)."""
    args = get_args()
    model = unwrap_model(model)

    # Standard lm loss
    loss_lm = _mask_loss(output_tensor, loss_mask)
    loss = loss_lm
    num_tokens = loss_mask.sum().clone().detach().to(torch.int)
    report = {'lm loss': torch.cat([loss_lm.clone().detach().view(1), num_tokens.view(1)])}

    if args.export_kd_teacher_load:
        # [ModelOpt]: Handle knowledge distillation
        losses = model.compute_kd_loss(
            student_loss=loss_lm,
            loss_reduction_fn=lambda x: _mask_loss(x, loss_mask),
        )
        report["total loss"] = torch.cat([losses["kd_loss"].clone().detach().view(1), num_tokens.view(1)])
        if model.training:
            loss = losses["kd_loss"]

    return loss, num_tokens, report
```

**torchtitan 损失**（`components/loss.py:32-62`）：vocab-parallel cross-entropy，通过 `_LossParallelCrossEntropy`（`loss.py:66-224`）在 TP 维度分布式计算 softmax，仅需 3 次 all-reduce（max / sumexp / gather）。`ChunkedLossWrapper`（`loss.py:509-729`）支持分 chunk 计算 lm_head 降低峰值显存。

**torchtitan 损失**（`components/loss.py:32-62`）：vocab-parallel cross-entropy，通过 `_LossParallelCrossEntropy`（`loss.py:66-224`）在 TP 维度分布式计算 softmax，仅需 3 次 all-reduce（max / sumexp / gather）。`ChunkedLossWrapper`（`loss.py:509-729`）支持分 chunk 计算 lm_head 降低峰值显存。

### 1.5 SFT 完整调用链（Megatron-LM）

```
finetune.py::__main__
  └─ parse_and_validate_args → add_finetune_args (L35)
  └─ pretrain(cfg, train_valid_test_sft_datasets_provider, forward_step, ...)
       ├─ train_valid_test_sft_datasets_provider (L438)
       │    └─ SFTDataset(num_packed_samples, hf_dataset, tokenizer, seq_length)
       │         ├─ _load_dataset_synchronized → datasets.load_dataset (L159)
       │         └─ __getitem__ → _process_and_pack_example (L332)
       │              └─ tokenizer.apply_chat_template + loss_mask
       └─ forward_step (L544)
            ├─ get_batch → build_lm_batch (L481)
            ├─ model(tokens, position_ids, attention_mask, labels)
            └─ loss_func(loss_mask, output_tensor, model) → _mask_loss
```

---

## 2. DPO（直接偏好优化）

### 2.1 概念原理

DPO 绕过显式 reward model，直接从偏好对 (chosen, rejected) 优化策略。损失函数：

```
L_DPO = -E[ log σ( β × ( log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x) ) ) ]
```

- `y_w` = chosen（偏好响应），`y_l` = rejected（拒绝响应）
- `π_ref` = 冻结的 reference model（通常是 SFT 模型），提供 KL 锚点
- `β` = KL 约束系数，控制偏离 SFT 模型的保守程度

### 2.2 DPO 在本研究栈中的现状

**四个仓库均无 DPO 训练损失实现。** 面试时应明确指出：

- **Megatron-LM**：`megatron/rl/agent/api.py:54-55` 定义了 `ContrastiveRollout`（含 `chosen_trajectory` / `rejected_trajectory`），但仅是 RL rollout 数据结构，**无 DPO 损失函数**。`megatron/training/datasets/varlen_dataset.py:45` 明确声明 "preference (chosen/rejected) datasets are out of scope"。
- **torchtitan / slime / miles**：均无 DPO 实现，聚焦 GRPO/PPO。

DPO 主流实现位于 HuggingFace `TRL`（`DPOTrainer`）/ `OpenRLHF`。若面试被问到，应坦诚本研究栈未覆盖，并描述 DPO 与 GRPO 的取舍：

| 维度 | DPO | GRPO |
|------|-----|------|
| Reward Model | 不需要 | 不需要（verifiable reward） |
| 数据形式 | 离线偏好对 (chosen, rejected) | 在线采样 + 规则奖励 |
| 训练稳定性 | 较稳定（静态数据） | 需调 KL / clip |
| 探索能力 | 无（离线） | 有（在线采样） |
| 工业界趋势 | 用于冷启动 | 主流（DeepSeek/Qwen 路线） |

### 2.3 Reference Model 管理

虽然无 DPO，但 GRPO/PPO 中的 reference model 逻辑类似：
- **slime**：`args.kl_coef` 控制 KL 惩罚强度，`compute_approx_kl(log_probs, ref_log_probs, kl_loss_type)`（`ppo_utils.py:12-51`）支持 k1/k2/k3/low_var_kl 四种 KL 估计器
- **Megatron-LM GRPO**：`calculate_grpo_loss` 接收 `ref_logprobs` 参数（`rl_utils.py:2774`），KL 项为 `kl_beta * (ref_diff.exp() - ref_diff - 1)`（`rl_utils.py:2843-2844`）

---

## 3. RLHF（基于人类反馈的强化学习）

### 3.1 概念原理

本研究栈中 RLHF 以 **GRPO** 为主流实现（取代 PPO），辅以 PPO 在 slime/miles 中的完整支持：

```
传统 RLHF 三阶段（本研究栈未完全采用）：
  SFT → Reward Model 训练 → PPO 策略优化

现代 GRPO 路线（本研究栈采用）：
  SFT → 在线采样 + Verifiable Rule Reward → GRPO 策略优化
```

GRPO 核心思想：**用同组采样的 reward 均值/方差归一化代替 learned reward model**：

```
advantage_i = (reward_i - mean(group)) / std(group)
```

### 3.2 仓库 RL 实现架构对比

| 仓库 | 入口文件 | 算法 | Off-policy 生成 | Reward 来源 |
|------|----------|------|----------------|-------------|
| Megatron-LM | `train_rl.py` | GRPO | ✓ `--rl-partial-rollouts` | 自定义 Agent |
| torchtitan | `experiments/rl/train.py` | GRPO/DAPO | ✗ (vLLM 同步) | 自定义 reward_fns |
| slime | `train_async.py` | PPO/GRPO/CISPO/GSPO/REINFORCE++ | ✓ async | Rule-based / RM |
| miles | `train_async.py` | PPO/GRPO/GSPO/REINFORCE++ | ✓ async | Reward Hub (统一接口) |

### 3.3 Reward Model / Reward Function 代码位置

**miles Reward Hub**（统一奖励接口）— `miles/rollout/`:

| 文件 | 功能 |
|------|------|
| `rm_hub/__init__.py:43` | `async_rm()` — 奖励类型分发中枢 |
| `rm_hub/__init__.py:59-86` | 支持类型：`deepscaler` / `dapo` / `math` / `gpqa` / `f1` / `remote_rm` / `ifbench` / `random` |
| `rm_hub/deepscaler.py:38` | `get_deepscaler_rule_based_reward` — 规则匹配奖励 |
| `rm_hub/gpqa.py:54` | `compute_gpqa_reward` — GPQA 选择题评分 |
| `ray/rollout/train_data_conversion.py:207` | `_normalize_rewards_by_rollout` — 组内奖励归一化 |
| `ray/rollout/train_data_conversion.py:257` | `_post_process_rewards` — GRPO 标准归一化开关 |

**slime Reward** — `slime/ray/rollout.py:279` `_post_process_rewards`：根据 `advantage_estimator` 选择是否进行 `grpo_std_normalization`（`rollout.py:298`）。

**Megatron-LM** — `megatron/rl/agent/reward_only_agent.py:51` `get_reward()` 为抽象方法，由子类实现自定义奖励函数，在 `_rollout_from_episode`（`reward_only_agent.py:180`）中计算 trajectory reward。

### 3.4 GRPO 训练循环与损失函数

#### Megatron-LM（核心实现）

**入口与调用链**：

```
train_rl.py::__main__
  └─ parse_and_validate_args(extra_args_provider=add_inference_args)
  └─ pretrain(cfg, None, ModelType.encoder_or_decoder, forward_step, model_provider)
       └─ forward_step (train_rl.py:190)
            ├─ load_packed_data_by_index → (tokens, advantages, old_logprobs, loss_mask, ref_logprobs, ...)
            ├─ get_logprobs(model, tokens, position_ids) → current_logprobs
            └─ calculate_grpo_loss(current, old, ref, advantages, ...) (rl_utils.py:2771)
                 ├─ ratios = (current_logprobs - old_logprobs).exp()
                 ├─ clamped_ratios = ratios.clamp(1-eps_lower, 1+eps_upper)
                 ├─ kl_term = (ref - current).exp() - (ref - current) - 1
                 └─ loss = -min(ratios*adv, clamped*adv) + kl_beta*kl_term - entropy_weight*entropy
```

**`calculate_grpo_loss` 核心实现**（`megatron/rl/rl_utils.py:2771-2862`）：

```python
# megatron/rl/rl_utils.py:2771
def calculate_grpo_loss(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clamp_eps_lower: float,
    clamp_eps_upper: float,
    kl_beta: float,
    entropy_weight: float,
    inference_logprobs: torch.Tensor | None = None,
    is_truncation_coef: float | None = None,
    seq_starts: list | None = None,
    seq_lengths: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get GRPO loss, the kl term of the loss and the pi/pi_{old} ratios."""
    # Ensure shapes match before computation
    if current_logprobs.shape != old_logprobs.shape:
        log_single_rank(logger, logging.WARNING, f"WARNING: Shape mismatch ...")

    ratios = (current_logprobs - old_logprobs).exp()                           # π/π_old
    clamped_ratios = ratios.clamp(1 - clamp_eps_lower, 1 + clamp_eps_upper)   # clip
    truncated_from_above = torch.gt(ratios, 1 + clamp_eps_upper)
    truncated_from_below = torch.lt(ratios, 1 - clamp_eps_lower)

    # Handle advantages based on whether this is packed or unpacked
    if seq_starts is not None and seq_lengths is not None:
        # Packed sequences: map each sequence's advantage to its tokens
        bin_size = current_logprobs.shape[1]
        packed_advantages = torch.zeros((1, bin_size), device=current_logprobs.device, dtype=current_logprobs.dtype)
        for seq_idx, (start, seq_len) in enumerate(zip(seq_starts, seq_lengths)):
            end = min(start + seq_len - 1, bin_size)
            if end > start:
                packed_advantages[0, start:end] = advantages[seq_idx].item()
        advantages = packed_advantages
    else:
        advantages = advantages.view(-1, 1)  # [batch, 1] broadcast to [batch, seq]

    ref_diff = ref_logprobs - current_logprobs
    kl_term = ref_diff.exp() - ref_diff - 1                    # KL(π||π_ref) 非负估计
    entropy_term = -current_logprobs.exp() * current_logprobs  # 熵正则

    is_weights = torch.tensor(1.0, dtype=old_logprobs.dtype).to(old_logprobs.device)
    if inference_logprobs is not None:
        is_weights = (old_logprobs - inference_logprobs).exp()  # importance sampling
        if is_truncation_coef is not None:
            is_weights = torch.min(is_weights, torch.tensor(is_truncation_coef, ...))

    loss = (
        -is_weights * torch.min(ratios * advantages, clamped_ratios * advantages)  # PPO-clip surrogate
        + kl_beta * kl_term                                                         # KL 正则
        - entropy_weight * entropy_term                                             # 熵正则
    )

    return loss, kl_term, ratios, entropy_term, truncated_from_above, truncated_from_below
```

**advantage 计算**（`rl_utils.py:1173-1218`）：

```python
def calculate_grpo_advantages(rewards, num_turns):
    reward_means = np.where(real_mask, rewards, 0.0).sum(axis=-1) / real_counts
    reward_stds = np.sqrt(((rewards - reward_means)**2).sum(axis=-1) / real_counts)
    advantages = (rewards - reward_means) / (1e-4 + reward_stds)  # 组内归一化
```

**Off-policy 生成**：`RolloutBank`（`rollout_bank.py:206`）实现单写入持久化存储，`--rl-generation-lag` 控制生成超前步数，`--rl-partial-rollouts` 启用部分 rollout 重叠。

#### torchtitan GRPO/DAPO

**DAPOLoss**（`experiments/rl/losses/dapo.py:23-136`）：

```python
# 核心计算（L88-105）
trainer_logprobs, token_entropy = compute_logprobs(logits, labels, return_entropy=True)
effective_loss_mask = loss_mask & torch.isfinite(generator_logprobs)
ratio = torch.exp(clamp(trainer_logprobs - generator_logprobs, -10, 10))
clipped_ratio = clamp(ratio, 1 - ratio_clip_low, 1 + ratio_clip_high)  # 非对称 clip
token_loss = -min(ratio * advantages, clipped_ratio * advantages)
loss = (token_loss * effective_loss_mask).sum() / global_valid_tokens
```

`GRPOLoss`（`losses/grpo.py:17-36`）继承 `DAPOLoss`，将 `ratio_clip_low == ratio_clip_high`（对称 clip）。DAPO 的 "clip-higher" 通过增大上界保留更多 up-weighted token 的概率质量，对抗 entropy collapse。

#### slime（最完整的算法集）

| 文件 | 算法 | 核心函数 |
|------|------|----------|
| `utils/ppo_utils.py:12` | KL 估计 | `compute_approx_kl` — k1/k2/k3/low_var_kl |
| `utils/ppo_utils.py:125` | PPO | `compute_policy_loss` — 标准 PPO clip + dual-clip |
| `utils/ppo_utils.py:152` | CISPO | `compute_cispo_loss` — 裁剪比 stop-grad，梯度流过 log_probs |
| `utils/ppo_utils.py:95` | GSPO | `compute_gspo_kl` — 序列级 KL 展开为 per-token |
| `utils/ppo_utils.py:361` | GRPO | `get_grpo_returns` — 常数 reward 广播 |
| `utils/ppo_utils.py:371` | REINFORCE++ | `get_reinforce_plus_plus_returns` — chunked discounted returns |
| `utils/ppo_utils.py:584` | GAE | `vanilla_gae` / `chunked_gae` — 用于 PPO 的 advantage 估计 |
| `utils/ppo_utils.py:716` | LogProbs | `calculate_log_probs_and_entropy` — vocab-parallel softmax |

#### miles（与 slime 共享架构）

| 文件 | 功能 |
|------|------|
| `backends/training_utils/loss.py:28` | `compute_advantages_and_returns` — 统一 advantage 计算调度 |
| `backends/training_utils/loss_hub/advantages.py:53` | `compute_advantages` — grpo/gspo/ppo/reinforce++ 分支 |
| `backends/training_utils/loss_hub/math_utils.py:453` | `get_grpo_returns` — GRPO return 计算 |
| `ray/rollout/train_data_conversion.py:243` | GRPO `grpo_std_normalization` 开关 |

### 3.5 KL 约束实现

KL 散度防止策略偏离参考模型过远（reward hacking）。三种形式：

**作为 reward penalty**（slime PPO）：
```
final_reward = reward - kl_coef * KL(pi_theta || pi_ref)
```
代码：`slime/utils/ppo_utils.py:30-41` — `compute_approx_kl` k3/low_var_kl: `kl = exp(-log_ratio) - 1 - (-log_ratio)`（非负无偏估计，Schulman blog）

**作为 loss 正则项**（Megatron-LM GRPO）：
```
loss = -surrogate + kl_beta * kl_term
```
代码：`megatron/rl/rl_utils.py:2843-2858` — `kl_term = ref_diff.exp() - ref_diff - 1`，其中 `ref_diff = ref_logprobs - current_logprobs`

**IS 修正的 KL**（DeepSeek-V3.2 风格，slime）：
```python
# ppo_utils.py:44-45
if importance_ratio is not None:
    kl = importance_ratio * kl  # 无偏 KL 估计
```

---

## 4. 后训练数据工程

### 4.1 Chat Template 与 Tokenization

**SFTTokenizer**（`sft_tokenizer.py:46-232`）支持多种 prompt 格式：

| 格式 | assistant_prefix_len | 说明 |
|------|---------------------|------|
| `nemotron-nano-v2` | 3 | `<SPECIAL_11>Assistant\n` 前缀掩码 |
| `nemotron-h-aligned` | 0 | 无前缀掩码 |
| `default` | 0 | 使用 tokenizer 自带 chat_template，不掩码 |

**对话分词流程**（`sft_tokenizer.py:130-209`）：
1. `apply_chat_template(conversation, tokenize=True)` → 完整 token 序列
2. 逐 turn 重新 tokenize 确定边界 → `turn_tokens`
3. 按 role 对 target 设 `IGNORE_INDEX`

### 4.2 数据加载与批处理

**Megatron-LM VarlenDataset 多源加载**（`varlen_dataset.py:1-48`）：
- 自动检测 4 种 schema：`openai-messages` / `sharegpt` / `alpaca-dolly` / `pretrain-text`
- 字段同义词映射：`instruction|prompt|query|question` + `output|response|completion|answer`
- 多源：HuggingFace Hub / 本地 parquet / 本地 jsonl（pandas 读避免 pyarrow schema 推断失败）

**torchtitan Grain 数据管道**（`hf_datasets/text_datasets.py`）：
- `HuggingFaceStreamingSource` / `HuggingFaceRandomAccessSource` — 数据源抽象
- `SampleProcessor` → `TextProcessor` / `ChatProcessor` — 样本级 tokenization
- `FirstFitPackIterDataset` — 序列打包（`packing.py`）
- `TextCollator` — 批次级拼接 + padding（`collators.py:41`）

**Megatron-LM SFT packing**（`sft_dataset.py:104-150`）：
- 多条样本拼接到 `sequence_length + 1`（+1 用于 input/label 错位）
- `cu_seqlens` 记录每条子序列边界
- CP（Context Parallel）场景下 padding 到 `cp_size * 2` 的倍数（`sft_dataset.py:127-132`）

### 4.3 后训练流水线 ASCII 图

```
            ┌─────────────────────────────────────────────────────────────┐
            │                   后训练数据流水线                           │
            └─────────────────────────────────────────────────────────────┘

  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
  │  Raw Data    │    │  Tokenize +  │    │   Packing /  │    │   Forward    │
  │  jsonl/HF    │───▶│  Loss Mask   │───▶│  Batching    │───▶│   + Loss     │
  │  parquet     │    │  IGNORE_INDEX│    │  cu_seqlens  │    │              │
  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
        │                    │                   │                    │
   SFTLowLevelDataset   SFTTokenizer      SFTDataset           loss_func
   VarlenDataset        ChatProcessor     FirstFitPack       calculate_grpo_loss
   (sft_dataset.py:17)  (sft_tokenizer    (sft_dataset.py:    (rl_utils.py:2771)
   (varlen_dataset.py)   .py:130)          104)


            ┌─────────────────────────────────────────────────────────────┐
            │                GRPO/PPO 训练循环                             │
            └─────────────────────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │ Generate │────▶│  Reward  │────▶│ Advantage│──┬─▶│  Policy  │
   │ Rollouts │     │ Function │     │ Compute  │  │  │  Update  │
   └──────────┘     └──────────┘     └──────────┘  │  └──────────┘
        │                                            │       │
   InferenceEngine              calculate_grpo_       │  calculate_grpo_loss
   (megatron/rl/                advantages            │  compute_policy_loss
    inference/)                 (rl_utils.py:1173)    │  (ppo_utils.py:125)
                                                    │
                                              ┌──────────┐
                                              │ KL Constraint│
                                              │ kl_beta *   │
                                              │ kl_term     │
                                              └──────────┘

   Off-policy 重叠 (Megatron-LM --rl-partial-rollouts):
   ┌──────────────────────────────────────────────────────────────┐
   │ Step N:   [Train N] ──────────────────────────────▶          │
   │ Step N+1:          [Train N+1] ──────────────────▶           │
   │ Gen:       [Gen N+1] [Gen N+2] [Gen N+3]                     │
   │            ──── lag=2 ────▶                                  │
   │ RolloutBank (rollout_bank.py:206) 持久化已完成 rollout       │
   └──────────────────────────────────────────────────────────────┘
```

---

## 5. 关键配置参数表

### 5.1 SFT 配置参数

| 参数 | 仓库 | 含义 | 典型值 |
|------|------|------|--------|
| `--seq-length` | Megatron-LM | 最大序列长度 | 2048-8192 |
| `--finetune-hf-dataset` | Megatron-LM | HuggingFace 数据集名称 | `HuggingFaceH4/ultrachat_200k` |
| `--finetune-data-split` | Megatron-LM | 数据集 split | `train` |
| `--micro-batch-size` | Megatron-LM | SFT 强制为 1 | 1 |
| `--num-tokens-per-batch` | torchtitan | packing 批次 token 数 | 视配置 |
| `--messages-fn` | torchtitan | 消息字段提取函数 | 自定义 |
| `--reset-position-ids` | Megatron-LM | 每条样本重置 position_ids | True |
| `--eod-mask-loss` | Megatron-LM | EOD token 掩码 | True |

### 5.2 GRPO 配置参数

| 参数 | 仓库 | 含义 | 典型值 |
|------|------|------|--------|
| `--grpo-prompts-per-step` | Megatron-LM | 每步采样的 prompt 数 | 32 |
| `--grpo-group-size` | Megatron-LM | 每个 prompt 采样响应数 | 2-8 |
| `--grpo-clamp-eps-lower` | Megatron-LM | 重要性比下界 clip | 0.01 |
| `--grpo-clamp-eps-upper` | Megatron-LM | 重要性比上界 clip | 0.01 (DAPO 可非对称) |
| `--grpo-kl-beta` | Megatron-LM | KL 正则系数 | 0.001 |
| `--grpo-entropy-term-weight` | Megatron-LM | 熵正则系数 | 0.0 |
| `--rl-partial-rollouts` | Megatron-LM | 启用 off-policy 生成重叠 | True |
| `--rl-generation-lag` | Megatron-LM | 生成超前步数 | 0-2 |
| `--rl-submission-granularity` | Megatron-LM | 生成提交粒度 (R/G/B) | B |
| `--ratio-clip-low` | torchtitan | DAPO 下界 clip | 0.2 |
| `--ratio-clip-high` | torchtitan | DAPO 上界 clip (可 > low) | 0.28 |

### 5.3 PPO/通用 RL 配置参数

| 参数 | 仓库 | 含义 | 典型值 |
|------|------|------|--------|
| `--advantage-estimator` | slime/miles | 算法选择 | grpo/ppo/reinforce++/cispo/gspo |
| `--kl-coef` | slime/miles | KL reward 惩罚系数 | 0.01-0.2 |
| `--kl-loss-coef` | slime/miles | KL loss 正则系数 | 0.01-0.05 |
| `--eps-clip` | slime/miles | PPO clip 范围 | 0.2 |
| `--eps-clip-high` | slime/miles | PPO 上界 clip（非对称） | 0.2-0.28 |
| `--gamma` | slime/miles | GAE 折扣因子 | 1.0 |
| `--lambd` | slime/miles | GAE lambda | 1.0 |
| `--entropy-coef` | slime/miles | 熵正则系数 | 0.0 |
| `--n-samples-per-prompt` | slime/miles | 每 prompt 采样响应数 | 4-16 |
| `--rollout-batch-size` | slime/miles | 每轮 rollout 的 prompt 数 | 视配置 |
| `--num-steps-per-rollout` | slime/miles | rollout 划分的训练步数 | 4-8 |
| `--normalize-advantages` | slime/mines | 跨 DP 组归一化 advantage | True |
| `--grpo-std-normalization` | slime/miles | GRPO 组内标准差归一化 | True |
| `--kl-loss-type` | slime/miles | KL 估计器类型 | k1/k2/k3/low_var_kl |

### 5.4 Reward Model 配置参数

| 参数 | 仓库 | 含义 | 典型值 |
|------|------|------|--------|
| `--rm-type` | miles | 奖励函数类型 | deepscaler/dapo/math/gpqa/f1/remote_rm |
| `--rm-url` | miles | 远程 RM 服务地址 | URL |
| `--custom-rm-path` | miles | 自定义奖励函数路径 | 文件路径 |
| `--opd-kl-coef` | slime/miles | On-Policy Distillation KL 系数 | 0.0 |
| `--custom-advantage-function-path` | slime | 自定义 advantage 函数路径 | 文件路径 |

---

## 6. 源码文件索引

### 6.1 Megatron-LM

| 文件路径 | 功能描述 | 关键行号 |
|----------|----------|----------|
| `megatron/training/datasets/sft_dataset.py` | SFT 数据集（jsonl messages + packing） | L17 `SFTLowLevelDataset`, L51 `SFTDataset`, L169 `loss_mask` |
| `megatron/training/datasets/varlen_dataset.py` | 变长多源 SFT 数据集 | L3 说明, L54 字段同义词 |
| `megatron/core/tokenizers/text/libraries/sft_tokenizer.py` | SFT 分词器 + chat template masking | L46 `SFTTokenizer`, L130 `tokenize_conversation` |
| `megatron/post_training/loss_func.py` | SFT 损失函数（+ KD 蒸馏） | L39 `loss_func`, L13 `_mask_loss` |
| `examples/post_training/modelopt/finetune.py` | SFT 入口脚本（HF datasets + packing） | L64 `SFTDataset`, L438 `train_valid_test_sft_datasets_provider` |
| `train_rl.py` | GRPO RL 训练入口 | L190 `forward_step`, L337 `train_valid_test_datasets_provider` |
| `megatron/rl/rl_utils.py` | GRPO 核心工具函数 | L1067 `get_logprobs`, L1173 `calculate_grpo_advantages`, L2771 `calculate_grpo_loss` |
| `megatron/rl/rollout_bank.py` | Rollout 持久化存储 | L206 `RolloutBank` |
| `megatron/rl/agent/reward_only_agent.py` | Reward-only Agent 基类 | L51 `get_reward`, L180 `_rollout_from_episode` |
| `megatron/rl/agent/api.py` | Agent/Rollout 类型定义 | L54 `ContrastiveRollout` |
| `megatron/rl/README.md` | Megatron-RL 设计文档 | 全文（off-policy lag 说明） |

### 6.2 torchtitan

| 文件路径 | 功能描述 | 关键行号 |
|----------|----------|----------|
| `torchtitan/hf_datasets/text_datasets.py` | SFT 数据处理（ChatProcessor） | L32 `TextProcessor`, L61 `ChatProcessor`, L100 `_tokenize_sample` |
| `torchtitan/components/loss.py` | 损失函数（CE / chunked / vocab-parallel） | L27 `IGNORE_INDEX`, L32 `cross_entropy_loss`, L326 `compute_logprobs`, L509 `ChunkedLossWrapper` |
| `torchtitan/components/data/collators.py` | 批次 collator | L41 `TextCollator`, L71 IGNORE_INDEX padding |
| `torchtitan/components/data/packing.py` | 序列打包 | L102 `IGNORE_INDEX`, L144 填充掩码 |
| `torchtitan/experiments/rl/train.py` | RL 训练入口（Monarch Actors） | L20 命令, L41 `breakable_cudagraph_env` |
| `torchtitan/experiments/rl/losses/dapo.py` | DAPO 损失（per-token clip-higher） | L23 `DAPOLoss`, L57 `__call__`, L99 核心 clip |
| `torchtitan/experiments/rl/losses/grpo.py` | GRPO 损失（DAPO 对称特例） | L17 `GRPOLoss`, L26 `clip_eps` |
| `torchtitan/experiments/rl/rollout_recorder.py` | Rollout 记录器（保留极端 reward） | L34 `keep_extreme_rewards` |

### 6.3 slime

| 文件路径 | 功能描述 | 关键行号 |
|----------|----------|----------|
| `slime/utils/ppo_utils.py` | PPO/GRPO/CISPO/GSPO 核心算法集 | L12 `compute_approx_kl`, L125 `compute_policy_loss`, L152 `compute_cispo_loss`, L361 `get_grpo_returns`, L584 `vanilla_gae`, L716 `calculate_log_probs_and_entropy` |
| `slime/ray/rollout.py` | Rollout 执行 + reward 后处理 | L279 `_post_process_rewards`, L298 GRPO normalization |
| `slime/utils/arguments.py` | RL 训练配置 | L1382 `--eps-clip`, L1470 `--entropy-coef`, L1839 kl_coef 校验 |

### 6.4 miles

| 文件路径 | 功能描述 | 关键行号 |
|----------|----------|----------|
| `miles/backends/training_utils/loss.py` | 统一 advantage + loss 调度 | L28 `compute_advantages_and_returns`, L36 算法列表 |
| `miles/backends/training_utils/loss_hub/advantages.py` | Advantage 计算分支 | L53 `compute_advantages` (grpo/gspo/ppo/reinforce++) |
| `miles/backends/training_utils/loss_hub/math_utils.py` | GRPO return + KL 工具 | L453 `get_grpo_returns` |
| `miles/rollout/rm_hub/__init__.py` | Reward Hub 统一接口 | L43 `async_rm`, L59-86 奖励类型分发 |
| `miles/rollout/rm_hub/deepscaler.py` | DeepScaler 规则奖励 | L38 `get_deepscaler_rule_based_reward` |
| `miles/ray/rollout/train_data_conversion.py` | 奖励归一化处理 | L207 `_normalize_rewards_by_rollout`, L257 `_post_process_rewards` |
| `miles/rollout/on_policy_distillation.py` | On-Policy Distillation | L88 `_get_reward_weight_mode`, L352 `reward_func` |

---

## 7. 面试高频要点

### 7.1 Loss Masking 为何必须？

不做 loss masking 会导致模型学习预测 prompt 内容（而非学习生成 response），指令跟随能力下降。`IGNORE_INDEX = -100` 是 PyTorch `cross_entropy` 的默认忽略索引。torchtitan 在 `components/loss.py:27`、Megatron-LM 在 `sft_dataset.py:14` 均定义为该值。

### 7.2 GRPO 取代 PPO 的原因

PPO 需要 learned reward model + value network（critic），训练成本高且不稳定。GRPO 用同组采样的 reward 均值/方差归一化代替，无需额外网络。代码体现：Megatron-LM `calculate_grpo_advantages`（`rl_utils.py:1173`）仅做组内 z-score，而 PPO 路径需要 `vanilla_gae`（`ppo_utils.py:584`）+ value head。

### 7.3 各仓库的设计哲学差异

- **Megatron-LM**：生产级，megatron/rl/ 模块与 training loop 深度集成，支持 off-policy 生成重叠（`--rl-partial-rollouts` + `RolloutBank`）
- **torchtitan**：研究级，PyTorch-native SPMD，RL 放在 experiments/，强调可组合性（DAPOLoss → GRPOLoss 继承）
- **slime**：算法最全（PPO/GRPO/CISPO/GSPO/REINFORCE++），`@torch.compile(dynamic=True)` 装饰核心损失函数
- **miles**：Reward Hub 统一抽象（`async_rm` 分发），生产级基础设施（async、LoRA、on-policy distillation）

### 7.4 后训练 vs 预训练的并行策略差异

后训练（尤其 RLHF）中生成阶段（inference）与训练阶段常使用不同并行策略：
- Megatron-LM off-policy 生成：`max_effective_lag = DP * engine.max_requests / (G * P) - 1`（`rl/README.md:76`）
- miles/slime async：`rollout_batch_size * n_samples_per_prompt // num_steps_per_rollout` 决定 global batch（`arguments.py:1965`）

---

## 附录：源码文件索引

按功能分类列出本文档引用的所有文件路径和核心类/函数：

### 数据集与分词

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `megatron/training/datasets/sft_dataset.py` | `SFTLowLevelDataset` (:17), `SFTDataset` (:51) | SFT 数据集（jsonl messages + packing） |
| `megatron/training/datasets/varlen_dataset.py` | `VarlenDataset` (:3) | 变长多源 SFT 数据集 |
| `megatron/core/tokenizers/text/libraries/sft_tokenizer.py` | `SFTTokenizer` (:46), `tokenize_conversation` (:130) | SFT 分词器 + chat template masking |
| `torchtitan/hf_datasets/text_datasets.py` | `TextProcessor` (:32), `ChatProcessor` (:61) | SFT 数据处理（前缀重分词） |
| `torchtitan/components/data/collators.py` | `TextCollator` (:41) | 批次 collator |
| `torchtitan/components/data/packing.py` | `FirstFitPackIterDataset` (:144) | 序列打包 |

### 损失函数

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `megatron/post_training/loss_func.py` | `_mask_loss` (:13), `loss_func` (:39) | SFT 损失函数（+ KD 蒸馏） |
| `torchtitan/components/loss.py` | `cross_entropy_loss` (:32), `compute_logprobs` (:326), `ChunkedLossWrapper` (:509) | CE / chunked / vocab-parallel 损失 |
| `megatron/rl/rl_utils.py` | `calculate_grpo_loss` (:2771), `calculate_grpo_advantages` (:1173), `get_logprobs` (:1067) | GRPO 核心工具函数 |
| `torchtitan/experiments/rl/losses/dapo.py` | `DAPOLoss` (:23) | DAPO 损失（per-token clip-higher） |
| `torchtitan/experiments/rl/losses/grpo.py` | `GRPOLoss` (:17) | GRPO 损失（DAPO 对称特例） |
| `slime/utils/ppo_utils.py` | `compute_approx_kl` (:12), `compute_policy_loss` (:125), `compute_cispo_loss` (:152), `get_grpo_returns` (:361), `vanilla_gae` (:584), `calculate_log_probs_and_entropy` (:716) | PPO/GRPO/CISPO/GSPO 算法集 |
| `miles/backends/training_utils/loss.py` | `compute_advantages_and_returns` (:28) | 统一 advantage + loss 调度 |
| `miles/backends/training_utils/loss_hub/advantages.py` | `compute_advantages` (:53) | Advantage 计算分支 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `get_grpo_returns` (:453) | GRPO return 计算 |

### RL 训练入口与 Agent

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `examples/post_training/modelopt/finetune.py` | `SFTDataset` (:64), `train_valid_test_sft_datasets_provider` (:438) | SFT 入口脚本 |
| `train_rl.py` | `forward_step` (:190), `train_valid_test_datasets_provider` (:337) | GRPO RL 训练入口 |
| `torchtitan/experiments/rl/train.py` | `breakable_cudagraph_env` (:41) | RL 训练入口（Monarch Actors） |
| `megatron/rl/agent/reward_only_agent.py` | `get_reward` (:51), `_rollout_from_episode` (:180) | Reward-only Agent 基类 |
| `megatron/rl/agent/api.py` | `ContrastiveRollout` (:54) | Agent/Rollout 类型定义 |
| `megatron/rl/rollout_bank.py` | `RolloutBank` (:206) | Rollout 持久化存储 |

### Reward 与 Rollout

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `miles/rollout/rm_hub/__init__.py` | `async_rm` (:43) | Reward Hub 统一接口 |
| `miles/rollout/rm_hub/deepscaler.py` | `get_deepscaler_rule_based_reward` (:38) | DeepScaler 规则奖励 |
| `miles/rollout/rm_hub/gpqa.py` | `compute_gpqa_reward` (:54) | GPQA 选择题评分 |
| `miles/ray/rollout/train_data_conversion.py` | `_normalize_rewards_by_rollout` (:207), `_post_process_rewards` (:257) | 奖励归一化处理 |
| `slime/ray/rollout.py` | `_post_process_rewards` (:279) | Rollout 执行 + reward 后处理 |
| `torchtitan/experiments/rl/rollout_recorder.py` | `keep_extreme_rewards` (:34) | Rollout 记录器 |
| `miles/rollout/on_policy_distillation.py` | `_get_reward_weight_mode` (:88), `reward_func` (:352) | On-Policy Distillation |

### 配置与参数

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `slime/utils/arguments.py` | `--eps-clip` (:1382), `--entropy-coef` (:1470), kl_coef 校验 (:1839) | RL 训练配置 |

### 设计文档

| 文件路径 | 功能 |
|---------|------|
| `megatron/rl/README.md` | Megatron-RL 设计文档（off-policy lag 说明） |

---

## 8. 后训练完整调用链总图

下图展示 SFT 或 GRPO 训练一个完整步骤中各模块的调用关系：

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              后训练完整调用链总图                                          │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          SFT 训练调用链 (Megatron-LM)                            │    │
│  ├─────────────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                                 │    │
│  │  finetune.py::__main__                                                          │    │
│  │    └─ pretrain(cfg, train_valid_test_sft_datasets_provider, forward_step, ...)   │    │
│  │         ├─ SFTDataset (sft_dataset.py:51)                                       │    │
│  │         │    └─ SFTTokenizer.tokenize_conversation (sft_tokenizer.py:130)        │    │
│  │         │         └─ loss masking → IGNORE_INDEX                                │    │
│  │         └─ forward_step (finetune.py:544)                                       │    │
│  │              ├─ get_batch → build_lm_batch                                      │    │
│  │              ├─ model(tokens, position_ids, attention_mask, labels)              │    │
│  │              │    └─ GPTModel.forward → TransformerBlock → Attention / MLP       │    │
│  │              └─ loss_func(loss_mask, output_tensor, model)                       │    │
│  │                   └─ _mask_loss → sum(losses * loss_mask) (loss_func.py:13)      │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                        GRPO 训练调用链 (Megatron-LM)                             │    │
│  ├─────────────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                                 │    │
│  │  train_rl.py::__main__                                                          │    │
│  │    └─ pretrain(cfg, None, ModelType.encoder_or_decoder, forward_step, ...)       │    │
│  │         │                                                                       │    │
│  │         │  ┌──────────────────────────────────────────────────────────────┐     │    │
│  │         │  │              Rollout 生成阶段 (Off-policy)                     │     │    │
│  │         │  │  InferenceEngine.generate()                                   │     │    │
│  │         │  │    └─ model.generate → tokens                                  │     │    │
│  │         │  │  RewardOnlyAgent._rollout_from_episode (reward_only_agent.py)  │     │    │
│  │         │  │    └─ get_reward() → rewards                                   │     │    │
│  │         │  │  calculate_grpo_advantages (rl_utils.py:1173)                   │     │    │
│  │         │  │    └─ advantage = (reward - mean) / std                        │     │    │
│  │         │  │  RolloutBank.store (rollout_bank.py:206)                       │     │    │
│  │         │  └──────────────────────────────────────────────────────────────┘     │    │
│  │         │                                                                       │    │
│  │         └─ forward_step (train_rl.py:190)                                       │    │
│  │              ├─ load_packed_data_by_index → (tokens, advantages, old_logprobs,   │    │
│  │              │                                 loss_mask, ref_logprobs, ...)     │    │
│  │              ├─ get_logprobs(model, tokens, position_ids) → current_logprobs     │    │
│  │              │    └─ model.forward → logits → log_softmax                       │    │
│  │              └─ calculate_grpo_loss(current, old, ref, advantages, ...)          │    │
│  │                   ├─ ratios = (current - old).exp()                             │    │
│  │                   ├─ clamped = ratios.clamp(1-eps_lower, 1+eps_upper)            │    │
│  │                   ├─ kl_term = (ref - current).exp() - (ref - current) - 1       │    │
│  │                   └─ loss = -min(ratios*adv, clamped*adv) + kl_beta*kl_term      │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          通用模块依赖关系                                         │    │
│  ├─────────────────────────────────────────────────────────────────────────────────┤    │
│  │                                                                                 │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │    │
│  │  │  Data Layer │    │  Tokenizer  │    │   Model     │    │   Loss      │       │    │
│  │  │  SFTDataset │───▶│  SFTTokenizer│───▶│  GPTModel   │───▶│  loss_func  │       │    │
│  │  │  VarlenDataset│  │  ChatProcessor│  │  (TP/SP/CP) │    │  calculate_ │       │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    │  grpo_loss  │       │    │
│  │       │                   │                  │              └─────────────┘       │    │
│  │       │                   │                  │                    │              │    │
│  │       ▼                   ▼                  ▼                    ▼              │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐       │    │
│  │  │  Packing    │    │  Loss Mask  │    │  Inference  │    │  Optimizer  │       │    │
│  │  │  cu_seqlens │    │  IGNORE_    │    │  Engine     │    │  AdamW      │       │    │
│  │  │  FirstFitPack│   │  INDEX=-100 │    │  (Rollout)  │    │  (分布式)   │       │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘       │    │
│  │                                                                                 │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录：源码文件索引

按功能分类列出本文档引用的所有文件路径和核心类/函数：

### 数据集与分词

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `megatron/training/datasets/sft_dataset.py` | `SFTLowLevelDataset` (:17), `SFTDataset` (:51) | SFT 数据集（jsonl messages + packing） |
| `megatron/training/datasets/varlen_dataset.py` | `VarlenDataset` (:3) | 变长多源 SFT 数据集 |
| `megatron/core/tokenizers/text/libraries/sft_tokenizer.py` | `SFTTokenizer` (:46), `tokenize_conversation` (:130) | SFT 分词器 + chat template masking |
| `torchtitan/hf_datasets/text_datasets.py` | `TextProcessor` (:32), `ChatProcessor` (:61) | SFT 数据处理（前缀重分词） |
| `torchtitan/components/data/collators.py` | `TextCollator` (:41) | 批次 collator |
| `torchtitan/components/data/packing.py` | `FirstFitPackIterDataset` (:144) | 序列打包 |

### 损失函数

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `megatron/post_training/loss_func.py` | `_mask_loss` (:13), `loss_func` (:39) | SFT 损失函数（+ KD 蒸馏） |
| `torchtitan/components/loss.py` | `cross_entropy_loss` (:32), `compute_logprobs` (:326), `ChunkedLossWrapper` (:509) | CE / chunked / vocab-parallel 损失 |
| `megatron/rl/rl_utils.py` | `calculate_grpo_loss` (:2771), `calculate_grpo_advantages` (:1173), `get_logprobs` (:1067) | GRPO 核心工具函数 |
| `torchtitan/experiments/rl/losses/dapo.py` | `DAPOLoss` (:23) | DAPO 损失（per-token clip-higher） |
| `torchtitan/experiments/rl/losses/grpo.py` | `GRPOLoss` (:17) | GRPO 损失（DAPO 对称特例） |
| `slime/utils/ppo_utils.py` | `compute_approx_kl` (:12), `compute_policy_loss` (:125), `compute_cispo_loss` (:152), `get_grpo_returns` (:361), `vanilla_gae` (:584), `calculate_log_probs_and_entropy` (:716) | PPO/GRPO/CISPO/GSPO 算法集 |
| `miles/backends/training_utils/loss.py` | `compute_advantages_and_returns` (:28) | 统一 advantage + loss 调度 |
| `miles/backends/training_utils/loss_hub/advantages.py` | `compute_advantages` (:53) | Advantage 计算分支 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `get_grpo_returns` (:453) | GRPO return 计算 |

### RL 训练入口与 Agent

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `examples/post_training/modelopt/finetune.py` | `SFTDataset` (:64), `train_valid_test_sft_datasets_provider` (:438) | SFT 入口脚本 |
| `train_rl.py` | `forward_step` (:190), `train_valid_test_datasets_provider` (:337) | GRPO RL 训练入口 |
| `torchtitan/experiments/rl/train.py` | `breakable_cudagraph_env` (:41) | RL 训练入口（Monarch Actors） |
| `megatron/rl/agent/reward_only_agent.py` | `get_reward` (:51), `_rollout_from_episode` (:180) | Reward-only Agent 基类 |
| `megatron/rl/agent/api.py` | `ContrastiveRollout` (:54) | Agent/Rollout 类型定义 |
| `megatron/rl/rollout_bank.py` | `RolloutBank` (:206) | Rollout 持久化存储 |

### Reward 与 Rollout

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `miles/rollout/rm_hub/__init__.py` | `async_rm` (:43) | Reward Hub 统一接口 |
| `miles/rollout/rm_hub/deepscaler.py` | `get_deepscaler_rule_based_reward` (:38) | DeepScaler 规则奖励 |
| `miles/rollout/rm_hub/gpqa.py` | `compute_gpqa_reward` (:54) | GPQA 选择题评分 |
| `miles/ray/rollout/train_data_conversion.py` | `_normalize_rewards_by_rollout` (:207), `_post_process_rewards` (:257) | 奖励归一化处理 |
| `slime/ray/rollout.py` | `_post_process_rewards` (:279) | Rollout 执行 + reward 后处理 |
| `torchtitan/experiments/rl/rollout_recorder.py` | `keep_extreme_rewards` (:34) | Rollout 记录器 |
| `miles/rollout/on_policy_distillation.py` | `_get_reward_weight_mode` (:88), `reward_func` (:352) | On-Policy Distillation |

### 配置与参数

| 文件路径 | 核心类/函数 | 功能 |
|---------|-----------|------|
| `slime/utils/arguments.py` | `--eps-clip` (:1382), `--entropy-coef` (:1470), kl_coef 校验 (:1839) | RL 训练配置 |

### 设计文档

| 文件路径 | 功能 |
|---------|------|
| `megatron/rl/README.md` | Megatron-RL 设计文档（off-policy lag 说明） |

