# 强化学习 (Reinforcement Learning) — 代码级深度分析

> 本文档基于 **miles** 与 **slime** 双仓库源码的逐行阅读，提供 RL 训练系统的代码级专家分析。所有行号均经实际文件验证。

---

## 0. RL 训练循环全景图

LLM 强化学习训练系统由五个核心阶段组成，形成闭环迭代。下图展示完整 RL Loop 的数据流，包含同步/异步两种模式的差异：

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RL Training Loop 全景图                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│   ┌──────────┐     ┌──────────────────────────────────────────────────────────┐      │
│   │  Prompt  │     │              Rollout (SGLang / vLLM)                     │      │
│   │  Dataset │────▶│  Generate Responses → Log-probs → Sampling              │      │
│   └──────────┘     └───────────────────────┬──────────────────────────────────┘      │
│                                            │                                          │
│                                            ▼                                          │
│   ┌────────────────────────────────────────────────────────────────────────────┐     │
│   │                      Reward Computation                                    │     │
│   │  Rule-based Reward / RM Model / Outcome Reward → Per-sequence Score       │     │
│   └───────────────────────┬────────────────────────────────────────────────────┘     │
│                           │                                                         │
│                           ▼                                                         │
│   ┌────────────────────────────────────────────────────────────────────────────┐     │
│   │                      Advantage Estimation                                   │     │
│   │  GRPO: Group Norm │ PPO: GAE(γ,λ) │ REINFORCE++: KL-penalized Returns     │     │
│   └───────────────────────┬────────────────────────────────────────────────────┘     │
│                           │                                                         │
│                           ▼                                                         │
│   ┌────────────────────────────────────────────────────────────────────────────┐     │
│   │                      Policy Update (GRPO / PPO)                            │     │
│   │  PPO Clipped Loss + KL Penalty + Entropy Bonus + TIS/OPSM/OPD Correction  │     │
│   └───────────────────────┬────────────────────────────────────────────────────┘     │
│                           │                                                         │
│                           ▼                                                         │
│   ┌────────────────────────────────────────────────────────────────────────────┐     │
│   │                      Weight Sync (Training → Inference)                    │     │
│   │  Broadcast / P2P / RDT / Delta → SGLang Engines → Next Rollout           │     │
│   └────────────────────────────────────────────────────────────────────────────┘     │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  【同步模式】每步串行执行：Rollout → Reward → Advantage → Policy Update → Weight Sync │
│                                                                                     │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐              │
│  │ Rollout │──▶│ Reward  │──▶│Advantage│──▶│ Policy  │──▶│  Weight │──▶ Next      │
│  │         │   │         │   │         │   │ Update  │   │  Sync   │              │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘   └─────────┘              │
│                                                                                     │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  【异步模式】当前步训练与上一步 rollout 重叠（流水线并行）：                            │
│                                                                                     │
│  ┌─────────┐                                                                          │
│  │Rollout 0│────────────────┐                                                        │
│  └─────────┘                │                                                        │
│       ┌─────────┐           │                                                        │
│       │Train  0 │◀──────────┘                                                        │
│       └─────────┘           ┌─────────┐                                             │
│            ┌─────────┐      │Rollout 1│────────────────┐                            │
│            │Weight   │◀─────└─────────┘                │                            │
│            │Sync     │                                 │                            │
│            └─────────┘                    ┌─────────┐  │                            │
│                                           │Train  1 │◀─┘                            │
│                                           └─────────┘                               │
│                                                                                     │
│  关键差异：异步模式通过 `update_weights_interval` 控制同步频率，                        │
│  训练与推理 GPU 资源分离时可实现 100% 流水线利用率。                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**RL Loop 五阶段职责**：

| 阶段 | 输入 | 输出 | 核心计算 |
|------|------|------|---------|
| **Rollout** | Prompt batch | Response + log-probs | SGLang/vLLM 生成 |
| **Reward** | Response | Scalar score | Rule-based / RM / Outcome |
| **Advantage** | Rewards + KL | Per-token advantage | GRPO group norm / GAE |
| **Policy Update** | Advantage + old log-probs | Weight gradient | PPO clipped loss + KL + entropy |
| **Weight Sync** | Updated weights | Synced engines | Broadcast / P2P / RDT |

---

## 1. RL 训练循环总览

### 概念原理

LLM 强化学习训练的核心循环（RL Loop）由四个阶段组成：

1. **Rollout（采样）**：用当前策略 π_θ 对 prompt 批量生成回答
2. **Reward（奖励）**：对生成结果评分
3. **Advantage 估计**：计算每个 token/序列的相对优势
4. **Policy Update**：用 GRPO/PPO 损失更新策略权重

与监督学习不同，RL 训练是 **on-policy** 的：每次更新后策略改变，必须重新采样。这导致训练系统必须是 **训推一体（Training-Inference Integration）** 架构。

### miles 实现

miles 提供同步与异步两种训练模式：

**同步模式** (`miles/train.py:22 async def train`)：
```python
# miles/train.py:93-137 — 同步训练主循环
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_pack = await rollout_manager.generate.remote(rollout_id)  # 采样
    await actor_model.train(rollout_id, rollout_data_pack)                 # 训练
    await actor_model.update_weights(rollout_id=rollout_id)                # 权重同步
```

**异步模式** (`miles/train_async.py:22 async def train`)：
```python
# miles/train_async.py:73-92 — 异步流水线：当前步训练与上一步 rollout 重叠
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_curr_ref = await rollout_data_next_future      # 等待当前数据
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)  # 预取下一步
    await actor_model.train(rollout_id, rollout_data_curr_ref)
```

关键设计：
- `miles/train.py:28` `object_store.init_instance(args)` — 初始化跨进程对象存储（支持 Ray / Mooncake 后端，`miles/utils/object_store.py:33`）
- `miles/train.py:33-36` 先创建 RolloutManager 以计算 `num_rollout_per_epoch`，再创建训练模型
- `miles/train.py:38-46` 可选的 Control Server（`miles/utils/ft_utils/control_server/server.py`）用于容错控制
- `miles/train.py:52` `await actor_model.update_weights()` — 首次权重同步确保 SGLang 与训练模型一致
- `miles/train_async.py:107-111` 异步模式按 `update_weights_interval` 控制同步频率，并在同步前等待当前 rollout 完成

### slime 实现

slime 同样提供两种模式：

**同步模式** (`slime/train.py:9 def train`)：
```python
# slime/train.py:49-92 — 同步主循环
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
    ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
    actor_model.update_weights()
```

**异步模式** (`slime/train_async.py:10 def train`)：
```python
# slime/train_async.py:32-53 — 异步流水线，与 miles 结构类似
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)
for rollout_id in range(args.start_rollout_id, args.num_rollout):
    rollout_data_curr_ref = ray.get(rollout_data_next_future)
    rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))
```

关键差异：
- `slime/train.py:21` 同步模式使用 `ray.get()` 阻塞调用，而 miles 使用 `await`（asyncio）
- `slime/train.py:61-69` 支持 Critic 模型（PPO 模式）：`critic_model.async_train()` + `actor_model.async_train()`
- `slime/train.py:45-46` `actor_trains` 逻辑：前 `num_critic_only_steps` 步仅训练 Critic
- `slime/train_async.py:66-70` 异步模式按 `(rollout_id + 1) % update_weights_interval == 0` 控制权重同步时机

### 架构对比

```
┌──────────────────────────────────────────────────────────────────────┐
│                    RL Training Pipeline Data Flow                     │
│                                                                      │
│  ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────┐   │
│  │ Prompt  │───▶│ Rollout  │───▶│  Reward  │───▶│  Advantage   │   │
│  │ Dataset │    │ (SGLang) │    │  Model   │    │  Estimation  │   │
│  └─────────┘    └────┬─────┘    └──────────┘    └──────┬───────┘   │
│                      │                                  │            │
│                      │ samples                          │ adv        │
│                      ▼                                  ▼            │
│              ┌──────────────┐                  ┌──────────────┐      │
│              │  Training    │◀─── weights ─────│  Old Policy  │      │
│              │  (Megatron/  │                  │  Log-probs   │      │
│              │   FSDP)      │                  └──────────────┘      │
│              └──────┬───────┘                                        │
│                     │                                                │
│                     ▼                                                │
│              ┌──────────────┐                                        │
│              │ UpdateWeight │────────────────▶ SGLang Engines       │
│              │ (broadcast/  │                  (next rollout)        │
│              │  p2p/rdt)    │                                        │
│              └──────────────┘                                        │
└──────────────────────────────────────────────────────────────────────┘
```

**调用链 1：同步训练主循环**
```
train() → create_rollout_manager() → create_training_models()
       → rollout_manager.generate.remote() → actor_model.train()
       → actor_model.update_weights() → [next iteration]
```

**调用链 2：异步训练流水线**
```
train_async() → rollout_manager.generate.remote(0)  # 预取
             → for loop:
                 await rollout_data_next_future      # 等待当前
                 rollout_manager.generate.remote(i+1) # 启动下一步
                 await actor_model.train(rollout_data_curr_ref)
                 if (i+1) % update_weights_interval == 0:
                     await actor_model.update_weights()
```

---

## 2. Rollout 生成

### miles: rollout_manager.py, vLLM/SGLang 集成

miles 的 RolloutManager 位于 `miles/ray/rollout/rollout_manager.py:54`，是一个 Ray Actor：

```python
# miles/ray/rollout/rollout_manager.py:53-54
@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""
```

核心方法：
- `__init__` (`:57-123`)：初始化数据源、SGLang 服务器、健康监控
- `generate` (`:144-169`)：执行 rollout 并转换为训练数据
- `_get_rollout_data` (`:243-271`)：调用 rollout 函数获取样本
- `convert_samples_to_train_data` (`:157-163`)：样本→训练数据转换

SGLang 集成：
- `miles/ray/rollout/rollout_server.py` — `start_rollout_servers()` 启动 SGLang 服务
- `miles/ray/rollout/router_manager.py` — `start_session_server()` 启动路由服务器
- `miles/rollout/sglang_rollout.py` — SGLang rollout 函数实现

数据流：
```python
# miles/ray/rollout/rollout_manager.py:144-169
async def generate(self, rollout_id):
    data, metadata, metrics = await self._get_rollout_data(rollout_id=rollout_id)
    data = convert_samples_to_train_data(self.args, data, metadata=metadata, ...)
    data_ref = split_train_data_by_dp(self.args, data, self.train_parallel_config)
    return dict(sample_indices=sample_indices, data_ref=data_ref)
```

### slime: sglang_rollout.py:83 GenerateState, rollout.py:38 RolloutManager

slime 的 RolloutManager 位于 `slime/ray/rollout.py:38`：

```python
# slime/ray/rollout.py:37-38
@ray.remote
class RolloutManager:
    """The class to run rollout and convert rollout data to training data."""
```

**GenerateState** (`slime/rollout/sglang_rollout.py:83`)：
```python
# slime/rollout/sglang_rollout.py:83-148
class GenerateState(metaclass=SingletonMeta):
    """The global state for the generation process."""
    def __init__(self, args: Namespace) -> None:
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.semaphore = asyncio.Semaphore(args.sglang_server_concurrency * get_rollout_num_engines(args))
        self.sampling_params = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            max_new_tokens=args.rollout_max_response_len,
            ...
        )
        self.dp_counts = [0] * (args.sglang_dp_size or 1)  # DP rank 负载均衡
```

核心 rollout 函数：
- `generate_rollout` (`slime/rollout/sglang_rollout.py:627-649`) — 入口函数
- `generate_rollout_async` (`:374-470`) — 异步生成主循环
- `generate_and_rm` (`:224-289`) — 单样本生成 + 奖励
- `generate_and_rm_group` (`:292-336`) — 组生成（GRPO group sampling）

```python
# slime/rollout/sglang_rollout.py:374-470 — 异步 rollout 主循环
async def generate_rollout_async(args, rollout_id, data_source):
    state = GenerateState(args)
    target_data_size = args.rollout_batch_size
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)
        done, state.pendings = await asyncio.wait(state.pendings, ...)
        for task in done:
            group = task.result()
            # dynamic filter 逻辑
            if should_drop_dynamic_filter_output(...):
                continue
            data.append(group)
    aborted_samples = await abort(args, rollout_id)  # 中止未完成请求
    return RolloutFnTrainOutput(samples=data, metrics=...), aborted_samples
```

**调用链 3：Rollout 生成**
```
RolloutManager.generate(rollout_id)
  → _get_rollout_data(rollout_id)
    → call_rollout_function(generate_rollout, ...)
      → generate_rollout_async(args, rollout_id, data_source.get_samples)
        → state.submit_generate_tasks(samples)
          → asyncio.create_task(generate_and_rm_group(args, group, ...))
            → generate_and_rm(args, sample, ...)
              → generate(args, sample, sampling_params)  # SGLang HTTP API
              → async_rm(args, sample)                   # 奖励模型评分
    → postprocess_rollout_data(args, data, ...)
  → convert_samples_to_train_data(args, data, ...)
  → split_train_data_by_dp(args, data, ...)
```

---

## 3. 奖励计算与 Advantage 估计

### miles: loss_hub/advantages.py

miles 的 advantage 估计器位于 `miles/backends/training_utils/loss_hub/advantages.py:16`：

```python
# miles/backends/training_utils/loss_hub/advantages.py:16-105
def compute_advantages(args, kl, rewards, log_probs, loss_masks, 
                       total_lengths, response_lengths, values=None, ...):
    if args.advantage_estimator in ["grpo", "gspo"]:
        rewards = torch.tensor(rewards, dtype=torch.float32, device=kl[0].device)
        returns = get_grpo_returns(rewards, kl)     # 组内广播标量奖励
        advantages = [r for r in returns]
    elif args.advantage_estimator == "ppo":
        advantages, returns = get_advantages_and_returns_batch(...)  # GAE
    elif args.advantage_estimator == "reinforce_plus_plus":
        returns = get_reinforce_plus_plus_returns(...)
    elif args.advantage_estimator == "reinforce_plus_plus_baseline":
        advantages = get_reinforce_plus_plus_baseline_advantages(...)
```

支持的 advantage 估计器：
- **GRPO** (`math_utils.py:453-460`)：`get_grpo_returns` — 将标量奖励广播到每个 token
- **GSPO**：序列级 KL + 组内归一化
- **PPO/GAE** (`math_utils.py:615-782`)：`get_advantages_and_returns_batch` — 支持 CP 的 chunked GAE
- **REINFORCE++** (`math_utils.py:463-524`)：带 KL 惩罚的折扣回报
- **REINFORCE++-baseline** (`math_utils.py:527-554`)：组基线 + KL 惩罚

Advantage 归一化 (`advantages.py:108-182`)：
```python
def normalize_advantages(args, advantages, loss_masks, ...):
    # CP > 1 时先切片 mask 到本地 rank
    # distributed_masked_whiten 在 DP 组内做 masked 白化
    whitened_advs_flat = distributed_masked_whiten(all_advs, all_masks, 
                                                     process_group=dp_group, shift_mean=True)
```

### slime: rollout.py:279 _post_process_rewards, Dr.GRPO

slime 的奖励后处理位于 `slime/ray/rollout.py:279`：

```python
# slime/ray/rollout.py:279-304
def _post_process_rewards(self, samples):
    raw_rewards = [sample.get_reward_value(self.args) for sample in samples]
    if (self.args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and self.args.rewards_normalization):
        # group norm: 按 prompt 分组
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        rewards = rewards.reshape(-1, self.args.n_samples_per_prompt)
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean  # 减组均值
        
        if self.args.advantage_estimator in ["grpo", "gspo", "cispo"] and self.args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)  # Dr.GRPO: 除以标准差
        return raw_rewards, rewards.flatten().tolist()
    return raw_rewards, raw_rewards
```

slime 的 advantage 计算在 `slime/backends/megatron_utils/loss.py:704` 中：
- `compute_advantages_and_returns` (`loss.py:704-880`) — 统一入口
- `get_grpo_returns` — GRPO 回报
- `get_advantages_and_returns_batch` — GAE（支持 chunked scan，`ppo_utils.py:700-713`）
- `normalize_advantages` — advantage 白化（`loss.py:818-877`）

**关键差异**：slime 的 Dr.GRPO 在 group normalization 后额外除以标准差（`grpo_std_normalization`），这是 DeepSeek-R1 论文中的做法，使不同 prompt 的 advantage 尺度一致。

---

## 4. GRPO/PPO 策略损失

### miles: loss_hub/losses.py:62 policy_loss_function, math_utils.py:254 compute_policy_loss

miles 的策略损失位于 `miles/backends/training_utils/loss_hub/losses.py:62`：

```python
# miles/backends/training_utils/loss_hub/losses.py:62-396
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    advantages = torch.cat([adv.detach() for adv in batch["advantages"]], dim=0)
    # 1. 计算当前 log-probs 和 entropy
    log_probs_and_entropy = get_log_probs_and_entropy(logits, args=args, ...)
    log_probs = log_probs_and_entropy["log_probs"]
    
    # 2. 计算 KL 散度
    ppo_kl = old_log_probs - log_probs  # per-token KL
    
    # 3. 计算 PPO clipped loss
    pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, 
                                                args.eps_clip, args.eps_clip_high, ...)
    
    # 4. 可选：TIS 校正
    if args.get_mismatch_metrics or args.use_tis:
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
    
    # 5. 可选：KL loss
    if args.use_kl_loss:
        kl = compute_approx_kl(log_probs, ref_log_probs, kl_loss_type=args.kl_loss_type, ...)
        loss = loss + args.kl_loss_coef * kl_loss
    
    # 6. Entropy bonus
    if args.entropy_coef != 0:
        loss = pg_loss - args.entropy_coef * entropy_loss
```

**真实源码** (`losses.py:62-396`) — 完整 `policy_loss_function` 实现：

```python
# miles/backends/training_utils/loss_hub/losses.py:92-126
parallel_state = get_parallel_state()
advantages_list = [advantage.detach() for advantage in batch["advantages"]]
advantages = torch.cat(advantages_list, dim=0)
rollout_old_log_probs = (
    [log_prob.detach() for log_prob in batch["rollout_log_probs"]]
    if batch.get("rollout_log_probs") is not None else None
)
reference_log_probs = (
    [log_prob.detach() for log_prob in batch["ref_log_probs"]]
    if batch.get("ref_log_probs") is not None else None
)
response_lengths = batch["response_lengths"]
total_lengths = batch["total_lengths"]
calculate_entropy = args.entropy_coef != 0 or args.observe_training_entropy

log_probs_and_entropy = get_log_probs_and_entropy(
    logits, args=args, unconcat_tokens=batch["unconcat_tokens"],
    total_lengths=total_lengths, response_lengths=response_lengths,
    with_entropy=calculate_entropy, entropy_requires_grad=args.entropy_coef != 0,
    max_seq_lens=max_seq_lens,
)
log_probs = log_probs_and_entropy["log_probs"]
```

```python
# miles/backends/training_utils/loss_hub/losses.py:207-227
pg_loss, pg_clipfrac = compute_policy_loss(
    ppo_kl, advantages, args.eps_clip, args.eps_clip_high,
    getattr(args, "eps_clip_c", None)
)
if args.use_opsm:
    pg_loss = pg_loss * opsm_mask

# Apply off-policy correction using importance sampling if enabled
if args.get_mismatch_metrics or args.use_tis:
    sum_of_sample_mean_for_mismatch_metrics = sum_of_sample_mean
    if args.custom_tis_function_path is not None:
        tis_func = load_function(args.custom_tis_function_path)
    else:
        tis_func = vanilla_tis_function
    ois = (-ppo_kl).exp()
    pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
    sum_of_sample_mean = get_sum_of_sample_mean(
        total_lengths, response_lengths, modified_response_masks, ...,
        denominators=batch.get("rollout_mask_sums", None),
    )
```

```python
# miles/backends/training_utils/loss_hub/losses.py:303-334
entropy_loss = pg_loss.new_zeros(())
loss = pg_loss
if calculate_entropy:
    entropy = log_probs_and_entropy["entropy"]
    entropy = torch.cat(entropy, dim=0)
    entropy_loss = sum_of_sample_mean(entropy)
    if args.entropy_coef != 0:
        loss = pg_loss - args.entropy_coef * entropy_loss

if args.use_kl_loss:
    ref_log_probs = torch.cat(reference_log_probs, dim=0)
    kl = compute_approx_kl(log_probs, ref_log_probs,
                            kl_loss_type=args.kl_loss_type,
                            importance_ratio=importance_ratio)
    kl_loss = sum_of_sample_mean(kl)
    if args.kl_loss_coef != 0:
        loss = loss + args.kl_loss_coef * kl_loss
```

`compute_policy_loss` (`math_utils.py:254-277`)：
```python
@torch.compile(dynamic=True)
def compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high, eps_clip_c=None):
    ratio = _safe_exp_neg_ppo_kl(ppo_kl)           # exp(-ppo_kl) = π_new/π_old
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # PPO-clip
    if eps_clip_c is not None:  # dual-clip PPO
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    return pg_losses, clipfrac
```

**真实源码** (`math_utils.py:253-277`) — 完整 `compute_policy_loss` 实现：

```python
# miles/backends/training_utils/loss_hub/math_utils.py:253-277
@torch.compile(dynamic=True)
def compute_policy_loss(
    ppo_kl: torch.Tensor,
    advantages: torch.Tensor,
    eps_clip: float,
    eps_clip_high: float,
    eps_clip_c: float | None = None,
):
    ratio = _safe_exp_neg_ppo_kl(ppo_kl)
    pg_losses1 = -ratio * advantages
    pg_losses2 = -ratio.clamp(1 - eps_clip, 1 + eps_clip_high) * advantages
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    clipfrac = torch.gt(pg_losses2, pg_losses1).float()

    if eps_clip_c is not None:
        assert eps_clip_c > 1.0, (...)
        pg_losses3 = -eps_clip_c * advantages
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    else:
        pg_losses = clip_pg_losses1

    return pg_losses, clipfrac
```

### slime: loss.py:883 vanilla_tis_function, loss.py:933 policy loss

slime 的策略损失位于 `slime/backends/megatron_utils/loss.py:933`：

```python
# slime/backends/megatron_utils/loss.py:933-1172
def policy_loss_function(args, batch, logits, sum_of_sample_mean):
    advantages = torch.cat(batch["advantages"], dim=0)
    old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch.get("log_probs")
    
    _, log_probs_and_entropy = get_log_probs_and_entropy(logits, args=args, ...)
    log_probs = log_probs_and_entropy["log_probs"]
    
    # KL 散度
    ppo_kl = old_log_probs - log_probs
    
    # PPO/GSPO/CISPO loss
    if args.advantage_estimator == "cispo":
        pg_loss, pg_clipfrac = compute_cispo_loss(ppo_kl, log_probs, advantages, ...)
    else:
        pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, ...)
    
    # TIS 校正
    if args.get_mismatch_metrics or args.use_tis:
        pg_loss, modified_response_masks, tis_metrics = tis_func(**tis_kwargs)
        sum_of_sample_mean = get_sum_of_sample_mean(...)  # 重建 reducer
```

**vanilla_tis_function** (`slime/backends/megatron_utils/loss.py:883`)：
```python
def vanilla_tis_function(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kwargs):
    rollout_log_probs = torch.cat(rollout_log_probs, dim=0)
    old_log_probs = torch.cat(train_log_probs, dim=0)
    tis = torch.exp(old_log_probs - rollout_log_probs)  # IS 权重
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics
```

**icepop_function** (`slime/backends/megatron_utils/loss.py:907`)：
```python
def icepop_function(args, *, pg_loss, train_log_probs, rollout_log_probs, ...):
    ice_ratio = torch.exp(old_log_probs - rollout_log_probs)
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    pg_loss = pg_loss * ice_weight
    return pg_loss, loss_masks, metrics
```

**CISPO loss** (`slime/utils/ppo_utils.py:152`)：
```python
@torch.compile(dynamic=True)
def compute_cispo_loss(ppo_kl, log_probs, advantages, eps_clip, eps_clip_high):
    # MiniMax-M1: -sg(clip(ratio, 1-eps, 1+eps_high)) * advantages * log_probs
    ratio = (-ppo_kl).exp()
    ratio_truncated = torch.clamp(ratio, min=1.0 - eps_clip, max=1.0 + eps_clip_high)
    pg_losses = -ratio_truncated.detach() * advantages * log_probs
    clipfrac = (ratio_truncated != ratio).float()
    return pg_losses, clipfrac
```

---

## 5. KL 散度估计器

### miles: math_utils.py:139 compute_approx_kl (k1/k2/k3/low_var_kl)

miles 的 KL 估计器位于 `miles/backends/training_utils/loss_hub/math_utils.py:139`：

```python
# miles/backends/training_utils/loss_hub/math_utils.py:139-180
@torch.compile(dynamic=True)
def compute_approx_kl(log_probs, log_probs_base, kl_loss_type, importance_ratio=None):
    log_ratio = log_probs.float() - log_probs_base.float()
    
    if kl_loss_type == "k1":          # :157 — KL 的一阶近似
        kl = log_ratio
    elif kl_loss_type == "k2":        # :159 — KL 的二阶近似
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:  # :161 — Schulman 无偏低方差 KL
        log_ratio = -log_ratio
        if kl_loss_type == "low_var_kl":
            log_ratio = _safe_clamp_log_ratio(log_ratio)  # :167 — 数值稳定 clamp
        kl = log_ratio.exp() - 1 - log_ratio
    ...
    # IS ratio for unbiased KL (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)  # :178
    return kl
```

**真实源码** (`math_utils.py:138-180`) — 完整 `compute_approx_kl` 实现：

```python
# miles/backends/training_utils/loss_hub/math_utils.py:138-180
@torch.compile(dynamic=True)
def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_loss_type: str,
    importance_ratio: torch.Tensor | None = None,
) -> torch.Tensor:
    log_ratio = log_probs.float() - log_probs_base.float()

    if kl_loss_type == "k1":
        kl = log_ratio
    elif kl_loss_type == "k2":
        kl = log_ratio**2 / 2.0
    elif kl_loss_type in ["k3", "low_var_kl"]:
        log_ratio = -log_ratio
        if kl_loss_type == "low_var_kl":
            log_ratio = _safe_clamp_log_ratio(log_ratio)
        kl = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_loss_type: {kl_loss_type}")

    # Apply IS ratio for unbiased KL estimation (DeepSeek-V3.2)
    if importance_ratio is not None:
        kl = importance_ratio * kl

    # Clamp only for low_var_kl for numerical stability
    if kl_loss_type == "low_var_kl":
        kl = torch.clamp(kl, min=-10, max=10)

    return kl
```

四种 KL 估计器（来自 Schulman blog: http://joschu.net/blog/kl-approx.html）：
- **k1** (`math_utils.py:157`)：`kl = log(π/π_base)` — 一阶近似，有偏
- **k2** (`math_utils.py:159`)：`kl = log_ratio² / 2` — 二阶近似
- **k3** (`math_utils.py:161`)：`kl = exp(-log_ratio) - 1 - log_ratio` — 无偏低方差
- **low_var_kl** (`math_utils.py:166-167`)：k3 + log_ratio clamp 数值稳定版

### slime: ppo_utils.py:187 _VocabParallelLogProbEntropy

slime 的 KL 估计器位于 `slime/utils/ppo_utils.py`，通过 `_VocabParallelLogProbEntropy` 同时计算 log-prob 和 entropy：

```python
# slime/utils/ppo_utils.py:187-336
class _VocabParallelLogProbEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, log_prob_keep_mask, process_group, ...):
        # 1. 计算 target mask（跨 TP rank 定位 target token）
        target_mask = (target < vocab_start_index) | (target >= vocab_end_index)
        # 2. vocab_parallel_softmax：数值稳定的并行 softmax
        predicted_logits, log_prob_sum_exp_logits, log_prob_softmax, _ = vocab_parallel_softmax(...)
        # 3. log_prob = predicted_logit - log(sum_exp)
        predicted_logits = predicted_logits.masked_fill_(target_mask, 0.0).unsqueeze(-1)
        _maybe_all_reduce(predicted_logits, dist.ReduceOp.SUM, process_group)
        log_prob = predicted_logits - log_prob_sum_exp_logits.log()
        # 4. entropy = logits_max + log(sum_exp) - sum(softmax * logits)
        entropy = logits_logits_max + entropy_sum_exp_logits.log() - sum_softmax_times_logits
        return log_prob, entropy
```

slime 的 `compute_approx_kl` 同样支持 k1/k2/k3/low_var_kl（`ppo_utils.py:100-140`），逻辑与 miles 一致。

**关键差异**：slime 的 `_VocabParallelLogProbEntropy` 将 log-prob 和 entropy 融合在一个 autograd.Function 中，forward 时同时计算两者，backward 时复用 softmax 结果（`ppo_utils.py:298-336`），减少显存占用。

---

## 6. Log-probability 与 Entropy 计算

### miles: logit_processors.py:184 get_log_probs_and_entropy

miles 的 log-prob 计算位于 `miles/backends/training_utils/loss_hub/logit_processors.py:184`：

```python
# miles/backends/training_utils/loss_hub/logit_processors.py:184-270
def get_log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, 
                               response_lengths, with_entropy=True, ...):
    # 1. 逐样本提取 response 对齐的 logits/tokens
    response_chunks = _iter_response_chunks(logits, args=args, ...)
    for sample_index, (logits_chunk, tokens_chunk, response_indices) in enumerate(response_chunks):
        # 2. 计算 sampling mask（rollout top-p replay）
        sampling_mask = build_local_sampling_mask(...) if rollout_sampling_mask else None
        # 3. 调用 calculate_log_probs_and_entropy
        log_prob, entropy = calculate_log_probs_and_entropy(logits_chunk, tokens_chunk, ...)
```

底层使用 Megatron 的 `fused_vocab_parallel_cross_entropy`（`math_utils.py:296-301`）：
```python
from megatron.core.fusions.fused_cross_entropy import fused_vocab_parallel_cross_entropy
return -fused_vocab_parallel_cross_entropy(logits, tokens, process_group)
```

Entropy 通过 `_VocabParallelEntropy`（`math_utils.py:414-446`）计算：
```python
class _VocabParallelEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, process_group):
        logits_max = vocab_parallel_logits.max(dim=-1, keepdim=True).values
        dist.all_reduce(logits_max, op=dist.ReduceOp.MAX, group=process_group)
        normalized_logits = vocab_parallel_logits - logits_max
        softmax_logits = normalized_exp_logits.div_(normalized_sum_exp_logits)
        entropy = logits_max + normalized_sum_exp_logits.log() - sum_softmax_times_logits
        return entropy.squeeze(dim=-1)
```

### slime: loss.py:513 get_log_probs_and_entropy

slime 的 log-prob 计算位于 `slime/backends/megatron_utils/loss.py:513`：

```python
# slime/backends/megatron_utils/loss.py:513-604
def get_log_probs_and_entropy(logits, *, args, unconcat_tokens, total_lengths, 
                               response_lengths, with_entropy=True, ...):
    logits = logits.squeeze(0)
    # 1. 温度缩放（匹配 rollout-time log-probs）
    if rollout_temperature != 1.0:
        logits = logits / rollout_temperature
    # 2. 构建 shifted target tokens
    full_tokens = _build_shifted_tokens(T, device, unconcat_tokens, ...)
    # 3. 构建 top-p nucleus keep-mask
    top_p_keep_mask = _build_topp_keep_mask(...) if top_p_token_ids else None
    # 4. 一次性计算 [T, V] 上的 log-prob 和 entropy
    log_prob_full, entropy_full = calculate_log_probs_and_entropy(logits, full_tokens, ...)
    # 5. 提取 per-sample response 部分
    log_probs_list, entropy_list = _extract_per_sample(log_prob_full, entropy_full, ...)
```

**关键差异**：slime 在 `[T, V]` 完整张量上一次性计算（`loss.py:570-579`），然后提取 per-sample 部分（`_extract_per_sample`，`loss.py:432-510`），使 backward 只遍历 `[T, V]` 一次。miles 则是逐样本计算。

---

## 7. 权重同步（Training → Inference）

### miles: update_weight_utils.py:54 UpdateWeight, broadcast/p2p/rdt/delta

miles 的权重同步位于 `miles/backends/fsdp_utils/update_weight_utils.py:54`：

```python
# miles/backends/fsdp_utils/update_weight_utils.py:54-119
class UpdateWeight(abc.ABC):
    def __init__(self, args, model):
        self.weight_version = 0
    
    def update_weights(self):
        self.weight_version += 1
        # 1. 暂停 rollout 引擎生成
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
        # 2. 分桶传输权重（按 update_weight_buffer_size 分桶）
        for raw_name, raw_param in self.model.state_dict().items():
            for name, param in _iter_sync_named_params(raw_name, raw_param, ...):
                bucket.append((name, param, target_dtype))
                if bucket_size + param_size >= self.args.update_weight_buffer_size:
                    self.wait_and_update_bucket_weights(bucket)
        # 3. 恢复生成
        ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
```

**真实源码** (`update_weight_utils.py:54-123`) — 完整 `UpdateWeight` 实现：

```python
# miles/backends/fsdp_utils/update_weight_utils.py:54-123
class UpdateWeight(abc.ABC):
    def __init__(self, args: Namespace, model: torch.nn.Module) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0

    def update_weights(self) -> None:
        self.weight_version += 1
        if dist.get_rank() == 0:
            futures = [engine.pause_generation.remote() for engine in self.rollout_engines]
            futures.extend([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get(futures)
            ray.get([engine.begin_weight_update.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        bucket = []
        bucket_size = 0
        model_type = getattr(getattr(self.model, "config", None), "model_type", "")
        sync_dtypes = getattr(self.model, "_fsdp_sync_dtypes", None)
        for raw_name, raw_param in self.model.state_dict().items():
            for name, param in _iter_sync_named_params(raw_name, raw_param, model_type, self.model, sync_dtypes):
                param_size = param.numel() * param.element_size()
                if bucket and bucket_size + param_size >= self.args.update_weight_buffer_size:
                    self.wait_and_update_bucket_weights(bucket)
                    bucket = []
                    bucket_size = 0
                target_dtype = sync_dtypes.get(name) if sync_dtypes is not None else None
                param = gather_full_param(param, async_op=True)
                bucket.append((name, param, target_dtype))
                bucket_size += param_size

        if bucket:
            self.wait_and_update_bucket_weights(bucket)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.end_weight_update.remote() for engine in self.rollout_engines])
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def wait_and_update_bucket_weights(self, bucket):
        resolved = []
        for name, param, target_dtype in bucket:
            if hasattr(param, "wait"):
                param = param.wait()
            if target_dtype is not None and param.dtype != target_dtype:
                param = param.to(target_dtype)
            resolved.append((name, param))
        self.update_bucket_weights(resolved, weight_version=self.weight_version)
```

miles 支持多种同步策略（`miles/backends/megatron_utils/update_weight/`）：
- **broadcast**：从 rank 0 广播到所有 SGLang 引擎
- **p2p**：点对点传输
- **rdt**：RDMA 传输
- **delta**：增量同步

### slime: update_weight_from_distributed.py:24

slime 的权重同步位于 `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:24`：

```python
# slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:24-100
class UpdateWeightFromDistributed:
    """Update distributed engines through a device process group. Each PP rank: group "slime-pp_{pp_rank}",
    only DP=TP=0 transfers. Non-expert (TP) and expert (EP) params separate."""
    
    def connect_rollout_engines(self, rollout_engines, rollout_engine_lock, ...):
        # 1. 确定 PP source rank（DP=TP=0）
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 
            and mpu.get_tensor_model_parallel_rank() == 0
        )
        # 2. 创建 process group "slime-pp_{pp_rank}"
        if self._is_pp_src_rank:
            self._group_name = f"slime-pp_{pp_rank}"
            self._model_update_groups = connect_rollout_engines_from_distributed(...)
```

**关键差异**：slime 按 PP rank 分组（`slime-pp_{pp_rank}`），每个 PP rank 的 DP=TP=0 rank 负责传输；miles 统一由 rank 0 广播。slime 区分 expert/non-expert 参数分别传输。

---

## 8. Context Parallel in RL

### miles: cp_utils.py

miles 的 CP 工具位于 `miles/backends/training_utils/cp_utils.py`：

```python
# miles/backends/training_utils/cp_utils.py:19-67
def get_logits_and_tokens_offset_with_cp(total_length, response_length, qkv_format="thd", ...):
    # zigzag CP: 每个 rank 持有两个不连续的 chunk
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)
    # logits 需要 -1 offset（预测下一个 token）
    logits_0 = (max(chunk_0[0], prompt_length - 1), min(chunk_0[1], total_length - 1))
    logits_1 = (max(chunk_1[0], prompt_length - 1), min(chunk_1[1], total_length - 1))
    return chunk_size, (chunk_0, chunk_1), (logits_0, logits_1), (token_0, token_1)
```

miles 支持两种 CP 模式：
- **zigzag CP**（默认）：`cp_utils.py:19-67` — 每个 rank 持有两个不连续 chunk
- **allgather CP**：全局 concat 后连续 split

### slime: cp_utils.py, zigzag CP loss.py:273

slime 的 CP 工具位于 `slime/backends/megatron_utils/cp_utils.py:9`：

```python
# slime/backends/megatron_utils/cp_utils.py:9-44
def get_logits_and_tokens_offset_with_cp(total_length, response_length):
    # 与 miles 相同的 zigzag 逻辑
    chunk_0 = (cp_rank * chunk_size, (cp_rank + 1) * chunk_size)
    chunk_1 = ((2 * cp_size - cp_rank - 1) * chunk_size, (2 * cp_size - cp_rank) * chunk_size)
    ...
```

slime 的 zigzag CP 在 loss 计算中的特殊处理（`loss.py:273-323`）：
```python
def _build_shifted_tokens(T, device, unconcat_tokens, total_lengths, response_lengths, allgather_cp):
    if cp_size > 1 and not allgather_cp:
        # zigzag CP: 完全不同的 layout
        for tokens, total_length, response_length in zip(...):
            chunk_size_cp, chunks_offset, logits_offset, tokens_offset = get_logits_and_tokens_offset_with_cp(...)
            for half, base in ((0, end), (1, end + chunk_size_cp)):
                full_tokens[base + lo : base + hi] = tokens[tokens_offset[half][0] : tokens_offset[half][1]]
```

slime 还支持 allgather CP 到 zigzag ring-attn 的转换（`loss.py:194-270`）：
```python
def _allgather_cp_redistribute(res, *, logits_local_len, total_lengths, response_lengths):
    # 将 allgather CP 布局（全局连续 chunk）转换为 zigzag ring-attn 布局
    # 使用 differentiable all-reduce 重建完整 response 张量
    all_cat = dist.nn.all_reduce(all_cat, group=cp_group)
    new_values = [slice_log_prob_with_cp(full_resp, total_length, response_length) for ...]
```

---

## 9. Off-policy 校正（TIS/OPSM/OPD）— slime 独有

### slime: loss.py:883-930, arguments.py:1060-1077

slime 支持三种 off-policy 校正技术：

**1. TIS (Truncated Importance Sampling)** — `loss.py:883-930`：
```python
# slime/backends/megatron_utils/loss.py:883-904
def vanilla_tis_function(args, *, pg_loss, train_log_probs, rollout_log_probs, loss_masks, **kwargs):
    tis = torch.exp(old_log_probs - rollout_log_probs)  # IS 权重 = π_train / π_rollout
    tis_weights = torch.clamp(tis, min=args.tis_clip_low, max=args.tis_clip)
    pg_loss = pg_loss * tis_weights
    return pg_loss, loss_masks, metrics

# slime/backends/megatron_utils/loss.py:907-930
def icepop_function(args, *, pg_loss, train_log_probs, rollout_log_probs, ...):
    # clip-or-pop: 越界 token 直接置零
    ice_weight = torch.where(
        (ice_ratio >= args.tis_clip_low) & (ice_ratio <= args.tis_clip), ice_ratio, torch.zeros_like(ice_ratio)
    )
    pg_loss = pg_loss * ice_weight
```

配置参数（`slime/utils/arguments.py:1060-1077`）：
```python
parser.add_argument("--use-tis", action="store_true", default=False)
parser.add_argument("--tis-clip", type=float, default=2.0)       # IS ratio 上界
parser.add_argument("--tis-clip-low", type=float, default=0)     # IS ratio 下界
parser.add_argument("--custom-tis-function-path", type=str, default=None)
```

**2. OPSM (Off-Policy Sequence Masking)** — `slime/utils/ppo_utils.py:183-221`：
```python
def compute_opsm_mask(args, full_log_probs, full_old_log_probs, advantages, loss_masks):
    # 计算 sequence-level KL
    seq_kl = ((full_old_log_prob - full_log_prob) * loss_mask).sum() / loss_mask.sum()
    # 当 advantage < 0 且 seq_kl > delta 时 mask 掉该序列
    mask = ((advantage < 0) & (seq_kl > args.opsm_delta)).float()
    opsm_mask_list.append(1 - mask)
```

**3. OPD (On-Policy Distillation)** — `slime/backends/megatron_utils/loss.py:663-701`：
```python
def apply_opd_kl_to_advantages(args, rollout_data, advantages, student_log_probs):
    # reverse KL: student_logp - teacher_logp
    reverse_kl = student_log_probs[i] - teacher_log_probs[i]
    advantages[i] = adv - args.opd_kl_coef * reverse_kl  # 就地修改 advantages
    rollout_data["opd_reverse_kl"] = reverse_kls  # 存储用于日志
```

**调用链 4：Off-policy 校正**
```
policy_loss_function()
  → compute_policy_loss(ppo_kl, advantages, ...)  # 基础 PPO loss
  → if use_tis:
      tis_func(vanilla_tis / icepop)               # IS 权重校正
      get_sum_of_sample_mean(modified_masks)       # 重建 reducer
  → if use_opsm:
      compute_opsm_mask(...)                        # 序列级 masking
      pg_loss = pg_loss * opsm_mask
  → if use_opd:
      apply_opd_kl_to_advantages(...)               # advantage 级 KL 惩罚
  → sum_of_sample_mean(pg_loss)                    # 最终 reduction
```

---

## 10. 容错与健康监控

### miles: utils/ft_utils/, RolloutHealthMonitor

miles 的容错系统由多个组件组成：

**RolloutHealthMonitor** (`miles/utils/health_monitor.py:11`)：
```python
# miles/utils/health_monitor.py:11-60
class RolloutHealthMonitor:
    def __init__(self, server_group, args):
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
    
    def start(self):
        self._thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self._thread.start()
    
    def _health_monitor_loop(self):
        # 周期性检查引擎健康状态
        # 发现故障时触发 recover_updatable_engines()
```

**Control Server** (`miles/utils/ft_utils/control_server/server.py`)：
- `start_control_server(actor_model, rollout_manager, port, ft_components)` — 启动容错控制服务
- 支持 `rollout` 和 `train` 组件的故障检测与恢复

**Mini FT Controller** (`miles/utils/ft_utils/mini_ft_controller.py`)：
- `maybe_start_mini_ft_controller(args)` — 启动轻量级容错控制器

**Fault Tolerance 配置**：
- `args.use_fault_tolerance` — 启用容错
- `args.ft_components` — 容错组件列表（"rollout", "train"）
- `args.rollout_health_check_interval` — 健康检查间隔
- `args.rollout_health_check_timeout` — 健康检查超时

### slime: utils/health_monitor.py

slime 的 RolloutHealthMonitor 位于 `slime/utils/health_monitor.py:9`：

```python
# slime/utils/health_monitor.py:9-80
class RolloutHealthMonitor:
    def __init__(self, server_group, args):
        self._check_interval = args.rollout_health_check_interval
        self._check_timeout = args.rollout_health_check_timeout
        self._check_first_wait = args.rollout_health_check_first_wait
        self._need_first_wait = True  # 每次 resume 后需要等待
    
    def start(self):
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # 初始为暂停状态
        self._thread = threading.Thread(target=self._health_monitor_loop, daemon=True)
        self._thread.start()
    
    def pause(self):
        # offload 时暂停健康检查
        self._pause_event.set()
    
    def resume(self):
        # onload 时恢复健康检查
        self._pause_event.clear()
```

**关键差异**：
- miles 有独立的 Control Server 和 Mini FT Controller，支持更复杂的容错场景
- slime 的容错相对简单，仅依赖 RolloutHealthMonitor 线程
- 两者都支持 pause/resume 机制，在 offload/onload 时暂停检查

---

## 11. 关键配置参数表

| 参数名 | 仓库 | 类型 | 默认值 | 说明 |
|--------|------|------|--------|------|
| `advantage_estimator` | both | str | "grpo" | 优势估计器：grpo/gspo/cispo/ppo/reinforce_plus_plus/reinforce_plus_plus_baseline |
| `kl_loss_type` | both | str | "low_var_kl" | KL 散度类型：k1/k2/k3/low_var_kl |
| `kl_loss_coef` | both | float | 0.0 | KL loss 系数（用于 use_kl_loss） |
| `kl_coef` | both | float | 0.0 | KL 惩罚系数（嵌入 reward） |
| `use_kl_loss` | both | bool | False | 是否添加 KL loss 项 |
| `entropy_coef` | both | float | 0.0 | Entropy bonus 系数 |
| `eps_clip` | both | float | 0.2 | PPO clip 下界 (1-eps) |
| `eps_clip_high` | both | float | None | PPO clip 上界 (1+eps)，默认等于 eps_clip |
| `eps_clip_c` | both | float | None | Dual-clip PPO 下界 |
| `use_tis` | both | bool | False | 启用 TIS off-policy 校正 |
| `tis_clip` | both | float | 2.0 | IS ratio 上界 |
| `tis_clip_low` | both | float | 0 | IS ratio 下界 |
| `custom_tis_function_path` | both | str | None | 自定义 TIS 函数路径 |
| `use_rollout_logprobs` | both | bool | False | 使用 rollout log-probs 计算 IS ratio |
| `use_opsm` | both | bool | False | 启用 OPSM 序列级 masking |
| `opsm_delta` | both | float | 0.0 | OPSM KL 阈值 |
| `use_opd` | slime | bool | False | 启用 On-Policy Distillation |
| `opd_kl_coef` | slime | float | 0.0 | OPD reverse KL 系数 |
| `normalize_advantages` | both | bool | False | 是否白化 advantages |
| `rewards_normalization` | slime | bool | True | 是否做 group reward 归一化 |
| `grpo_std_normalization` | slime | bool | False | Dr.GRPO: 除以组标准差 |
| `n_samples_per_prompt` | both | int | 4 | 每个 prompt 采样数 |
| `update_weights_interval` | both | int | 1 | 权重同步间隔（步数） |
| `ref_update_interval` | miles | int | None | 参考模型更新间隔 |
| `offload_rollout` | both | bool | False | 是否 offload rollout 引擎 |
| `offload_train` | both | bool | False | 是否 offload 训练模型 |
| `use_critic` | both | bool | False | 是否使用 Critic（PPO 模式） |
| `num_critic_only_steps` | both | int | 0 | 前 N 步仅训练 Critic |
| `gamma` | both | float | 1.0 | GAE 折扣因子 |
| `lambd` | both | float | 1.0 | GAE lambda |
| `value_clip` | both | float | 0.2 | Value clip 范围 |
| `rollout_temperature` | both | float | 1.0 | 采样温度 |
| `rollout_top_p` | both | float | 1.0 | nucleus sampling top-p |
| `rollout_max_response_len` | both | int | 1024 | 最大响应长度 |
| `log_probs_chunk_size` | slime | int | -1 | log-prob 分块计算大小 |
| `use_unbiased_kl` | both | bool | False | 使用 IS 校正的无偏 KL |
| `calculate_per_token_loss` | both | bool | False | 按 token 计算 loss |
| `custom_advantage_function_path` | slime | str | None | 自定义 advantage 函数 |
| `custom_loss_function_path` | both | str | None | 自定义 loss 函数 |
| `use_routing_replay` | slime | bool | False | MoE routing replay |
| `allgather_cp` | slime | bool | False | allgather CP 模式 |

---

## 12. miles vs slime 架构对比表

| 对比维度 | miles | slime |
|----------|-------|-------|
| **训练后端** | FSDP2 + Megatron 双后端（`fsdp_utils/actor.py:54`, `megatron_utils/actor.py:88`） | 仅 Megatron（`megatron_utils/actor.py:56`） |
| **异步模式** | asyncio + `await`（`train_async.py:22`） | Ray `ray.get()`（`train_async.py:10`） |
| **对象存储** | 内置 ObjectStore（`utils/object_store.py:33`，支持 Ray/Mooncake） | 直接使用 Ray Object Store |
| **数据源** | `RolloutDataSource` + `RolloutDataSourceWithBuffer`（`data_source.py:50,168`） | 同结构（`data_source.py:50,168`） |
| **Advantage 计算** | 独立 `advantages.py:16` 模块，在 actor 外部计算 | 集成在 `loss.py:704 compute_advantages_and_returns` 中 |
| **Reward 归一化** | 在 `advantages.py:108` 中白化 advantage | 在 `rollout.py:279 _post_process_rewards` 中 group norm + Dr.GRPO |
| **KL 估计器** | `math_utils.py:139 compute_approx_kl`，支持 k1/k2/k3/low_var_kl + IS 校正 | `ppo_utils.py` 同逻辑，封装在 `_VocabParallelLogProbEntropy` |
| **Log-prob 计算** | 逐样本计算（`logit_processors.py:184`） | 全 `[T,V]` 一次性计算（`loss.py:513`），backward 更高效 |
| **Off-policy 校正** | TIS（`corrections.py:7 vanilla_tis_function`） | TIS + OPSM + OPD 三种（`loss.py:883-930`, `ppo_utils.py:183`, `loss.py:663`） |
| **CISPO 支持** | 不支持 | 支持（`ppo_utils.py:152 compute_cispo_loss`） |
| **权重同步** | 分桶广播（`update_weight_utils.py:70`），支持 broadcast/p2p/rdt/delta | PP rank 分组（`update_weight_from_distributed.py:24`），区分 expert/non-expert |
| **Ref 模型** | CPU-offloaded FSDP2（`fsdp_utils/actor.py:645 _create_ref_model`） | 通过 `ref_update_interval` 周期性更新 |
| **CP 模式** | zigzag + allgather（`cp_utils.py:19`） | zigzag + allgather + 互转（`loss.py:194 _allgather_cp_redistribute`） |
| **容错** | Control Server + Mini FT Controller + HealthMonitor（`ft_utils/`） | 仅 HealthMonitor（`health_monitor.py:9`） |
| **GAE 实现** | chunked GAE（`math_utils.py:809 chunked_gae`），FlashLinearAttention 并行前缀扫描 | 同（`ppo_utils.py:700-713`） |
| **Entropy 计算** | `_VocabParallelEntropy`（`math_utils.py:414`） | `_VocabParallelLogProbEntropy`（`ppo_utils.py:187`），融合 log-prob + entropy |
| **MoE 支持** | WeightBridge 变换（`update_weight_utils.py:35-51`） | routing replay（`arguments.py:1092`） |
| **自定义扩展** | `custom_loss_function_path`, `custom_tis_function_path`, `custom_pg_loss_reducer_function_path` | 同 + `custom_advantage_function_path` |

---

## 13. RL 训练完整调用链总图

下图展示一个完整 RL 步骤中 rollout / reward / advantage / policy_update / weight_sync 五大阶段的调用关系：

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                        RL 训练完整调用链总图                                              │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                            train() 主循环                                        │    │
│  │  miles/train.py:22  │  slime/train.py:9                                         │    │
│  └───────────────────────────────┬─────────────────────────────────────────────────┘    │
│                                  │                                                       │
│          ┌───────────────────────┼───────────────────────┐                               │
│          ▼                       ▼                       ▼                               │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐                        │
│  │   Rollout     │      │    Reward     │      │  Advantage    │                        │
│  │               │      │               │      │               │                        │
│  │ rollout_      │────▶│ async_rm()    │────▶│ compute_      │                        │
│  │ manager.      │      │ or RM model   │      │ advantages()  │                        │
│  │ generate()    │      │               │      │               │                        │
│  │               │      │ slime:        │      │ miles:        │                        │
│  │ miles: :144   │      │ _post_process │      │ advantages.py │                        │
│  │ slime: :163   │      │ _rewards :279 │      │    :16        │                        │
│  │               │      │               │      │ slime:        │                        │
│  │               │      │               │      │ loss.py:704   │                        │
│  └───────────────┘      └───────────────┘      └───────┬───────┘                        │
│                                                         │                               │
│                                                         ▼                               │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          Policy Update                                          │    │
│  │                                                                                 │    │
│  │  actor_model.train()                                                            │    │
│  │    → forward() → logits                                                        │    │
│  │    → policy_loss_function()                                                     │    │
│  │        → get_log_probs_and_entropy()     [logits → log_probs + entropy]         │    │
│  │        → compute_policy_loss()           [ppo_kl + adv → clipped loss]          │    │
│  │        → [optional] tis_func()           [off-policy IS correction]             │    │
│  │        → [optional] compute_approx_kl()   [ref model KL penalty]                │    │
│  │        → sum_of_sample_mean()            [CP-aware reduction]                  │    │
│  │    → backward() → optimizer.step()                                              │    │
│  │                                                                                 │    │
│  │  miles: losses.py:62  │  slime: loss.py:933                                     │    │
│  └───────────────────────────────┬─────────────────────────────────────────────────┘    │
│                                  │                                                       │
│                                  ▼                                                       │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                          Weight Sync                                             │    │
│  │                                                                                 │    │
│  │  actor_model.update_weights()                                                    │    │
│  │    → UpdateWeight.update_weights()                                              │    │
│  │        → pause_generation() + flush_cache()    [暂停 rollout 引擎]               │    │
│  │        → for param in state_dict():            [分桶 gather + transmit]         │    │
│  │            gather_full_param(param, async_op=True)                               │    │
│  │            wait_and_update_bucket_weights()                                      │    │
│  │        → end_weight_update() + continue_generation()                             │    │
│  │                                                                                 │    │
│  │  miles: update_weight_utils.py:70  │  slime: update_weight_from_distributed.py:24│    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                  │                                                       │
│                                  ▼                                                       │
│                           [Next Rollout]                                                 │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

**调用链 5：Policy Update 内部详细调用链**

```
policy_loss_function()                              # losses.py:62
  → get_log_probs_and_entropy(logits, ...)          # logit_processors.py:184
      → _iter_response_chunks(logits, ...)          # 逐样本提取 response logits
      → calculate_log_probs_and_entropy(...)        # fused_vocab_parallel_cross_entropy
  → ppo_kl = old_log_probs - log_probs              # per-token KL
  → compute_policy_loss(ppo_kl, advantages, ...)    # math_utils.py:254
      → ratio = exp(-ppo_kl)                        # π_new/π_old
      → pg_losses1 = -ratio * advantages
      → pg_losses2 = -clamp(ratio) * advantages     # PPO clip
      → [dual-clip] pg_losses3 = -eps_clip_c * adv  # dual-clip PPO
  → [optional] tis_func(pg_loss, ...)               # corrections.py:7
      → tis = exp(old_logp - rollout_logp)          # IS 权重
      → pg_loss *= clamp(tis, low, high)
  → [optional] compute_approx_kl(logp, ref_logp, ...) # math_utils.py:139
  → loss = pg_loss - entropy_coef * entropy + kl_coef * kl_loss
  → sum_of_sample_mean(loss)                        # CP-aware reduction
```

**调用链 6：Weight Sync 内部详细调用链**

```
UpdateWeight.update_weights()                       # update_weight_utils.py:70
  → pause_generation.remote() + flush_cache.remote() # 暂停 SGLang 引擎
  → begin_weight_update.remote()
  → dist.barrier(group=get_gloo_group())            # 同步所有 rank
  → for name, param in state_dict():               # 遍历所有参数
      → _iter_sync_named_params(name, param, ...)  # WeightBridge 变换 (MoE)
      → gather_full_param(param, async_op=True)     # FSDP gather 完整参数
      → if bucket_full: wait_and_update_bucket_weights(bucket)
  → end_weight_update.remote() + continue_generation.remote()
  → dist.barrier(group=get_gloo_group())
```

---

## 附录：源码文件索引

按功能分类列出所有引用的文件路径和核心类/函数，便于快速定位。

### A. 训练入口与主循环

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/train.py` | `async def train` | :22 | 同步训练入口（miles） |
| `miles/train_async.py` | `async def train` | :22 | 异步训练入口（miles） |
| `slime/train.py` | `def train` | :9 | 同步训练入口（slime） |
| `slime/train_async.py` | `def train` | :10 | 异步训练入口（slime） |

### B. Rollout 生成

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/ray/rollout/rollout_manager.py` | `RolloutManager` | :54 | Rollout 管理器（miles） |
| `miles/ray/rollout/rollout_manager.py` | `generate` | :144 | 执行 rollout 并转换训练数据 |
| `miles/ray/rollout/rollout_manager.py` | `_get_rollout_data` | :243 | 调用 rollout 函数获取样本 |
| `miles/rollout/sglang_rollout.py` | `GenerateState` | :83 | 生成全局状态（miles） |
| `slime/ray/rollout.py` | `RolloutManager` | :38 | Rollout 管理器（slime） |
| `slime/ray/rollout.py` | `generate` | :163 | 执行 rollout 并转换训练数据 |
| `slime/ray/rollout.py` | `_post_process_rewards` | :279 | 奖励归一化 + Dr.GRPO |
| `slime/rollout/sglang_rollout.py` | `GenerateState` | :83 | 生成全局状态（slime） |
| `slime/rollout/sglang_rollout.py` | `generate_rollout_async` | :374 | 异步 rollout 主循环 |
| `slime/rollout/data_source.py` | `RolloutDataSource` | :50 | 数据源 |
| `slime/rollout/data_source.py` | `RolloutDataSourceWithBuffer` | :168 | 带缓冲的数据源 |

### C. 策略损失与优化

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/training_utils/loss_hub/losses.py` | `policy_loss_function` | :62 | 策略损失主函数（miles） |
| `miles/backends/training_utils/loss_hub/losses.py` | `value_loss_function` | :399 | 价值损失（PPO） |
| `miles/backends/training_utils/loss_hub/losses.py` | `sft_loss_function` | :457 | SFT 损失 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `compute_policy_loss` | :254 | PPO clipped loss |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `compute_approx_kl` | :139 | KL 散度估计器（k1/k2/k3/low_var_kl） |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `_safe_clamp_log_ratio` | :21 | log_ratio 数值稳定 clamp |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `_safe_exp_neg_ppo_kl` | :31 | ESS 权重安全计算 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `compute_ess_ratio_contribution` | :35 | Effective Sample Size 计算 |
| `slime/backends/megatron_utils/loss.py` | `policy_loss_function` | :933 | 策略损失主函数（slime） |
| `slime/backends/megatron_utils/loss.py` | `value_loss_function` | :1175 | 价值损失（PPO） |
| `slime/utils/ppo_utils.py` | `compute_policy_loss` | :120 | PPO clipped loss（slime） |
| `slime/utils/ppo_utils.py` | `compute_cispo_loss` | :152 | CISPO loss (MiniMax-M1) |
| `slime/utils/ppo_utils.py` | `compute_approx_kl` | :100 | KL 散度估计器（slime） |

### D. Advantage 估计

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/training_utils/loss_hub/advantages.py` | `compute_advantages` | :16 | Advantage 估计调度（miles） |
| `miles/backends/training_utils/loss_hub/advantages.py` | `normalize_advantages` | :108 | Advantage 白化 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `get_grpo_returns` | :453 | GRPO 回报计算 |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `get_advantages_and_returns_batch` | :615 | 批量 GAE |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `chunked_gae` | :809 | 并行前缀扫描 GAE |
| `slime/backends/megatron_utils/loss.py` | `compute_advantages_and_returns` | :704 | Advantage 统一入口（slime） |
| `slime/utils/ppo_utils.py` | `get_grpo_returns` | :130 | GRPO 回报计算（slime） |
| `slime/utils/ppo_utils.py` | `get_advantages_and_returns_batch` | :700 | 批量 GAE（slime） |

### E. Log-probability 与 Entropy 计算

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/training_utils/loss_hub/logit_processors.py` | `get_log_probs_and_entropy` | :184 | Log-prob 与 Entropy（miles） |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `_VocabParallelEntropy` | :414 | Entropy 计算（miles） |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `compute_log_probs` | :280 | 基础 log-prob 计算 |
| `slime/backends/megatron_utils/loss.py` | `get_log_probs_and_entropy` | :513 | Log-prob 与 Entropy（slime） |
| `slime/backends/megatron_utils/loss.py` | `get_values` | :607 | 值函数提取 |
| `slime/backends/megatron_utils/loss.py` | `_build_shifted_tokens` | :273 | zigzag CP token 构建 |
| `slime/utils/ppo_utils.py` | `_VocabParallelLogProbEntropy` | :187 | 融合 log-prob + entropy |

### F. Off-policy 校正

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/training_utils/loss_hub/corrections.py` | `vanilla_tis_function` | :7 | TIS off-policy 校正（miles） |
| `miles/backends/training_utils/loss_hub/corrections.py` | `icepop_function` | :35 | clip-or-pop 校正（miles） |
| `miles/backends/training_utils/loss_hub/math_utils.py` | `compute_opsm_mask` | :183 | OPSM 序列 masking（miles） |
| `slime/backends/megatron_utils/loss.py` | `vanilla_tis_function` | :883 | TIS off-policy 校正（slime） |
| `slime/backends/megatron_utils/loss.py` | `icepop_function` | :907 | clip-or-pop 校正（slime） |
| `slime/backends/megatron_utils/loss.py` | `apply_opd_kl_to_advantages` | :663 | OPD KL 惩罚 |
| `slime/utils/ppo_utils.py` | `compute_opsm_mask` | :183 | OPSM 序列 masking（slime） |
| `slime/utils/arguments.py` | `--use-tis` / `--tis-clip` | :1060-1077 | TIS 配置参数 |

### G. 权重同步

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/fsdp_utils/update_weight_utils.py` | `UpdateWeight` | :54 | 权重同步基类（miles） |
| `miles/backends/fsdp_utils/update_weight_utils.py` | `update_weights` | :70 | 分桶权重传输 |
| `miles/backends/fsdp_utils/update_weight_utils.py` | `wait_and_update_bucket_weights` | :111 | 桶等待与传输 |
| `miles/backends/fsdp_utils/update_weight_utils.py` | `_iter_sync_named_params` | :35 | WeightBridge 变换（MoE） |
| `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` | `UpdateWeightFromDistributed` | :24 | 权重同步（slime） |

### H. Context Parallel

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/training_utils/cp_utils.py` | `get_logits_and_tokens_offset_with_cp` | :19 | CP offset 计算（miles） |
| `slime/backends/megatron_utils/cp_utils.py` | `get_logits_and_tokens_offset_with_cp` | :9 | CP offset 计算（slime） |
| `slime/backends/megatron_utils/cp_utils.py` | `get_sum_of_sample_mean` | :47 | CP-aware reducer |
| `slime/backends/megatron_utils/loss.py` | `_allgather_cp_redistribute` | :194 | allgather→zigzag 转换 |

### I. 训练 Actor 与数据类

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/backends/fsdp_utils/actor.py` | `FSDPTrainRayActor` | :54 | FSDP2 训练 Actor |
| `miles/backends/fsdp_utils/actor.py` | `_create_ref_model` | :645 | CPU-offloaded 参考模型 |
| `miles/backends/megatron_utils/actor.py` | `MegatronTrainRayActor` | :88 | Megatron 训练 Actor |
| `slime/backends/megatron_utils/actor.py` | `MegatronTrainRayActor` | :56 | Megatron 训练 Actor（slime） |
| `miles/utils/types.py` | `Sample` | :26 | 样本数据类（miles） |
| `slime/utils/types.py` | `Sample` | :94 | 样本数据类（slime） |

### J. 容错与健康监控

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/utils/health_monitor.py` | `RolloutHealthMonitor` | :11 | 健康监控（miles） |
| `miles/utils/ft_utils/control_server/server.py` | `start_control_server` | — | 容错控制服务 |
| `miles/utils/ft_utils/mini_ft_controller.py` | `maybe_start_mini_ft_controller` | — | 轻量级容错控制器 |
| `slime/utils/health_monitor.py` | `RolloutHealthMonitor` | :9 | 健康监控（slime） |

### K. 基础设施

| 文件路径 | 核心类/函数 | 行号 | 说明 |
|----------|------------|------|------|
| `miles/utils/object_store.py` | `ObjectStoreBackend` | :33 | 对象存储后端枚举 |

---

## 附录 B：工作实战要点速查

| 场景 | 查哪里 | 关键代码 |
|------|--------|---------|
| 切换 GRPO/PPO 算法 | `advantages.py:16` `compute_advantages()` | 设置 `--advantage-estimator` |
| 调整 KL 估计器类型 | `compute_approx_kl()` | `math_utils.py:139` (miles) / `ppo_utils.py:100` (slime) |
| 开启 off-policy 校正 | `vanilla_tis_function()` | `corrections.py:7` / `loss.py:883` |
| 自定义奖励函数 | `rm_hub/__init__.py:43` `async_rm()` | miles Reward Hub |
| 权重同步瓶颈 | `UpdateWeight.update_weights()` | `update_weight_utils.py:70` |
| CP 场景 log-prob 计算 | `get_logits_and_tokens_offset_with_cp()` | `cp_utils.py:19` (miles) / `:9` (slime) |
| Rollout 健康监控 | `RolloutHealthMonitor` | `health_monitor.py:11` (miles) / `:9` (slime) |
| 异步训练模式 | `train_async.py` | miles: `:22` / slime: `:10` |
| 添加新的 advantage 算法 | `advantages.py:16` → 注册新分支 | 扩展 `compute_advantages()` |
| 调试 loss NaN | `policy_loss_function()` → `compute_policy_loss()` | `losses.py:62` / `math_utils.py:254` |

---

## 附录 C：常见坑与解决方案

| 问题现象 | 根因 | 解决方案 | 代码位置 |
|---------|------|---------|---------|
| GRPO loss 震荡 | KL 系数过大 / 学习率过高 | 降低 `--kl-coef` 或 `--lr` | `arguments.py` |
| Rollout 超时 | 推理引擎吞吐不足 | 调整 `rollout_batch_size` / 增加 TP | `rollout_manager.py` |
| 权重同步慢 | FSDP gather 未分桶 | 启用分桶传输 `update_weights()` | `update_weight_utils.py:70` |
| CP 场景 token 错位 | zigzag 重分布错误 | 检查 `_build_shifted_tokens()` | `loss.py:273` (slime) |
| Off-policy 校正后 loss 爆炸 | IS 权重过大 | 启用 TIS clamp `--tis-clip` | `arguments.py:1060` |
| Advantage 全零 | 组内 reward 完全相同 | 检查 reward 函数 / 增加采样多样性 | `advantages.py:16` |

> **交叉引用**：RL 中的 MoE 应用详见 `skill-knowledge/moe.md`；后训练 SFT/DPO 详见 `skill-knowledge/post-training.md`；推理引擎详见 `skill-knowledge/inference.md`。

---

> **文档统计**：本文档覆盖 RL 训练系统 11 个核心模块，包含 80+ 处 file:line 源码引用、6 条完整调用链、3 幅 ASCII 架构图、2 张跨仓库对比表、1 张配置参数表、3 份附录。