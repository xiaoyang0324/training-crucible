# 源码仓代码地图 (Source Repo Code Map)

> 本文档是全仓库的代码级地图，用于快速定位任何训练问题的代码位置。
> 所有行号均基于本地克隆的 `C:\y30062407\workspace\local\面试\train\` 目录下的实际源码。

---

## 1. Megatron-LM

### 1.1 定位与设计理念

Megatron-LM 是大规模预训练与后训练的**工业级参考实现**，由 NVIDIA 维护。核心价值在于：
- **全栈并行**：TP + PP + DP + EP + CP 五维并行原生支持
- **高性能优化**：DistributedOptimizer (ZeRO-1)、CUDA Graph、FP8/FP4、MoE
- **生产级特性**：dist checkpointing、recompute、telemetry、fault tolerance

**面试定位**：当问题涉及"大规模预训练如何做并行"、"DistributedOptimizer 原理"、"1F1B 调度"、"MoE 实现"时，**首选引用此仓**。

### 1.2 核心架构（ASCII 图）

```
pretrain_gpt.py::pretrain()
  │
  ├─ initialize_megatron()          ← parallel_state.initialize_model_parallel()
  │     ├─ TP group / PP group / DP group / EP group / CP group
  │     └─ initialize_process_groups()
  │
  ├─ setup_model_and_optimizer()
  │     ├─ model_provider()          ← GPTModel / TransformerBlock
  │     ├─ get_megatron_optimizer()  ← DistributedOptimizer (ZeRO-1)
  │     └─ optimizer_param_scheduler()
  │
  └─ train()
        └─ train_step()  (循环)
              │
              ├─ forward_backward_func()     ← get_forward_backward_func()
              │     ├─ forward_backward_no_pipelining()
              │     ├─ forward_backward_pipelining_without_interleaving()  (1F1B)
              │     ├─ forward_backward_pipelining_with_interleaving()    (Interleaved)
              │     └─ forward_backward_pipelining_with_hybrid_cp()
              │
              ├─ forward_step()
              │     └─ TransformerBlock.forward()
              │           └─ TransformerLayer.forward()
              │                 ├─ _forward_attention() → Attention.forward()
              │                 └─ _forward_mlp()       → MLP.forward()
              │
              ├─ backward_step()               ← DDP._make_backward_post_hook()
              │     └─ grad reduce-scatter / all-reduce
              │
              └─ optimizer.step()
                    └─ DistributedOptimizer.step_with_ready_grads()
                          └─ param all-gather (overlap with compute)
```

### 1.3 核心调用链

**训练入口链：**
```
pretrain_gpt.py:508  if __name__ == "__main__"
  → training.py:1500  pretrain()
    → training.py:4167  train()
      → training.py:3010  train_step()  (per-iteration)
        → schedules.py:53  get_forward_backward_func()
```

**前向传播链：**
```
train_step() → forward_backward_func() → forward_step_func()
  → transformer_block.py:267  TransformerBlock.forward()
    → transformer_layer.py:802  TransformerLayer.forward()
      → transformer_layer.py:812  _forward_attention()
        → attention.py:1279  Attention.forward()
          → dot_product_attention.py  DotProductAttention.forward()
      → transformer_layer.py:814  _forward_mlp()
        → mlp.py:257  MLP.forward()
```

**反向传播链：**
```
backward_step()
  → distributed_data_parallel.py:87  DistributedDataParallel
    → _make_backward_post_hook()  ← 注册为 autograd hook
      → param_and_grad_buffer.py  bucket grad reduce-scatter
        → finalize_model_grads.py  finalize_model_grads()
```

**优化器链：**
```
optimizer/__init__.py:1002  get_megatron_optimizer()
  → optimizer/distrib_optimizer.py:113  DistributedOptimizer
    → step_with_ready_grads()
      → param all-gather (per param group, overlap with next forward)
```

**MoE 链：**
```
moe/moe_layer.py:625  MoELayer.forward()
  → router.py  TopKRouter.route()       ← routing
  → token_dispatcher.py  dispatch()     ← all-to-all dispatch
  → experts.py  GroupedGemmExperts()    ← expert compute
  → token_dispatcher.py  combine()      ← all-to-all combine
```

### 1.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 训练入口 | `megatron/training/training.py:1500` | `pretrain()`, `train():4167`, `train_step():3010` |
| TP 层 | `megatron/core/tensor_parallel/layers.py:986` | `ColumnParallelLinear`, `RowParallelLinear:1382` |
| TP 通信 | `megatron/core/tensor_parallel/layers.py:634` | `LinearWithGradAccumulationAndAsyncCommunication` |
| PP 调度 | `megatron/core/pipeline_parallel/schedules.py:53` | `get_forward_backward_func()` |
| 1F1B | `megatron/core/pipeline_parallel/schedules.py:2147` | `forward_backward_pipelining_without_interleaving()` |
| Interleaved | `megatron/core/pipeline_parallel/schedules.py:1019` | `forward_backward_pipelining_with_interleaving()` |
| TransformerBlock | `megatron/core/transformer/transformer_block.py:267` | `TransformerBlock.forward()` |
| TransformerLayer | `megatron/core/transformer/transformer_layer.py:802` | `TransformerLayer.forward()` |
| Attention | `megatron/core/transformer/attention.py:1279` | `Attention.forward()` |
| MLP | `megatron/core/transformer/mlp.py:257` | `MLP.forward()` |
| Optimizer | `megatron/core/optimizer/__init__.py:1002` | `get_megatron_optimizer()` |
| DistribOptimizer | `megatron/core/optimizer/distrib_optimizer.py:113` | `DistributedOptimizer` |
| DDP | `megatron/core/distributed/distributed_data_parallel.py:87` | `DistributedDataParallel` |
| Recompute | `megatron/core/recompute.py:22` | `checkpointed_forward()` |
| MoE Layer | `megatron/core/transformer/moe/moe_layer.py:625` | `MoELayer.forward()` |
| FP8 Config | `megatron/core/transformer/transformer_config.py:588` | `fp8` field |
| FP8 Utils | `megatron/core/fp8_utils.py:122` | `is_float8tensor()` |
| Parallel Groups | `megatron/core/parallel_state.py:600` | `initialize_model_parallel()` |
| Grad Finalize | `megatron/core/distributed/finalize_model_grads.py` | `finalize_model_grads()` |
| P2P Comm | `megatron/core/pipeline_parallel/p2p_communication.py` | `recv_forward()`, `send_forward()` |
| CUDA Graph | `megatron/core/transformer/cuda_graphs.py` | CUDA Graph capture/replay |
| Config Container | `megatron/training/pretrain_gpt.py` | `PretrainConfigContainer` |
| Model Provider | `megatron/core/models/gpt/gpt_model.py` | `GPTModel` |

### 1.5 配置参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--tensor-model-parallel-size` | 1 | TP 维度 |
| `--pipeline-model-parallel-size` | 1 | PP 维度 |
| `--context-parallel-size` | 1 | CP 维度 |
| `--expert-model-parallel-size` | 1 | EP 维度 |
| `--use-distributed-optimizer` | False | 启用 ZeRO-1 |
| `--recompute-activations` | False | 激活重算 |
| `--fp8-format` | None | FP8 格式 (e4m3/hybrid) |
| `--num-layers` | None | Transformer 层数 |
| `--hidden-size` | None | 隐藏层维度 |
| `--num-attention-heads` | None | 注意力头数 |

---

## 2. torchtitan

### 2.1 定位与设计理念

torchtitan 是 Meta 维护的 **PyTorch-native 预训练框架**，强调"纯 PyTorch"理念：
- **原生 FSDP2**：基于 DTensor/DeviceMesh，不依赖 Megatron
- **模块化设计**：parallelism、model、optimizer、checkpoint 解耦
- **多模型支持**：Llama 3/4, Qwen3, DeepSeek V3, GPT-OSS, Flux
- **RL 实验**：`torchtitan/experiments/rl/` 下有 TitanRL

**面试定位**：当问题涉及"FSDP2 原理"、"PyTorch DeviceMesh"、"native TP/PP"、"activation checkpoint 策略"时，**首选引用此仓**。

### 2.2 核心架构（ASCII 图）

```
train.py:17  main()
  │
  ├─ ConfigManager()              ← 配置解析 (tyro-based)
  │
  └─ Trainer(config)
        │
        ├─ __init__()
        │     ├─ ParallelDims.build_mesh()     ← DeviceMesh
        │     ├─ ModelSpec.build()             ← Llama3Model / etc.
        │     ├─ apply_fsdp_to_decoder()       ← FSDP2 wrapping
        │     ├─ apply_tp()                    ← Tensor Parallel
        │     ├─ apply_compile()               ← torch.compile
        │     ├─ apply_activation_checkpointing() ← FullAC / SelectiveAC
        │     └─ OptimizersContainer()         ← optimizer init
        │
        └─ train()  (循环)
              ├─ train_step()
              │     ├─ forward()
              │     ├─ backward()
              │     ├─ optimizer.step()
              │     └─ lr_scheduler.step()
              ├─ checkpointer.save()
              └─ profiler.step()
```

### 2.3 核心调用链

**训练入口链：**
```
train.py:17  main()
  → trainer.py:68  Trainer()
    → trainer.py:277  Trainer.__init__()
      → parallel_dims.py:132  ParallelDims.build_mesh()
    → trainer.py:1010  Trainer.train()
      → trainer.py  train_step()
```

**并行化链：**
```
Trainer.__init__()
  → fsdp.py:168  apply_fsdp_to_decoder()       ← FSDP2 (DTensor)
  → tensor_parallel.py:19  NoParallel           ← TP 标记
  → linear.py:47  AllGatherLinear               ← TP AllGather
  → pipeline_parallel.py:65  pipeline_llm()     ← PP
  → cudagraph.py:189  CUDAGraphWrapper          ← CUDA Graph
  → activation_checkpoint.py:166  FullAC        ← AC
  → activation_checkpoint.py:185  SelectiveAC   ← 选择性 AC
  → activation_checkpoint.py:290  MemoryBudgetAC ← 内存预算 AC
```

**优化与 Loss 链：**
```
optimizer/optimizer.py:79  OptimizersContainer.__init__()
  → optimizer/lr_scheduler.py:130  linear_warmup_stable_decay()
  → loss.py:282  CrossEntropyLoss.forward()
  → checkpointer/dcp.py:89  CheckpointManager
```

### 2.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 入口 | `torchtitan/train.py:17` | `main()` |
| Trainer | `torchtitan/trainer.py:68` | `Trainer`, `Trainer.__init__:277`, `train():1010` |
| ParallelDims | `torchtitan/distributed/parallel_dims.py:132` | `ParallelDims`, `build_mesh():211` |
| FSDP | `torchtitan/distributed/fsdp.py:168` | `apply_fsdp_to_decoder()` |
| TP | `torchtitan/distributed/tensor_parallel.py:19` | `NoParallel` |
| TP Linear | `torchtitan/distributed/linear.py:47` | `AllGatherLinear` |
| PP | `torchtitan/distributed/pipeline_parallel.py:65` | `pipeline_llm()` |
| CUDA Graph | `torchtitan/distributed/cudagraph.py:189` | `CUDAGraphWrapper` |
| Full AC | `torchtitan/distributed/activation_checkpoint.py:166` | `FullAC` |
| Selective AC | `torchtitan/distributed/activation_checkpoint.py:185` | `SelectiveAC` |
| Memory Budget AC | `torchtitan/distributed/activation_checkpoint.py:290` | `MemoryBudgetAC` |
| Float8 | `torchtitan/components/quantization/float8.py:53` | `Float8LinearConverter` |
| Optimizer | `torchtitan/components/optimizer/optimizer.py:79` | `OptimizersContainer` |
| LR Scheduler | `torchtitan/components/optimizer/lr_scheduler.py:130` | `linear_warmup_stable_decay()` |
| Loss | `torchtitan/components/loss.py:282` | `CrossEntropyLoss` |
| Checkpoint | `torchtitan/components/checkpointer/dcp.py:89` | `CheckpointManager` |
| Model | `torchtitan/models/llama3/model.py:53` | `Llama3Model` |
| Config | `torchtitan/config/configs.py:122` | `ParallelismConfig` |
| Profiler | `torchtitan/tools/profiler.py:99` | `Profiler` |
| Compile | `torchtitan/distributed/compile.py` | `apply_compile()` |
| CP | `torchtitan/distributed/context_parallel/` | Context Parallel |

### 2.5 配置参数速查

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--training.steps` | None | 训练步数 |
| `--parallelism.data_parallel_replicate_degree` | 1 | DP replicate |
| `--parallelism.data_parallel_shard_degree` | -1 | DP shard (FSDP) |
| `--parallelism.tensor_parallel_degree` | 1 | TP |
| `--parallelism.pipeline_parallel_degree` | 1 | PP |
| `--parallelism.context_parallel_degree` | 1 | CP |
| `--activation_checkpoint.mode` | "full" | AC 模式 (full/selective/none) |
| `--training.compile` | False | torch.compile |
| `--float8.enable_float8_linear` | False | Float8 量化 |

---

## 3. miles

### 3.1 定位与设计理念

miles 是一个 **RL (GRPO/PPO) 后训练框架**，强调 true-on-policy 训练：
- **双后端支持**：FSDP2 + Megatron-LM 作为训练后端
- **SGLang 集成**：rollout 生成引擎
- **True On-Policy**：`miles/true_on_policy/` 数值精确性保障
- **Ray 编排**：actor/rollout 分布式调度
- **Fault Tolerance**：训练容错与恢复

**面试定位**：当问题涉及"GRPO/PPO 训练循环"、"rollout 生成"、"policy loss 计算"、"KL 散度估计"、"weight sync"时，**首选引用此仓**。

### 3.2 核心架构（ASCII 图）

```
train.py:22  train()  /  train_async.py  (异步入口)
  │
  ├─ create_placement_groups()        ← Ray PG 分配
  ├─ create_rollout_manager()         ← RolloutManager (SGLang)
  └─ create_training_models()
        │
        ├─ MegatronTrainRayActor       ← Megatron 后端
        │     └─ megatron_utils/actor.py:88
        │
        └─ FSDPTrainRayActor           ← FSDP2 后端
              └─ fsdp_utils/actor.py:88

训练循环 (per step):
  │
  ├─ 1. rollout_manager.generate()    ← SGLang 生成 response
  │     └─ ray/rollout/rollout_manager.py:54
  │
  ├─ 2. actor.train()                 ← policy update
  │     ├─ get_log_probs_and_entropy()  ← logit_processors.py:184
  │     ├─ compute_policy_loss()        ← math_utils.py:254
  │     └─ compute_approx_kl()           ← math_utils.py:139
  │
  ├─ 3. policy_loss_function()        ← losses.py:62
  │
  └─ 4. actor.update_weights()        ← weight sync to SGLang
        └─ fsdp_utils/update_weight_utils.py:54  UpdateWeight
```

### 3.3 核心调用链

**训练入口链：**
```
train.py:22  train()
  → create_placement_groups()
  → ray/rollout/rollout_manager.py:54  RolloutManager (Ray actor)
  → backends/megatron_utils/actor.py:88  MegatronTrainRayActor.init()
    → megatron/training/training.py  pretrain()
  → backends/fsdp_utils/actor.py:88  FSDPTrainRayActor.init()
```

**Rollout → Training 链：**
```
RolloutManager.generate()
  → sglang_rollout  GenerateState     ← 推理引擎生成
  → rollout_data_conversion  postprocess_rollout_data()
  → train_data_conversion  convert_samples_to_train_data()
  → actor.train()
    → get_log_probs_and_entropy()     ← logit_processors.py:184
    → compute_policy_loss()           ← math_utils.py:254
    → compute_approx_kl()             ← math_utils.py:139
    → policy_loss_function()          ← losses.py:62
```

**Weight Sync 链：**
```
FSDPTrainRayActor.update_weights()
  → fsdp_utils/update_weight_utils.py:54  UpdateWeight
    → connect_rollout_engines()       ← 连接到 SGLang engine
    → update_weights()                ← 逐层权重同步
  → sglang engine.update_weights_from_tensor()
```

### 3.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 入口 | `train.py:22`, `train_async.py` | `train()` |
| Rollout Manager | `miles/ray/rollout/rollout_manager.py:54` | `RolloutManager` (Ray actor) |
| FSDP Actor | `miles/backends/fsdp_utils/actor.py:88` | `FSDPTrainRayActor` |
| Megatron Actor | `miles/backends/megatron_utils/actor.py:88` | `MegatronTrainRayActor` |
| Policy Loss | `miles/backends/training_utils/loss_hub/losses.py:62` | `policy_loss_function()` |
| Compute Policy Loss | `miles/backends/training_utils/loss_hub/math_utils.py:254` | `compute_policy_loss()` |
| Compute Approx KL | `miles/backends/training_utils/loss_hub/math_utils.py:139` | `compute_approx_kl()` |
| Log Probs | `miles/backends/training_utils/loss_hub/logit_processors.py:184` | `get_log_probs_and_entropy()` |
| Update Weight | `miles/backends/fsdp_utils/update_weight_utils.py:54` | `UpdateWeight` |
| Ref Model | `miles/backends/fsdp_utils/actor.py:645` | `_create_ref_model()` |
| CP Utils | `miles/backends/training_utils/cp_utils.py` | `get_logits_and_tokens_offset_with_cp()` |
| Types | `miles/utils/types.py:26` | `Sample`, `RewardSpec`, `AdapterRef` |
| Object Store | `miles/utils/object_store.py` | `MooncakeDistributedStore` |
| Data Source | `miles/rollout/data_source.py` | `RolloutDataSource` |
| Filter Hub | `miles/rollout/filter_hub/dynamic_sampling_filters.py` | dynamic sampling |
| Fully Async | `miles/rollout/fully_async_rollout.py` | async rollout |
| Parallel | `miles/backends/megatron_utils/parallel.py` | `ParallelState` |
| Model | `miles/backends/megatron_utils/model.py` | model provider |
| Checkpoint | `miles/backends/megatron_utils/checkpoint.py` | checkpoint utils |

### 3.5 配置参数速查

| 参数 | 说明 |
|------|------|
| `--rollout_batch_size` | 每步 rollout 的样本数 |
| `--n_resp_per_prompt` | 每个 prompt 生成的 response 数 |
| `--kl_loss_coef` | KL 损失系数 |
| `--entropy_loss_coef` | 熵损失系数 |
| `--eps_clip` | PPO clip epsilon |
| `--grpo` | 使用 GRPO 而非 PPO |
| `--fully_async` | 异步训练模式 |
| `--offload_train` | 训练 offload |
| `--use_fault_tolerance` | 启用容错 |

---

## 4. slime

### 4.1 定位与设计理念

slime 是一个 **RL 后训练框架**，支持 GRPO/PPO 和 agentic RL：
- **Megatron 后端**：基于 Megatron-LM 的训练
- **SGLang 集成**：rollout 生成
- **Agentic RL**：`slime/agent/` 多智能体 RL
- **Observability**：tracing, profiling, reproducibility
- **Weight Sync**：支持分布式 engine 权重同步

**面试定位**：当问题涉及"RL 数据生成流程"、"agentic RL"、"train-infer 集成"、"reward 后处理"时，**首选引用此仓**。

### 4.2 核心架构（ASCII 图）

```
train.py:9  train()  /  train_async.py
  │
  ├─ create_placement_groups()        ← Ray PG 分配
  ├─ create_rollout_manager()         ← RolloutManager (SGLang)
  │     └─ ray/rollout.py:38
  │
  └─ create_training_models()
        └─ MegatronTrainRayActor
              └─ backends/megatron_utils/actor.py:56

训练循环 (per step):
  │
  ├─ 1. rollout_manager.rollout()    ← SGLang 生成
  │     └─ rollout/sglang_rollout.py:83  GenerateState
  │
  ├─ 2. _post_process_rewards()       ← reward 后处理
  │     └─ ray/rollout.py:279
  │
  ├─ 3. actor.train_step()            ← policy update
  │     ├─ get_log_probs_and_entropy()  ← loss.py:513
  │     ├─ vanilla_tis_function()       ← loss.py:883
  │     └─ _VocabParallelLogProbEntropy ← ppo_utils.py:187
  │
  └─ 4. actor.update_weights()        ← weight sync
        └─ update_weight/update_weight_from_distributed.py:24
```

### 4.3 核心调用链

**训练入口链：**
```
train.py:9  train()
  → ray/rollout.py:38  RolloutManager (Ray actor)
  → backends/megatron_utils/actor.py:56  MegatronTrainRayActor.init()
    → megatron/training/training.py  pretrain()
```

**Rollout 链：**
```
RolloutManager.rollout()
  → rollout/sglang_rollout.py:83  GenerateState.__init__()
  → sglang engine.generate()
  → ray/rollout.py:279  _post_process_rewards()
    → reward normalization (GRPO/GSPO/CISPO)
```

**Loss 链：**
```
actor.train_step()
  → backends/megatron_utils/loss.py:513  get_log_probs_and_entropy()
  → slime/utils/ppo_utils.py:187  _VocabParallelLogProbEntropy.forward()
  → backends/megatron_utils/loss.py:883  vanilla_tis_function()
```

**Weight Sync 链：**
```
actor.update_weights()
  → backends/megatron_utils/update_weight/update_weight_from_distributed.py:24
    UpdateWeightFromDistributed
    → connect_rollout_engines()
    → _send_weights()
```

### 4.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 入口 | `train.py:9`, `train_async.py` | `train()` |
| Rollout Manager | `slime/ray/rollout.py:38` | `RolloutManager` (Ray actor) |
| GenerateState | `slime/rollout/sglang_rollout.py:83` | `GenerateState` (Singleton) |
| Megatron Actor | `slime/backends/megatron_utils/actor.py:56` | `MegatronTrainRayActor` |
| Log Probs | `slime/backends/megatron_utils/loss.py:513` | `get_log_probs_and_entropy()` |
| Vanilla TIS | `slime/backends/megatron_utils/loss.py:883` | `vanilla_tis_function()` |
| PPO Utils | `slime/utils/ppo_utils.py:187` | `_VocabParallelLogProbEntropy` |
| Reward Post-process | `slime/ray/rollout.py:279` | `_post_process_rewards()` |
| CP Utils | `slime/backends/megatron_utils/cp_utils.py` | `get_logits_and_tokens_offset_with_cp()` |
| Update Weight | `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py:24` | `UpdateWeightFromDistributed` |
| Data Source | `slime/rollout/data_source.py:50` | `RolloutDataSource` |
| Arguments | `slime/utils/arguments.py` | `parse_args()` |
| SGLang Utils | `slime/backends/sglang_utils/` | SGLang integration |
| Agent | `slime/agent/` | Agent-based RL |
| Observability | `slime/observability/` | tracing, profiling |

### 4.5 配置参数速查

| 参数 | 说明 |
|------|------|
| `--rollout_batch_size` | 每步 rollout 的样本数 |
| `--n_rollout_per_epoch` | 每 epoch rollout 数 |
| `--kl_coef` | KL 系数 |
| `--clip_ratio` | PPO clip ratio |
| `--reward_type` | reward 类型 |
| `--advantage_estimator` | advantage 估计器 (grpo/gspo/cispo) |
| `--offload_rollout` | rollout offload |
| `--release_train` | release train mode |

---

## 5. torchada

### 5.1 定位与设计理念

torchada 是 **NVIDIA Ada / RTX 消费级 GPU 的 PyTorch 后端适配层**：
- **平台透明**：CUDA → MUSA 的 API 翻译层
- **Graph Rotation**：CUDA Graph 的 LRU 轮换机制（解决 graph 数量限制）
- **Triton Kernels**：FP8 量化、Fused MoE 等高性能 kernel
- **C++ 扩展**：统一的 CUDA/MUSA extension 构建

**面试定位**：当问题涉及"硬件适配层设计"、"CUDA Graph 轮换"、"跨平台 kernel 开发"、"FP8 量化实现"时，**首选引用此仓**。

### 5.2 核心架构（ASCII 图）

```
__init__.py:57  apply_patches()       ← 入口，自动应用所有 patch
  │
  ├─ _platform.py:21  detect_platform()  ← 平台检测
  │     ├─ TORCHADA_PLATFORM env var
  │     ├─ MUSA availability
  │     ├─ CUDA availability
  │     └─ CPU fallback
  │
  ├─ _cpp_ops.py:73  load_cpp_ops()     ← 加载 C++ 算子
  │
  ├─ _patch.py:2103  apply_patches()    ← 注册所有 patch
  │     ├─ patch_function()  decorator  ← _patch.py:48
  │     ├─ torch.device("cuda") → "musa"
  │     ├─ torch.cuda.* → torch.musa.*
  │     ├─ nccl → mccl backend
  │     └─ ctypes.CDLL → PatchedCDLL
  │
  └─ _graph_rotation.py:138  _Rotation  ← CUDA Graph 轮换
        ├─ _DEFAULT_CAP = 1900           ← 默认 graph 上限
        └─ LRU rotation of live graphs
```

### 5.3 核心调用链

**初始化链：**
```
__init__.py:57  apply_patches()
  → _platform.py:21  detect_platform()    ← 检测 GPU 平台
  → _cpp_ops.py:73  load_cpp_ops()        ← 加载 C++ 算子
  → _patch.py:2103  apply_patches()       ← 应用所有 patch
    → _patch.py:48  patch_function()      ← 装饰器注册
```

**Graph Rotation 链：**
```
_graph_rotation.py:44  _DEFAULT_CAP = 1900
  → _graph_rotation.py:48  _read_cap()     ← 读取 graph 上限
    → TORCHADA_GRAPH_EXEC_CAP env var
    → TORCHADA_GRAPH_AUTOPROBE=1  ← 自动探测
  → _graph_rotation.py:138  _Rotation     ← LRU 轮换
    → cap: int                             ← 最大 graph 数
    → _live: dict[id(graph), weakref]      ← LRU 顺序
```

**FP8 量化链：**
```
triton/kernels/quant/fp8.py:55  per_token_group_quant_fp8()
  → Triton kernel: per_token_quant_fp8_kernel
  → 输入: x [M, K], group_size
  → 输出: y [M, K], s [M, K/group_size]
```

**Fused MoE 链：**
```
triton/runtime/fused_moe/fused_moe.py:331  fused_experts_impl()
  → Triton kernel: fused_moe_kernel
  → 支持: topk_weights, topk_ids, 多 expert 融合
```

### 5.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 入口 | `src/torchada/__init__.py:57` | `apply_patches()` |
| Patch 注册 | `src/torchada/_patch.py:2103` | `apply_patches()` |
| Patch 装饰器 | `src/torchada/_patch.py:48` | `patch_function()` |
| 平台检测 | `src/torchada/_platform.py:21` | `detect_platform()` |
| C++ 算子 | `src/torchada/_cpp_ops.py:73` | `load_cpp_ops()` |
| Graph Rotation | `src/torchada/_graph_rotation.py:138` | `_Rotation` |
| Default Cap | `src/torchada/_graph_rotation.py:44` | `_DEFAULT_CAP = 1900` |
| FP8 量化 | `src/torchada/triton/kernels/quant/fp8.py:55` | `per_token_group_quant_fp8()` |
| Fused MoE | `src/torchada/triton/runtime/fused_moe/fused_moe.py:331` | `fused_experts_impl()` |
| CUDA Extension | `src/torchada/utils/cpp_extension.py:1023` | `CUDAExtension` |
| 映射 | `src/torchada/_mapping.py` | API 映射表 |
| Runtime | `src/torchada/_runtime.py` | 运行时管理 |

### 5.5 配置参数速查

| 参数/环境变量 | 说明 |
|--------------|------|
| `TORCHADA_PLATFORM` | 强制指定平台 (cuda/musa/cpu) |
| `TORCHADA_GRAPH_EXEC_CAP` | CUDA Graph 上限 (默认 1900) |
| `TORCHADA_GRAPH_AUTOPROBE` | 自动探测 graph 上限 |
| `_DEFAULT_MARGIN` | graph 容量余量 (默认 128) |

---

## 6. torch_musa

### 6.1 定位与设计理念

torch_musa 是 **Moore Threads MUSA GPU 的 PyTorch 后端**：
- **C++ 底层**：完整的 Device/Stream/Event/Allocator 实现
- **MCCL**：MUSA 的 NCCL 等价通信库
- **Inductor**：MUSA 的 Triton codegen 适配
- **FSDP**：MUSA 的 FSDP grad scaler 适配
- **CUDA Graph**：MUSA Graph 支持

**面试定位**：当问题涉及"硬件后端开发"、"C++ 算子实现"、"内存分配器设计"、"跨硬件移植"时，**首选引用此仓**。

### 6.2 核心架构（ASCII 图）

```
__init__.py  (18-step init sequence)
  │
  ├─ torch.utils.rename_privateuse1_backend("musa")
  ├─ import torch_musa._MUSAC          ← C++ extension
  ├─ align_rng_to_nv_gpu()             ← RNG 对齐
  ├─ torch.__setattr__("musa", ...)    ← 注册 torch.musa
  ├─ register_musa_hook()              ← hook 注册
  │
  ├─ Device 层 (csrc/core/)
  │     ├─ Device.cpp                  ← 设备管理
  │     ├─ GuardImpl.h                 ← device guard
  │     └─ MUSAFunctions.h             ← 设备函数
  │
  ├─ Memory 层 (csrc/core/)
  │     └─ MUSACachingAllocator.cpp    ← 内存分配器 (4219 行)
  │
  ├─ Stream 层 (csrc/core/)
  │     ├─ MUSAStream.cpp              ← 流管理
  │     └─ MUSAEvent.h                 ← 事件同步
  │
  ├─ Graph 层
  │     ├─ csrc/aten/musa/MUSAGraph.cpp  ← C++ Graph
  │     └─ musa_graph/graphs.py          ← Python Graph API
  │
  ├─ Communication 层
  │     └─ PythonMCCL.cpp              ← MCCL 通信
  │
  └─ Inductor 层
        └─ _inductor/codegen/wrapper.py:71  MUSATritonWrapperCodeGen
```

### 6.3 核心调用链

**初始化链：**
```
__init__.py
  → rename_privateuse1_backend("musa")
  → import _MUSAC                      ← C++ extension 加载
  → core.random.align_rng_to_nv_gpu()  ← RNG 对齐
  → core.device.register_musa_hook()   ← hook 注册
  → _apply_distributed_patch()         ← 分布式 patch
```

**内存分配链：**
```
torch.musa.empty_cache()
  → MUSACachingAllocator.cpp           ← C++ caching allocator
    ├── allocate()                     ← 分配
    ├── free()                         ← 释放
    └── garbage_collect()              ← GC
```

**Graph 捕获链：**
```
musa_graph/graphs.py
  → is_current_stream_capturing()      ← 检测 capture 状态
  → update_musa_graph_with_profile()   ← 更新 graph
  → csrc/aten/musa/MUSAGraph.cpp       ← C++ Graph 实现
    ├── capture_begin()
    ├── capture_end()
    └── replay()
```

**Inductor 链：**
```
torch.compile()
  → _inductor/codegen/wrapper.py:71  MUSATritonWrapperCodeGen
    ├── write_header()                 ← 生成 wrapper 头
    └── generate()                     ← 生成 kernel 调用
```

### 6.4 关键文件索引（按功能分类）

| 功能 | 文件路径 | 关键类/函数 |
|------|---------|------------|
| 入口 | `torch_musa/__init__.py` | 18-step init sequence |
| Device | `torch_musa/csrc/core/Device.cpp` | `Device` 管理 |
| GuardImpl | `torch_musa/csrc/core/GuardImpl.h` | `DeviceGuardImpl` |
| Memory | `torch_musa/csrc/core/MUSACachingAllocator.cpp` | C++ caching allocator (4219 行) |
| Stream | `torch_musa/csrc/core/MUSAStream.cpp` | `MUSAStream` |
| Event | `torch_musa/csrc/core/MUSAEvent.h` | `MUSAEvent` |
| Graph C++ | `torch_musa/csrc/aten/musa/MUSAGraph.cpp` | `MUSAGraph` |
| Graph Python | `torch_musa/musa_graph/graphs.py` | `is_current_stream_capturing()` |
| MCCL | `torch_musa/csrc/core/PythonMCCL.cpp` | MCCL 通信 |
| Inductor | `torch_musa/_inductor/codegen/wrapper.py:71` | `MUSATritonWrapperCodeGen` |
| FSDP | `torch_musa/distributed/fsdp/sharded_grad_scaler.py` | `GradScaler` |
| AMP | `torch_musa/core/amp/` | AMP 支持 |
| Profiler | `torch_musa/profiler/` | 性能分析 |

### 6.5 配置参数速查

| 参数/环境变量 | 说明 |
|--------------|------|
| `TORCH_RNG_ALIGN` | RNG 对齐配置 |
| `TORCH_ARCH_MP_COUNT` | MP 数量 |
| `TORCH_ARCH_MAXTHREADS_PER_MP` | 每 MP 最大线程 |
| `MUSA_VISIBLE_DEVICES` | 可见设备 |
| `MCCL_BACKEND` | MCCL 后端配置 |

---

## 7. 跨仓库技术对比表

| 技术点 | Megatron-LM | torchtitan | miles | slime | torchada | torch_musa |
|--------|------------|-----------|-------|-------|----------|-----------|
| **并行策略** | TP+PP+DP+EP+CP | TP+PP+DP+CP | 继承后端 | 继承后端 | N/A | N/A |
| **TP 实现** | ColumnParallelLinear | DTensor-based | 继承 Megatron | 继承 Megatron | N/A | N/A |
| **PP 调度** | 1F1B / Interleaved | pipeline_llm | 继承 Megatron | 继承 Megatron | N/A | N/A |
| **DP 实现** | DistributedOptimizer (ZeRO-1) | FSDP2 (DTensor) | FSDP2 / 继承 | 继承 Megatron | N/A | N/A |
| **EP 实现** | MoE 原生 EP | N/A | 继承 Megatron | 继承 Megatron | N/A | N/A |
| **CP 实现** | 原生 CP + Hybrid CP | context_parallel | cp_utils.py | cp_utils.py | N/A | N/A |
| **Recompute** | checkpointed_forward | FullAC / SelectiveAC / MemoryBudgetAC | 继承后端 | 继承后端 | N/A | N/A |
| **CUDA Graph** | 原生 CUDA Graph | CUDAGraphWrapper | 继承后端 | 继承后端 | _Rotation (LRU) | MUSAGraph |
| **FP8 支持** | Transformer Engine FP8 | Float8LinearConverter | N/A | N/A | Triton FP8 kernel | N/A |
| **FP4 支持** | NVFP4 | N/A | N/A | N/A | N/A | N/A |
| **MoE 实现** | MoELayer + TopKRouter | N/A | 继承 Megatron | 继承 Megatron | Fused MoE Triton | N/A |
| **RL 支持** | 仅训练后端 | TitanRL 实验 | GRPO/PPO 完整 | GRPO/PPO + Agentic | N/A | N/A |
| **Weight Sync** | N/A | N/A | UpdateWeight | UpdateWeightFromDistributed | N/A | N/A |
| **Rollout 引擎** | N/A | N/A | SGLang | SGLang | N/A | N/A |
| **True On-Policy** | N/A | N/A | true_on_policy/ | N/A | N/A | N/A |
| **Checkpoint** | Dist checkpointing | DCP (Distributed CP) | 继承后端 | 继承后端 | N/A | N/A |
| **编译** | N/A | torch.compile | 继承后端 | 继承后端 | N/A | Inductor |
| **硬件支持** | NVIDIA GPU | NVIDIA GPU | NVIDIA GPU | NVIDIA GPU | Ada/RTX + MUSA | MUSA GPU |
| **Ray 编排** | N/A | N/A | 完整支持 | 完整支持 | N/A | N/A |
| **Fault Tolerance** | 基础容错 | N/A | 完整 FT | 基础容错 | N/A | N/A |
| **内存优化** | ZeRO-1 + Recompute | FSDP2 + AC | 继承后端 | 继承后端 | Graph Rotation | CachingAllocator |
| **配置系统** | argparse + ConfigContainer | tyro-based | argparse | argparse | env vars | env vars |

---

## 8. "遇到问题时查哪里" 路由表

| 问题类型 | 首选仓库 | 关键文件 | 说明 |
|---------|---------|---------|------|
| **TP 并行原理** | Megatron-LM | `core/tensor_parallel/layers.py:986` | ColumnParallelLinear / RowParallelLinear |
| **TP 通信重叠** | Megatron-LM | `core/tensor_parallel/layers.py:634` | LinearWithGradAccumulationAndAsyncCommunication |
| **PP 1F1B 调度** | Megatron-LM | `core/pipeline_parallel/schedules.py:2147` | 标准 1F1B 实现 |
| **PP Interleaved** | Megatron-LM | `core/pipeline_parallel/schedules.py:1019` | Virtual pipeline 调度 |
| **FSDP2 原理** | torchtitan | `distributed/fsdp.py:168` | DTensor-based FSDP2 |
| **DeviceMesh 构建** | torchtitan | `distributed/parallel_dims.py:211` | build_mesh() |
| **ZeRO-1 优化器** | Megatron-LM | `core/optimizer/distrib_optimizer.py:113` | DistributedOptimizer |
| **DDP grad sync** | Megatron-LM | `core/distributed/distributed_data_parallel.py:87` | bucket reduce-scatter |
| **Activation Checkpoint** | torchtitan | `distributed/activation_checkpoint.py:166` | FullAC / SelectiveAC / MemoryBudgetAC |
| **Recompute 实现** | Megatron-LM | `core/recompute.py:22` | checkpointed_forward |
| **MoE 路由** | Megatron-LM | `core/transformer/moe/moe_layer.py:625` | TopKRouter + token dispatch |
| **MoE Fused Kernel** | torchada | `triton/runtime/fused_moe/fused_moe.py:331` | Triton fused_experts_impl |
| **FP8 量化** | torchada | `triton/kernels/quant/fp8.py:55` | per_token_group_quant_fp8 |
| **FP8 训练** | Megatron-LM | `core/transformer/transformer_config.py:588` | FP8 recipe 配置 |
| **CUDA Graph** | torchtitan | `distributed/cudagraph.py:189` | CUDAGraphWrapper |
| **Graph Rotation** | torchada | `_graph_rotation.py:138` | LRU _Rotation |
| **Policy Loss (PPO/GRPO)** | miles | `backends/training_utils/loss_hub/losses.py:62` | policy_loss_function |
| **KL 散度估计** | miles | `backends/training_utils/loss_hub/math_utils.py:139` | compute_approx_kl |
| **Log Probs 计算** | slime | `backends/megatron_utils/loss.py:513` | get_log_probs_and_entropy |
| **PPO Loss 实现** | slime | `utils/ppo_utils.py:187` | _VocabParallelLogProbEntropy |
| **Reward 后处理** | slime | `ray/rollout.py:279` | _post_process_rewards |
| **Weight Sync** | miles | `backends/fsdp_utils/update_weight_utils.py:54` | UpdateWeight |
| **Ref Model** | miles | `backends/fsdp_utils/actor.py:645` | _create_ref_model |
| **True On-Policy** | miles | `true_on_policy/` | 数值精确性保障 |
| **Rollout 生成** | miles | `ray/rollout/rollout_manager.py:54` | RolloutManager |
| **SGLang 集成** | slime | `rollout/sglang_rollout.py:83` | GenerateState |
| **硬件适配层** | torchada | `_patch.py:2103` | apply_patches |
| **平台检测** | torchada | `_platform.py:21` | detect_platform |
| **C++ 后端** | torch_musa | `csrc/core/Device.cpp` | Device 管理 |
| **内存分配器** | torch_musa | `csrc/core/MUSACachingAllocator.cpp` | CachingAllocator (4219 行) |
| **MCCL 通信** | torch_musa | `csrc/core/PythonMCCL.cpp` | MCCL 通信 |
| **Inductor Codegen** | torch_musa | `_inductor/codegen/wrapper.py:71` | MUSATritonWrapperCodeGen |
| **LR Scheduler** | torchtitan | `components/optimizer/lr_scheduler.py:130` | linear_warmup_stable_decay |
| **Loss 实现** | torchtitan | `components/loss.py:282` | CrossEntropyLoss |
| **Checkpoint** | torchtitan | `components/checkpointer/dcp.py:89` | DCP CheckpointManager |
| **Profiler** | torchtitan | `tools/profiler.py:99` | Profiler |
| **Parallel Groups** | Megatron-LM | `core/parallel_state.py:600` | initialize_model_parallel |
| **Grad Finalize** | Megatron-LM | `core/distributed/finalize_model_grads.py` | finalize_model_grads |
| **P2P Communication** | Megatron-LM | `core/pipeline_parallel/p2p_communication.py` | recv_forward / send_forward |
| **Config 系统** | torchtitan | `config/configs.py:122` | ParallelismConfig |
| **Agentic RL** | slime | `agent/` | 多智能体 RL |
| **Fault Tolerance** | miles | `utils/ft_utils/` | 训练容错 |
| **Async Rollout** | miles | `rollout/fully_async_rollout.py` | 异步 rollout |
| **Dynamic Sampling** | miles | `rollout/filter_hub/dynamic_sampling_filters.py` | 动态采样过滤 |

---

## 9. 仓库间依赖关系

```
                    ┌─────────────────────────────────────────────────────┐
                    │                   训练栈全景                         │
                    └─────────────────────────────────────────────────────┘

  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │   Megatron-LM    │     │   torchtitan     │     │    torchada      │
  │   (NVIDIA)       │     │   (Meta)         │     │   (NVIDIA Ada)   │
  │                  │     │                  │     │                  │
  │  全栈并行参考     │     │  PyTorch-native  │     │  硬件适配层       │
  │  TP/PP/DP/EP/CP  │     │  FSDP2 + TP + PP │     │  CUDA→MUSA 翻译  │
  │  ZeRO-1 + FP8    │     │  torch.compile   │     │  Triton Kernels  │
  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
           │                        │                        │
           │ 被 miles/slime 依赖     │ 独立实现                │ 被 torch_musa 依赖
           │                        │                        │ (可选)
           ▼                        ▼                        ▼
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │      miles       │     │      slime       │     │   torch_musa     │
  │   (RL 后训练)     │     │   (RL 后训练)     │     │  (MUSA GPU)      │
  │                  │     │                  │     │                  │
  │  GRPO/PPO 训练   │     │  GRPO/PPO +      │     │  C++ 后端实现     │
  │  双后端(FSDP/MT)  │     │  Agentic RL      │     │  Device/Stream/  │
  │  SGLang Rollout  │     │  SGLang Rollout  │     │  Event/Allocator │
  │  True On-Policy  │     │  Observability   │     │  MCCL 通信       │
  └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
           │                        │                        │
           │ 共享 SGLang 依赖        │ 共享 SGLang 依赖        │ 独立 C++ 栈
           │                        │                        │
           ▼                        ▼                        ▼
  ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
  │     SGLang       │     │   (外部依赖)      │     │  Moore Threads   │
  │  (推理引擎)       │     │   vLLM / HF      │     │  MUSA GPU 硬件   │
  │                  │     │   transformers    │     │                  │
  │  Rollout 生成    │     │                  │     │  MTT S5000 等    │
  │  KV Cache 管理   │     │                  │     │                  │
  └──────────────────┘     └──────────────────┘     └──────────────────┘

依赖关系说明：
  - Megatron-LM ← miles, slime (作为训练后端)
  - torchtitan 独立，不依赖其他仓
  - torchada ← torchada 的 MUSA 部分依赖 torch_musa 的 C++ 算子
  - torch_musa 独立 C++ 栈，torchada 可选依赖
  - miles ↔ slime 共享架构模式（Ray + SGLang + Megatron），但独立实现
  - SGLang 作为 miles/slime 的 rollout 引擎（外部依赖，通过 Ray Actor 调用）
```

Now let me count the lines and verify the file is complete.