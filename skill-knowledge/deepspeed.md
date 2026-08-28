# DeepSpeed — 代码级深度分析

## 1. DeepSpeed 总览与设计哲学

DeepSpeed 是微软开源的分布式训练库，核心差异化是 **ZeRO（Zero Redundancy Optimizer）** 系列优化器，通过将模型状态（优化器状态、梯度、参数）分片到数据并行维度，消除传统 DDP 中每个 rank 持有完整模型状态的冗余。与 Megatron-LM 的"模型并行优先"和 torchtitan 的"FSDP 原生"路线不同，DeepSpeed 走的是"数据并行 + 状态分片"路线——用户代码几乎不变，只需 JSON 配置即可启用 ZeRO-1/2/3、Pipeline Parallel、MoE、Autotuning 等特性。

### 分层架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        deepspeed launcher                           │
│                     (deepspeed/launcher/run.py)                     │
└────────────────────────────────┬────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DeepSpeedEngine                                │
│                  (runtime/engine.py:235)                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  FP16/BF16   │  │  ZeRO-1/2    │  │  ZeRO-3 / ZeRO-Offload   │  │
│  │  Optimizer   │  │  Optimizer   │  │  (stage3.py / offload)   │  │
│  │(fp16/bf16.py)│  │(stage_1_and_ │  │  (stage3.py:148)         │  │
│  │              │  │  2.py:134)   │  │  (parameter_offload.py)  │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Pipeline    │  │    MoE       │  │    Autotuning            │  │
│  │  Engine      │  │  (moe/)      │  │  (autotuning/)           │  │
│  │(pipe/engine) │  │  layer.py    │  │  autotuner.py            │  │
│  │  schedule.py │  │  sharded_moe │  │                          │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │
│  │  Inference   │  │  Ops Layer   │  │    Checkpoint            │  │
│  │  Engine      │  │  CPUAdam     │  │    Engine                │  │
│  │(inference/)  │  │  FusedAdam   │  │  (checkpoint_engine/)    │  │
│  └──────────────┘  └──────────────┘  └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### 核心模块表

| 模块 | 文件路径 | 核心类/函数 | 职责 |
|------|----------|-------------|------|
| Engine | `runtime/engine.py` | `DeepSpeedEngine` | 训练主循环、配置分发、优化器/调度器构建 |
| Config | `runtime/config.py` | `DeepSpeedConfig` | JSON 配置解析、参数校验 |
| ZeRO-1/2 | `runtime/zero/stage_1_and_2.py` | `DeepSpeedZeroOptimizer` | 优化器状态/梯度分片 |
| ZeRO-3 | `runtime/zero/stage3.py` | `DeepSpeedZeroOptimizer_Stage3` | 参数分片 + all-gather on demand |
| Offload | `runtime/zero/parameter_offload.py` | `DeepSpeedZeRoOffload` | 参数 offload 到 CPU/NVME |
| Partition | `runtime/zero/partition_parameters.py` | `Init`, `register_external_parameter` | 参数分片/收集原语 |
| Pipeline | `runtime/pipe/engine.py` | `PipelineEngine` | 1F1B schedule、P2P 通信 |
| MoE | `moe/layer.py` | `MoE` | MoE 层封装 |
| MoE Gate | `moe/sharded_moe.py` | `TopKGate`, `top1gating`, `top2gating`, `topkgating` | 门控路由 |
| EP Router | `moe/ep_router.py` | `TokenChoiceTopKRouter` | Expert Parallel 路由 |
| Autotuning | `autotuning/autotuner.py` | `Autotuner` | 自动调优配置 |
| Inference | `inference/engine.py` | `InferenceEngine` | 推理引擎、kernel injection |
| CPU Adam | `ops/adam/cpu_adam.py` | `DeepSpeedCPUAdam` | CPU 端 Adam 优化器 |
| Fused Adam | `ops/adam/fused_adam.py` | `FusedAdam` | GPU 融合 Adam |

---

## 2. DeepSpeedEngine（核心引擎）

### 2.1 初始化调用链

`DeepSpeedEngine.__init__` 是入口，完成从配置解析到优化器构建的全流程：

```
DeepSpeedEngine.__init__()                          # engine.py:238
  ├── _do_args_sanity_check(args)                  # engine.py:535 — 校验 LOCAL_RANK
  ├── _configure_with_arguments(args, mpu)         # engine.py:517 — 设置 local_rank
  ├── _do_sanity_check()                           # engine.py:566 — 校验精度/优化器兼容性
  ├── _configure_expert_parallel(model)            # engine.py:535 — AutoEP 初始化
  ├── _configure_distributed_model(model)          # engine.py:1628 — 模型分发
  │     ├── _cast_module_mixed_precision()         # engine.py:1426 — 精度转换
  │     └── _broadcast_model()                     # engine.py:1594 — 参数广播
  ├── _configure_optimizer(optimizer, params)      # engine.py:1902 — 优化器配置
  │     ├── _configure_basic_optimizer()           # engine.py:1961 — 构建基础优化器
  │     │     ├── DeepSpeedCPUAdam                 # cpu_adam.py:13
  │     │     └── FusedAdam                        # fused_adam.py:18
  │     └── _do_optimizer_sanity_check()           # engine.py:1854 — 决定优化器 wrapper
  │           ├── ZERO_OPTIMIZATION → _configure_zero_optimizer()  # engine.py:2235
  │           │     ├── stage≤2 → DeepSpeedZeroOptimizer           # stage_1_and_2.py:134
  │           │     └── stage=3 → DeepSpeedZeroOptimizer_Stage3     # stage3.py:148
  │           ├── AMP → amp.initialize()
  │           ├── FP16 → FP16_Optimizer
  │           └── BFLOAT16 → BF16_Optimizer
  ├── _configure_lr_scheduler()                    # engine.py:1447 — 调度器配置
  └── _configure_checkpointing()                   # engine.py:1463 — Checkpoint 引擎
```

**关键设计点**：`_configure_zero_optimizer` (`engine.py:2235`) 是 ZeRO 的核心分发点。根据 `zero_stage` 选择：
- `stage ≤ 2`：构建 `DeepSpeedZeroOptimizer`（`stage_1_and_2.py:134`）
- `stage = 3`：构建 `DeepSpeedZeroOptimizer_Stage3`（`stage3.py:148`）或 `DeepSpeedZeRoOffload`（`parameter_offload.py:117`）

### 2.2 训练循环

DeepSpeed 的训练循环由 `forward → backward → step` 三步组成，核心调用链：

```
# Forward
DeepSpeedEngine.forward(*inputs, **kwargs)          # engine.py:2676
  ├── _forward_prologue(inputs, kwargs)             # engine.py:2594 — ZeRO-3 in_forward=True
  ├── autocast_if_enabled(self)                     # torch_autocast — 混合精度
  └── self.module(*inputs, **kwargs)                # 实际前向

# Backward
DeepSpeedEngine.backward(loss)                      # engine.py:3067
  ├── _backward_prologue()                          # engine.py:2801 — 启动 timer
  │     └── optimizer.backward_prologue()           # ZeRO 优化器进入 backward
  ├── loss.backward()                               # PyTorch 反向传播
  │     └── grad_handling_hook()                    # stage_1_and_2.py:1098 — 梯度 hook
  │           └── reduce_independent_p_g_buckets()  # stage_1_and_2.py:1125 — 梯度分片 reduce
  └── _backward_epilogue()                          # engine.py:2835 — allreduce + 梯度同步
        └── allreduce_gradients()                   # engine.py:2757
              └── optimizer.overlapping_partition_gradients_reduce_epilogue()
                    └── independent_gradient_partition_epilogue()  # stage_1_and_2.py:898

# Step
DeepSpeedEngine.step()                              # engine.py:3242
  ├── _take_model_step(lr_kwargs)                   # engine.py:3169
  │     ├── clip_grad_norm_()                       # 梯度裁剪
  │     └── optimizer.step()                        # 优化器更新
  ├── optimizer.zero_grad()                         # 梯度清零
  └── lr_scheduler.step()                           # 学习率更新
```

### 2.3 关键类/函数表

| 类/函数 | 文件:行号 | 职责 |
|---------|-----------|------|
| `DeepSpeedEngine` | `engine.py:235` | 主引擎类，继承 `nn.Module` |
| `EngineTimers` | `engine.py:192` | wall-clock timer 管理 |
| `forward()` | `engine.py:2676` | 前向传播入口 |
| `backward()` | `engine.py:3067` | 反向传播入口，处理 loss scaling |
| `step()` | `engine.py:3242` | 参数更新入口 |
| `scale()` | `engine.py:3014` | 手动 backward 时的 loss scaling |
| `no_sync()` | `engine.py:2898` | 禁用梯度同步的上下文管理器 |
| `coalesce_grad_reduction()` | `engine.py:2918` | 合并多次 backward 的梯度 reduce |
| `train_batch()` | PipelineEngine 使用 | Pipeline 训练入口 |

---

## 3. ZeRO 优化器（核心差异化）

### 3.1 ZeRO 三阶段原理

ZeRO（Zero Redundancy Optimizer）的核心思想：将模型状态按数据并行维度分片，每个 rank 只存储和更新 1/N 的状态。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          ZeRO 三阶段对比                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Baseline DDP (每个 rank 持有完整状态):                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │ Optim St │  │  Grads   │  │  Params  │  × N ranks                  │
│  └──────────┘  └──────────┘  └──────────┘                              │
│                                                                         │
│  ZeRO-1: 优化器状态分片                                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │ Optim St │  │  Grads   │  │  Params  │                              │
│  │  1/N     │  │  allred  │  │  full    │                              │
│  └──────────┘  └──────────┘  └──────────┘                              │
│  节省: 优化器状态内存 (Adam 2×模型大小)                                  │
│                                                                         │
│  ZeRO-2: + 梯度分片                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │ Optim St │  │  Grads   │  │  Params  │                              │
│  │  1/N     │  │  1/N     │  │  full    │                              │
│  └──────────┘  └──────────┘  └──────────┘                              │
│  节省: 优化器状态 + 梯度内存                                             │
│                                                                         │
│  ZeRO-3: + 参数分片                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                              │
│  │ Optim St │  │  Grads   │  │  Params  │                              │
│  │  1/N     │  │  1/N     │  │  1/N     │                              │
│  └──────────┘  └──────────┘  └──────────┘                              │
│  节省: 全部模型状态内存 (线性缩放)                                       │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 ZeRO-1/2 实现

`DeepSpeedZeroOptimizer`（`stage_1_and_2.py:134`）是 ZeRO-1/2 的核心类。

**初始化调用链**：

```
DeepSpeedZeroOptimizer.__init__()                   # stage_1_and_2.py:146
  ├── 参数分组 & flatten                            # stage_1_and_2.py:373-465
  │     ├── flatten_dense_tensors_aligned()         # stage_1_and_2.py:1120 — 对齐 flatten
  │     └── get_data_parallel_partitions()          # 按 DP 维度切分
  ├── _update_model_bit16_weights()                # stage_1_and_2.py:774 — 设置模型权重为 flat slice
  ├── single_partition_of_fp32_groups              # stage_1_and_2.py:491 — FP32 master weights
  ├── initialize_gradient_partitioning_data_structures()  # stage_1_and_2.py:874
  ├── create_gradient_handling_hooks()             # stage_1_and_2.py:1084 — 注册梯度 hook
  ├── initialize_optimizer_states()                # stage_1_and_2.py:816
  └── _link_all_hp_params()                        # stage_1_and_2.py:717 — 链接 FP16/FP32 参数
```

**step() 流程**（梯度 reduce + 参数更新）：

```
reduce_gradients()                                  # stage_1_and_2.py:840
  ├── [非 overlap] 遍历参数，reduce_ready_partitions_and_remove_grads()
  └── overlapping_partition_gradients_reduce_epilogue()    # stage_1_and_2.py:1032
        └── independent_gradient_partition_epilogue()      # stage_1_and_2.py:898
              ├── reduce_ipg_grads()                       # stage_1_and_2.py:1615
              │     └── average_tensor()                   # stage_1_and_2.py:1277
              │           ├── gradient_reduction_w_predivide()  # allreduce
              │           └── allreduce_and_scatter()       # reduce-scatter
              └── _clear_param_grad_only()                 # 清理梯度
```

**关键数据结构**：
- `bit16_groups`：FP16 模型参数分组
- `single_partition_of_fp32_groups`：每个 rank 持有的 FP32 master weight 分片
- `ipg_buckets`（`IPGBucket`，`stage_1_and_2.py:113`）：梯度 reduce 的连续缓冲区
- `param_to_partition_ids`：参数到分片 partition 的映射

### 3.3 ZeRO-3 实现

`DeepSpeedZeroOptimizer_Stage3`（`stage3.py:148`）实现参数分片。与 ZeRO-1/2 的关键区别：

1. **参数也分片**：每个 rank 只持有 1/N 的参数
2. **All-gather on demand**：forward/backward 时按需收集完整参数
3. **Parameter Offload**：参数可 offload 到 CPU/NVME

**核心调用链**：

```
DeepSpeedZeroOptimizer_Stage3.__init__()            # stage3.py:160
  ├── initialize_ds_offload()                       # stage3.py:269
  │     └── DeepSpeedZeRoOffload.__init__()         # parameter_offload.py:119
  │           ├── _convert_to_zero_parameters()     # 参数转换为 ds_tensor
  │           ├── _inject_parameters()              # 注入 ZeROOrderedDict
  │           └── PartitionedParameterCoordinator   # parameter_offload.py:193
  ├── _create_fp16_partitions_with_defragmentation()  # stage3.py:433
  └── _setup_for_real_optimizer()                  # stage3.py:485
```

**DeepSpeedZeRoOffload**（`parameter_offload.py:117`）是 ZeRO-3 的参数管理核心：

- `ZeROOrderedDict`（`parameter_offload.py:40`）：替代 `nn.Module._parameters`，在 `__getitem__` 时触发 `all_gather()` 收集完整参数
- `PartitionedParameterCoordinator`：管理参数 prefetch 和释放
- `mark_persistent_parameters()`：标记常驻内存的参数

**参数分片原语**（`partition_parameters.py`）：
- `register_external_parameter()`（`partition_parameters.py:142`）：注册跨模块使用的参数
- `_dist_allgather_fn()`（`partition_parameters.py:108`）：all-gather 实现
- `NoGatherHandle` / `NoGatherCoalescedHandle`：不收集/合并收集 handle

### 3.4 ZeRO 对比表（vs Megatron DistributedOptimizer vs torchtitan FSDP）

| 特性 | DeepSpeed ZeRO | Megatron DistributedOptimizer | torchtitan FSDP |
|------|---------------|------------------------------|-----------------|
| **分片粒度** | 参数组级 flatten | 参数级 | 参数级 |
| **通信原语** | allreduce / reduce-scatter | reduce-scatter | all-gather + reduce-scatter |
| **参数重聚** | all-gather on demand | all-gather on demand | all-gather on demand |
| **Offload** | CPU / NVME | 无 | 无 |
| **Master Weights** | FP32 flat buffer | FP32 per-param | 同参数精度 |
| **梯度累积** | `grad_accum` attribute | `grad_accum` buffer | 原生支持 |
| **实现复杂度** | 高（手动 hook） | 中（手动 hook） | 低（PyTorch 原生） |
| **代码位置** | `runtime/zero/stage_1_and_2.py` | `megatron/core/distributed/` | `torch/distributed/fsdp` |

---

## 4. Pipeline Parallel

### 4.1 PipelineEngine

`PipelineEngine`（`runtime/pipe/engine.py:60`）继承 `DeepSpeedEngine`，专用于流水线并行训练。

**初始化关键步骤**：

```
PipelineEngine.__init__()                           # pipe/engine.py:72
  ├── super().__init__()                            # DeepSpeedEngine
  ├── assert isinstance(model, PipelineModule)      # pipe/engine.py:74
  ├── enable_backward_allreduce = False             # pipe/engine.py:80 — PP 自己管理 reduce
  ├── grid = module._grid                           # pipe/engine.py:102 — 获取 PP layout
  ├── p2p.init_process_groups(grid)                 # pipe/engine.py:174 — P2P 通信组
  └── _build_data_iter()                            # pipe/engine.py:264 — 数据迭代器
```

**train_batch() 调用链**：

```
PipelineEngine.train_batch(data_iter)               # pipe/engine.py:337
  ├── module.train()
  ├── TrainSchedule(micro_batches, stages, stage_id)    # schedule.py
  │     └── steps() 生成 1F1B 指令序列
  └── _exec_schedule(sched)                         # 执行 schedule
        ├── ForwardPass(buffer_id)                  # 前向
        │     └── forward_step() → module forward
        ├── BackwardPass(buffer_id)                 # 反向
        │     └── backward_step() → loss.backward()
        ├── ReduceGrads()                           # 梯度 reduce
        │     └── _exec_reduce_grads()
        └── OptimizerStep()                         # 参数更新
              └── self.step()
```

### 4.2 1F1B Schedule

1F1B（One Forward One Backward）是 DeepSpeed 的默认流水线调度：

```
# TrainSchedule 在 runtime/pipe/schedule.py 中定义
# 核心逻辑：交替执行 forward 和 backward，保持 pipeline 充满

时间线 (4 stages, 8 micro-batches):
Stage 0: F0 F1 F2 F3 | B0 F4 B1 F5 B2 F6 B3 F7 B4 B5 B6 B7
Stage 1: .. F0 F1 F2 | B0 F3 B1 F4 B2 F5 B3 F6 B4 F7 B5 B6 B7
Stage 2: .. .. F0 F1 | B0 F2 B1 F3 B2 F4 B3 F5 B4 F6 B5 F7 B6 B7
Stage 3: .. .. .. F0 | B0 F1 B1 F2 B2 F3 B3 F4 B4 F5 B5 F6 F7 B7

F = Forward, B = Backward
```

### 4.3 与 Megatron PP 对比

| 特性 | DeepSpeed PP | Megatron PP |
|------|-------------|-------------|
| **Schedule** | 1F1B (TrainSchedule) | 1F1B / Interleaved 1F1B |
| **通信** | 自定义 p2p | torch.distributed send/recv |
| **梯度 reduce** | 手动控制 | 手动控制 |
| **兼容性** | ZeRO-1 only | 全 ZeRO stages |
| **代码位置** | `runtime/pipe/engine.py` | `megatron/core/pipeline_parallel/` |

---

## 5. MoE 支持

### 5.1 DeepSpeed MoE 架构

DeepSpeed 提供两种 MoE 实现：原生 `MoE` 类和 AutoEP（自动 Expert Parallel）。

**MoE → MOELayer → TopKGate → Experts 调用链**：

```
MoE.__init__()                                      # moe/layer.py:38
  ├── Experts(expert, num_local_experts)            # moe/layer.py:72 — 本地专家
  ├── TopKGate(hidden_size, num_experts, k)         # moe/layer.py:73 — 门控网络
  └── MOELayer(gate, experts, ep_group_name)        # moe/layer.py:73 — MoE 层

MoE.forward(hidden_states)                          # moe/layer.py:105
  └── self.deepspeed_moe(hidden_states)             # moe/layer.py:123
        └── MOELayer.forward()                      # moe/sharded_moe.py
              ├── gate(hidden_states) → topkgating() # sharded_moe.py:382
              │     ├── F.softmax(logits)           # 计算门控权重
              │     ├── torch.topk(gates, k)        # 选择 top-k 专家
              │     └── 计算 l_aux (负载均衡 loss)
              ├── _AllToAll.apply()                 # 分发 token 到专家
              ├── experts(hidden_states)            # 专家计算
              └── _AllToAll.apply()                 # 收集结果
```

**TopKGate**（`moe/sharded_moe.py:474`）：
- `wg`：门控线性层 (`nn.Linear`)
- 支持 `top1gating`、`top2gating`、`topkgating` 三种门控函数

**门控函数对比**：

| 函数 | 文件:行号 | 采样方式 | 负载均衡 |
|------|-----------|----------|----------|
| `top1gating` | `sharded_moe.py:184` | argmax + Random Token Selection | `l_aux = Σ(me × ce) × num_experts` |
| `top2gating` | `sharded_moe.py:291` | top-1 + Gumbel-max 2nd | `l_aux = mean(me × ce) × E²` |
| `topkgating` | `sharded_moe.py:382` | torch.topk + capacity drop | `l_aux = mean(me × ce) × E² / k` |

**AllToAll 通信**（`sharded_moe.py:97`）：
- 自定义 `torch.autograd.Function`，forward/backward 均执行 `dist.all_to_all_single`
- 实现 token 在 expert parallel group 间的分发与收集

### 5.2 TokenChoiceTopKRouter

`TokenChoiceTopKRouter`（`moe/ep_router.py:27`）是从 TorchTitan 移植的 Token-choice 路由器，支持 **node-limited routing**（分组限制路由）。

**核心特性**：
- `gate = nn.Linear(dim, num_experts)`（`ep_router.py:65`）：门控投影
- 支持 `softmax` / `sigmoid` 评分函数（`ep_router.py:157-162`）
- `_get_node_limited_routing_scores()`（`ep_router.py:82`）：先选 top groups，再在组内选 top-k experts
- `e_score_correction_bias`（`ep_router.py:76`）：可训练的专家分数校正（DeepSeek-V3 noaux_tc 风格）

**调用链**：

```
TokenChoiceTopKRouter.forward(x, expert_bias)       # ep_router.py:136
  ├── scores = self.gate(x)                         # ep_router.py:154 — (T, num_experts)
  ├── sigmoid/softmax 评分                           # ep_router.py:157-162
  ├── expert_bias 叠加                               # ep_router.py:164
  ├── e_score_correction_bias 校正                   # ep_router.py:167-168
  ├── _get_node_limited_routing_scores()             # ep_router.py:171-172 — 分组限制
  │     ├── group_scores (max / top2_sum)            # ep_router.py:109-118
  │     ├── topk(group_scores, num_limited_groups)   # ep_router.py:121
  │     └── masked_fill(-inf) for non-selected groups
  ├── topk(scores_for_choice, top_k)                # ep_router.py:175 — 选择专家
  ├── route_norm 归一化                              # ep_router.py:181-183
  └── count_tokens_per_expert()                      # ep_router.py:187 — 统计 token 分布
```

### 5.3 与 Megatron MoE 对比表

| 特性 | DeepSpeed MoE | Megatron MoE |
|------|---------------|--------------|
| **门控类型** | TopKGate (top1/2/k) | TopKGate (top1/2/k) |
| **路由实现** | Token-choice + group limited | Token-choice / Expert-choice |
| **AllToAll** | 自定义 `_AllToAll` | `MoEAlltoAllTokenDispatcher` |
| **Expert Parallel** | AutoEP / native EP | 原生 EP |
| **负载均衡** | `l_aux` + `expert_bias` | `l_aux` + `z_loss` |
| **代码位置** | `moe/layer.py`, `moe/sharded_moe.py` | `megatron/core/transformer/moe/` |

---

## 6. Autotuning

DeepSpeed Autotuning 通过自动实验寻找最优配置（ZeRO stage、micro-batch size 等）。

**核心类**：`Autotuner`（`autotuning/autotuner.py:42`）

**工作流程**：

```
Autotuner.__init__(args, active_resources)          # autotuner.py:47
  ├── _get_user_config(args.user_args)              # autotuner.py:158 — 解析 DS 配置
  ├── DeepSpeedAutotuningConfig(user_config)        # autotuner.py:58 — 调优配置
  ├── _get_resource_manager(active_resources)       # autotuner.py:188 — GPU 资源
  └── _get_exp_resources(args)                      # autotuner.py:95 — 实验资源分配

# 调优 tuner 类型（autotuning/tuner/）：
#   - GridSearchTuner: 网格搜索
#   - RandomTuner: 随机搜索
#   - ModelBasedTuner: 模型-based 搜索
```

**关键配置项**（`autotuning/config.py`）：
- `enabled`：启用 autotuning
- `exps_dir`：实验结果目录
- `results_dir`：最终结果目录
- `metric`：优化指标（如 `throughput`）
- `model_info`：模型信息 profiling

**与 Megatron 手动调优对比**：DeepSpeed Autotuning 自动搜索配置空间，减少人工试错；Megatron 依赖经验调优。

---

## 7. Inference Engine

`InferenceEngine`（`inference/engine.py:40`）是 DeepSpeed 的推理引擎。

**核心架构**：

```
InferenceEngine.__init__(model, config)             # inference/engine.py:45
  ├── 精度转换 → _convert_to_dtype(config)           # inference/engine.py:113
  ├── TP group 创建 → _create_model_parallel_group   # inference/engine.py:119
  ├── EP group 创建 → _create_ep_parallel_group      # inference/engine.py:128
  ├── Kernel Injection:
  │     ├── replace_transformer_layer()              # 替换为 DS 优化层
  │     ├── DeepSpeedSelfAttention                  # 自定义 attention
  │     └── DeepSpeedTransformerInference            # Transformer 推理层
  ├── CUDA Graph 支持                                # inference/engine.py:107
  └── Weight Quantization                            # inference/engine.py:92
```

**关键特性**：
- **Kernel Injection**：替换 HF 模型 Linear/Normalize 为 `LinearAllreduce`、`LinearLayer`
- **Tensor Parallelism**：推理时 TP 支持
- **Expert Parallelism**：MoE 推理 EP 支持
- **Weight Quantization**：推理权重量化
- **CUDA Graph**：减少 kernel launch 开销

**与 vLLM/SGLang 对比**：DeepSpeed Inference 更偏向"训练引擎的推理模式"，而 vLLM/SGLang 是专用推理引擎，Continuous Batching 更成熟。

---

## 8. Ops（算子层）

### 8.1 CPUAdam / FusedAdam

DeepSpeed 提供两种 Adam 实现，分别针对 CPU offload 和 GPU 场景。

**DeepSpeedCPUAdam**（`ops/adam/cpu_adam.py:13`）：

```
DeepSpeedCPUAdam.__init__(model_params, ...)        # cpu_adam.py:16
  ├── CPU 厂商检测 (AMD/Intel)                       # cpu_adam.py:75-85
  ├── CPUAdamBuilder().load()                        # cpu_adam.py:91 — 加载 C++ 扩展
  └── ds_opt_adam.create_adam(opt_id, lr, betas, eps, weight_decay, adamw_mode)
                                                    # cpu_adam.py:93

DeepSpeedCPUAdam.step()                             # cpu_adam.py:107
  └── ds_opt_adam.adam_update(...)                   # cpu_adam.py:160 — C++ 内核
```

**FusedAdam**（`ops/adam/fused_adam.py:18`）：

```
FusedAdam.__init__(params, ...)                     # fused_adam.py:76
  ├── FusedAdamBuilder().load()                      # fused_adam.py:94 — 加载 CUDA 扩展
  └── multi_tensor_adam = fused_adam_cuda.multi_tensor_adam  # fused_adam.py:97

FusedAdam.step()                                    # fused_adam.py:107
  └── 按 dtype 分组 (fp16/bf16/fp32)                 # fused_adam.py:136-138
      └── multi_tensor_adam(...)                     # 单 kernel 多参数更新
```

**对比表**：

| 特性 | DeepSpeedCPUAdam | FusedAdam |
|------|------------------|-----------|
| **运行设备** | CPU | GPU |
| **实现** | C++ 内核 | CUDA 内核 |
| **加速比** | 5-7× vs torch AdamW | 融合 elementwise ops |
| **多 tensor** | 逐参数 | MultiTensorApply 批处理 |
| **使用场景** | ZeRO-Offload | ZeRO-1/2 标准训练 |

### 8.2 Transformer Ops

DeepSpeed 提供优化的 Transformer 算子（`ops/transformer/`）：

- `DeepSpeedSelfAttention`： fused attention kernel
- `DeepSpeedTransformerInference`：推理优化 Transformer 层
- Kernel Injection 通过 `replace_transformer_layer` 替换原生层

---

## 9. 关键配置参数表

### ZeRO 配置

| 参数名 (JSON key) | 代码位置 | 类型 | 默认值 | 说明 |
|-------------------|----------|------|--------|------|
| `zero_optimization.stage` | `engine.py:1154` | int | 0 | ZeRO 阶段 (0/1/2/3) |
| `zero_optimization.offload_optimizer` | `engine.py:1126` | dict | None | 优化器 offload 配置 |
| `zero_optimization.offload_param` | `engine.py:1129` | dict | None | 参数 offload 配置 |
| `zero_optimization.contiguous_gradients` | `engine.py:1190` | bool | True | 梯度连续存储 |
| `zero_optimization.overlap_comm` | `engine.py:1123` | bool | False | 通信计算重叠 |
| `zero_optimization.reduce_scatter` | `engine.py:1120` | bool | True | 使用 reduce-scatter |
| `zero_optimization.reduce_bucket_size` | `engine.py:1168` | int | 500M | reduce bucket 大小 |
| `zero_optimization.allgather_bucket_size` | `engine.py:1174` | int | 5B | all-gather bucket 大小 |
| `zero_optimization.max_live_parameters` | `engine.py:1202` | int | 1B | ZeRO-3 最大活跃参数 |
| `zero_optimization.max_reuse_distance` | `engine.py:1205` | int | 1B | ZeRO-3 参数复用距离 |
| `zero_optimization.prefetch_bucket_size` | `engine.py:1208` | int | 50M | ZeRO-3 prefetch 大小 |
| `zero_optimization.param_persistence_threshold` | `engine.py:1214` | int | 100K | 参数常驻阈值 |
| `zero_optimization.round_robin_gradients` | `engine.py:1323` | bool | False | 梯度轮询分区 |

### 优化器配置

| 参数名 (JSON key) | 代码位置 | 类型 | 说明 |
|-------------------|----------|------|------|
| `optimizer.type` | `engine.py:1082` | str | 优化器类型 (Adam/AdamW/Lamb/...) |
| `optimizer.params.lr` | `engine.py:1085` | float | 学习率 |
| `optimizer.params.betas` | `engine.py:1085` | tuple | Adam betas |
| `optimizer.params.eps` | `engine.py:1085` | float | Adam epsilon |
| `optimizer.params.weight_decay` | `engine.py:1085` | float | 权重衰减 |
| `optimizer.params.adam_w_mode` | `engine.py:93` | bool | AdamW 模式 (默认 True) |

### FP16/BF16 配置

| 参数名 (JSON key) | 代码位置 | 类型 | 说明 |
|-------------------|----------|------|------|
| `fp16.enabled` | `engine.py:1244` | bool | 启用 FP16 |
| `fp16.loss_scale` | `engine.py:1278` | float | 静态 loss scale (0=动态) |
| `fp16.initial_dynamic_scale` | `engine.py:1360` | float | 初始动态 scale |
| `fp16.dynamic_loss_scale_args` | `engine.py:1363` | dict | 动态 scale 参数 |
| `bf16.enabled` | `engine.py:1248` | bool | 启用 BF16 |

### Pipeline Parallel 配置

| 参数名 (JSON key) | 代码位置 | 类型 | 说明 |
|-------------------|----------|------|------|
| `pipeline.activation_checkpoint_interval` | `engine.py:211` | int | activation checkpoint 间隔 |
| `pipeline.pipe_partitioned` | `engine.py:143` | bool | pipe 分区 |
| `pipeline.grad_partitioned` | `engine.py:144` | bool | 梯度分区 |

### 其他关键配置

| 参数名 (JSON key) | 代码位置 | 类型 | 说明 |
|-------------------|----------|------|------|
| `train_batch_size` | `engine.py:1076` | int | 全局 batch size |
| `train_micro_batch_size_per_gpu` | `engine.py:1079` | int | 每 GPU micro batch |
| `gradient_accumulation_steps` | `engine.py:1282` | int | 梯度累积步数 |
| `gradient_clipping` | `engine.py:1354` | float | 梯度裁剪阈值 |
| `steps_per_print` | `engine.py:1318` | int | 打印频率 |
| `wall_clock_breakdown` | `engine.py:1011` | bool | 详细计时 |
| `memory_breakdown` | `engine.py:1041` | bool | 内存分解 |

---

## 10. DeepSpeed vs Megatron vs torchtitan 总对比表

| 维度 | DeepSpeed | Megatron-LM | torchtitan |
|------|-----------|-------------|------------|
| **核心定位** | 数据并行 + ZeRO 状态分片 | 模型并行 (TP/PP) 优先 | PyTorch 原生 FSDP2 |
| **ZeRO 实现** | 手动实现 ZeRO-1/2/3 | DistributedOptimizer (类似 Z2) | PyTorch FSDP2 原生 |
| **参数分片** | `ds_tensor` + flat buffer | per-param buffer | per-param shard |
| **参数重聚** | `all_gather()` on demand | `all_gather()` on demand | `all_gather()` on demand |
| **Pipeline** | `PipelineEngine` + 1F1B | `PipelineSchedule` (1F1B/Interleaved) | 无原生 PP |
| **MoE** | `MoE` + AutoEP + `TokenChoiceTopKRouter` | `MoE` + `MoEAlltoAllTokenDispatcher` | 无原生 MoE |
| **Autotuning** | 内置 Autotuner | 无 | 无 |
| **Inference** | `InferenceEngine` (kernel injection) | 无 | 无 |
| **Offload** | CPU / NVME | 无 | CPU (FSDP offload) |
| **混合精度** | FP16/BF16 + loss scaling | FP16/BF16 + loss scaling | 原生 torch AMP |
| **Checkpoint** | 自定义 Checkpoint Engine | Distributed Checkpoint | PyTorch DCP |
| **代码侵入** | 低 (JSON 配置) | 中 (需要适配 Megatron 模型) | 低 (PyTorch 原生) |
| **通信原语** | `deepspeed.comm as dist` | `torch.distributed` | `torch.distributed` |
| **适用场景** | 超大模型训练 (ZeRO-3) | 大规模并行 (TP+PP) | 中等规模 FSDP |
| **代码位置** | `deepspeed/` | `megatron/` | `torchtitan/` |

---

## 附录：关键代码位置索引

| 功能 | 文件路径 | 行号 |
|------|----------|------|
| DeepSpeedEngine 类定义 | `runtime/engine.py` | 235 |
| Engine `__init__` | `runtime/engine.py` | 238 |
| Engine `forward` | `runtime/engine.py` | 2676 |
| Engine `backward` | `runtime/engine.py` | 3067 |
| Engine `step` | `runtime/engine.py` | 3242 |
| _configure_zero_optimizer | `runtime/engine.py` | 2235 |
| DeepSpeedConfig | `runtime/config.py` | 100 |
| DEEPSPEED_OPTIMIZERS | `runtime/config.py` | 84 |
| ZeRO-1/2 Optimizer 类定义 | `runtime/zero/stage_1_and_2.py` | 134 |
| ZeRO-1/2 `__init__` | `runtime/zero/stage_1_and_2.py` | 146 |
| IPGBucket | `runtime/zero/stage_1_and_2.py` | 113 |
| reduce_gradients | `runtime/zero/stage_1_and_2.py` | 840 |
| independent_gradient_partition_epilogue | `runtime/zero/stage_1_and_2.py` | 898 |
| reduce_ipg_grads | `runtime/zero/stage_1_and_2.py` | 1615 |
| average_tensor | `runtime/zero/stage_1_and_2.py` | 1277 |
| ZeRO-3 Optimizer 类定义 | `runtime/zero/stage3.py` | 148 |
| ZeRO-3 `__init__` | `runtime/zero/stage3.py` | 160 |
| IPGBucketZ3 | `runtime/zero/stage3.py` | 124 |
| DeepSpeedZeRoOffload 类定义 | `runtime/zero/parameter_offload.py` | 117 |
| ZeROOrderedDict | `runtime/zero/parameter_offload.py` | 40 |
| PartitionedParameterCoordinator | `runtime/zero/partition_parameters.py` | (引用) |
| register_external_parameter | `runtime/zero/partition_parameters.py` | 142 |
| _dist_allgather_fn | `runtime/zero/partition_parameters.py` | 108 |
| NoGatherHandle | `runtime/zero/partition_parameters.py` | 58 |
| NoGatherCoalescedHandle | `runtime/zero/partition_parameters.py` | 78 |
| PipelineEngine 类定义 | `runtime/pipe/engine.py` | 60 |
| PipelineEngine `train_batch` | `runtime/pipe/engine.py` | 337 |
| PipeSchedule 基类 | `runtime/pipe/schedule.py` | 11 |
| InferenceSchedule | `runtime/pipe/schedule.py` | 135 |
| MoE 类定义 | `moe/layer.py` | 17 |
| MoE `forward` | `moe/layer.py` | 105 |
| MOELayer | `moe/sharded_moe.py` | (类定义) |
| TopKGate | `moe/sharded_moe.py` | 474 |
| top1gating | `moe/sharded_moe.py` | 184 |
| top2gating | `moe/sharded_moe.py` | 291 |
| topkgating | `moe/sharded_moe.py` | 382 |
| _AllToAll (autograd Function) | `moe/sharded_moe.py` | 97 |
| TokenChoiceTopKRouter | `moe/ep_router.py` | 27 |
| TokenChoiceTopKRouter `forward` | `moe/ep_router.py` | 136 |
| _get_node_limited_routing_scores | `moe/ep_router.py` | 82 |
| Autotuner 类定义 | `autotuning/autotuner.py` | 42 |
| InferenceEngine 类定义 | `inference/engine.py` | 40 |
| DeepSpeedCPUAdam 类定义 | `ops/adam/cpu_adam.py` | 13 |
| DeepSpeedCPUAdam `step` | `ops/adam/cpu_adam.py` | 107 |
| FusedAdam 类定义 | `ops/adam/fused_adam.py` | 18 |
| FusedAdam `step` | `ops/adam/fused_adam.py` | 107 |
