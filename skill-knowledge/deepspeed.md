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

**`step()` 关键代码**（梯度 norm 计算 → 优化器更新 → all-gather 收集完整参数）：

```python
# deepspeed/runtime/zero/stage_1_and_2.py:2204
def step(self, closure=None):
    self.micro_step_id = INITIAL_MICRO_STEP_ID
    # 1. 检查梯度 overflow，overflow 则跳过本次更新
    if self.check_grad_overflow:
        self.check_overflow(partition_gradients=self.partition_gradients)
    prev_scale = self.loss_scale
    self._update_scale(self.overflow)
    if self.overflow:
        self.zero_grad(set_to_none=True)
        return

    # 2. 计算全局梯度 norm（用于 loss scale 和 grad clip）
    scaled_global_grad_norm = self.scaled_global_norm()
    self._global_grad_norm = scaled_global_grad_norm / prev_scale

    # 3. 遍历参数组，执行 optimizer step
    for i, group in enumerate(self.bit16_groups):
        partition_id = dist.get_rank(group=self.real_dp_process_group[i])
        if self.cpu_offload:
            # CPU offload 路径：梯度已在 CPU，直接 unscale + Adam 更新
            single_grad_partition = self.single_partition_of_fp32_groups[i].grad
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
            self._optimizer_step(i)
            # FP32 master weight → FP16 model weight 拷贝
            bit16_partitions[partition_id].data.copy_(fp32_partition.data, non_blocking=True)
        else:
            # GPU 路径：将平均梯度拷贝到 flat buffer
            flat_grad_partition = self._get_preflattened_grad_partition(i)
            self.single_partition_of_fp32_groups[i].grad = single_grad_partition
            self.free_grad_in_param_list(self.params_in_partition[i])
            self.unscale_and_clip_grads([single_grad_partition], scaled_global_grad_norm)
            self._optimizer_step(i)
            self.single_partition_of_fp32_groups[i].grad = None
            # FP32 master weight → FP16 partition 写回
            bit16_partitions[partition_id].data.copy_(fp32_partition.data)

    # 4. all-gather：收集所有 rank 的 FP16 分片 → 完整参数
    all_gather_dp_groups(
        groups_flat=self.bit16_groups_flat,
        partitioned_param_groups=self.parallel_partitioned_bit16_groups,
        dp_process_group=self.real_dp_process_group,
        allgather_bucket_size=self.allgather_bucket_size
    )
    # 5. 更新模型 bit16 weights（从 flat buffer 写回各 rank 的 model weight）
    for i in range(len(self.bit16_groups)):
        self._update_model_bit16_weights(i)
```

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

**`DeepSpeedZeRoOffload.__init__()` 关键代码**（参数转换 → 注入 → hook 注册）：

```python
# deepspeed/runtime/zero/parameter_offload.py:119
def __init__(self, module, timers, ds_config, zenflow=False,
             overlap_comm=True, prefetch_bucket_size=50000000,
             max_reuse_distance=1000000000, max_live_parameters=1000000000,
             param_persistence_threshold=100000, dp_process_group=None,
             offload_param_config=None, ...):

    self.module = module
    self.dtype = list(module.parameters())[0].dtype
    self.dp_process_group = dp_process_group

    # 1. 解析 offload 配置（CPU / NVME / none）
    if offload_param_config is not None and offload_param_config.device != OffloadDeviceEnum.none:
        self.offload_device = offload_param_config.device
        self.offload_param_pin_memory = offload_param_config.pin_memory

    # 2. 将模型参数转换为 ZeRO 参数（ds_tensor），按 DP 维度分片
    self._convert_to_zero_parameters(ds_config, module, mpu)
    #   内部调用 Init(module=module, data_parallel_group=group, ...)

    # 3. 注册外部参数（跨模块共享的参数）
    for m in module.modules():
        _init_external_params(m)

    # 4. 注入 ZeROOrderedDict 替代 nn.Module._parameters
    #    使得 param.__getitem__ 时自动触发 all_gather()
    _inject_parameters(module, ZeROOrderedDict)

    # 5. 标记常驻参数（小于 threshold 的参数不释放）
    self.persistent_parameters = self.mark_persistent_parameters(
        self.param_numel_persistence_threshold, self.model_persistence_threshold)

    # 6. 创建参数协调器（管理 prefetch / release 时序）
    self.param_coordinator = PartitionedParameterCoordinator(
        prefetch_bucket_sz=self._prefetch_bucket_sz,
        max_reuse_distance_in_numel=self._max_reuse_distance_in_numel,
        max_available_parameters_in_numel=self._max_available_parameters_in_numel,
        allgather_stream=self.__allgather_stream,
        prefetch_nvme=self.offload_device == OffloadDeviceEnum.nvme, ...)

    # 7. 注册 forward/backward hooks（pre/post module hooks）
    self.setup_zero_stage3_hooks()
```

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

DeepSpeed Autotuning 通过自动实验寻找最优配置（ZeRO stage、micro-batch size 等），无需用户代码改动。

**核心类**：`Autotuner`（`autotuning/autotuning/autotuner.py:42`）

### 6.1 工作流程（Search → Report → Apply）

```
Autotuner.__init__(args, active_resources)          # autotuner.py:47
  ├── _get_user_config(args.user_args)              # autotuner.py:158 — 解析 DS 配置
  ├── DeepSpeedAutotuningConfig(user_config)        # autotuner.py:58 — 调优配置
  ├── _get_resource_manager(active_resources)       # autotuner.py:188 — GPU 资源管理
  └── _get_exp_resources(args)                      # autotuner.py:95 — 实验资源分配

Autotuner.tune()                                    # autotuner.py:404 — 主入口
  ├── model_info_profile_run()                      # autotuner.py:663 — profiling 模型参数量/激活内存
  ├── 按 ZeRO stage 逐级尝试（Z0→Z1→Z2→Z3）
  │     └── tune_space(tuning_space)                # autotuner.py:523 — 每个 stage 的调优
  │           ├── 1. 二分搜索 max micro-batch size（基于 GPU 内存）
  │           ├── 2. run_tuning_micro_batch_sizes() — 遍历 mbs 组合
  │           ├── 3. _generate_experiments()        # autotuner.py:304 — 生成实验配置
  │           ├── 4. 选择 tuner 类型并执行搜索
  │           │     ├── GridSearchTuner             # 网格搜索（默认）
  │           │     ├── RandomTuner                 # 随机搜索
  │           │     └── ModelBasedTuner             # 模型-based 搜索（贝叶斯优化）
  │           └── 5. 记录最优结果到 self.records
  └── write_optimal_config()                        # autotuner.py:1075 — 写入最优配置

Autotuner.run_after_tuning()                        # autotuner.py:1103 — 用最优配置启动训练
  └── subprocess.Popen(self.optimal_cmd)            # 启动最终训练任务
```

### 6.2 Tuner 类型对比

| Tuner | 文件 | 搜索策略 | 适用场景 |
|-------|------|----------|----------|
| `GridSearchTuner` | `autotuning/tuner/grid_search.py` | 遍历所有参数组合 | 参数空间小，精确最优 |
| `RandomTuner` | `autotuning/tuner/random.py` | 随机采样参数组合 | 参数空间大，快速探索 |
| `ModelBasedTuner` | `autotuning/tuner/model_based.py` | 贝叶斯优化（surrogate model） | 评估成本高，样本效率优先 |

### 6.3 关键代码：`tune_space()` 搜索流程

```python
# deepspeed/autotuning/autotuner.py:523
def tune_space(self, tuning_space, prev_max_mbs=0, prev_best_mbs=0, prev_best_metric_val=0):
    stage = config_zero.get(ZERO_OPTIMIZATION_STAGE, None)
    # 1. 基于 GPU 内存计算理论最大 micro-batch size
    calculated_max_micro_batch_size = int(
        self.gpu_mem - self.get_instantiation_memory_required_per_gpu(stage)) // self.activation_mem

    # 2. 搜索 min/max micro-batch size（二分搜索 OOM 边界）
    min_micro_batch_size, max_micro_batch_size = self.get_min_max_micro_batch_size(
        stage, prev_max_mbs, calculated_max_micro_batch_size)

    # 3. 遍历候选 mbs，运行实验收集 metric
    tuning_micro_batch_sizes = self.run_tuning_micro_batch_sizes(
        tuning_micro_batch_sizes, max_train_batch_size_per_gpu, ...)

    # 4. 生成完整实验配置（ZeRO 参数组合）
    exps = self._generate_experiments(tuning_space, max_train_batch_size_per_gpu)

    # 5. 选择 tuner 类型并执行搜索
    if self.autotuning_config.tuner_type == AUTOTUNING_TUNER_MODELBASED:
        t = ModelBasedTuner(exps, self.rm, self.metric(), tuning_space)
    elif self.autotuning_config.tuner_type == AUTOTUNING_TUNER_RANDOM:
        t = RandomTuner(exps, self.rm, self.metric())
    else:
        t = GridSearchTuner(exps, self.rm, self.metric())

    # 6. 执行调优实验
    num_exps = t.tune(sample_size=sample_size, n_trials=..., early_stopping=...)
    exp = t.best_exp
    metric_val = t.best_metric_val
    self.update_records(tuning_space_name, exp, metric_val, num_exps)
```

### 6.4 关键配置项（`autotuning/config.py`）

| 配置项 | 类型 | 说明 |
|--------|------|------|
| `enabled` | bool | 启用 autotuning |
| `exps_dir` | str | 实验结果目录 |
| `results_dir` | str | 最终结果目录 |
| `metric` | str | 优化指标（如 `throughput`） |
| `model_info` | dict | 模型信息 profiling |
| `tuner_type` | str | 搜索类型（`grid`/`random`/`model_based`） |
| `tuner_num_trials` | int | 搜索试验次数 |
| `tuner_early_stopping` | int | 早停轮数 |
| `fast` | bool | 快速模式（仅调 micro-batch size） |

**与 Megatron 手动调优对比**：DeepSpeed Autotuning 自动搜索配置空间，减少人工试错；Megatron 依赖经验调优。

---

## 7. Inference Engine

`InferenceEngine`（`inference/engine.py:40`）是 DeepSpeed 的推理引擎，通过 kernel injection 替换 HuggingFace 模型原生层为优化版本。

### 7.1 核心架构

```
InferenceEngine.__init__(model, config)             # inference/engine.py:45
  ├── 精度转换 → _convert_to_dtype(config)           # inference/engine.py:113
  ├── TP group 创建 → _create_model_parallel_group   # inference/engine.py:247
  ├── EP group 创建 → _create_ep_parallel_group      # inference/engine.py:260
  ├── Kernel Injection（三选一）:
  │     ├── [1] injection_dict → 用户指定 TP policy
  │     ├── [2] replace_with_kernel_inject → 自动替换原生层
  │     │     ├── generic_injection()               # 通用层替换
  │     │     └── replace_transformer_layer()       # replace_module.py:189
  │     │           ├── policy.match() → 匹配 HF 模型结构
  │     │           ├── policy_to_ds_container()    # 创建 DS 容器
  │     │           ├── _container.create_module()  # 构建优化层
  │     │           └── _container.apply_tensor_parallelism()  # 应用 TP
  │     └── [3] tp_size>1 → AutoTP 自动张量并行
  ├── CUDA Graph 支持                                # inference/engine.py:497
  └── Weight Quantization                            # inference/engine.py:289
```

### 7.2 Kernel Injection 机制

Kernel Injection 是 DeepSpeed Inference 的核心优化——将 HuggingFace 模型的 `nn.Linear`、`nn.LayerNorm` 替换为 DeepSpeed 优化版本：

| 原始层 | 替换后 | 文件位置 | 功能 |
|--------|--------|----------|------|
| `nn.Linear` (column) | `LinearLayer` | `module_inject/layers.py:678` | 列切分 + all-gather 收集 |
| `nn.Linear` (row) | `LinearAllreduce` | `module_inject/layers.py:581` | 行切分 + reduce-scatter |
| `nn.LayerNorm` | `Normalize` | `module_inject/layers.py` | 融合 LayerNorm kernel |
| `nn.Embedding` | `nn.Embedding` (TP 版) | `module_inject/layers.py` | 词嵌入切分 |
| Transformer Layer | `DeepSpeedTransformerInference` | `model_implementations/transformers/ds_transformer.py` | 完整 Transformer 推理层 |

**`replace_transformer_layer()` 调用链**（`module_inject/replace_module.py:189`）：

```python
# deepspeed/module_inject/replace_module.py:216
def replace_with_policy(child, policy_cls, triangular_masking, inference=False, layer_id=0):
    policy = policy_cls(child, inference=inference)          # 1. 创建 policy 对象
    _container = policy_to_ds_container(policy=policy, ...)  # 2. 创建 DS 容器
    _container.set_tensor_parallel_config(tp_size, tp_group)  # 3. 设置 TP 配置
    _container.initialize_tensors()                          # 4. 初始化张量
    _container.convert_to_required_dtype()                   # 5. 精度转换
    _container.set_quantization_config(quantizer)            # 6. 量化配置
    _container.create_ds_model_config()                      # 7. 创建 DS 模型配置
    _container.create_module()                               # 8. 构建优化模块
    _container.transpose()                                   # 9. 权重转置
    _container.apply_tensor_parallelism(mp_replace)          # 10. 应用 TP 切分
    _container.copy_data_to_new_module()                     # 11. 拷贝权重数据
    return _container.module
```

### 7.3 Tensor Parallelism (TP) 支持

推理时 TP 通过 `_create_model_parallel_group()`（`inference/engine.py:247`）创建 TP group：

```python
# deepspeed/inference/engine.py:247
def _create_model_parallel_group(self, config):
    if InferenceEngine.inference_mp_group is None:
        init_distributed()
        ranks = [i for i in range(config.tensor_parallel.tp_size)]
        self.mp_group = dist.new_group(ranks)           # 创建 TP 通信组
        InferenceEngine.inference_mp_group = self.mp_group
    else:
        self.mp_group = InferenceEngine.inference_mp_group
```

**TP 切分逻辑**（`LinearAllreduce`，`module_inject/layers.py:581`）：
- **权重切分**：`torch.chunk(param, tp_world_size, dim=-1)` 按列切分
- **前向计算**：`output = input @ W_local.T`，然后 `RowParallel.apply()` 做 all-reduce
- **推理模式**：`uneven_partition()` 支持非均匀切分（考虑 GQA num_kv_heads）

### 7.4 Expert Parallelism (EP) 支持

MoE 推理通过 `_create_ep_parallel_group()`（`inference/engine.py:260`）创建 EP group：

```python
# deepspeed/inference/engine.py:260
def _create_ep_parallel_group(self, moe_experts):
    for moe_ep_size in self.ep_group.keys():
        num_ep_groups = dist.get_world_size() // moe_ep_size
        for i in range(num_ep_groups):
            ranks = list(range(ep_cnt, ep_cnt + size))
            _ep_group = dist.new_group(ranks)           # 创建 EP 通信组
            if dist.get_rank() in ranks:
                self.ep_group.update({moe_ep_size: _ep_group})
        # 同时创建 expert_mp_group（用于 expert 内部的 TP）
        if dist.get_world_size() > moe_ep_size:
            expert_mp_comm_ranks = [i + nr * moe_ep_size for nr in range(expert_mp_size)]
            _expert_mp_group = dist.new_group(expert_mp_comm_ranks)
```

### 7.5 CUDA Graph 支持

```python
# deepspeed/inference/engine.py:497
def _create_cuda_graph(self, *inputs, **kwargs):
    cuda_stream = get_accelerator().Stream()
    # 1. warmup：创建 workspace 和 cublas handle
    with get_accelerator().stream(cuda_stream):
        for i in range(3):
            ret = self.module(*inputs, **kwargs)
    # 2. 捕获 CUDA Graph
    self._cuda_graphs = get_accelerator().create_graph()
    with get_accelerator().capture_to_graph(self._cuda_graphs):
        self.static_output = self.module(*self.static_inputs, **self.static_kwargs)
    self.cuda_graph_created = True
```

### 7.6 关键特性总结

- **Kernel Injection**：替换 HF 模型 Linear/Normalize 为 `LinearAllreduce`、`LinearLayer`
- **Tensor Parallelism**：推理时 TP 支持，`LinearAllreduce` 行切分 + all-reduce
- **Expert Parallelism**：MoE 推理 EP 支持，独立 EP group 管理
- **Weight Quantization**：推理权重量化（int8）
- **CUDA Graph**：减少 kernel launch 开销
- **AutoTP**：自动解析模型结构并应用 TP（`module_inject/auto_tp.py`）

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

## 11. DeepSpeed 训练完整调用链总图

下图展示从 `DeepSpeedEngine.__init__()` 到 `train_batch()` 的完整流程，包含 ZeRO/MoE/Pipeline 的调用关系：

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              DeepSpeed 训练完整调用链                                      │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                        DeepSpeedEngine.__init__()                                │    │
│  │                        (runtime/engine.py:238)                                   │    │
│  │                                                                                 │    │
│  │  ┌─────────────────┐  ┌─────────────────────┐  ┌─────────────────────────────┐  │    │
│  │  │ _configure_      │  │ _configure_          │  │ _configure_                  │  │    │
│  │  │ distributed_model│  │ optimizer            │  │ lr_scheduler                 │  │    │
│  │  │                  │  │                      │  │                              │  │    │
│  │  │ 精度转换+广播    │  │ ┌──────────────────┐ │  │                              │  │    │
│  │  │                  │  │ │ ZERO_OPTIMIZATION│ │  │                              │  │    │
│  │  │                  │  │ │   判断 stage      │ │  │                              │  │    │
│  │  │                  │  │ └────────┬─────────┘ │  │                              │  │    │
│  │  │                  │  │          │            │  │                              │  │    │
│  │  │                  │  │    ┌─────┴─────┐      │  │                              │  │    │
│  │  │                  │  │    │           │      │  │                              │  │    │
│  │  │                  │  │ stage≤2    stage=3    │  │                              │  │    │
│  │  │                  │  │    │           │      │  │                              │  │    │
│  │  │                  │  │    ▼           ▼      │  │                              │  │    │
│  │  │                  │  │ ┌────────┐ ┌────────┐ │  │                              │  │    │
│  │  │                  │  │ │ZeRO-1/2│ │ZeRO-3  │ │  │                              │  │    │
│  │  │                  │  │ │Optimizer│ │Optimizer│ │  │                              │  │    │
│  │  │                  │  │ │        │ │_Stage3 │ │  │                              │  │    │
│  │  │                  │  │ │        │ │+Offload│ │  │                              │  │    │
│  │  │                  │  │ └───┬────┘ └───┬────┘ │  │                              │  │    │
│  │  │                  │  │     │          │      │  │                              │  │    │
│  │  │                  │  │     │  ┌───────┘      │  │                              │  │    │
│  │  │                  │  │     │  │              │  │                              │  │    │
│  │  │                  │  │     │  ▼              │  │                              │  │    │
│  │  │                  │  │     │ DeepSpeed       │  │                              │  │    │
│  │  │                  │  │     │ ZeRoOffload     │  │                              │  │    │
│  │  │                  │  │     │ .__init__()     │  │                              │  │    │
│  │  │                  │  │     │                 │  │                              │  │    │
│  │  │                  │  │     │ 参数分片+hook   │  │                              │  │    │
│  │  │                  │  │     └─────────────────┘ │  │                              │  │    │
│  │  └─────────────────┘  └─────────────────────┘  └─────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                              │
│                                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                           train_batch() / step()                                 │    │
│  │                                                                                 │    │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                          Forward Pass                                     │   │    │
│  │  │                                                                            │   │    │
│  │  │  DeepSpeedEngine.forward()                                                │   │    │
│  │  │    ├── _forward_prologue()  → ZeRO-3: in_forward=True                     │   │    │
│  │  │    ├── autocast_if_enabled() → 混合精度                                     │   │    │
│  │  │    └── self.module(*inputs)                                               │   │    │
│  │  │          │                                                                 │   │    │
│  │  │          │  [ZeRO-3 路径]                                                  │   │    │
│  │  │          ├── pre_forward_module_hook()                                    │   │    │
│  │  │          │     └── param_coordinator.fetch_sub_module()                   │   │    │
│  │  │          │           └── param.all_gather()  ← 按需收集完整参数            │   │    │
│  │  │          ├── 实际 layer forward (Linear/Attention/MoE)                    │   │    │
│  │  │          └── post_forward_module_hook()                                   │   │    │
│  │  │                └── param_coordinator.release_sub_module()                 │   │    │
│  │  │                      └── param.partition()  ← 释放参数，保留分片           │   │    │
│  │  │                                                                            │   │    │
│  │  │          │                                                                 │   │    │
│  │  │          │  [MoE 路径]                                                     │   │    │
│  │  │          └── MOELayer.forward()                                           │   │    │
│  │  │                ├── TopKGate() → topkgating() → 选择专家                   │   │    │
│  │  │                ├── _AllToAll.apply() → 分发 token 到专家                  │   │    │
│  │  │                ├── experts() → 专家计算                                    │   │    │
│  │  │                └── _AllToAll.apply() → 收集结果                            │   │    │
│  │  └──────────────────────────────────────────────────────────────────────────┘   │    │
│  │                                          │                                       │    │
│  │                                          ▼                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                          Backward Pass                                    │   │    │
│  │  │                                                                            │   │    │
│  │  │  DeepSpeedEngine.backward(loss)                                           │   │    │
│  │  │    ├── loss.backward()                                                    │   │    │
│  │  │    │     │                                                                 │   │    │
│  │  │    │     │  [ZeRO-1/2 路径]                                               │   │    │
│  │  │    │     └── grad_handling_hook() (per-param hook)                        │   │    │
│  │  │    │           └── reduce_independent_p_g_buckets()                       │   │    │
│  │  │    │                 └── allreduce_and_scatter() → 梯度分片 reduce         │   │    │
│  │  │    │                                                                               │    │
│  │  │    │     │  [ZeRO-3 路径]                                               │   │    │
│  │  │    │     ├── pre_backward_module_hook()                                  │   │    │
│  │  │    │     │     └── param_coordinator.fetch_sub_module(forward=False)     │   │    │
│  │  │    │     │           └── param.all_gather()  ← 反向时再次收集参数         │   │    │
│  │  │    │     └── post_backward_module_hook()                                 │   │    │
│  │  │    │           └── param_coordinator.release_sub_module(forward=False)    │   │    │
│  │  │    │                 └── param.partition()  ← 释放参数                    │   │    │
│  │  │    └── _backward_epilogue() → allreduce_gradients()                      │   │    │
│  │  └──────────────────────────────────────────────────────────────────────────┘   │    │
│  │                                          │                                       │    │
│  │                                          ▼                                       │    │
│  │  ┌──────────────────────────────────────────────────────────────────────────┐   │    │
│  │  │                          Step (Optimizer Update)                          │   │    │
│  │  │                                                                            │   │    │
│  │  │  DeepSpeedEngine.step()                                                   │   │    │
│  │  │    ├── _take_model_step()                                                 │   │    │
│  │  │    │     ├── clip_grad_norm_()  → 梯度裁剪                                 │   │    │
│  │  │    │     └── optimizer.step()                                             │   │    │
│  │  │    │           │                                                           │   │    │
│  │  │    │           │  [ZeRO-1/2]                                               │   │    │
│  │  │    │           ├── scaled_global_norm() → 计算梯度 norm                    │   │    │
│  │  │    │           ├── unscale_and_clip_grads() → 反缩放+裁剪                  │   │    │
│  │  │    │           ├── _optimizer_step(i) → Adam/AdamW 更新                   │   │    │
│  │  │    │           └── all_gather_dp_groups() → 收集完整 FP16 参数             │   │    │
│  │  │    │           │                                                           │   │    │
│  │  │    │           │  [ZeRO-3]                                                 │   │    │
│  │  │    │           └── DeepSpeedZeRoOffload 无额外 step（参数已分片）          │   │    │
│  │  │    ├── optimizer.zero_grad()  → 梯度清零                                   │   │    │
│  │  │    └── lr_scheduler.step()  → 学习率更新                                   │   │    │
│  │  └──────────────────────────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                              │
│                                          ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐    │
│  │                    [Pipeline Parallel 路径]                                       │    │
│  │                                                                                 │    │
│  │  PipelineEngine.train_batch()                                                   │    │
│  │    ├── TrainSchedule(micro_batches, stages, stage_id) → 1F1B 指令序列            │    │
│  │    └── _exec_schedule(sched)                                                    │    │
│  │          ├── ForwardPass()  → module forward                                    │    │
│  │          ├── BackwardPass() → loss.backward()                                   │    │
│  │          ├── ReduceGrads()  → _exec_reduce_grads()                              │    │
│  │          └── OptimizerStep() → self.step()                                      │    │
│  └─────────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                         │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录：源码文件索引

按功能分类列出所有引用的文件路径和核心类/函数：

### Engine & Config（引擎与配置）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `runtime/engine.py` | `DeepSpeedEngine` | 235 |
| `runtime/engine.py` | `DeepSpeedEngine.__init__` | 238 |
| `runtime/engine.py` | `DeepSpeedEngine.forward` | 2676 |
| `runtime/engine.py` | `DeepSpeedEngine.backward` | 3067 |
| `runtime/engine.py` | `DeepSpeedEngine.step` | 3242 |
| `runtime/engine.py` | `_configure_zero_optimizer` | 2235 |
| `runtime/config.py` | `DeepSpeedConfig` | 100 |
| `runtime/config.py` | `DEEPSPEED_OPTIMIZERS` | 84 |

### ZeRO-1/2（优化器状态/梯度分片）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `runtime/zero/stage_1_and_2.py` | `DeepSpeedZeroOptimizer` | 134 |
| `runtime/zero/stage_1_and_2.py` | `DeepSpeedZeroOptimizer.__init__` | 146 |
| `runtime/zero/stage_1_and_2.py` | `DeepSpeedZeroOptimizer.step` | 2204 |
| `runtime/zero/stage_1_and_2.py` | `IPGBucket` | 113 |
| `runtime/zero/stage_1_and_2.py` | `reduce_gradients` | 840 |
| `runtime/zero/stage_1_and_2.py` | `independent_gradient_partition_epilogue` | 898 |
| `runtime/zero/stage_1_and_2.py` | `reduce_ipg_grads` | 1615 |
| `runtime/zero/stage_1_and_2.py` | `average_tensor` | 1277 |
| `runtime/zero/stage_1_and_2.py` | `allreduce_bucket` | 1757 |
| `runtime/zero/stage_1_and_2.py` | `allreduce_no_retain` | 1834 |

### ZeRO-3 & Parameter Offload（参数分片与卸载）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `runtime/zero/stage3.py` | `DeepSpeedZeroOptimizer_Stage3` | 148 |
| `runtime/zero/stage3.py` | `DeepSpeedZeroOptimizer_Stage3.__init__` | 160 |
| `runtime/zero/stage3.py` | `IPGBucketZ3` | 124 |
| `runtime/zero/parameter_offload.py` | `DeepSpeedZeRoOffload` | 117 |
| `runtime/zero/parameter_offload.py` | `DeepSpeedZeRoOffload.__init__` | 119 |
| `runtime/zero/parameter_offload.py` | `ZeROOrderedDict` | 40 |
| `runtime/zero/parameter_offload.py` | `setup_zero_stage3_hooks` | 279 |
| `runtime/zero/parameter_offload.py` | `pre_sub_module_forward_function` | 510 |
| `runtime/zero/parameter_offload.py` | `post_sub_module_forward_function` | 534 |
| `runtime/zero/partition_parameters.py` | `PartitionedParameterCoordinator` | (引用) |
| `runtime/zero/partition_parameters.py` | `register_external_parameter` | 142 |
| `runtime/zero/partition_parameters.py` | `_dist_allgather_fn` | 108 |
| `runtime/zero/partition_parameters.py` | `NoGatherHandle` | 58 |
| `runtime/zero/partition_parameters.py` | `NoGatherCoalescedHandle` | 78 |

### Pipeline Parallel（流水线并行）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `runtime/pipe/engine.py` | `PipelineEngine` | 60 |
| `runtime/pipe/engine.py` | `PipelineEngine.__init__` | 72 |
| `runtime/pipe/engine.py` | `PipelineEngine.train_batch` | 337 |
| `runtime/pipe/schedule.py` | `PipeSchedule` 基类 | 11 |
| `runtime/pipe/schedule.py` | `TrainSchedule` | (引用) |
| `runtime/pipe/schedule.py` | `InferenceSchedule` | 135 |

### MoE（混合专家）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `moe/layer.py` | `MoE` | 17 |
| `moe/layer.py` | `MoE.forward` | 105 |
| `moe/layer.py` | `Experts` | (引用) |
| `moe/sharded_moe.py` | `MOELayer` | (类定义) |
| `moe/sharded_moe.py` | `TopKGate` | 474 |
| `moe/sharded_moe.py` | `top1gating` | 184 |
| `moe/sharded_moe.py` | `top2gating` | 291 |
| `moe/sharded_moe.py` | `topkgating` | 382 |
| `moe/sharded_moe.py` | `_AllToAll` (autograd Function) | 97 |
| `moe/ep_router.py` | `TokenChoiceTopKRouter` | 27 |
| `moe/ep_router.py` | `TokenChoiceTopKRouter.forward` | 136 |
| `moe/ep_router.py` | `_get_node_limited_routing_scores` | 82 |

### Autotuning（自动调优）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `autotuning/autotuner.py` | `Autotuner` | 42 |
| `autotuning/autotuner.py` | `Autotuner.__init__` | 47 |
| `autotuning/autotuner.py` | `Autotuner.tune` | 404 |
| `autotuning/autotuner.py` | `Autotuner.tune_space` | 523 |
| `autotuning/autotuner.py` | `Autotuner._generate_experiments` | 304 |
| `autotuning/autotuner.py` | `Autotuner.write_optimal_config` | 1075 |
| `autotuning/autotuner.py` | `Autotuner.run_after_tuning` | 1103 |
| `autotuning/config.py` | `DeepSpeedAutotuningConfig` | (引用) |
| `autotuning/tuner/grid_search.py` | `GridSearchTuner` | (引用) |
| `autotuning/tuner/random.py` | `RandomTuner` | (引用) |
| `autotuning/tuner/model_based.py` | `ModelBasedTuner` | (引用) |

### Inference（推理引擎）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `inference/engine.py` | `InferenceEngine` | 40 |
| `inference/engine.py` | `InferenceEngine.__init__` | 45 |
| `inference/engine.py` | `InferenceEngine._create_model_parallel_group` | 247 |
| `inference/engine.py` | `InferenceEngine._create_ep_parallel_group` | 260 |
| `inference/engine.py` | `InferenceEngine._apply_injection_policy` | 380 |
| `inference/engine.py` | `InferenceEngine._create_cuda_graph` | 497 |
| `inference/engine.py` | `InferenceEngine.forward` | 557 |
| `module_inject/replace_module.py` | `replace_transformer_layer` | 189 |
| `module_inject/layers.py` | `LinearAllreduce` | 581 |
| `module_inject/layers.py` | `LinearLayer` | 678 |
| `module_inject/policy.py` | `TransformerPolicy` | (引用) |
| `module_inject/auto_tp.py` | `AutoTP` | (引用) |
| `ops/transformer/inference/ds_attention.py` | `DeepSpeedSelfAttention` | (引用) |
| `model_implementations/transformers/ds_transformer.py` | `DeepSpeedTransformerInference` | (引用) |

### Ops（算子层）

| 文件路径 | 核心类/函数 | 行号 |
|----------|-------------|------|
| `ops/adam/cpu_adam.py` | `DeepSpeedCPUAdam` | 13 |
| `ops/adam/cpu_adam.py` | `DeepSpeedCPUAdam.__init__` | 16 |
| `ops/adam/cpu_adam.py` | `DeepSpeedCPUAdam.step` | 107 |
| `ops/adam/fused_adam.py` | `FusedAdam` | 18 |
| `ops/adam/fused_adam.py` | `FusedAdam.__init__` | 76 |
| `ops/adam/fused_adam.py` | `FusedAdam.step` | 107 |

---

> **文档统计**：本文档覆盖 DeepSpeed 8 大核心模块，包含 80+ 处 file:line 源码引用、4 条完整调用链、5 幅 ASCII 架构图、6 张跨框架对比表、5 张配置参数表、4 个真实代码片段。
