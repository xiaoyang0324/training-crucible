# 预训练 (Pre-training) — 代码级深度分析

> 本文以 **Megatron-LM**（NVIDIA，生产级分布式训练框架）和 **torchtitan**（Meta，PyTorch-native 训练框架）为双主线，从源码调用链层面拆解大模型预训练的全流程。所有文件路径均标注实际行号（已对照源码验证），可直接用于面试中的代码级问答。

---

## 0. 预训练全流程全景图

下图展示从 `pretrain()` 入口到 `optimizer.step()` 完成的完整数据流和模块调用关系，涵盖 Megatron-LM 与 torchtitan 两条路径：

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                            预训练全流程全景图                                          │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                         阶段一：初始化 (Initialize)                          │    │
│  │                                                                             │    │
│  │  pretrain() ──→ initialize_megatron() ──→ initialize_model_parallel()      │    │
│  │       │               │                          │                          │    │
│  │       │               │                          ├─ TP 进程组               │    │
│  │       │               │                          ├─ PP 进程组               │    │
│  │       │               │                          ├─ CP 进程组               │    │
│  │       │               │                          ├─ EP 进程组               │    │
│  │       │               │                          └─ DP 进程组               │    │
│  │       │               └─ 随机种子 / timers / 分布式后端                      │    │
│  │       │                                                                     │    │
│  │  [torchtitan] main() ──→ ConfigManager ──→ Trainer.__init__()               │    │
│  │                              │                │                             │    │
│  │                              │                ├─ ParallelDims.build_mesh()  │    │
│  │                              │                └─ DeviceMesh 多维拓扑        │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                          │
│                                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                    阶段二：模型与优化器构建 (Setup)                           │    │
│  │                                                                             │    │
│  │  setup_model_and_optimizer()                                                │    │
│  │    ├─ get_model() ──→ TransformerBlock ──→ [TransformerLayer × N]           │    │
│  │    │                      ├─ Attention (QKV + CoreAttn + OutProj)           │    │
│  │    │                      └─ MLP/MoE (fc1 + activation + fc2)               │    │
│  │    ├─ get_megatron_optimizer() ──→ DistributedOptimizer                     │    │
│  │    │    ├─ grad buffer 按 DP world size 分片                                 │    │
│  │    │    └─ _build_model_gbuf_param_range_map()                              │    │
│  │    └─ OptimizerParamScheduler (warmup + decay)                              │    │
│  │                                                                             │    │
│  │  [torchtitan] Trainer.__init__()                                            │    │
│  │    ├─ model_config.build() ──→ meta init                                    │    │
│  │    ├─ model_spec.parallelize_fn() ──→ TP + FSDP + AC + Compile              │    │
│  │    ├─ OptimizersContainer ──→ regex 参数组匹配                               │    │
│  │    └─ LRSchedulersContainer ──→ warmup + stable + decay                     │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                          │
│                                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                      阶段三：训练循环 (Train Loop)                            │    │
│  │                                                                             │    │
│  │  train() ──→ while iteration < train_iters:                                 │    │
│  │    └─ train_step() ──┬─ optimizer.zero_grad()                               │    │
│  │                      ├─ forward_backward_func() ──┬─ forward_step()         │    │
│  │                      │          (PP schedule)      │    ├─ get_batch()       │    │
│  │                      │                             │    ├─ model.forward()   │    │
│  │                      │                             │    │   ├─ Attn fwd      │    │
│  │                      │                             │    │   └─ MLP/MoE fwd   │    │
│  │                      │                             │    └─ loss_func()       │    │
│  │                      │                             └─ backward_step()         │    │
│  │                      │                                  └─ loss.backward()    │    │
│  │                      ├─ grad sync (async reduce-scatter / AllGather)         │    │
│  │                      ├─ clip_grad_norm_()                                    │    │
│  │                      ├─ optimizer.step() ──→ step_with_ready_grads()        │    │
│  │                      │    ├─ AdamW 参数更新                                   │    │
│  │                      │    └─ param AllGather (overlap with next fwd)         │    │
│  │                      └─ lr_scheduler.step()                                  │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                          │                                          │
│                                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │                      阶段四：Checkpoint 与评估 (Save/Eval)                    │    │
│  │                                                                             │    │
│  │  if iteration % save_interval == 0:                                         │    │
│  │    ├─ save_checkpoint() ──→ model.state_dict() + optimizer.state_dict()     │    │
│  │    └─ [torchtitan] dcp.save() ──→ PyTorch DCP 标准格式                      │    │
│  │                                                                             │    │
│  │  if iteration % eval_interval == 0:                                         │    │
│  │    └─ evaluate() ──→ 在验证集上计算 loss / perplexity                       │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
│                                                                                     │
│  ═══ 数据流 ═══                                                                     │
│  Token IDs → Embedding → [TransformerBlock × N] → LayerNorm → LM Head → Logits     │
│                    │    Attention: QKV proj → CoreAttn → Out proj                  │
│                    │    MLP/MoE: fc1 → activation → fc2 (或 Router → Experts)      │
│                    └─ 每层输出: hidden_states [seq, batch, hidden]                  │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**全景图核心路径总结：**

| 阶段 | Megatron-LM 入口 | torchtitan 入口 | 输出 |
|------|-----------------|-----------------|------|
| 初始化 | `pretrain() → initialize_megatron()` | `main() → Trainer.__init__()` | 进程组、DeviceMesh |
| 构建 | `setup_model_and_optimizer()` | `model_config.build() + parallelize_fn()` | 模型、优化器、调度器 |
| 训练循环 | `train() → train_step()` | `Trainer.train() → train_step()` | 参数更新 |
| 持久化 | `save_checkpoint()` | `dcp.save()` | checkpoint 文件 |

---

## 1. 训练入口与启动流程

### 概念原理

预训练的工程入口承担四件事：**解析配置 → 初始化分布式环境 → 构建模型/优化器/调度器 → 进入训练循环**。两个框架的设计哲学差异显著：

- **Megatron-LM**：采用"全局 args + 过程式函数"风格，所有状态通过 `get_args()` 全局访问，启动链为 `pretrain() → initialize_megatron() → setup_model_and_optimizer() → train()`。
- **torchtitan**：采用"dataclass config + 面向对象"风格，所有组件封装在 `Trainer` 类中，启动链为 `main() → config.build() → Trainer.__init__() → trainer.train()`。

### Megatron-LM 实现

**完整调用链（入口 → 训练循环）：**

```
pretrain_gpt.py:508  __main__ 块
  ├─ pretrain_gpt.py:525  parse_and_validate_args()          # 解析并校验 CLI 参数
  ├─ pretrain_gpt.py:536  pretrain(config, ...)              # training/training.py:1500
  │    ├─ training/initialize.py  initialize_megatron()     # 初始化分布式、随机种子、timers
  │    │    └─ parallel_state.py:601  initialize_model_parallel()
  │    │         ├─ 创建 TP/PP/CP/EP/DP 进程组
  │    │         └─ order="tp-cp-ep-dp-pp" (line 617)
  │    ├─ training.py:2665  setup_model_and_optimizer()
  │    │    ├─ get_model() 或 builder.build_distributed_models()
  │    │    ├─ optimizer/__init__.py:1002  get_megatron_optimizer()
  │    │    │    └─ distrib_optimizer.py:113  DistributedOptimizer
  │    │    └─ OptimizerParamScheduler                        # LR 调度器
  │    ├─ BlendedMegatronDatasetBuilder.build()              # 构建数据集
  │    └─ training.py:4167  train()
  │         └─ training.py:4566  while iteration < args.train_iters:  # 主循环
  │              └─ training.py:3010  train_step()
  │                   ├─ optimizer.zero_grad()
  │                   ├─ schedules.py:53  get_forward_backward_func()
  │                   └─ forward_backward_pipelining_without_interleaving()
  └─ pretrain_gpt.py:110  get_batch()                        # 数据批次生成（被 forward_step 调用）
```

**关键类/函数表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `pretrain_gpt.py:110` | `get_batch()` | 数据批次生成，含 TP/CP 分片广播 |
| `pretrain_gpt.py:222` | `loss_func()` | 语言模型损失 + NaN/Inf/spiky loss 检查 |
| `pretrain_gpt.py:295` | `forward_step()` | 单步前向：get_batch → model() → output_tensor |
| `training.py:1500` | `pretrain()` | 总入口，编排 init → model → train |
| `training.py:2665` | `setup_model_and_optimizer()` | 构建模型 + 优化器 + LR 调度器 |
| `training.py:3010` | `train_step()` | 单步训练：zero_grad → forward_backward → optimizer step |
| `training.py:4167` | `train()` | 训练主函数，管理迭代循环、checkpoint、eval |
| `training.py:4566` | `while iteration < args.train_iters` | 训练主循环入口 |
| `parallel_state.py:601` | `initialize_model_parallel()` | 初始化所有并行进程组 |

### torchtitan 实现

**完整调用链（入口 → 训练循环）：**

```
torchtitan/train.py:17  main()
  ├─ train.py:28  ConfigManager()
  ├─ train.py:44  config.build() → 构造 Trainer
  │    └─ torchtitan/trainer.py:277  Trainer.__init__()
  │         ├─ trainer.py:293  init_distributed()
  │         │    ├─ dist_utils.init_distributed()
  │         │    └─ parallel_dims.py:132  ParallelDims.from_config()
  │         │         └─ parallel_dims.py:211  build_mesh()   # 构建多维 DeviceMesh
  │         ├─ trainer.py:360  model_config.build() (meta init)
  │         ├─ trainer.py:479  model_spec.parallelize_fn()   # 应用 TP/FSDP/AC/Compile
  │         │    ├─ fsdp.py:168  apply_fsdp_to_decoder()
  │         │    ├─ tensor_parallel.py / linear.py           # TP 应用
  │         │    └─ activation_checkpoint.py                 # AC 应用
  │         ├─ optimizer.py:79  OptimizersContainer()
  │         ├─ lr_scheduler.py  LRSchedulersContainer()
  │         └─ checkpointer/dcp.py:89  CheckpointManager()
  └─ train.py:58  trainer.train()
       └─ trainer.py:1010  Trainer.train()
            └─ trainer.py:1029  while should_continue_training():
                 └─ trainer.py:842  train_step()
                      ├─ optimizers.zero_grad()
                      ├─ forward_backward_step()
                      ├─ clip_grad_norm_()
                      └─ optimizers.step()
```

**关键类/函数表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `train.py:17` | `main()` | 入口函数，构建 Trainer 并启动训练 |
| `trainer.py:68` | `class Trainer` | 训练器主类，管理所有组件 |
| `trainer.py:277` | `Trainer.__init__()` | 初始化分布式、模型、优化器、checkpoint |
| `trainer.py:842` | `train_step()` | 单步训练：数据获取 → fwd/bwd → 梯度裁剪 → 优化器更新 |
| `trainer.py:1010` | `Trainer.train()` | 训练主循环 |
| `trainer.py:1073` | `should_continue_training()` | 判断是否继续训练 |
| `parallel_dims.py:132` | `ParallelDims` | 并行维度定义与校验 |
| `parallel_dims.py:211` | `build_mesh()` | 构建多维 DeviceMesh |

### 跨仓库对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 入口文件 | `pretrain_gpt.py` | `train.py` |
| 配置系统 | argparse + `arguments.py` | tyro + dataclass `Config` |
| 训练循环 | `training.py:train()` 函数式 | `Trainer.train()` 面向对象 |
| 模型构建 | `model_provider()` 回调 | `ModelSpec` 协议 + `model_config.build()` |
| 分布式初始化 | `initialize_model_parallel()` | `ParallelDims.build_mesh()` |
| 数据批次 | `get_batch()` 全局函数 | `DataLoader` 组件化 |
| 状态管理 | 全局 `get_args()` | `Trainer` 实例属性 |

---

## 2. 并行策略

### 2.1 并行维度定义

| 缩写 | 全称 | 切分对象 | 通信类型 |
|------|------|----------|----------|
| DP | Data Parallel | 数据 | AllReduce/ReduceScatter 梯度 |
| TP | Tensor Parallel | 权重矩阵列/行 | AllGather/ReduceScatter 激活 |
| PP | Pipeline Parallel | 模型层 | P2P Send/Recv |
| CP | Context Parallel | 序列维度 | AllGather/ReduceScatter |
| EP | Expert Parallel | MoE 专家 | AlltoAll |

**框架结构图：**

```
┌─────────────────────────────────────────────────────────────────┐
│                      Global Cluster (World Size)                │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │              Pipeline Parallel (PP)                        │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐   │  │
│  │  │ Stage 0 │──▶│ Stage 1 │──▶│ Stage 2 │──▶│ Stage 3 │   │  │
│  │  │ TP×DP×CP│   │ TP×DP×CP│   │ TP×DP×CP│   │ TP×DP×CP│   │  │
│  │  └─────────┘   └─────────┘   └─────────┘   └─────────┘   │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                  │
│  World Size = TP × PP × DP × CP  (Megatron)                     │
│  World Size = TP × PP × dp_replicate × dp_shard × CP (torchtitan)│
│  Megatron group order="tp-cp-ep-dp-pp" (parallel_state.py:617)  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 TP (Tensor Parallel)

#### 概念原理

TP 将单层权重沿特征维度切分到多个 GPU，每个 GPU 计算部分结果，通过通信聚合。核心操作：**ColumnParallel**（按列切分权重，输入完整，输出需 gather）和 **RowParallel**（按行切分权重，输入已切分，输出需 reduce-scatter/all-reduce）。

#### Megatron-LM 实现

**调用链：**

```
forward_step()                                      # pretrain_gpt.py:295
  → model.forward()
    → transformer_block.py:267  TransformerBlock.forward()
      → transformer_layer.py:802  TransformerLayer.forward()
        → _forward_attention()
        │    → attention.py:1279  Attention.forward()
        │         ├─ layers.py:986  ColumnParallelLinear (QKV projection)
        │         │    └─ layers.py:634  LinearWithGradAccumulationAndAsyncCommunication.forward()
        │         │         ├─ AllGather input (if sequence_parallel, line 677-683)
        │         │         └─ GEMM: Y = X @ W.T
        │         └─ layers.py:1382  RowParallelLinear (output projection)
        │              └─ ReduceScatter output
        → _forward_mlp()
             → mlp.py:257  MLP.forward()
                  ├─ layers.py:986  ColumnParallelLinear (fc1)
                  └─ layers.py:1382  RowParallelLinear (fc2)
```

**关键类表：**

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `layers.py:986` | `ColumnParallelLinear` | 列并行线性层，沿输出维度切分权重 |
| `layers.py:1382` | `RowParallelLinear` | 行并行线性层，沿输入维度切分权重 |
| `layers.py:634` | `LinearWithGradAccumulationAndAsyncCommunication` | 自定义 autograd Function，融合梯度累积与异步通信 |

**ColumnParallelLinear 核心逻辑：**
- 权重形状: `[output_size_per_partition, input_size]`
- `output_size_per_partition = output_size / world_size`
- Sequence Parallel 时先 AllGather 输入（layers.py:677-683），通过 `get_global_memory_buffer()` 获取通信缓冲区
- 支持 gradient accumulation fusion（layers.py:654-657）

**RowParallelLinear 核心逻辑：**
- 权重形状: `[output_size, input_size_per_partition]`
- 前向输出做 ReduceScatter（SP）或 AllReduce（非 SP）
- `input_size_per_partition = input_size / world_size`

#### torchtitan 实现

**调用链：**

```
Trainer.__init__() → model_spec.parallelize_fn()
  → tensor_parallel.py:19  NoParallel (应用于不需要TP的模块，如 MoE router gate)
  → linear.py:47  AllGatherLinear (列并行)
    → all_gather(x_shard_m) → GEMM
  → linear.py:307  LinearReduceScatter (行并行)
    → GEMM → reduce_scatter
```

**关键类表：**

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `tensor_parallel.py:19` | `NoParallel` | 复制计算，仅设置 DTensor 并插入 hooks |
| `linear.py:47` | `AllGatherLinear` | AllGather + 列并行 GEMM |
| `linear.py:307` | `LinearReduceScatter` | 行并行 GEMM + ReduceScatter |

**AllGatherLinear 数据流 (linear.py:47-80)：**
```
x_shard_m [M/R, K] → all_gather → [M, K] @ w_shard_n.T [K, N/R] → y_shard_n [M, N/R]
```
- 通过 `@spmd.register_autograd_function` 注册自定义 autograd
- SPMD 类型标注：`x S(0)@TP, w S(0)@TP → y S(1)@TP`
- wgrad 复用 forward 保存的 `x_shard_k`（避免 R 倍激活内存），backward 时沿 K 维度 re-gather（linear.py:68-79）

#### 跨仓库 TP 对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 实现方式 | 自定义 `autograd.Function` | `spmd.register_autograd_function` + DTensor |
| 通信原语 | `dist_all_gather_func` / `dist_reduce_scatter_func` | `torch.ops.symm_mem` (symmetric memory) |
| SP 支持 | `config.sequence_parallel=True` | `parallelism.enable_sequence_parallel=True` |
| 权重存储 | `[out_per_partition, in]` / `[out, in_per_partition]` | `[N/R, K]` / `[N, K/R]` |
| 类型安全 | 运行时推断 | `spmd.S(n)` / `spmd.R` 静态类型标注 |

### 2.3 PP (Pipeline Parallel)

#### 概念原理

PP 将模型层按垂直方向切分到不同 GPU，每个 GPU 持有部分层。核心调度策略：**1F1B**（一前一后，steady-state 阶段交替执行前向和反向）和 **Interleaved**（虚拟流水线，每个 GPU 持有多个 micro-chunk）。

#### Megatron-LM 实现

**调用链：**

```
training.py:3010  train_step()
  → schedules.py:53  get_forward_backward_func()
    → schedules.py:2147  forward_backward_pipelining_without_interleaving()
      → schedules.py:2334  warmup forward passes
      → schedules.py:2379  1F1B steady state
        → forward_step() → backward_step()
      → schedules.py:2450  cooldown backward passes
```

**1F1B Schedule 示意图 (schedules.py:2379)：**

```
1F1B Schedule (micro-batch=4, PP=2):

GPU0 (Stage0): [F0][F1][F2][F3][B0][B1][B2][B3]
GPU1 (Stage1): [ID][F0][F1][F2][F3][B0][B1][B2][B3]
                ↑ bubble = (PP-1) / (micro_batch + PP - 1)

warmup = total_stages - current_stage - 1  (schedules.py:2277)
steady = num_microbatches - warmup           (schedules.py:2279)
```

**关键函数表：**

| 文件:行号 | 函数 | 职责 |
|-----------|------|------|
| `schedules.py:53` | `get_forward_backward_func()` | 根据配置选择 PP 调度策略 |
| `schedules.py:2147` | `forward_backward_pipelining_without_interleaving()` | 非交错 1F1B 调度 |
| `schedules.py:2379` | 1F1B steady state 循环 | 前向/反向交替执行 |
| `schedules.py:2277` | warmup 计算 | `num_warmup = total_stages - current_stage - 1` |

#### torchtitan 实现

**调用链：**

```
Trainer.__init__() → model_spec.pipelining_fn()
  → pipeline_parallel.py:65  pipeline_llm()
    → pipeline_parallel.py:97  _pipeline_module_split()
    → pipeline_parallel.py:125  _build_pipeline_schedule()
```

**关键函数表：**

| 文件:行号 | 函数 | 职责 |
|-----------|------|------|
| `pipeline_parallel.py:65` | `pipeline_llm()` | PP 入口，拆分模型并构建 schedule |
| `pipeline_parallel.py:97` | `_pipeline_module_split()` | 按 stage 切分模型 |
| `pipeline_parallel.py:125` | `_build_pipeline_schedule()` | 构建 pipeline schedule |

#### 跨仓库 PP 对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 调度策略 | 1F1B / Interleaved / ZB | Looped schedules |
| P2P 通信 | `P2PCommunicator` 类 | `_PipelineSchedule` |
| 交错调度 | `combined_1f1b.py` | `module_fqns_per_model_part` |
| Bubble 控制 | `num_microbatches_with_partial_activation_checkpoints` | `num_pp_microbatches` |

### 2.4 CP (Context Parallel)

#### 概念原理

CP 将长序列切分到多个 GPU，每个 GPU 处理序列的一个 chunk。Attention 计算需要全局信息，因此 CP 组内需要通信交换 KV。Megatron-LM 通过 `get_batch_on_this_cp_rank()` 在数据加载阶段完成序列切分。

#### Megatron-LM 实现

| 文件:行号 | 函数/类 | 职责 |
|-----------|---------|------|
| `parallel_state.py:601` | `initialize_model_parallel()` | CP 组初始化，`context_parallel_size` 参数 |
| `pretrain_gpt.py:180` | `get_batch_on_this_cp_rank()` | 序列维度分片到 CP rank |

#### torchtitan 实现

| 文件:行号 | 函数/类 | 职责 |
|-----------|---------|------|
| `parallel_dims.py:132` | `ParallelDims.cp` | CP 维度配置 |
| `parallel_dims.py:211` | `build_mesh()` | CP mesh 维度构建 |
| `distributed/context_parallel.py` | `prepare_context_parallel_input()` | CP 输入准备 |

### 2.5 EP (Expert Parallel) 与 MoE

#### 概念原理

EP 将 MoE 层的专家分布到不同 GPU，每个 GPU 持有部分专家。Token 通过 router 分配到对应专家的 GPU 上，需要 all-to-all 通信。

**MoE 完整数据流**（详见 `moe.md`）：

```
Input [S, B, H]
  → Router (Linear + TopK) → routing_map [T, E]
  → Token Dispatch (AllGather / AllToAll) → 分发到对应 expert 的 GPU
  → Expert Compute (FFN)
  → Token Combine (AllGather / AllToAll) → 汇总回原 GPU
  → Weighted Combine → Output [S, B, H]
```

**MoE 三大核心子系统**（每个都有独立的代码级分析，见 `moe.md`）：

| 子系统 | 功能 | 关键技术 |
|--------|------|----------|
| Router | 决定 token 去哪个 expert | TopK / TokenChoice / ExpertChoice |
| Load Balancing | 避免 expert 闲置/过载 | Aux Loss / Z-Loss / Sinkhorn / Expert Bias |
| Token Dispatch | 跨 GPU 分发/汇总 token | AllGather / AllToAll / HybridEP |

#### Megatron-LM 实现

**MoE 层调用链**：

```
TransformerLayer._forward_mlp()
  → moe_layer.py:625  MoELayer.forward()
       ├─ router.py:849  TopKRouter.routing()      # TopK + expert bias + aux loss
       │    ├─ router.py:750  _apply_expert_bias()  # 可训练偏置（DeepSeek-V3 风格）
       │    ├─ router.py:646  apply_z_loss()        # z-loss 正则
       │    └─ router.py:284  sinkhorn_load_balancing()  # Sinkhorn 平衡
       ├─ token_dispatcher.py:64  MoETokenDispatcher.token_dispatch()
       │    ├─ MoEAllGatherTokenDispatcher   # AllGather 模式
       │    └─ MoEAlltoAllTokenDispatcher   # AllToAll 模式
       └─ experts.py  MLPExperts.forward()     # 本地 expert 计算
```

**关键类表**：

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `moe/moe_layer.py:625` | `MoELayer.forward()` | MoE 层前向：route → dispatch → compute → combine |
| `moe/router.py:148` | `TopKRouter` | TopK 路由 + aux loss + z-loss + sinkhorn |
| `moe/router.py:750` | `_apply_expert_bias()` | 可训练 expert 偏置（noaux_tc） |
| `moe/token_dispatcher.py:233` | `MoEAllGatherTokenDispatcher` | AllGather 分发模式 |
| `moe/token_dispatcher.py:375` | `MoEAlltoAllTokenDispatcher` | AllToAll 分发模式 |
| `parallel_state.py:610` | `expert_model_parallel_size` | EP 组大小配置 |

#### torchtitan 实现

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `parallel_dims.py:139` | `ParallelDims.ep` | EP 维度配置 |
| `models/common/token_dispatcher.py` | `HybridEPTokenDispatcher` / `MinimalAsyncEPTokenDispatcher` | Token 分发器 |

#### DeepSpeed 实现

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `deepspeed/moe/layer.py:17` | `MoE` | MoE 层封装 |
| `deepspeed/moe/sharded_moe.py:474` | `TopKGate` | TopK 门控 |
| `deepspeed/moe/sharded_moe.py:563` | `MOELayer` | MoE 层前向 |
| `deepspeed/moe/ep_router.py:27` | `TokenChoiceTopKRouter` | Token-choice 路由（ported from torchtitan） |

> **深入分析见 `moe.md`**：包含 Router 类型对比、Load Balancing 四级技术（Aux Loss/Z-Loss/Sinkhorn/Expert Bias）、三种 Token Dispatch 通信模式的 ASCII 图、以及 18 维度 × 4 仓库对比总表。

### 2.6 DP / FSDP (Data Parallel)

#### 概念原理

数据并行将数据切分到多个 GPU，每个 GPU 持有完整模型副本，梯度通过 all-reduce 或 reduce-scatter 同步。FSDP（Fully Sharded Data Parallel）将模型参数、梯度、优化器状态也分片，显著降低显存。

#### Megatron-LM DDP 实现

**调用链：**

```
setup_model_and_optimizer()                             # training.py:2665
  → distributed_data_parallel.py:87  DDP.__init__()
    → 参数分组到 bucket_groups
    → _make_backward_post_hook() 注册反向 hook (line 553)
  → train_step()
    → backward hook (distributed_data_parallel.py:553)
      → bucket_group.register_grad_ready()
      → 触发 async all-reduce / reduce-scatter
```

**关键类表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `distributed_data_parallel.py:87` | `DDP` | DDP 包装器，连续梯度缓冲区 |
| `distributed_data_parallel.py:553` | `_make_backward_post_hook()` | 反向 hook，触发梯度同步 |
| `distributed_data_parallel.py:542` | `_finish_param_sync_for_bucket_group()` | 完成参数 AllGather |
| `distributed_data_parallel.py:588` | `no_sync()` | 关闭梯度同步的上下文管理器 |

#### torchtitan FSDP 实现

**调用链：**

```
Trainer.__init__() → model_spec.parallelize_fn()
  → fsdp.py:168  apply_fsdp_to_decoder()
    → fully_shard() (PyTorch FSDP2)
    → MixedPrecisionPolicy(param_dtype, reduce_dtype)
```

**关键函数表：**

| 文件:行号 | 函数 | 职责 |
|-----------|------|------|
| `fsdp.py:168` | `apply_fsdp_to_decoder()` | 对 decoder 应用 FSDP2 |
| `fsdp.py:223` | `MixedPrecisionPolicy` | 混合精度策略配置 |
| `fsdp.py:234` | `get_fsdp_reshard_after_forward_policy()` | reshard 策略 |

#### 跨仓库 DP 对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 实现 | 自定义 DDP + bucket | PyTorch FSDP2 (`fully_shard`) |
| 梯度同步 | async all-reduce / reduce-scatter | FSDP reduce-scatter |
| 参数同步 | `start_param_sync()` / forward pre-hook | FSDP all-gather |
| 精度 | `grad_reduce_in_fp32` | `MixedPrecisionPolicy` |

---

## 3. 模型架构

### 概念原理

标准 Transformer Decoder 架构：

```
Input → Embedding → [TransformerBlock × N] → LayerNorm → LM Head → Logits
```

每个 TransformerBlock 包含：
- Pre-LN → Self-Attention → Residual Add
- Pre-LN → MLP → Residual Add

### Megatron-LM 实现

**完整调用链（模型前向）：**

```
GPTModel.forward()
  → transformer_block.py:267  TransformerBlock.forward()
    → for layer in self.layers:
         transformer_layer.py:802  TransformerLayer.forward()
           ├─ _forward_attention()
           │    ├─ input_layernorm()
           │    └─ attention.py:1279  Attention.forward()
           │         ├─ get_query_key_value_tensors() → QKV projection (ColumnParallelLinear)
           │         ├─ CoreAttention / FlashAttention
           │         └─ output projection (RowParallelLinear)
           └─ _forward_mlp()
                ├─ pre_mlp_layernorm()
                └─ mlp.py:257  MLP.forward()
                     ├─ linear_fc1 (ColumnParallelLinear) → activation → linear_fc2 (RowParallelLinear)
                     └─ (或 MoE: moe_layer.py:625 MoELayer.forward())
```

**关键类表：**

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `transformer_block.py:267` | `TransformerBlock` | Transformer 块，包含多层 TransformerLayer |
| `transformer_block.py:336` | `_build_layers()` | 构建层列表 |
| `transformer_layer.py:802` | `TransformerLayer.forward()` | 单层 Transformer 前向 |
| `transformer_layer.py:615` | `_forward_attention()` | 注意力子层前向 |
| `transformer_layer.py:899` | `_forward_mlp()` | MLP 子层前向 |
| `attention.py:1279` | `Attention.forward()` | 注意力计算 |
| `mlp.py:257` | `MLP.forward()` | MLP 前向 |

**TransformerBlock 结构 (transformer_block.py:267-334)：**
- `self.layers = nn.ModuleList([build_layer(...)])` (line 375)
- `self.final_layernorm` (line 388)
- `self.checkpoint_core_attention` (line 299)

**TransformerLayer 结构 (transformer_layer.py)：**
- `self.input_layernorm` → `self.self_attention` → `self.pre_mlp_layernorm` → `self.mlp`
- Recompute 标志: `recompute_input_layernorm`, `recompute_pre_mlp_layernorm`, `recompute_mlp` (lines 474-476)

### torchtitan 实现

**完整调用链（模型前向）：**

```
model_config.build()
  → llama3/model.py:53  Llama3Model (继承 Decoder)
    → models/common/decoder.py:62  Decoder.__init__()
      → tok_embeddings + layers + norm + lm_head
  → Decoder.forward()
    → TransformerBlock.forward()                    # models/common/decoder.py:48
      ├─ attention_norm(x)
      ├─ Attention.forward()                         # models/common/attention.py
      ├─ ffn_norm(h)
      └─ FeedForward.forward() / MoE.forward()
```

**关键类表：**

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `llama3/model.py:53` | `Llama3Model` | Llama3 模型定义 |
| `models/common/decoder.py:62` | `Decoder` | 通用 Decoder 基类 |
| `models/common/decoder.py:48` | `TransformerBlock` | Transformer 块 |
| `models/common/attention.py` | `FlexAttention` / `VarlenAttention` | 注意力实现 |
| `models/common/feed_forward.py` | `FeedForward` | 前馈网络 |

### 框架结构图（模型架构）

```
┌─────────────────────────────────────────────────────────┐
│                    GPTModel / Llama3Model                │
│  ┌───────────────────────────────────────────────────┐  │
│  │              TransformerBlock                      │  │
│  │  ┌─────────────────────────────────────────────┐  │  │
│  │  │           TransformerLayer                    │  │  │
│  │  │  ┌──────────────┐    ┌──────────────────┐   │  │  │
│  │  │  │ LayerNorm     │───▶│  Attention        │   │  │  │
│  │  │  │ (input_layern)│    │  ├─ QKV proj (CP) │   │  │  │
│  │  │  └──────────────┘    │  ├─ Core Attn      │   │  │  │
│  │  │  ┌──────────────┐    │  └─ Out proj (RP) │   │  │  │
│  │  │  │ LayerNorm     │───▶│                    │   │  │  │
│  │  │  │ (pre_mlp)     │    │  MLP / MoE         │   │  │  │
│  │  │  └──────────────┘    │  ├─ fc1 (CP)       │   │  │  │
│  │  │                      │  ├─ activation     │   │  │  │
│  │  │                      │  └─ fc2 (RP)       │   │  │  │
│  │  │                      └──────────────────┘   │  │  │
│  │  └─────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│  CP = ColumnParallel, RP = RowParallel                   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. 优化器与学习率调度

### 概念原理

预训练通常使用 **AdamW 优化器**，配合 **warmup + cosine/linear decay** 学习率调度。分布式场景下，优化器状态也需要分片（DistributedOptimizer）以节省显存。

### Megatron-LM 实现

**调用链：**

```
setup_model_and_optimizer()                              # training.py:2665
  → optimizer/__init__.py:1002  get_megatron_optimizer()
    ├─ distrib_optimizer.py:113  DistributedOptimizer.__init__()
    │    → 参数分片到 DP ranks
    │    → 构建 grad buffer (连续内存)
    │    → _build_model_gbuf_param_range_map() (line 134)
    └─ OptimizerParamScheduler                           # LR 调度器

train_step()                                             # training.py:3010
  → optimizer.zero_grad()
  → forward_backward()
  → distrib_optimizer.py:3177  step_with_ready_grads()
    ├─ super().step_with_ready_grads()                   # 执行 AdamW 更新
    ├─ start_param_sync_for_bucket_group_subset()        # 参数 AllGather
    └─ timers('params-all-gather')                       # 计时
```

**关键类表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `optimizer/__init__.py:1002` | `get_megatron_optimizer()` | 构建 Megatron 优化器 |
| `distrib_optimizer.py:113` | `DistributedOptimizer` | 分布式优化器，状态分片 |
| `distrib_optimizer.py:3177` | `step_with_ready_grads()` | 梯度就绪时执行优化器 step |
| `distrib_optimizer.py:134` | `_build_model_gbuf_param_range_map()` | 构建参数到 grad buffer 的映射 |
| `optimizer_param_scheduler.py` | `OptimizerParamScheduler` | LR 调度器 |

**DistributedOptimizer 核心设计：**
- 将 grad buffer 按 DP world size 等分，每个 DP rank 拥有连续区域
- 参数更新只针对 owned params，然后 AllGather 同步完整参数
- 支持 `overlap_param_gather`：参数 AllGather 与下一步 forward 重叠

`step_with_ready_grads()` 实际代码（distrib_optimizer.py:3177-3208）：

```python
# megatron/core/optimizer/distrib_optimizer.py:3177
@torch.no_grad()
def step_with_ready_grads(self) -> bool:
    """Step the optimizer with ready gradients, return successful.
    Under the hood, either launch synchronous param all-gathers or get ready to launch
    asynchorous all-gathers that get overlapped with the next forward pass.
    """
    update_successful = super().step_with_ready_grads()

    should_sync_params = not self.ddp_config.overlap_param_gather and not getattr(
        self, '_defer_param_sync', False
    )
    timers = self.config.timers
    if timers is not None and (self.ddp_config.use_megatron_fsdp or should_sync_params):
        timers('params-all-gather', log_level=1).start(barrier=self.config.barrier_with_L1_time)
    if self.ddp_config.use_megatron_fsdp:
        for model_chunk in self.model_chunks:
            model_chunk.start_param_sync()
    else:
        if should_sync_params:
            self.start_param_sync_for_bucket_group_subset()
    if timers is not None and (self.ddp_config.use_megatron_fsdp or should_sync_params):
        timers('params-all-gather').stop()

    return update_successful
```

**代码要点：**
1. 先调用 `super().step_with_ready_grads()` 执行底层 AdamW 更新
2. 根据 `overlap_param_gather` 配置决定同步/异步 AllGather
3. 异步模式下，AllGather 在 `optimizer.zero_grad()` 或 forward pre-hook 中启动
4. `timers('params-all-gather')` 用于性能分析

### torchtitan 实现

**调用链：**

```
Trainer.__init__()
  → optimizer.py:79  OptimizersContainer.__init__()
    → 按 regex pattern 匹配参数到 param group
    → 同类型 optimizer 批量融合
  → lr_scheduler.py  LRSchedulersContainer()

train_step()                                             # trainer.py:842
  → optimizers.zero_grad()
  → forward_backward_step()
  → clip_grad_norm_()
  → optimizers.step()
  → lr_schedulers.step()
```

**关键类表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `optimizer.py:79` | `OptimizersContainer` | 优化器容器，支持混合优化器类型 |
| `optimizer.py:378` | `default_adamw()` | 便捷创建 AdamW 配置 |
| `lr_scheduler.py:130` | `linear_warmup_stable_decay()` | warmup + stable + decay 调度 |

**OptimizersContainer 核心设计 (optimizer.py:79-100)：**
- 按 regex `pattern` 匹配参数到 param group（first match wins）
- 同类型 optimizer 批量融合为单个实例
- 支持 `fused` / `foreach` / `for-loop` / `fused_opt_states_bf16` 实现
- 默认 AdamW 参数：`betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1` (optimizer.py:391-394)

**LR 调度 (lr_scheduler.py:130-179)：**
- 三阶段：linear warmup → stable → decay
- 支持三种 decay：`linear` / `sqrt` / `cosine`
- 返回乘法因子 `curr_adjustment`（LambdaLR 兼容）

### 跨仓库优化器对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 优化器类型 | Adam / SGD / Muon (LayerWise) | AdamW (可扩展) |
| 状态分片 | DistributedOptimizer (grad buffer 分片) | FSDP 自动分片 |
| 实现 | 自定义 CUDA kernel | PyTorch fused/foreach |
| LR 调度 | warmup + cosine/linear | warmup + stable + decay (3-stage) |
| 参数组 | 全局统一 | regex 匹配多 group |

---

## 5. 精度与混合精度

### 概念原理

混合精度训练使用 FP16/BF16 进行前向和反向计算，FP32 维护主权重和优化器状态，通过 loss scaling 防止梯度下溢。FP8（8-bit 浮点）是新一代精度，通过 Transformer Engine 支持。

### Megatron-LM 实现

| 文件:行号 | 配置/类 | 职责 |
|-----------|---------|------|
| `transformer_config.py:588` | `fp8` | FP8 格式：`e4m3` / `hybrid` |
| `transformer_config.py:595` | `fp8_recipe` | FP8 recipe：`delayed` / `tensorwise` / `mxfp8` / `blockwise` |
| `transformer_config.py:603` | `fp8_param` | 参数保持 FP8 精度 |
| `fp8_utils.py` | FP8 工具函数 | amax 历史校正 |
| `Float16Module` | `transformer/module.py` | FP16 模块包装器 |

Transformer Engine FP8 配置实际代码（transformer_config.py:588-618）：

```python
# megatron/core/transformer/transformer_config.py:588
fp8: Optional[Literal['e4m3', 'hybrid']] = field(
    default=None, metadata={"argparse_meta": {"arg_names": ["--fp8-format"]}}
)
"""If set, enables the use of FP8 precision through Transformer Engine. There are 2 predefined
choices (1) 'e4m3' uniformly uses e4m3 for all FP8 tensors, (2) 'hybrid' uses e4m3 for all FP8
activation and weight tensors and e5m2 for all FP8 output activation gradient tensors."""

fp8_recipe: Optional[Literal['tensorwise', 'delayed', 'mxfp8', 'blockwise', 'custom']] = (
    "delayed"
)
"""If set, enables the use of FP8 precision through Transformer Engine. There are 5 predefined
choices (1) 'tensorwise' uses per tensor current scaling recipe, (2) 'delayed'
uses delayed scaling recipe, 3) 'mxfp8' for Blackwell architecture only,
4) 'blockwise' for blockwise scaling recipe, 5) 'custom' for custom quantization recipe."""

fp8_param: bool = False
"""If set, keep the parameters in fp8 precision to save memory. This option must be used
together with fp8 mode (i.e., TransformerConfig.fp8 is not None). Note that not all parameters
will be converted to fp8; for example, biases will remain unchanged."""

fp8_margin: int = 0
"""Margin for the scaling factor computation."""

fp8_interval: int = 1
"""DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
Controls how often the scaling factor is recomputed."""
```

**FP8 配置要点：**
- `fp8` 选择数据格式：`e4m3`（统一格式）或 `hybrid`（前向 e4m3，梯度 e5m2）
- `fp8_recipe` 控制 amax 缩放策略：`delayed`（延迟缩放，默认）适合大多数场景
- `fp8_param=True` 将参数保持 FP8 精度，进一步节省显存
- `fp8_margin` 和 `fp8_interval` 控制缩放因子计算的保守程度和频率

### torchtitan 实现

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `float8.py:53` | `Float8LinearConverter` | 将 Linear 替换为 Float8Linear |
| `float8.py:58` | `recipe_name` | Float8 recipe：`rowwise` / `rowwise_with_gw_hp` |
| `fsdp.py:223` | `MixedPrecisionPolicy` | FSDP 混合精度策略 |

### 跨仓库精度对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| FP8 实现 | Transformer Engine (TE) | torchao Float8Linear |
| FP8 recipe | delayed / tensorwise / mxfp8 / blockwise | rowwise / rowwise_with_gw_hp |
| 混合精度 | Float16Module + MixedPrecisionOptimizer | MixedPrecisionPolicy (FSDP) |
| BF16 支持 | 原生支持 | 原生支持 |

---

## 6. 激活重计算与内存优化

### 概念原理

激活重计算（Activation Checkpointing / Recomputation）通过不保存中间激活、在反向时重新计算来节省显存。三种粒度：
- **Full**：整个 transformer block 重计算
- **Selective**：选择性重计算特定 ops（如 core attention）
- **Memory-budget**：编译器自动划分 compute/memory 权衡

### Megatron-LM 实现

**调用链：**

```
TransformerBlock.forward()                               # transformer_block.py:267
  → recompute.py:22  checkpointed_forward()
    → custom(start, end) 函数
      → for index in range(start, end):
           layer = self.layers[index]
           hidden_states = layer(hidden_states, ...)     # 不保存中间激活

TransformerLayer.__init__()                              # transformer_layer.py:474-476
  → self.recompute_input_layernorm = False
  → self.recompute_pre_mlp_layernorm = False
  → self.recompute_mlp = False
  → if config.recompute_granularity == 'selective':
       if "layernorm" in config.recompute_modules: ...
```

**关键配置表：**

| 配置参数 | 说明 |
|----------|------|
| `recompute_granularity` | `full` / `selective` / `uniform` |
| `recompute_method` | `uniform` / `block` |
| `recompute_num_layers` | 重计算的层数 |
| `recompute_modules` | 选择性重计算的模块列表（如 `core_attn`, `layernorm`, `mlp`） |

`checkpointed_forward()` 实际代码（recompute.py:22-133）：

```python
# megatron/core/recompute.py:22
def checkpointed_forward(
    self: MegatronModule,
    hidden_states: Tensor,
    attention_mask: Tensor,
    context: Optional[Tensor],
    context_mask: Optional[Tensor],
    rotary_pos_emb: Tensor,
    attention_bias: Optional[Tensor],
    packed_seq_params: PackedSeqParams,
    use_inner_quantization_context: bool,
    padding_mask: Optional[Tensor] = None,
    extract_layer_indices: Optional[Set[int]] = None,
    layer_offset: int = 0,
    cp_layout_state: Optional[ContextParallelLayoutState] = None,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Forward method with activation checkpointing."""
    if extract_layer_indices is None:
        extract_layer_indices = set()
    intermediate_hidden_states: List[Tensor] = []

    is_dual_rope = isinstance(rotary_pos_emb, (tuple, list))
    rotary_pos_emb = rotary_pos_emb if is_dual_rope else (None, rotary_pos_emb)

    def custom(start: int, end: int):
        def custom_forward(
            hidden_states, attention_mask, context, context_mask,
            rotary_pos_emb_local, rotary_pos_emb_global, padding_mask=None,
        ):
            rotary_pos_emb = (
                (rotary_pos_emb_local, rotary_pos_emb_global)
                if is_dual_rope
                else rotary_pos_emb_global
            )
            for index in range(start, end):
                layer = self.layers[index]
                # ... FP8/FP4 量化上下文处理 ...
                if isinstance(layer, TransformerLayer):
                    hidden_states, context = layer(**layer_kwargs)
                else:  # MambaLayer (HybridStack)
                    hidden_states = layer(**layer_kwargs)
            return hidden_states, context
        return custom_forward

    # chunk_runner: 对 [start, end) 区间应用 torch.utils.checkpoint
```

**代码要点：**
1. `custom(start, end)` 返回一个闭包 `custom_forward`，供 `torch.utils.checkpoint` 调用
2. 闭包内逐层计算 `layer(**layer_kwargs)`，不保存中间激活
3. 支持 Dual RoPE：将 `(rotary_pos_emb_local, rotary_pos_emb_global)` 元组解包
4. 支持 FP8/FP4 量化上下文：`get_fp8_context(config, layer_number)`
5. 支持混合架构：`TransformerLayer` 和 `MambaLayer`（HybridStack）分支处理
6. `extract_layer_indices` 用于从中间层提取特征（如 distillation）

### torchtitan 实现

**调用链：**

```
Trainer.__init__() → model_spec.parallelize_fn()
  → activation_checkpoint.py
    ├─ FullAC.apply()               # line 166: 整个 block 重计算
    ├─ SelectiveAC.apply()          # line 185: 选择性重计算
    └─ MemoryBudgetAC.apply()       # line 290: 编译器自动划分
```

**关键类表：**

| 文件:行号 | 类 | 职责 |
|-----------|-----|------|
| `activation_checkpoint.py:166` | `FullAC` | 整个 transformer block 重计算 |
| `activation_checkpoint.py:185` | `SelectiveAC` | 选择性重计算（per-op） |
| `activation_checkpoint.py:290` | `MemoryBudgetAC` | 编译器自动划分 compute/memory |

**SelectiveAC 核心设计 (activation_checkpoint.py:185-229)：**
- `get_save_ops()` 返回需要保存激活的 ops 集合
- `force_recompute_mm_shapes_by_fqns` 强制重计算特定 FQN 的 matmul（默认 `moe.router.gate`）
- 使用 `ptd_checkpoint_wrapper` 包装模块

**MemoryBudgetAC 核心设计 (activation_checkpoint.py:290-330)：**
- `memory_budget` 参数（0.0-1.0）控制 compute/memory 权衡
- 0.0 = 全部重计算，1.0 = 默认运行时优化
- 通过 `torch._functorch.config.activation_memory_budget` 配置

### 跨仓库 AC 对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| Full AC | `recompute_granularity='full'` | `FullAC` |
| Selective AC | `recompute_granularity='selective'` + `recompute_modules` | `SelectiveAC` + `get_save_ops()` |
| Memory budget | 无 | `MemoryBudgetAC` (编译器集成) |
| 粒度 | layer 级 / module 级 | block 级 / op 级 |

---

## 7. 分布式通信与计算重叠

### 概念原理

大模型训练的性能瓶颈常在通信。关键优化：
- **梯度通信与反向重叠**：bucket-based async all-reduce/reduce-scatter
- **参数 AllGather 与 forward 重叠**：`overlap_param_gather`
- **PP 通信隐藏**：1F1B schedule 中的 P2P

### Megatron-LM 实现

**梯度通信与反向重叠：**

```
DDP.__init__()                                           # distributed_data_parallel.py:87
  → 参数分组到 bucket_groups (按 bucket_size)
  → _make_backward_post_hook() 注册 hook (line 553)

backward pass:
  → backward hook (distributed_data_parallel.py:553)
    → param.main_grad.add_(param.grad.data)               # 累积到 main_grad
    → bucket_group.register_grad_ready()                  # 标记梯度就绪
    → 当 bucket 内所有梯度就绪：触发 async reduce-scatter
```

**关键机制：**
- `overlap_grad_reduce=True`：梯度 reduce-scatter 与反向计算重叠
- `overlap_param_gather=True`：参数 AllGather 与 forward 重叠（通过 forward pre-hook）
- `bucket_size` 控制桶大小（默认 `max(40M, 1M * dp_size)`，line 134）

`LinearWithGradAccumulationAndAsyncCommunication` 实际代码（layers.py:634-687）：

```python
# megatron/core/tensor_parallel/layers.py:634
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """See linear_with_grad_accumulation_and_async_allreduce"""

    @staticmethod
    @custom_fwd
    def forward(
        ctx, input, weight, bias, gradient_accumulation_fusion,
        allreduce_dgrad, sequence_parallel, grad_output_buffer,
        wgrad_deferral_limit, tp_group, gtp_remat_size, output_dtype,
    ):
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None
        ctx.save_for_backward(input, weight)
        ctx.main_grad = main_grad
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group

        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()
            all_gather_buffer = get_global_memory_buffer().get_tensor(dim_size, input.dtype, "mpu")
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        return _linear_forward(total_input, weight, bias, output_dtype)
```

**代码要点：**
1. 自定义 `torch.autograd.Function`，融合梯度累积与异步通信
2. Sequence Parallel 时先 AllGather 输入（`dist_all_gather_func`），通过 `get_global_memory_buffer()` 获取通信缓冲区
3. `gradient_accumulation_fusion`：将梯度直接累积到 `weight.main_grad`，避免额外拷贝
4. `ctx.save_for_backward(input, weight)` 保存反向所需张量
5. GTP (Gradient Tensor Parallelism) 支持：`gtp_remat_size > 1` 时 all-gather weight

**参数 AllGather 重叠 (distrib_optimizer.py:3177)：**
```
step_with_ready_grads()
  ├─ super().step_with_ready_grads()                     # 执行优化器更新
  ├─ start_param_sync_for_bucket_group_subset()          # 启动异步 AllGather
  └─ 下次 forward pre-hook 完成 AllGather
```

### torchtitan 实现

**CUDA Graph 捕获与回放：**

```
Trainer.__init__()
  → cudagraph.py:189  CUDAGraphWrapper
    → capture: 记录 forward+backward 到 CUDA graph
    → replay: 重放 graph（避免 kernel launch 开销）

train_step()                                             # trainer.py:842
  → wrap_with_cuda_graph()                               # cudagraph.py
    → 首次：capture graph
    → 后续：replay graph
```

**CUDAGraphWrapper 核心设计 (cudagraph.py:189-229)：**
- `static_input_indices`：标记地址不变的输入（如模型权重）
- `example_inputs`：定义固定输入结构
- `should_check_address`：调试时验证静态输入地址
- 要求固定形状输入，无 CPU↔GPU 同步

**FSDP 通信重叠：**
- FSDP2 自动管理参数 all-gather 与 forward 的重叠
- `reshard_after_forward_policy` 控制 reshard 时机（`default` / `always` / `never`）

### 跨仓库通信重叠对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 梯度重叠 | bucket-based async reduce-scatter | FSDP reduce-scatter |
| 参数重叠 | `overlap_param_gather` + forward pre-hook | FSDP all-gather |
| CUDA Graph | TECudaGraphHelper / FullCudaGraphWrapper | CUDAGraphWrapper |
| PP 通信 | P2P send/recv | _PipelineSchedule |

---

## 8. Checkpoint 与状态管理

### 概念原理

Checkpoint 是训练容错和断点续训的关键。分布式场景下需要处理：
- **模型状态**：参数 + buffer
- **优化器状态**：momentum + variance（分布式下需处理分片 FQN 映射）
- **训练状态**：iteration、rng_state、dataloader state

### Megatron-LM 实现

**调用链：**

```
train()                                                  # training.py:4167
  → training.py:4566  while iteration < args.train_iters:
    → training.py:4612  save_checkpoint() (if iteration % save_interval == 0)
      → training.py:4700  save_checkpoint()
        ├─ get_checkpoint_state()                         # 收集所有状态
        │    ├─ model.state_dict()
        │    ├─ optimizer.state_dict()                    # distrib_optimizer.py:127
        │    └─ iteration, rng_state, opt_param_scheduler
        └─ ckpt/manager.py  save()                        # 实际保存到磁盘

load_checkpoint()                                        # training.py:2500
  → ckpt/manager.py  load()                               # 加载 checkpoint
    → model.load_state_dict()
    → optimizer.load_state_dict()                         # FQN 重映射
```

**关键函数表：**

| 文件:行号 | 函数 | 职责 |
|-----------|------|------|
| `training.py:4700` | `save_checkpoint()` | 保存 checkpoint 总入口 |
| `training.py:2500` | `load_checkpoint()` | 加载 checkpoint |
| `distrib_optimizer.py:127` | `state_dict()` | 优化器状态导出（含 FQN 映射） |
| `ckpt/manager.py` | `CheckpointManager` | checkpoint 文件管理 |

**分布式 Checkpoint 关键问题：**
- PP 场景下每个 stage 只保存自己的参数分片
- FQN (Fully Qualified Name) 需要跨 stage 统一命名
- `use_distributed_optimizer=True` 时，优化器状态按 DP rank 分片保存

### torchtitan 实现

**调用链：**

```
Trainer.__init__()
  → checkpointer/dcp.py:89  CheckpointManager.__init__()
    → async_mode 配置
    → checkpoint 目录设置

train()                                                  # trainer.py:1010
  → save_checkpoint() (trainer.py:1076)
    → state_dict()                                       # 收集状态
    │    ├─ model.state_dict()
    │    ├─ optimizers.state_dict()
    │    └─ training_state (iteration, step, rng)
    → dcp.save()                                         # PyTorch DCP 保存

load_checkpoint()
  → dcp.load()                                           # PyTorch DCP 加载
    → model.load_state_dict()
    → optimizers.load_state_dict()
```

**关键类表：**

| 文件:行号 | 类/函数 | 职责 |
|-----------|---------|------|
| `checkpointer/dcp.py:89` | `CheckpointManager` | checkpoint 管理器 |
| `checkpointer/dcp.py:137` | `async_mode` | 异步保存模式 |
| `trainer.py:1076` | `state_dict()` | 收集训练状态 |
| `trainer.py:1090` | `load_state_dict()` | 恢复训练状态 |

**CheckpointManager 核心设计 (dcp.py:89-160)：**
- 基于 PyTorch DCP (Distributed Checkpoint) 标准
- 支持 `async_mode`：异步保存不阻塞训练
- `state_dict` 返回嵌套 dict：`{model, optimizer, training_state}`
- 自动处理 FSDP 分片状态的合并/拆分

### 跨仓库 Checkpoint 对比

| 维度 | Megatron-LM | torchtitan |
|------|-------------|------------|
| 存储格式 | 自定义 (torch.save) | PyTorch DCP 标准 |
| 分布式处理 | 手动 FQN 映射 | DCP 自动处理 |
| 异步保存 | 不支持 | `async_mode` 支持 |
| PP 处理 | 每个 stage 独立保存 | DCP 统一处理 |
| 优化器状态 | 按 DP rank 分片 | FSDP 自动分片 |

---

## 9. 关键配置参数表

### Megatron-LM 核心配置

| 配置参数 | 代码位置 | 说明 | 可选值 |
|----------|----------|------|--------|
| `tensor_model_parallel_size` | `parallel_state.py:601` | TP 并行度 | 整数 (1, 2, 4, 8...) |
| `pipeline_model_parallel_size` | `parallel_state.py:601` | PP 并行度 | 整数 |
| `context_parallel_size` | `parallel_state.py:601` | CP 并行度 | 整数 |
| `expert_model_parallel_size` | `parallel_state.py:610` | EP 并行度 | 整数 |
| `recompute_granularity` | `transformer_config.py` | 重计算粒度 | `full` / `selective` / `uniform` |
| `recompute_method` | `transformer_config.py` | 重计算方法 | `uniform` / `block` |
| `recompute_num_layers` | `transformer_config.py` | 重计算层数 | 整数 |
| `recompute_modules` | `transformer_config.py` | 选择性重计算模块 | `core_attn`, `layernorm`, `mlp` |
| `fp8` | `transformer_config.py:588` | FP8 格式 | `e4m3` / `hybrid` |
| `fp8_recipe` | `transformer_config.py:595` | FP8 recipe | `delayed` / `tensorwise` / `mxfp8` / `blockwise` |
| `fp8_param` | `transformer_config.py:603` | 参数保持 FP8 | bool |
| `sequence_parallel` | `transformer_config.py` | Sequence Parallel | bool |
| `overlap_grad_reduce` | `distributed_data_parallel.py` | 梯度通信重叠 | bool |
 参数 AllGather 重叠 | bool |
| `use_distributed_optimizer` | `distrib_optimizer.py` | 分布式优化器（状态分片） | bool |
| `num_microbatches_with_partial_activation_checkpoints` | `schedules.py` | PP 部分激活重计算 microbatch 数 | 整数 |

### torchtitan 核心配置

| 配置参数 | 代码位置 | 说明 | 可选值 |
|----------|----------|------|--------|
| `tensor_parallel_degree` | `parallel_dims.py:132` | TP 并行度 | 整数 |
| `pipeline_parallel_degree` | `parallel_dims.py:132` | PP 并行度 | 整数 |
| `data_parallel_shard_degree` | `parallel_dims.py:132` | DP 分片度 | 整数 |
| `data_parallel_replicate_degree` | `parallel_dims.py:132` | DP 复制度 | 整数 |
| `context_parallel_degree` | `parallel_dims.py:132` | CP 并行度 | 整数 |
| `expert_parallel_degree` | `parallel_dims.py:139` | EP 并行度 | 整数 |
| `enable_activation_checkpoint` | `activation_checkpoint.py` | AC 模式 | `full` / `selective` / `memory_budget` |
| `memory_budget` | `activation_checkpoint.py:290` | 内存预算 (0.0-1.0) | float |
| `float8.recipe_name` | `float8.py:58` | Float8 recipe | `rowwise` / `rowwise_with_gw_hp` |
| `float8.enable_fsdp_float8_all_gather` | `float8.py` | FSDP Float8 AllGather | bool |
| `enable_sequence_parallel` | `parallel_dims.py` | Sequence Parallel | bool |
| `num_tokens_per_microbatch_per_dp_rank` | `config` | 每 DP rank 每 microbatch token 数 | 整数 |
| `cudagraph` | `cudagraph.py` | CUDA Graph 模式 | `none` / `segment` / `full` |

---

## 10. 预训练完整调用链总图

下图展示一个训练步骤中所有核心模块的调用关系和数据流：

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                              预训练完整训练步骤调用链                                        │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐        │
│  │   数据加载    │     │   分布式初始化 │     │   模型构建    │     │   优化器构建  │        │
│  │  get_batch() │     │ initialize_  │     │ get_model()  │     │ get_megatron │        │
│  │              │     │ model_parallel│     │              │     │ _optimizer() │        │
│  └──────┬───────┘     └──────┬───────┘     └──────┬───────┘     └──────┬───────┘        │
│         │                    │                    │                    │                │
│         │                    │                    │                    │                │
│         ▼                    ▼                    ▼                    ▼                │
│  ┌──────────────────────────────────────────────────────────────────────────────┐        │
│  │                           train_step() 单步训练                              │        │
│  │                                                                              │        │
│  │  ┌─────────────────────────────────────────────────────────────────────┐     │        │
│  │  │                    forward_backward_func()                          │     │        │
│  │  │                    (PP schedule: 1F1B / Interleaved)                │     │        │
│  │  │                                                                     │     │        │
│  │  │  ┌─────────────┐    ┌─────────────────────────────────────────┐     │     │        │
│  │  │  │forward_step │    │         model.forward()                  │     │     │        │
│  │  │  │             │───▶│                                         │     │     │        │
│  │  │  │  get_batch()│    │  TransformerBlock → TransformerLayer     │     │     │        │
│  │  │  │  → model()  │    │    ├─ _forward_attention()              │     │     │        │
│  │  │  │  → loss_func│    │    │    ├─ QKV (ColumnParallelLinear)   │     │     │        │
│  │  │  └─────────────┘    │    │    ├─ CoreAttn / FlashAttention    │     │     │        │
│  │  │                      │    │    └─ OutProj (RowParallelLinear)  │     │     │        │
│  │  │                      │    │    └─ _forward_mlp()               │     │     │        │
│  │  │                      │    │         ├─ fc1 (ColumnParallel)   │     │     │        │
│  │  │                      │    │         ├─ activation (GeLU/SiLU) │     │     │        │
│  │  │                      │    │         └─ fc2 (RowParallel)      │     │     │        │
│  │  │                      │    │    (或 MoE: Router → Dispatch →     │     │     │        │
│  │  │                      │    │          Expert Compute → Combine)  │     │     │        │
│  │  │                      │    └─────────────────────────────────────────┘     │     │        │
│  │  │                      │                                                      │     │        │
│  │  │                      │    ┌─────────────────────────────────────────┐     │     │        │
│  │  │                      │    │      backward_step() → loss.backward()  │     │     │        │
│  │  │                      │    │                                         │     │     │        │
│  │  │                      │    │  autograd 引擎遍历 grad_fn 链            │     │     │        │
│  │  │                      │    │    ├─ LinearWithGradAccumulation...     │     │     │        │
│  │  │                      │    │    │    .backward()                      │     │     │        │
│  │  │                      │    │    │    ├─ weight.T @ grad_output (wgrad)│     │     │        │
│  │  │                      │    │    │    └─ input.T @ grad_output (dgrad) │     │     │        │
│  │  │                      │    │    ├─ LayerNorm.backward()               │     │     │        │
│  │  │                      │    │    └─ Attention.backward()               │     │     │        │
│  │  │                      │    │                                         │     │     │        │
│  │  │                      │    │  ► backward hook 触发梯度同步:           │     │     │        │
│  │  │                      │    │    bucket_group.register_grad_ready()   │     │     │        │
│  │  │                      │    │    → async reduce-scatter / all-reduce  │     │     │        │
│  │  │                      │    └─────────────────────────────────────────┘     │     │        │
│  │  └──────────────────────────────────────────────────────────────────────────┘     │        │
│  │                                                                                  │        │
│  │  ┌─────────────────────────────────────────────────────────────────────┐         │        │
│  │  │                    optimizer.step()                                 │         │        │
│  │  │                                                                     │         │        │
│  │  │  step_with_ready_grads()                                           │         │        │
│  │  │    ├─ super().step_with_ready_grads()   # AdamW 更新               │         │        │
│  │  │    │    ├─ exp_avg = β1*exp_avg + (1-β1)*grad                     │         │        │
│  │  │    │    ├─ exp_avg_sq = β2*exp_avg_sq + (1-β2)*grad²              │         │        │
│  │  │    │    └─ param -= lr * exp_avg / (√exp_avg_sq + ε) + λ*param   │         │        │        │
│  │  │    └─ start_param_sync_for_bucket_group_subset()                   │         │        │
│  │  │         └─ param AllGather (overlap with next forward)            │         │        │
│  │  └─────────────────────────────────────────────────────────────────────┘         │        │
│  └──────────────────────────────────────────────────────────────────────────────────┘        │
│                                                                                             │
│  ═══ 通信原语 ═══                                                                          │
│  TP: AllGather (SP 输入) / ReduceScatter (SP 输出) / AllReduce (非 SP)                      │
│  PP: P2P Send/Recv (跨 stage 传递 activation)                                              │
│  DP: ReduceScatter (梯度同步) / AllGather (参数同步)                                        │
│  EP: AllToAll (Token 分发/汇总)                                                            │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**训练步骤关键路径时序：**

| 时序 | 操作 | 通信 | 可重叠 |
|------|------|------|--------|
| T0 | `optimizer.zero_grad()` | - | - |
| T1 | `forward_step()` → TP AllGather | TP AllGather input | - |
| T2 | Attention + MLP | - | - |
| T3 | `backward_step()` | DP reduce-scatter (per bucket) | 梯度与反向重叠 |
| T4 | `optimizer.step()` | param AllGather | 与下一步 forward 重叠 |
| T5 | `lr_scheduler.step()` | - | - |

---

## 附录：面试高频问题与代码定位

### Q1: Megatron-LM 如何初始化分布式环境？

**代码定位：**
- `parallel_state.py:601` `initialize_model_parallel()`
- 进程组创建顺序：`tp-cp-ep-dp-pp` (line 617)
- 参数：`tensor_model_parallel_size`, `pipeline_model_parallel_size`, `context_parallel_size`, `expert_model_parallel_size`

### Q2: TP 中 ColumnParallelLinear 和 RowParallelLinear 的区别？

**代码定位：**
- `layers.py:986` ColumnParallelLinear：权重 `[out_per_partition, in]`，输入完整，输出按 partition 切分
- `layers.py:1382` RowParallelLinear：权重 `[out, in_per_partition]`，输入已切分，输出需 reduce/all-reduce
- 配合使用：ColumnParallel 输出 → RowParallel 输入，中间无需通信

### Q3: PP 的 1F1B 调度如何减少 bubble？

**代码定位：**
- `schedules.py:2379` 1F1B steady state 循环
- warmup = `total_stages - current_stage - 1` (line 2277)
- bubble 比例 = `(PP-1) / (micro_batch + PP - 1)`
- 增加 microbatch 数可降低 bubble 比例

### Q4: DistributedOptimizer 如何实现状态分片？

**代码定位：**
- `distrib_optimizer.py:113` DistributedOptimizer.__init__()
- grad buffer 按 DP world size 等分，每个 rank 拥有连续区域
- `_build_model_gbuf_param_range_map()` (line 134) 构建参数到 buffer 的映射
- `step_with_ready_grads()` (line 3177) 只更新 owned params，然后 AllGather

### Q5: torchtitan 中 FSDP2 与 Megatron DDP 的核心差异？

**代码定位：**
- torchtitan: `fsdp.py:168` `apply_fsdp_to_decoder()` 使用 `fully_shard()` (PyTorch 原生)
- Megatron: `distributed_data_parallel.py:87` 自定义 DDP + bucket 机制
- FSDP2 自动管理参数 all-gather/reshard，Megatron 手动控制

### Q6: 激活重计算的 Full、Selective、MemoryBudget 三种模式如何选择？

**代码定位：**
- Megatron: `recompute_granularity` 配置 (`transformer_config.py`)
- torchtitan: `activation_checkpoint.py`
  - `FullAC` (line 166)：整个 block 重计算，显存节省最大
  - `SelectiveAC` (line 185)：选择性重计算特定 ops
  - `MemoryBudgetAC` (line 290)：编译器自动权衡

### Q7: FP8 训练中 amax 是什么？如何更新？

**代码定位：**
- Megatron: `fp8_utils.py` amax 历史校正
- `transformer_config.py:595` `fp8_recipe` 控制 amax 更新策略
- `delayed`：延迟缩放，使用历史 amax
- `tensorwise`：逐张量 amax

### Q8: torchtitan 如何实现 TP 的类型安全？

**代码定位：**
- `linear.py:47` AllGatherLinear 使用 `@spmd.register_autograd_function`
- SPMD 类型标注：`x S(0)@TP, w S(0)@TP → y S(1)@TP`
- `tensor_parallel.py:19` NoParallel 使用 `spmd.R` (Replicate)

### Q9: MoE 的 Expert Parallel 如何实现 Token 分发？

**代码定位：**
- Megatron: `moe/moe_layer.py:625` MoELayer.forward()
  - route → dispatch (all-to-all) → compute → combine (all-to-all)
- torchtitan: `models/common/token_dispatcher.py`
  - `HybridEPTokenDispatcher` / `MinimalAsyncEPTokenDispatcher`

### Q10: 如何在 PP 场景下正确保存和加载 checkpoint？

**代码定位：**
- Megatron: `training.py:4700` `save_checkpoint()`
  - 每个 stage 独立保存自己的参数分片
  - FQN 需要跨 stage 统一命名
- torchtitan: `checkpointer/dcp.py:89` `CheckpointManager`
  - 基于 PyTorch DCP 标准，自动处理分片
  - `async_mode` 支持异步保存

---

---

## 附录 B：源码文件索引

下表按功能分类列出本文引用的所有核心文件及其包含的关键类/函数（面试快速定位用）：

### B.1 训练入口与生命周期

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `pretrain_gpt.py` | `get_batch()` (:110), `loss_func()` (:222), `forward_step()` (:295) | 数据批次、损失、单步前向 |
| `megatron/training/training.py` | `pretrain()` (:1500), `setup_model_and_optimizer()` (:2665), `train_step()` (:3010), `train()` (:4167), `save_checkpoint()` (:4700), `load_checkpoint()` (:2500) | 训练主逻辑 |
| `megatron/training/initialize.py` | `initialize_megatron()` | 分布式初始化、随机种子 |
| `megatron/core/parallel_state.py` | `initialize_model_parallel()` (:601) | TP/PP/CP/EP/DP 进程组创建 |

### B.2 并行策略

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/tensor_parallel/layers.py` | `ColumnParallelLinear` (:986), `RowParallelLinear` (:1382), `LinearWithGradAccumulationAndAsyncCommunication` (:634) | TP 线性层 + 自定义 autograd |
| `megatron/core/tensor_parallel/mappings.py` | `_CopyToModelParallelRegion`, `_ReduceFromModelParallelRegion` | TP 通信原语封装 |
| `megatron/core/pipeline_parallel/schedules.py` | `get_forward_backward_func()` (:53), `forward_backward_pipelining_without_interleaving()` (:2147) | PP 调度策略 |
| `megatron/core/pipeline_parallel/p2p_communication.py` | `P2PCommunicator` | PP P2P Send/Recv |
| `megatron/core/distributed/distributed_data_parallel.py` | `DDP` (:87), `_make_backward_post_hook()` (:553) | DDP + 梯度 bucket 同步 |

### B.3 模型架构

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/transformer/transformer_block.py` | `TransformerBlock` (:267), `_build_layers()` (:336) | Transformer 块，多层容器 |
| `megatron/core/transformer/transformer_layer.py` | `TransformerLayer.forward()` (:802), `_forward_attention()` (:615), `_forward_mlp()` (:899) | 单层 Transformer 前向 |
| `megatron/core/transformer/attention.py` | `Attention.forward()` (:1279) | 注意力计算 |
| `megatron/core/transformer/mlp.py` | `MLP.forward()` (:257) | MLP 前向 |
| `megatron/core/transformer/transformer_config.py` | `TransformerConfig` (fp8:588, recompute, sequence_parallel) | Transformer 配置中心 |

### B.4 优化器与学习率

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/optimizer/__init__.py` | `get_megatron_optimizer()` (:1002) | 优化器构建入口 |
| `megatron/core/optimizer/distrib_optimizer.py` | `DistributedOptimizer` (:113), `step_with_ready_grads()` (:3177), `_build_model_gbuf_param_range_map()` (:134) | 分布式优化器，状态分片 |
| `megatron/core/optimizer/optimizer_param_scheduler.py` | `OptimizerParamScheduler` | LR 调度器 (warmup + decay) |

### B.5 激活重计算与内存

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/recompute.py` | `checkpointed_forward()` (:22) | 激活重计算核心逻辑 |

### B.6 精度与混合精度

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/fp8_utils.py` | `get_fp8_context()`, amax 历史校正 | FP8 工具函数 |
| `megatron/core/transformer/module.py` | `Float16Module` | FP16 模块包装器 |

### B.7 MoE 专家并行

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/transformer/moe/moe_layer.py` | `MoELayer.forward()` (:625) | MoE 层前向 |
| `megatron/core/transformer/moe/router.py` | `TopKRouter` (:148), `_apply_expert_bias()` (:750) | TopK 路由 + 负载均衡 |
| `megatron/core/transformer/moe/token_dispatcher.py` | `MoEAllGatherTokenDispatcher` (:233), `MoEAlltoAllTokenDispatcher` (:375) | Token 分发器 |

### B.8 Checkpoint

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `megatron/core/optimizer/distrib_optimizer.py` | `state_dict()` (:127) | 优化器状态导出（含 FQN 映射） |
| `megatron/core/ckpt/manager.py` | `CheckpointManager` | checkpoint 文件管理 |

### B.9 torchtitan 核心文件

| 文件路径 | 核心类/函数 | 职责 |
|----------|------------|------|
| `torchtitan/train.py` | `main()` (:17) | 入口函数 |
| `torchtitan/trainer.py` | `Trainer` (:68), `train_step()` (:842), `train()` (:1010) | 训练器主类 |
| `torchtitan/distributed/parallel_dims.py` | `ParallelDims` (:132), `build_mesh()` (:211) | 并行维度 + DeviceMesh |
| `torchtitan/distributed/fsdp.py` | `apply_fsdp_to_decoder()` (:168) | FSDP2 应用 |
| `torchtitan/distributed/tensor_parallel.py` | `NoParallel` (:19) | TP 基础 |
| `torchtitan/distributed/linear.py` | `AllGatherLinear` (:47), `LinearReduceScatter` (:307) | TP 线性层 |
| `torchtitan/distributed/activation_checkpoint.py` | `FullAC` (:166), `SelectiveAC` (:185), `MemoryBudgetAC` (:290) | 激活重计算 |
| `torchtitan/distributed/cudagraph.py` | `CUDAGraphWrapper` (:189) | CUDA Graph 捕获回放 |
| `torchtitan/components/optimizer/optimizer.py` | `OptimizersContainer` (:79), `default_adamw()` (:378) | 优化器容器 |
| `torchtitan/components/optimizer/lr_scheduler.py` | `linear_warmup_stable_decay()` (:130) | LR 调度器 |
| `torchtitan/components/checkpointer/dcp.py` | `CheckpointManager` (:89) | Checkpoint 管理器 |
| `torchtitan/components/quantization/float8.py` | `Float8LinearConverter` (:53) | Float8 量化 |
| `torchtitan/models/llama3/model.py` | `Llama3Model` (:53) | Llama3 模型 |
| `torchtitan/models/common/decoder.py` | `Decoder` (:62), `TransformerBlock` (:48) | 通用 Decoder |

### B.10 文件路径速查（按字母排序）

| 文件路径 | 模块分类 | 关键符号 |
|----------|---------|---------|
| `megatron/core/distributed/distributed_data_parallel.py` | DP | `DDP` |
| `megatron/core/optimizer/distrib_optimizer.py` | Optimizer | `DistributedOptimizer` |
| `megatron/core/optimizer/optimizer_param_scheduler.py` | Optimizer | `OptimizerParamScheduler` |
| `megatron/core/pipeline_parallel/schedules.py` | PP | `forward_backward_pipelining_*` |
| `megatron/core/parallel_state.py` | Init | `initialize_model_parallel` |
| `megatron/core/recompute.py` | Memory | `checkpointed_forward` |
| `megatron/core/tensor_parallel/layers.py` | TP | `ColumnParallelLinear` |
| `megatron/core/transformer/attention.py` | Model | `Attention` |
| `megatron/core/transformer/mlp.py` | Model | `MLP` |
| `megatron/core/transformer/moe/moe_layer.py` | MoE | `MoELayer` |
| `megatron/core/transformer/moe/router.py` | MoE | `TopKRouter` |
| `megatron/core/transformer/moe/token_dispatcher.py` | MoE | `MoEAlltoAllTokenDispatcher` |
| `megatron/core/transformer/transformer_block.py` | Model | `TransformerBlock` |
| `megatron/core/transformer/transformer_config.py` | Config | `TransformerConfig` |
| `megatron/core/transformer/transformer_layer.py` | Model | `TransformerLayer` |
| `megatron/training/training.py` | Training | `pretrain`, `train_step` |
| `pretrain_gpt.py` | Entry | `get_batch`, `forward_step` |
| `torchtitan/trainer.py` | Training | `Trainer` |

---

> **文档版本：** v3.0 (代码级深度版 — 含全景图 + 代码片段 + 总调用链 + 源码索引)
> **生成日期：** 2026-08-28
> **覆盖框架：** Megatron-LM (NVIDIA) + torchtitan (Meta)
> **总计：** 11 章 + 2 附录，60+ file:line 引用，12+ 调用链，15+ 对比表，4 个真实代码片段，2 个全景 ASCII 图
