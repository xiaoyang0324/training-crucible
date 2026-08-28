---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → skill-performance/SKILL.md
description: >
  性能优化专家——5 步优化工作流，覆盖吞吐低、OOM、MFU 低、扩展效率差、
  通信瓶颈等性能问题。每个步骤带代码位置映射 (Code Location Mapping)，
  可直接定位到源码仓的具体实现。
  触发条件：用户报告训练速度慢、GPU 利用率低、内存不足、扩展效率差、
  通信开销大、算子性能差等性能相关问题。
---

# 性能优化专家 — 5 步优化工作流 (Code-Level)

## The Iron Law

```
优化性能问题前，必须先拿到四样东西：
  1. 性能基线（吞吐、MFU、内存占用、训练步长）
  2. 瓶颈画像（compute/memory/communication/I/O/launch 哪类）
  3. 环境信息（模型规模、并行配置、硬件、框架版本）
  4. 优化目标（提吞吐 / 降内存 / 提扩展效率）

没有瓶颈画像不做优化建议。绝不凭猜测推荐技术。
所有建议必须带源码仓 file:line 引用。
```

---

## 触发条件

| 症状 | 典型表现 | 紧急度 |
|------|---------|--------|
| **吞吐低** | 训练步长 > 预期，tokens/s 低 | 🟠 高 |
| **OOM** | CUDA out of memory / 分配失败 | 🔴 紧急 |
| **MFU 低** | GPU 利用率 < 40%，远低于理论峰值 | 🟠 高 |
| **扩展效率差** | 增加 GPU 后效率不线性提升 | 🟡 中 |
| **通信瓶颈** | comm/compute 比例过高 | 🟠 高 |
| **I/O 瓶颈** | 数据加载跟不上计算 | 🟡 中 |
| **Launch 瓶颈** | kernel launch 开销大 | 🟡 中 |

---

## 5 步优化工作流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    性能优化工作流 (带代码位置映射)                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Step 1: 画像 (Profile)                                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ torchtitan/tools/profiler.py:99 Profiler          ← Kineto profiler     │    │
│  │ torchtitan/tools/profiler.py:38 MemoryProfiler    ← 内存快照            │    │
│  │ torchtitan/components/metrics.py:38 DeviceMemoryMonitor ← 实时内存监控  │    │
│  │ torchtitan/components/metrics.py:260 MetricsProcessor ← MFU/吞吐计算    │    │
│  │ Megatron: megatron/training/training.py (profile args) ← profiling开关  │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                     │
│           ▼                                                                     │
│  Step 2: 识别 (Identify)                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ bottleneck-taxonomy.md ← 瓶颈分类体系                                    │    │
│  │ 计算 comm/compute ratio: metrics.py:481-493 tps/mfu 计算                 │    │
│  │ 热点算子: torch.profiler → kernel 耗时排序                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                     │
│           ▼                                                                     │
│  Step 3: 匹配 (Match)                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ sota-techniques.md ← SOTA 技术 (每技术带 file:line)                      │    │
│  │ skill-tickets/ ← 历史案例 (type: performance)                            │    │
│  │ 按预期收益排序候选技术                                                    │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                     │
│           ▼                                                                     │
│  Step 4: 适配 (Adapt)                                                           │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 配置入口: torchtitan/config/configs.py (ParallelismConfig/TrainingConfig)│    │
│  │ 硬件适配: torchada/_graph_rotation.py / torch_musa/                      │    │
│  │ 给出具体配置参数 + 代码路径                                               │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                     │
│           ▼                                                                     │
│  Step 5: 验证 (Validate)                                                        │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ 验证指标: metrics.py:455-541 MetricsProcessor.log() ← MFU/吞吐/内存      │    │
│  │ 数值验证: scripts/loss_compare.py ← loss/grad_norm 对比                  │    │
│  │ 归档: skill-tickets/TEMPLATE.md                                          │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 1: 画像 (Profile)

**目标:** 收集完整性能基线，建立量化基准。

**向用户确认的问题：**
- 当前吞吐 (tokens/s)？目标是多少？
- MFU 是多少？GPU 利用率如何？
- 训练步长 (step time) 是多少 ms？
- 是否使用 profiling 工具（Nsight / torch profiler）做过分析？

**需要收集的指标：**
- 吞吐：tokens/s, samples/s, steps/s
- 计算效率：MFU (Model FLOPs Utilization), GPU SM 利用率
- 内存：峰值 GPU 内存、显存碎片
- 通信：allreduce/allgather 耗时、comm/compute ratio
- I/O：数据加载耗时、dataloader worker 数量
- 并行配置：TP/PP/DP/CP/EP、micro-batch size、gradient accumulation

**代码位置映射:**

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| Kineto profiler 生命周期 | `torchtitan/tools/profiler.py:99 Profiler` | torch.profiler 封装，控制 profiling schedule |
| 内存快照 (memory snapshot) | `torchtitan/tools/profiler.py:38 MemoryProfiler` | `memory._record_memory_history` + `_snapshot()` 定期 dump |
| 实时 GPU 内存监控 | `torchtitan/components/metrics.py:38 DeviceMemoryMonitor` | `get_device_properties` + `memory_stats` 每步采集 |
| MFU/吞吐 计算与日志 | `torchtitan/components/metrics.py:455 MetricsProcessor.log()` | L481-493: tps → tflops → mfu 计算; L501-516: 指标聚合 |
| 数据加载耗时统计 | `torchtitan/components/metrics.py:496-497` | `time_data_loading` / `time_data_loading_pct` |
| Megatron profiling 开关 | `megatron/training/training.py:3552` | `args.profile` + `args.profile_ranks` 控制 |
| Megatron RL profiler | `megatron/training/training.py:2001 initialize_rl_profiler` | RL 场景专用 profiler |

---

### Step 2: 识别 (Identify)

**目标:** 分类瓶颈类型 (compute/memory/communication/I/O/launch-bound)，定位热点算子/层/操作。

**瓶颈判定规则:**

| 观察到的现象 | 瓶颈类型 |
|-------------|---------|
| GPU SM 利用率 > 80%，内存尚有空间 | Compute-bound |
| 内存接近上限，GPU 利用率波动 | Memory-bound |
| comm/compute ratio > 30% | Communication-bound |
| GPU 空闲等待数据，util 波动大 | I/O-bound |
| 大量小 kernel，launch 开销占比高 | Launch-bound |

**定位热点方法：**
- 使用 `torch.profiler` 获取 kernel 耗时排序
- 使用 Nsight Systems/Compute 分析 GPU 执行 timeline
- 对比计算耗时与通信耗时

**代码位置映射:**

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| 瓶颈分类体系 | `bottleneck-taxonomy.md` | 5 类瓶颈 × 3 训练阶段的完整分类 |
| comm/compute ratio 计算 | `torchtitan/components/metrics.py:481-493` | `tps = ntokens / time_delta`; `mfu = flops / peak_flops` |
| 内存指标采集 | `torchtitan/components/metrics.py:61-89 DeviceMemoryMonitor.get_peak_stats()` | `active_bytes.all.peak` + `num_ooms` |
| 通信耗时分析 | Megatron: `megatron/core/pipeline_parallel/schedules.py:539 backward_step` | grad reduce hook 计时 |
| DDP bucketed grad reduce | `megatron/core/distributed/distributed_data_parallel.py:87 DistributedDataParallel` | bucket 化 all-reduce / reduce-scatter |
| TP 通信重叠实现 | `megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication` | all-gather + matmul 重叠 |

---

### Step 3: 匹配 (Match)

**目标:** 根据瓶颈类型匹配 SOTA 技术，检索历史案例，按预期收益排序候选。

**匹配 SOTA 技术：**
- 查阅 `skill-performance/references/sota-techniques.md`
- 检索 `skill-tickets/` 中 `type: performance` 的历史案例
- 按预期收益排序候选技术

**常见技术映射：**
- Compute-bound → Flash Attention、Fused Operators、CUDA Graph、FP8
- Memory-bound → Activation Recompute、ZeRO、Gradient Checkpointing
- Communication-bound → Comm Overlap、Sequence Packing、Async PP
- I/O-bound → 数据预取、多 worker dataloader、数据重排
- Launch-bound → CUDA Graph、算子融合、减少 kernel 数量

**代码位置映射:**

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| SOTA 技术目录 (带源码引用) | `sota-techniques.md` | 10 项核心技术，每项带完整调用链 file:line |
| 历史案例库 | `skill-tickets/` (type: performance) | 按 `bottleneck_type` + `technique` 检索 |
| Flash Attention 调用链 | Megatron: `attention.py:499 _run_core_attention` → L1000 `_resolve_flash_version` | FA4 > FA3 > FA2 自动选择 |
| Activation Recompute | torchtitan: `activation_checkpoint.py:166 FullAC` / `:185 SelectiveAC` / `:290 MemoryBudgetAC` | 三种 AC 策略 |
| CUDA Graph | torchtitan: `cudagraph.py:189 CUDAGraphWrapper` / `:335 wrap_with_cuda_graph` | 图录制重放核心实现 |
| FP8 训练 | torchtitan: `float8.py:53 Float8LinearConverter` | torchao FP8 recipe 集成 |
| Comm Overlap | Megatron: `layers.py:634` (async all-reduce) | grad accumulation + async comm |
| FSDP 配置 | torchtitan: `fsdp.py:168 apply_fsdp_to_decoder` | per-layer 分片策略 |
| Async PP | Megatron: `schedules.py:2147 forward_backward_pipelining_without_interleaving` | 1F1B 调度 |
| MoE Fused | torchada: `fused_moe.py:331 fused_experts_impl` | triton fused MoE kernel |
| Graph Rotation | torchada: `_graph_rotation.py:138 _Rotation` | CUDA graph LRU 轮换 |

---

### Step 4: 适配 (Adapt)

**目标:** 根据用户硬件/框架调整方案，给出具体配置参数建议。

**适配原则：**
- 根据用户硬件（NVIDIA/Ascend/MUSA）选择可用技术
- 根据用户框架（Megatron/torchtitan/miles/slime）引用具体实现
- 给出具体配置参数建议

**引用源码：**
- 每个推荐技术必须附源码仓路径
- 说明该技术的配置入口和关键参数

**代码位置映射:**

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| torchtitan 并行配置 | `torchtitan/config/configs.py:122 ParallelismConfig` | DP/TP/PP/CP/EP 度数配置 |
| torchtitan 训练配置 | `torchtitan/config/configs.py:77 disable_cuda_graphs` | CUDA graph 开关 |
| FSDP reshard 策略 | `torchtitan/config/configs.py:147 fsdp_reshard_after_forward` | "default"/"always"/"never" |
| FSDP symm mem | `torchtitan/config/configs.py:162 enable_fsdp_symm_mem` | symmetric-memory 优化 |
| compile 配置 | `torchtitan/config/configs.py:298 enable_compile` | torch.compile 开关 |
| AC 配置入口 | `torchtitan/trainer.py:485 ac_config=config.activation_checkpoint` | Selective/Full/MemoryBudget |
| Megatron 参数解析 | `megatron/training/arguments.py:117 parse_args` | `--fp8-format`, `--overlap-grad-reduce` 等 |
| Megatron overlap grad reduce | `megatron/training/arguments.py:1909` | `overlap_grad_reduce` 校验 |
| torchada graph rotation | `torchada/_graph_rotation.py:138 _Rotation` | MUSA 架构 CUDA graph 轮换 |
| torchada FP8 量化 | `torchada/triton/kernels/quant/fp8.py:55 per_token_group_quant_fp8` | per-token-group FP8 |
| Sequence Packing | `torchtitan/components/data/packing.py:21 ConcatThenSplitPackingConfig` / `:58 FirstFitPackingConfig` | 两种 packing 策略 |

---

### Step 5: 验证 (Validate)

**目标:** 对比优化前后的吞吐 / MFU / 内存，确认优化未引入精度问题，归档到 `skill-tickets/`。

**验证标准：**
- 对比优化前后的吞吐 / MFU / 内存
- 确认优化未引入精度问题（loss/grad_norm bit-wise 一致）
- 验证优化在目标硬件上稳定

**归档要求：**
- 按 `skill-tickets/TEMPLATE.md` 格式创建 ticket
- 记录优化前后对比数据
- 引用源码路径和配置变更

**代码位置映射:**

| 功能 | 代码位置 | 说明 |
|------|---------|------|
| 验证指标日志 | `torchtitan/components/metrics.py:455 MetricsProcessor.log()` | tps / tflops / mfu / memory / data_loading_pct |
| MFU 计算公式 | `torchtitan/components/metrics.py:489-493` | `mfu = 100 * num_flops_per_token * tps / gpu_peak_flops` |
| 内存统计 | `torchtitan/components/metrics.py:499 device_mem_stats` | `max_active_gib` / `max_reserved_gib` / `num_ooms` |
| 数值对比脚本 | `torchtitan/scripts/loss_compare.py` | loss + grad_norm 逐 step 对比 |
| 确定性训练验证 | torchtitan CLI: `--debug.seed=42 --debug.deterministic` | bit-wise 可复现验证 |
| 验证指标 (Step 1→5 对比) | `torchtitan/components/metrics.py:501-516 metrics dict` | 完整指标字典 |

---

## 引用

| 文件 | 内容 |
|------|------|
| `skill-performance/references/bottleneck-taxonomy.md` | 性能瓶颈分类体系 (每类带代码位置) |
| `skill-performance/references/sota-techniques.md` | SOTA 优化技术目录 (每技术带调用链) |
| `skill-tickets/TEMPLATE.md` | 问题归档模板 |
| `skill-references/source-repo-map.md` | 源码仓路由（用于定位代码引用） |

---

## 源码仓路由速查

| 关注点 | 主要代码仓 | 关键路径 |
|--------|-----------|---------|
| 训练循环 + 指标 | torchtitan | `torchtitan/trainer.py`, `torchtitan/components/metrics.py` |
| Profiling | torchtitan | `torchtitan/tools/profiler.py` |
| FSDP / AC / CP | torchtitan | `torchtitan/distributed/` |
| CUDA Graph | torchtitan | `torchtitan/distributed/cudagraph.py` |
| FP8 量化 | torchtitan | `torchtitan/components/quantization/float8.py` |
| Sequence Packing | torchtitan | `torchtitan/components/data/packing.py` |
| PP Schedule (1F1B) | Megatron-LM | `megatron/core/pipeline_parallel/schedules.py` |
| TP Comm Overlap | Megatron-LM | `megatron/core/tensor_parallel/layers.py` |
| DDP Bucketed Reduce | Megatron-LM | `megatron/core/distributed/distributed_data_parallel.py` |
| Flash Attention | Megatron-LM | `megatron/core/transformer/attention.py` |
| MoE Layer | Megatron-LM | `megatron/core/transformer/moe/moe_layer.py` |
| Graph Rotation | torchada | `src/torchada/_graph_rotation.py` |
| Fused MoE (triton) | torchada | `src/torchada/triton/runtime/fused_moe/fused_moe.py` |
| FP8 Quant (triton) | torchada | `src/torchada/triton/kernels/quant/fp8.py` |
