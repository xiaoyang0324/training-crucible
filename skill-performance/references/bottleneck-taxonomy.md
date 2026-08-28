# 性能瓶颈分类 (Performance Bottleneck Taxonomy)

> 本文档提供性能瓶颈的多维分类体系，用于 Step 2 (Identify) 快速识别瓶颈类型。
> 每类瓶颈带代码位置映射：出现位置 + profiling 指标代码 + 排查代码路径。

---

## 1. Compute-bound (计算瓶颈)

### 1.1 特征

- GPU SM 利用率 > 80%
- 计算 FLOPS 接近硬件峰值
- 内存带宽尚有空间

### 1.2 典型场景

- 大矩阵乘法（Attention、MLP）占主导
- 模型大、batch size 大
- 算子未充分优化

### 1.3 诊断指标

- MFU > 50% 但吞吐仍不达标
- Nsight Compute 显示 compute pipeline 繁忙

### 1.4 代码位置映射

| 关注点 | 代码位置 | 说明 |
|--------|---------|------|
| MFU 计算 | `torchtitan/components/metrics.py:489-493` | `mfu = 100 * num_flops_per_token * tps / gpu_peak_flops` |
| 峰值 FLOPS 获取 | `torchtitan/components/metrics.py:351-353` | `utils.get_peak_flops(device_name)` |
| GPU 内存实时状态 | `torchtitan/components/metrics.py:61-89 DeviceMemoryMonitor.get_peak_stats()` | `active_bytes.all.peak` |
| Attention 热点定位 | `megatron/core/transformer/attention.py:499 _run_core_attention` | attention kernel 入口 |
| Megatron MFU 日志 | `megatron/training/training.py` (iteration log) | MFU 输出到日志 |
| Profiling 启动 | `torchtitan/tools/profiler.py:99 Profiler` | torch.profiler 封装 |

### 1.5 排查时需要检查的代码路径

1. **MFU 是否真实?** → `metrics.py:489-493` (确认 `num_flops_per_token` 正确)
2. **哪个 kernel 耗时最多?** → `profiler.py:274 build_torch_profiler` (kernel 排序)
3. **Attention 是否用了 Flash?** → `attention.py:1000 _resolve_flash_version` (FA 版本)
4. **是否启用 FP8?** → `float8.py:53 Float8LinearConverter` (FP8 状态)
5. **算子是否融合?** → 检查 `compile` 配置 + `torch.compile` 日志

### 1.6 优化方向

- Flash Attention（减少 HBM 访问）
- Fused Operators（减少 kernel launch 和内存读写）
- FP8/FP4 训练（提升计算吞吐）

---

## 2. Memory-bound (内存瓶颈)

### 2.1 特征

- GPU 内存接近上限
- OOM 或频繁 OOM
- 内存碎片化

### 2.2 典型场景

- 大模型 + 大 batch size
- 激活值占用过多
- 权重 + 优化器状态占用过多

### 2.3 诊断指标

- `torch.cuda.max_memory_allocated()` 接近 GPU 总内存
- 内存分配/释放频繁
- `num_alloc_retries` > 0 或 `num_ooms` > 0

### 2.4 代码位置映射

| 关注点 | 代码位置 | 说明 |
|--------|---------|------|
| 峰值内存统计 | `torchtitan/components/metrics.py:61-89 DeviceMemoryMonitor.get_peak_stats()` | `max_reserved_gib`, `max_reserved_pct` |
| OOM 计数 | `torchtitan/components/metrics.py:72-80` | `num_ooms` / `num_alloc_retries` |
| 内存快照 dump | `torchtitan/tools/profiler.py:38 MemoryProfiler` | `_record_memory_history` + `_snapshot()` |
| AC 策略选择 | `torchtitan/distributed/activation_checkpoint.py:166/185/290` | FullAC / SelectiveAC / MemoryBudgetAC |
| AC 应用入口 | `torchtitan/trainer.py:485 ac_config=config.activation_checkpoint` | AC 配置注入 |
| FSDP 分片 | `torchtitan/distributed/fsdp.py:168 apply_fsdp_to_decoder` | per-layer 分片策略 |
| CPU Offload | `torchtitan/config/configs.py:72 enable_cpu_offload` | CPU offload 开关 |
| MoE load balancing hook | `torchtitan/components/optimizer/optimizer.py:402 register_moe_load_balancing_hook` | MoE 辅助 loss 钩子 |

### 2.5 排查时需要检查的代码路径

1. **内存主要消耗在哪?** → `profiler.py:38 MemoryProfiler` (memory snapshot 可视化)
2. **激活值 vs 权重 vs 优化器?** → `metrics.py:499 device_mem_stats` (按类别分析)
3. **AC 是否启用?** → `activation_checkpoint.py:152 apply(model)` (AC 状态)
4. **FSDP 分片度数?** → `fsdp.py:168` + config `data_parallel_shard_degree`
5. **是否 OOM retry?** → `metrics.py:75-80` (alloc_retries / ooms 日志)

### 2.6 优化方向

- Activation Recompute（重算换内存）
- ZeRO-1/2/3（分片优化器状态/梯度/权重）
- CPU Offload（将状态卸载到 CPU）
- Gradient Checkpointing

---

## 3. Communication-bound (通信瓶颈)

### 3.1 特征

- comm/compute ratio > 30%
- 多卡训练扩展效率差
- GPU 等待通信完成

### 3.2 典型场景

- 大模型分布式训练（TP/PP/DP 通信）
- 梯度同步开销大
- All-to-All 通信（MoE）

### 3.3 诊断指标

- Nsight Systems 显示通信 kernel 占比高
- 增加 GPU 后吞吐不线性增长

### 3.4 代码位置映射

| 关注点 | 代码位置 | 说明 |
|--------|---------|------|
| DDP bucketed grad reduce | `megatron/core/distributed/distributed_data_parallel.py:87 DistributedDataParallel` | bucket 化 all-reduce / reduce-scatter |
| backward hook (grad reduce) | `megatron/core/distributed/distributed_data_parallel.py:553 _make_backward_post_hook` | 梯度 ready 时触发 reduce |
| TP async all-reduce | `megatron/core/tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication` | async comm + grad accumulation |
| TP weight prefetch | `megatron/core/tensor_parallel/layers.py:660-661` | `weight.all_gather_and_prefetch(fwd=True)` |
| PP schedule | `megatron/core/pipeline_parallel/schedules.py:2147 forward_backward_pipelining_without_interleaving` | 1F1B 调度 |
| PP interleaved | `megatron/core/pipeline_parallel/schedules.py:1019 forward_backward_pipelining_with_interleaving` | interleaved 1F1B |
| MoE layer forward | `megatron/core/transformer/moe/moe_layer.py:625 MoELayer.forward` | route → dispatch → compute → combine |
| Megatron overlap 校验 | `megatron/training/arguments.py:1909` | `overlap_grad_reduce` 参数校验 |

### 3.5 排查时需要检查的代码路径

1. **comm/compute ratio?** → `metrics.py:481-493` (对比 tps vs 理论峰值)
2. **哪种通信占比最高?** → Nsight Systems (TP all-gather / DP all-reduce / PP P2P)
3. **梯度 reduce 是否 overlap?** → `distributed_data_parallel.py:581 overlap_grad_reduce`
4. **PP bubble 占比?** → `schedules.py:2147` + Nsight Systems timeline
5. **MoE dispatch 开销?** → `moe_layer.py:625` (dispatch/combine 通信)

### 3.6 优化方向

- Communication-Overlap（通信与计算重叠）
- Sequence Packing（减少 padding 通信）
- Async Pipeline（异步流水线）
- 优化并行策略（减少通信量）

---

## 4. I/O-bound (数据加载瓶颈)

### 4.1 特征

- GPU 空闲等待数据
- GPU 利用率波动大
- 数据加载耗时 > 计算耗时

### 4.2 典型场景

- 大规模数据集读取
- 数据预处理复杂
- dataloader worker 不足

### 4.3 诊断指标

- torch profiler 显示 `DataLoader` 耗时占比高
- GPU util 间歇性下降

### 4.4 代码位置映射

| 关注点 | 代码位置 | 说明 |
|--------|---------|------|
| 数据加载耗时统计 | `torchtitan/components/metrics.py:496-497` | `time_data_loading` / `time_data_loading_pct` |
| 数据加载时间记录 | `torchtitan/components/metrics.py:355 self.data_loading_times` | 每步累积加载时间 |
| Sequence Packing | `torchtitan/components/data/packing.py:21 ConcatThenSplitPackingConfig` | concat-then-split packing |
| FirstFit Packing | `torchtitan/components/data/packing.py:58 FirstFitPackingConfig` | first-fit 装箱 |
| Packing bins 配置 | `torchtitan/components/data/packing.py:63 num_packing_bins` | 同时开放的行数 |
| 数据集构建 | `torchtitan/components/data/dataset.py` | DatasetConfig + build |

### 4.5 排查时需要检查的代码路径

1. **数据加载占比?** → `metrics.py:496-497` (`time_data_loading_pct` > 20%?)
2. **是否启用 packing?** → `packing.py` (packing recipe 配置)
3. **dataloader worker 数?** → torchtitan config `dataloader.num_workers`
4. **预处理是否太重?** → `dataset.py` (map 操作耗时)

### 4.6 优化方向

- 增加 dataloader worker 数量
- 数据预取（prefetch）
- 数据格式优化（如 mmap）
- 数据重排（减少随机读取）

---

## 5. Launch-bound (启动瓶颈)

### 5.1 特征

- 大量小 kernel 执行
- kernel launch 开销占比高
- GPU 利用率低但无明显热点

### 5.2 典型场景

- 复杂计算图（大量小算子）
- 动态 shape 导致无法用 CUDA Graph
- Python 开销大

### 5.3 诊断指标

- Nsight Systems 显示大量小 kernel
- kernel 间间隙大

### 5.4 代码位置映射

| 关注点 | 代码位置 | 说明 |
|--------|---------|------|
| CUDA Graph wrapper | `torchtitan/distributed/cudagraph.py:189 CUDAGraphWrapper` | 图录制重放核心 |
| CUDA Graph 入口 | `torchtitan/distributed/cudagraph.py:335 wrap_with_cuda_graph` | 装饰 fwd_bwd_fn |
| CUDA Graph 开关 | `torchtitan/config/configs.py:77 disable_cuda_graphs` | 禁用 CUDA graph |
| torch.compile 配置 | `torchtitan/config/configs.py:298 enable_compile` | torch.compile 开关 |
| compile components | `torchtitan/config/configs.py:304 compile_components` | 编译组件选择 |
| Graph Rotation | `torchada/_graph_rotation.py:138 _Rotation` | MUSA CUDA graph 轮换 |
| Profiler kernel 分析 | `torchtitan/tools/profiler.py:274 build_torch_profiler` | kernel 排序输出 |

### 5.5 排查时需要检查的代码路径

1. **CUDA Graph 是否启用?** → `cudagraph.py:335` + `configs.py:77` (检查禁用条件)
2. **kernel 数量?** → `profiler.py` (torch.profiler kernel 统计)
3. **compile 是否生效?** → `configs.py:298 enable_compile`
4. **MUSA graph cap?** → `_graph_rotation.py:138 _Rotation.stats`

### 5.6 优化方向

- CUDA Graph（消除 kernel launch 开销）
- 算子融合（减少 kernel 数量）
- torch.compile（图优化）

---

## 6. 按训练阶段分类

### 6.1 预训练 (Pre-training)

| 常见瓶颈 | 典型表现 | 高发场景 | 排查代码路径 |
|---------|---------|---------|-------------|
| **内存瓶颈** | OOM | 大模型 + 大 batch | `metrics.py:72-80` (num_ooms) |
| **通信瓶颈** | 扩展效率差 | 大规模并行 (>100 GPU) | `distributed_data_parallel.py:553` |
| **计算瓶颈** | MFU 低 | 算子未优化 | `metrics.py:489-493` (mfu) |
| **I/O 瓶颈** | GPU 空闲 | 大规模数据集 | `metrics.py:496-497` (data_loading_pct) |

### 6.2 后训练 (Post-training: SFT/DPO/RLHF)

| 常见瓶颈 | 典型表现 | 高发场景 | 排查代码路径 |
|---------|---------|---------|-------------|
| **内存瓶颈** | OOM | 长序列 + 大 batch | `activation_checkpoint.py:152` |
| **计算瓶颈** | 推理慢 | 生成式推理 | `attention.py:1000` (FA 版本) |
| **通信瓶颈** | 同步延迟 | 多卡 DPO 训练 | `schedules.py:2147` (PP schedule) |

### 6.3 强化学习 (RL: GRPO/PPO)

| 常见瓶颈 | 典型表现 | 高发场景 | 排查代码路径 |
|---------|---------|---------|-------------|
| **I/O 瓶颈** | rollout 等待 | 生成式 rollout | `metrics.py:496-497` (data_loading_pct) |
| **通信瓶颈** | 权重同步延迟 | 训练-推理分离架构 | `distributed_data_parallel.py:553` |
| **内存瓶颈** | OOM | 多模型共存 (actor + critic + reward) | `activation_checkpoint.py:166/185/290` |

---

## 7. 诊断方法

### 7.1 Profiling 工具

| 工具 | 用途 | 输出 | 代码入口 |
|------|------|------|---------|
| **torch.profiler** | Python 层性能分析 | kernel 耗时、调用栈、时间线 | `torchtitan/tools/profiler.py:99 Profiler` |
| **Nsight Systems** | 系统级性能分析 | CPU/GPU 时间线、通信 kernel | 外部工具 |
| **Nsight Compute** | 单 kernel 详细分析 | SM 利用率、内存带宽、寄存器 | 外部工具 |
| **nvidia-smi** | 实时监控 | GPU 利用率、内存占用、功耗 | 外部工具 |
| **Memory Snapshot** | 内存分配可视化 | 内存时序图 | `torchtitan/tools/profiler.py:38 MemoryProfiler` |

### 7.2 关键指标

| 指标 | 含义 | 健康范围 | 代码位置 |
|------|------|---------|---------|
| **MFU** | 模型 FLOPS 利用率 | > 40% 良好，> 50% 优秀 | `torchtitan/components/metrics.py:489-493` |
| **吞吐** | tokens/s | 与基线对比 | `torchtitan/components/metrics.py:481-483` |
| **步长** | 单步训练耗时 (ms) | 与基线对比 | `torchtitan/components/metrics.py:495` |
| **comm/compute** | 通信计算比 | < 20% 良好 | 计算: `metrics.py` vs Nsight |
| **GPU util** | GPU 利用率 | > 70% 良好 | `nvidia-smi` / Nsight |
| **内存占用** | GPU 内存使用 | < 90% 总内存 | `torchtitan/components/metrics.py:61-89` |
| **data_loading_pct** | 数据加载占比 | < 10% 良好 | `torchtitan/components/metrics.py:496-497` |

### 7.3 诊断流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        性能诊断流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 收集基线指标 (吞吐/MFU/内存)                                     │
│     → torchtitan/components/metrics.py:455 MetricsProcessor.log()  │
│     → metrics dict: tps, tflops, mfu, memory, data_loading_pct     │
│           │                                                         │
│           ▼                                                         │
│  2. 运行 torch.profiler 获取 kernel 耗时                             │
│     → torchtitan/tools/profiler.py:99 Profiler                     │
│     → profiler.step() 每步采集                                      │
│           │                                                         │
│           ▼                                                         │
│  3. 判断瓶颈类型:                                                    │
│     ├─ GPU util 高 + 内存有空间 → Compute-bound                     │
│     │   检查: metrics.py:489-493 (MFU 计算)                         │
│     ├─ 内存接近上限 → Memory-bound                                  │
│     │   检查: metrics.py:72-80 (num_ooms / num_alloc_retries)       │
│     ├─ comm 占比高 → Communication-bound                            │
│     │   检查: Nsight Systems + distributed_data_parallel.py:553     │
│     ├─ GPU 空闲等数据 → I/O-bound                                   │
│     │   检查: metrics.py:496-497 (data_loading_pct)                 │
│     └─ 大量小 kernel → Launch-bound                                 │
│         检查: profiler.py (kernel 数量) + cudagraph.py:333 (CUDA Graph)│
│           │                                                         │
│           ▼                                                         │
│  4. 使用 Nsight 深入分析热点                                         │
│           │                                                         │
│           ▼                                                         │
│  5. 匹配优化技术 (sota-techniques.md)                                │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 8. 瓶颈→技术→预期收益 速查表

| 瓶颈类型 | 推荐技术 | 预期加速/节省 | 关键代码位置 |
|---------|---------|--------------|-------------|
| **Compute-bound** | Flash Attention 2/3/4 | 2-4x attention 加速 | `attention.py:499 _run_core_attention` |
| **Compute-bound** | Fused Operators | 10-30% 整体加速 | `torch.compile` + `cudagraph.py` |
| **Compute-bound** | CUDA Graph | 10-20% launch 开销消除 | `cudagraph.py:189 CUDAGraphWrapper` |
| **Compute-bound** | FP8/FP4 训练 | 1.5-2x 计算吞吐 | `float8.py:53 Float8LinearConverter` |
| **Memory-bound** | Activation Recompute | 30-60% 内存节省 | `activation_checkpoint.py:166/185/290` |
| **Memory-bound** | ZeRO-1/2/3 (FSDP) | 线性内存节省 | `fsdp.py:168 apply_fsdp_to_decoder` |
| **Memory-bound** | CPU Offload | 释放 GPU 内存 | `configs.py:72 enable_cpu_offload` |
| **Communication-bound** | Comm Overlap | 20-40% 通信隐藏 | `layers.py:634` (async all-reduce) |
| **Communication-bound** | Sequence Packing | 减少 padding 通信 | `packing.py:21/58` |
| **Communication-bound** | Async Pipeline | 减少 bubble | `schedules.py:2147` (1F1B) |
| **Communication-bound** | MoE Fused | 优化 All-to-All | `fused_moe.py:331 fused_experts_impl` |
| **I/O-bound** | 数据预取 + 多 worker | 隐藏 I/O 延迟 | `dataset.py` + `packing.py` |
| **Launch-bound** | CUDA Graph | 消除 launch 开销 | `cudagraph.py:189 CUDAGraphWrapper` |
| **Launch-bound** | torch.compile | 图优化 + 融合 | `configs.py:298 enable_compile` |

---

## 引用

- `skill-performance/SKILL.md` — 5 步优化工作流 (带代码位置映射)
- `skill-performance/references/sota-techniques.md` — SOTA 优化技术目录（每技术带 file:line 调用链）
- `skill-tickets/` — 历史案例库（按 `type: performance` 检索）
