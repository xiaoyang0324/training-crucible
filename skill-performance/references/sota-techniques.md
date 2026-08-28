# SOTA 优化技术目录 (State-of-the-Art Optimization Techniques)

> 本文档收录训练性能优化的 SOTA 技术，每个技术包含：
> - 完整代码调用链（带 file:line）
> - 该技术在哪个仓库的哪个文件实现
> - 配置参数与代码入口
> - 预期性能收益与验证方法
>
> 用于 Step 3 (Match) 和 Step 4 (Adapt) 匹配和适配优化方案。

---

## 1. Flash Attention

### 1.1 原理

通过 tiling 和 kernel fusion 减少 HBM 访问。将 attention 计算分解为小块在 SRAM 中完成，将 HBM 读写从 O(N²) 降至 O(N)。
- FA2: 标准实现，支持 Ampere+ 架构
- FA3: Hopper 架构优化，利用 TMA/async copy
- FA4: 最新版本，进一步利用 Hopper 特性

### 1.2 适用场景

- 所有 Transformer 模型
- 长序列训练（序列越长收益越大）
- Compute-bound 场景

### 1.3 完整代码调用链

```
用户配置: --flash-attention-version (2/3/4)
    │
    ▼
Megatron: attention.py:321  self.flash_attention_version = config.flash_attention_version
    │
    ▼
Megatron: attention.py:1000 _resolve_flash_version() -> (use_fa4, use_fa3)
    │  自动选择: FA4 > FA3 > FA2 (按安装情况)
    │
    ▼
Megatron: attention.py:499 _run_core_attention(query, key, value, attention_mask, ...)
    │
    ├── FA4 → attention.py:1091 flash_attn4_varlen_func(...)
    ├── FA3 → attention.py:1108 _flash_attention_3_forward_wrapper(...)
    │           └→ flash_attn_3.flash_attn_interface._flash_attn_forward
    └── FA2 → attention.py:1130 flash_attn_varlen_func(...)
```

**torchtitan 侧:** 通过 SDPA (Scaled Dot-Product Attention) 调用，底层自动 dispatch 到 flash attention:
- `torchtitan/models/common/attention.py` — 标准 attention 实现
- PyTorch SDPA 自动选择 flash 后端

### 1.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `flash_attention_version` | Megatron: `transformer_config.py` | 2/3/4 或 None (auto) |
| `window_size` | Megatron: `transformer_config.py` | sliding window attention |
| `--flash-attention-version` | Megatron CLI | 命令行入口 |

### 1.5 预期性能收益与验证

- **Attention 计算加速:** 2-4x
- **HBM 读写减少:** O(N) vs O(N²)
- **验证方法:** 对比 attention kernel 耗时 (Nsight Compute)

---

## 2. Activation Recompute (激活重算)

### 2.1 原理

前向传播时不保存中间激活值，反向传播时重新计算。用计算换内存，显著降低激活值内存占用。

三种策略：
- **FullAC:** 重算整个 transformer block
- **SelectiveAC:** 按 op 类型选择性重算（保存 compute/comm 密集 op）
- **MemoryBudgetAC:** 编译器 partitioner 按内存预算自动决策

### 2.2 适用场景

- 大模型训练，激活值占用大量内存
- 长序列训练（激活值随序列长度线性增长）
- 内存不足但计算资源有冗余

### 2.3 完整代码调用链

```
用户配置: --activation-checkpoint (selective/full/memory-budget)
    │
    ▼
torchtitan/trainer.py:485  ac_config=config.activation_checkpoint
    │
    ▼
torchtitan/models/*/parallelize.py → model_spec.parallelize_fn(ac_config=...)
    │
    ▼
torchtitan/distributed/activation_checkpoint.py:152 ActivationCheckpointing.apply(model)
    │
    ├── FullAC (L166)    → ptd_checkpoint_wrapper(module, ...)  [全量重算]
    ├── SelectiveAC (L185) → create_selective_checkpoint_contexts(policy)  [按 op 选择性]
    │       L209: get_save_ops() → _get_default_save_ops()  [SDPA/linear/topk/reduce_scatter]
    │       L243: _get_custom_policy() → wrapped_policy()  [每 2 个 mm 重算 1 个]
    └── MemoryBudgetAC (L290) → torch._functorch.config.activation_memory_budget = config.memory_budget
```

**Megatron 侧:**
```
Megatron: transformer_layer.py:999  te_checkpoint / tensor_parallel.checkpoint
    │
    ├── recompute_granularity: "full" / "selective"
    ├── recompute_method: "uniform" / "block"
    └── recompute_num_layers: int
```

### 2.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `activation_checkpoint` | `torchtitan/trainer.py:108` | SelectiveAC/FullAC/MemoryBudgetAC/None |
| `memory_budget` | `activation_checkpoint.py:298` | 0.0-1.0，仅 MemoryBudgetAC |
| `force_recompute_mm_shapes_by_fqns` | `activation_checkpoint.py:196` | 强制重算指定 fqn 的 mm |
| `--recompute-granularity` | Megatron CLI | full / selective |
| `--recompute-method` | Megatron CLI | uniform / block |

### 2.5 预期性能收益与验证

- **内存节省:** 30-60%（取决于 recompute 比例）
- **计算开销增加:** 约 10-30%
- **验证方法:** `--debug.seed=42 --debug.deterministic` 对比 loss/grad_norm bit-wise 一致

---

## 3. CUDA Graph

### 3.1 原理

将多个 kernel 录制为图，一次性提交。消除 kernel launch 开销和 Python 调度开销。

### 3.2 适用场景

- 固定 shape 的训练循环
- 大量小 kernel 的场景
- Launch-bound 场景

### 3.3 完整代码调用链

```
用户配置: (默认启用) / --disable-cuda-graphs
    │
    ▼
torchtitan/config/configs.py:77  disable_cuda_graphs: bool = False
    │
    ▼
torchtitan/trainer.py → wrap_with_cuda_graph(fwd_bwd_fn)
    │
    ▼
torchtitan/distributed/cudagraph.py:335 wrap_with_cuda_graph(fwd_bwd_fn)
    │  首次调用时:
    │  L368: extra_input_spec = CUDAGraphInputSpec(extra_kwargs)
    │  L395: graph_wrapper = CUDAGraphWrapper(flat_fwd_bwd, example_inputs)
    │
    ▼
torchtitan/distributed/cudagraph.py:189 CUDAGraphWrapper.__call__(*args)
    │  L295-302: warmup (首次 1 步)
    │  L304-315: capture (录制图)
    │  L322-325: replay (重放图)
    │           L323: self._args[i].copy_(args[i])  [更新动态输入]
    │           L324: self._graph.replay()
```

**torchada 侧 (MUSA 架构):**
```
torchada/_graph_rotation.py:138 _Rotation
    │  L146-160: LRU 轮换 live CUDA-graph executables
    │  L162-178: _ensure_aux() → load_graph_rotation_ops()
    │  解决 MUSA ~2048 CUDA-graph cap 限制
```

### 3.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `disable_cuda_graphs` | `torchtitan/config/configs.py:77` | 禁用 CUDA graph |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `torchada/_graph_rotation.py:122` | graph rotation margin |

### 3.5 预期性能收益与验证

- **整体加速:** 10-20%
- **Launch 开销几乎消除**
- **验证方法:** 对比 step time (warmup 后稳定步长)

---

## 4. FP8 Training

### 4.1 原理

使用 FP8 (8-bit) 进行矩阵乘法，利用 Hopper/Ada 架构的 Tensor Core 支持。配合 per-tensor/per-channel scaling 保持精度。

### 4.2 适用场景

- Hopper (H100) 或 Ada (RTX 4090) 架构
- 大模型训练（计算密集型）
- Compute-bound 场景

### 4.3 完整代码调用链

```
用户配置: --fp8-format e4m3/hybrid
    │
    ▼
torchtitan/components/quantization/float8.py:53 Float8LinearConverter.__init__()
    │  L86-96: 检查硬件支持 (SM89+ / gfx942+)
    │  L113: TorchAOFloat8LinearConfig.from_recipe_name(recipe_name)
    │  L121-123: rowwise recipe → emulate_precision_casts = True
    │
    ▼
torchtitan/components/quantization/float8.py:151 Float8LinearConverter.convert(model_config)
    │  L156: model_config.traverse(Linear.Config)
    │  L158: filter_fn 过滤不适用的 Linear
    │  L159-164: 替换为 Float8Linear.Config
    │
    ▼
torchtitan/components/quantization/float8.py:216 Float8GroupedExpertsConverter
    │  L247: model_config.traverse(GroupedExperts.Config)
    │  L249-250: _get_float8_grouped_experts_cls() → Float8GroupedExperts
    │  L201-208: _grouped_mm() → _quantize_then_scaled_grouped_mm()
```

**Megatron 侧:**
```
Megatron: transformer_config.py:588  fp8: Optional[Literal['e4m3', 'hybrid']]
Megatron: transformer_config.py:595  fp8_recipe: tensorwise/delayed/mxfp8/blockwise/custom
    │
    ▼
Transformer Engine (TE) 底层 FP8 GEMM
```

**torchada 侧 (MUSA):**
```
torchada/triton/kernels/quant/fp8.py:55 per_token_group_quant_fp8(x, group_size, eps, dtype)
    │  L75-78: 检查 group_size 整除 + contiguous
    │  返回: (quantized_tensor, scale)
```

### 4.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `fp8` | Megatron: `transformer_config.py:588` | e4m3 / hybrid |
| `fp8_recipe` | Megatron: `transformer_config.py:595` | delayed (default) |
| `recipe_name` | `float8.py:58` | rowwise / rowwise_with_gw_hp |
| `emulate` | `float8.py:68` | 软件模拟 FP8 (测试用) |
| `filter_fqns` | `float8.py:61` | 跳过指定 fqn 的 Linear |

### 4.5 预期性能收益与验证

- **计算吞吐提升:** 1.5-2x
- **内存占用减少** (激活值/权重)
- **验证方法:** 对比 loss 收敛曲线，确保精度不退化

---

## 5. Comm Overlap (通信重叠)

### 5.1 原理

将通信操作（allreduce/allgather）与计算操作重叠执行，利用 NCCL 的异步通信能力。

### 5.2 适用场景

- 通信占比高的分布式训练
- 梯度同步、权重同步
- Communication-bound 场景

### 5.3 完整代码调用链

```
Megatron: tensor_parallel/layers.py:634 LinearWithGradAccumulationAndAsyncCommunication
    │
    ├── forward (L637-687):
    │   L660-661: weight.all_gather_and_prefetch(fwd=True)  [TP weight prefetch]
    │   L677-683: dist_all_gather_func()  [sequence parallel all-gather]
    │   L687: _linear_forward(total_input, weight, bias)
    │
    ├── backward (L689-):
    │   allreduce_dgrad → async all-reduce 与 wgrad 重叠
    │   grad_output_buffer → 梯度预取 buffer
    │   wgrad_deferral_limit → wgrad 延迟计算
    │
    ▼
Megatron: distributed_data_parallel.py:553 _make_backward_post_hook(param)
    │  L581-584: overlap_grad_reduce → register_grad_ready()
    │  bucket 化 all-reduce / reduce-scatter
    │
    ▼
Megatron: distributed_data_parallel.py:87 DistributedDataParallel.__init__()
    │  L133-134: bucket_size = max(40M, 1M * dp_size)
    │  L136-137: 不 overlap 时 bucket_size = None
```

### 5.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `overlap_grad_reduce` | Megatron: `arguments.py:1909` | 梯度 all-reduce 与反向重叠 |
| `overlap_param_gather` | Megatron: `arguments.py:1027` | 参数 all-gather 与优化器重叠 |
| `gradient_accumulation_fusion` | Megatron: `layers.py:654` | 梯度累加融合 |
| `sequence_parallel` | Megatron: `layers.py:677` | SP all-gather |

### 5.5 预期性能收益与验证

- **通信隐藏:** 20-40%
- **整体加速:** 10-30%
- **验证方法:** Nsight Systems 查看 comm/compute 重叠区域

---

## 6. Sequence Packing (序列打包)

### 6.1 原理

将多个短序列打包到一个长序列中，减少 padding 带来的无效计算和通信。

### 6.2 适用场景

- 变长序列训练
- 数据长度差异大
- Communication-bound 场景（减少 padding 通信）

### 6.3 完整代码调用链

```
torchtitan/components/data/packing.py
    │
    ├── L21 ConcatThenSplitPackingConfig:
    │   L46: grain.experimental.ConcatThenSplitIterDataset()
    │   拼接后按固定长度切分
    │
    ├── L58 FirstFitPackingConfig:
    │   L93: grain.experimental.FirstFitPackIterDataset()
    │   贪心 first-fit 装箱
    │
    ├── L113 _packing_output_is_full(): 检查是否填满
    │
    ├── L118 _text_sequence_to_packing_input(): 转换输入格式
    │
    └── L139 _packing_output_to_text_sequence():
        L144: padding 位置 labels 设为 IGNORE_INDEX
        L148-153: 恢复 position ids (segment 内从 0 开始)
```

### 6.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `num_packing_bins` | `packing.py:63` | 同时开放的行数 (FirstFit) |
| packing recipe | torchtitan config | concat-then-split / first-fit |

### 6.5 预期性能收益与验证

- **计算效率提升:** 10-30%（取决于 padding 比例）
- **通信量减少**
- **验证方法:** 对比有效 token 占比 (non-padding tokens / total tokens)

---

## 7. FSDP (Fully Sharded Data Parallelism)

### 7.1 原理

PyTorch 原生的 ZeRO-3 等价实现。权重、梯度、优化器状态全分片，支持 per-layer 控制。

### 7.2 适用场景

- PyTorch 原生训练
- 大规模模型训练
- Memory-bound 场景

### 7.3 完整代码调用链

```
torchtitan/distributed/fsdp.py:168 apply_fsdp_to_decoder(model, dp_mesh, ...)
    │
    ├── L223-227: MixedPrecisionPolicy(param_dtype, reduce_dtype)
    │
    ├── L238-250: weight tying → tok_embeddings + output 合并为一个 FSDP unit
    │
    ├── L267-366: 遍历 transformer_block
    │   ├── L274: moe_enabled → expert 参数特殊处理
    │   │   L282-291: ep_degree > 1 → edp_mesh + Shard(0)/Shard(1)
    │   │   L319-360: _shard_placement_fn() → per-param mesh
    │   └── L361-366: dense block → 普通 fully_shard
    │
    ├── L368: fully_shard(model)  [根节点]
    │
    ├── L370-371: enable_symm_mem → enable_fsdp_symm_mem(model)
    │
    ├── L374: disable_fsdp_gradient_division(model)
    │
    └── L386-424: EP prefetch 设置
```

### 7.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `data_parallel_shard_degree` | `torchtitan/config/configs.py:135` | FSDP 分片度数 |
| `fsdp_reshard_after_forward` | `torchtitan/config/configs.py:147` | default/always/never |
| `enable_fsdp_symm_mem` | `torchtitan/config/configs.py:162` | symmetric-memory 优化 |
| `enable_cpu_offload` | `torchtitan/config/configs.py:72` | CPU offload |

### 7.5 预期性能收益与验证

- **内存节省:** 线性于 DP 并行度
- **验证方法:** 对比单卡 vs 多卡内存占用

---

## 8. Async Pipeline (异步流水线)

### 8.1 原理

将模型分为多个 stage，异步执行，减少流水线 bubble。

### 8.2 适用场景

- 大规模模型（需要 PP 并行）
- 多节点训练
- Communication-bound 场景

### 8.3 完整代码调用链

```
Megatron: pipeline_parallel/schedules.py:53 get_forward_backward_func()
    │
    ├── pp_size == 1 → L723 forward_backward_no_pipelining
    │
    ├── vp_size > 1 → L1019 forward_backward_pipelining_with_interleaving
    │   L1486: combined_1f1b_schedule_for_interleaved_pipelining()
    │
    └── pp_size > 1, vp == None → L2147 forward_backward_pipelining_without_interleaving
        │  标准 1F1B 调度
        │
        ▼
Megatron: pipeline_parallel/combined_1f1b.py:35 combined_1f1b_schedule_for_no_pipelining
Megatron: pipeline_parallel/combined_1f1b.py:138 combined_1f1b_schedule_for_interleaved_pipelining
Megatron: pipeline_parallel/combined_1f1b.py:281 combined_forward_backward_step
```

### 8.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `--pipeline-model-parallel-size` | Megatron CLI | PP 度数 |
| `--virtual-pipeline-model-parallel-size` | Megatron CLI | interleaved VP 度数 |
| `--overlap-p2p-communication` | Megatron CLI | P2P 通信重叠 |

### 8.5 预期性能收益与验证

- **Bubble 减少:** 10-30%
- **验证方法:** 对比 PP stage 间 bubble 占比 (Nsight Systems)

---

## 9. MoE Fused (Triton Fused MoE)

### 9.1 原理

将 MoE 的 dispatch + expert GEMM + combine 融合为单个 triton kernel，减少 kernel launch 和 HBM 访问。

### 9.2 适用场景

- MoE 模型训练
- Expert Parallelism
- Compute-bound 场景

### 9.3 完整代码调用链

```
torchada/triton/runtime/fused_moe/fused_moe.py:331 fused_experts_impl(
    hidden_states, w1, w2, topk_weights, topk_ids, ...)
    │
    ├── L361-363: padded_size 计算 (FP8: 128, 非 FP8: 0)
    │
    ├── L366-373: 约束检查 (shape/contiguous/dtype)
    │
    ├── 内部: grouped GEMM (w1) → activation (silu) → grouped GEMM (w2)
    │
    └── 可选: fuse_sum_all_reduce (融合 all-reduce)
```

### 9.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `use_fp8_w8a8` | `fused_moe.py:343` | FP8 量化 MoE |
| `use_int8_w8a8` | `fused_moe.py:344` | INT8 量化 MoE |
| `filter_expert` | `fused_moe.py:359` | 过滤无效 expert |
| `fuse_sum_all_reduce` | `fused_moe.py:326` | 融合 sum + all-reduce |

### 9.5 预期性能收益与验证

- **MoE 计算加速:** 2-4x (vs 非融合实现)
- **验证方法:** 对比 MoE layer 耗时

---

## 10. Graph Rotation (CUDA Graph LRU 轮换)

### 10.1 原理

MUSA 架构下 CUDA graph executable 数量有上限 (~2048)。通过 LRU 轮换机制，在显存中只保留最近使用的 graph executable，超出时驱逐最久未使用的。

### 10.2 适用场景

- MUSA 架构训练
- 深层模型 (层数多 → graph 数量多)
- 推理部署 (多 batch shape)

### 10.3 完整代码调用链

```
torchada/_graph_rotation.py:138 _Rotation
    │
    ├── L146-160: __init__()
    │   L147: cap — 最大 live graph 数
    │   L149: _live: OrderedDict[id(graph) → weakref(graph)]  (LRU 顺序)
    │   L155-160: stats — register/evict/reinstantiate/build_failed 计数
    │
    ├── L162-178: _ensure_aux()
    │   L167: load_graph_rotation_ops()  ← C++ 扩展加载
    │   L172-173: 失败时 _aux_failed = True, stats["build_failed"] += 1
    │
    ├── L119-135: auto-probe cap
    │   L122: TORCHADA_GRAPH_EXEC_MARGIN 环境变量
    │   L125: cap = max(64, limit - margin)
    │
    └── 核心操作:
        register(graph) → 加入 _live
        evict() → 驱逐 LRU 末尾超出 cap 的 graph
        reinstantiate(graph) → 重新实例化被驱逐的 graph
```

### 10.4 配置参数与代码入口

| 参数 | 代码位置 | 说明 |
|------|---------|------|
| `TORCHADA_GRAPH_EXEC_MARGIN` | `_graph_rotation.py:122` | 轮换 margin |
| `_DEFAULT_CAP` | `_graph_rotation.py` | 默认 cap 值 |
| `_DEFAULT_MARGIN` | `_graph_rotation.py` | 默认 margin 值 |

### 10.5 预期性能收益与验证

- **避免 MUSA ~2048 graph cap OOM**
- **深层模型可训练**
- **验证方法:** 检查 stats["evict"] / stats["reinstantiate"] 计数

---

## 11. 速查表

| 类别 | 技术 | 原理 | 预期收益 | 源码仓 (关键路径) |
|------|------|------|---------|------------------|
| **计算** | Flash Attention 2/3/4 | Tiling + Fusion 减少 HBM 访问 | 2-4x attention 加速 | Megatron `attention.py:499 _run_core_attention` |
| **计算** | FP8 Training | 低精度 Tensor Core | 1.5-2x 计算吞吐 | torchtitan `float8.py:53 Float8LinearConverter` |
| **计算** | CUDA Graph | 图录制重放 | 10-20% launch 消除 | torchtitan `cudagraph.py:189 CUDAGraphWrapper` |
| **计算** | MoE Fused | triton 融合 MoE kernel | 2-4x MoE 加速 | torchada `fused_moe.py:331 fused_experts_impl` |
| **计算** | Graph Rotation | CUDA graph LRU 轮换 | 避免 MUSA cap OOM | torchada `_graph_rotation.py:138 _Rotation` |
| **内存** | Activation Recompute | 重算换内存 | 30-60% 内存节省 | torchtitan `activation_checkpoint.py:166/185/290` |
| **内存** | FSDP2 | 全分片 (ZeRO-3 等价) | 线性内存节省 | torchtitan `fsdp.py:168 apply_fsdp_to_decoder` |
| **通信** | Comm Overlap | 计算通信重叠 | 20-40% 通信隐藏 | Megatron `layers.py:634` (async all-reduce) |
| **通信** | Sequence Packing | 序列打包减少 padding | 10-30% 效率提升 | torchtitan `packing.py:21/58` |
| **通信** | Async Pipeline | 1F1B 异步流水线 | 10-30% bubble 减少 | Megatron `schedules.py:2147` |

---

## 引用

- `skill-performance/SKILL.md` — 5 步优化工作流 (带代码位置映射)
- `skill-performance/references/bottleneck-taxonomy.md` — 性能瓶颈分类体系
- `skill-tickets/` — 历史案例库