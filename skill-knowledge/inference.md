# 推理优化 (Inference) — 代码级深度分析

> 本文档聚焦本地仓库中**真实存在**的推理优化代码（torchlorada、torch_musa、Megatron-LM），所有文件路径均可溯源。vLLM/SGLang 仅作为外部知识引用并明确标注。

---

## 1. KV Cache 原理与实现

### 1.1 概念原理

**KV Cache** 缓存已计算的 Key-Value 对，避免 decode 阶段的重复计算：

```
每 token 每层 KV 显存 = 2 × num_kv_heads × head_dim × dtype_size

示例：Llama-3-70B, 80层, head_dim=128, FP16
单 token KV ≈ 2 × 8 × 128 × 2 bytes = 4 KB
单请求 4096 token ≈ 4096 × 4KB × 80 = 1.25 GB
```

**PagedAttention** 借鉴 OS 虚拟内存：将 KV Cache 切分为固定大小 block（通常 8-32 tokens），通过 block table 映射逻辑连续 → 物理离散：

```
┌──────────────────────────────────────────────────────────────┐
│                    PagedAttention 内存布局                     │
│                                                               │
│  逻辑视图（每个序列连续）：                                     │
│  Seq1: [KV₀ KV₁ KV₂ KV₃ KV₄]                                │
│  Seq2: [KV₀ KV₁ KV₂]                                         │
│                                                               │
│  Block Table（每序列一个）：                                   │
│  Seq1 → [Blk2][Blk5][Blk1][Blk7][Blk3]                       │
│  Seq2 → [Blk4][Blk6][Blk8]                                   │
│                                                               │
│  物理内存池（blocks 按需分配）：                                │
│  [0:_][1:S1][2:S1][3:S1][4:S2][5:S1][6:S2][7:S1][8:S2][9:_] │
│                                                               │
│  优势：消除内部碎片、支持动态长度、显存利用率 >95%              │
└──────────────────────────────────────────────────────────────┘
```

**Prefix Caching（前缀缓存）**：共享 system prompt / 多轮对话的公共前缀 block，通过 hash → block_id 映射实现 O(1) 查找。

### 1.2 各仓库中的 KV Cache 代码

#### Megatron-LM — KV Block Allocator（PagedAttention 核心）

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py:17` | `KVBlockAllocator` 类定义 — block 池管理 |
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py:64` | `block_bag = torch.arange(pool_size)` — block ID 栈初始化 |
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py:69` | `block_hashes = torch.full((pool_size,), -1)` — prefix caching hash 跟踪 |
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py:72` | `kv_hash_to_block_id: Dict[int, int]` — O(1) 前缀查找映射 |
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py:75` | `block_ref_counts` — 引用计数（0 = 可驱逐，>0 = 活跃使用） |

#### Megatron-LM — Triton KV Append Kernel（分页写入）

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py:21` | `_append_kv_cache_kernel` — Triton kernel：KV → paged cache 散射写入 |
| `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py:50` | "Each program handles one head of one token" — 2D grid 设计 |
| `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py:67` | `block_idx = tl.load(block_idx_ptr + token_idx)` — 加载目标 block 索引 |
| `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py:77` | `tl.load(key_head_ptr + offs_h * stride_key_hdim)` — 按 head 加载 K/V |

#### Megatron-LM — Speculative KV Rewind

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:27` | `_rewind_kv_cache_kernel` — 投机解码失败时 KV 状态回滚 |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:72` | `num_to_rewind = tl.where(prefill == 1, 0, NUM_SPEC_TOKENS - accepted)` — 计算回滚量 |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:77` | `new_offset = ((diff % BLOCK_SIZE_TOKENS) + BLOCK_SIZE_TOKENS) % BLOCK_SIZE_TOKENS` — Python 风格模运算处理负数 |

---

## 2. 量化推理

### 2.1 FP8/INT8/INT4 量化原理

```
┌───────────────────────────────────────────────────────────────┐
│                   量化方案对比                                  │
│                                                                │
│  方案    │ 权重  │ 激活  │ 压缩比 │ 硬件要求                    │
│  ────────┼───────┼───────┼────────┼───────────────────────────│
│  W4A16   │ INT4  │ FP16  │ ~3.5×  │ 通用                       │
│  W8A8    │ INT8  │ INT8  │ 2×     │ 通用                       │
│  FP8 E4M3│ FP8   │ FP8   │ 2×     │ H100/H200 原生 TensorCore │
│  MXFP8   │ FP8   │ FP8   │ 2×     │ Blackwell+ (block scale)  │
│  FP4     │ FP4   │ FP4   │ 4×     │ Blackwell 原生             │
│                                                                │
│  量化公式（per-group）：                                        │
│  scale = max(|x|) / dtype_max                                  │
│  x_q = clamp(round(x / scale), dtype_min, dtype_max)          │
│  反量化：x̂ = x_q × scale                                       │
└───────────────────────────────────────────────────────────────┘
```

**FP8 两种格式**：
- **E4M3**（4 exp + 3 mantissa）：前向激活/权重，精度高
- **E5M2**（5 exp + 2 mantissa）：梯度，动态范围大

**MXFP8（Microscaling FP8）**：每 32 个元素共享一个 8-bit e8m0 scale（2 的幂），cuBLAS 需要 swizzled 布局（128×4 macro-tile）。

### 2.2 torchada FP8 Quant Kernel

| 文件:行 | 功能 |
|---------|------|
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:9` | `fp8_dtype = torch.float8_e4m3fn` — 默认 FP8 类型 |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:12` | `_per_token_group_quant_8bit` — Triton kernel：per-token-group 量化核心 |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:46` | `_absmax = tl.maximum(tl.max(tl.abs(y)), eps)` — 计算 group absmax |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:49` | `y_q = tl.clamp(y * y_s_inv, bit8_min, bit8_max)` — 量化 + clamp |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:55` | `per_token_group_quant_fp8` — Python 入口函数 |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:94` | `M = x.numel() // group_size; N = group_size` — 计算 grid 维度 |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py:150` | `per_token_quant_int8` — INT8 per-token 量化变体 |

### 2.3 Megatron-LM MXFP8 Quant Kernel

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/quantization/mxfp8_quantize.py:17` | `MXFP8_BLOCK_SIZE = 32` — 32 元素共享 1 个 scale |
| `Megatron-LM/megatron/core/inference/quantization/mxfp8_quantize.py:41` | `_mxfp8_quant_swizzle_kernel` — Triton kernel + 融合 scale swizzle |
| `Megatron-LM/megatron/core/inference/quantization/mxfp8_quantize.py:54` | "round up in scale calculation" — 上取整策略（参考 MXFP8 论文） |
| `Megatron-LM/megatron/core/inference/quantization/mxfp8_quantize.py:60` | cuBLAS swizzled scale 布局说明（128×4 macro-tile 交错） |
| `Megatron-LM/megatron/core/inference/quantization/utils.py:62` | `resolve_mxfp8_backend` — 根据 GEMM 后端选择 MXFP8 quantizer |
| `Megatron-LM/megatron/core/inference/quantization/utils.py:17` | `MXFP8Backend`, `MXFP8Tensor` 导入 — TE/FlashInfer/torch 后端抽象 |

---

## 3. Graph Capture 在推理中的应用

### 3.1 Graph Capture 原理

```
┌───────────────────────────────────────────────────────────────┐
│               推理 Graph Capture 流水线                         │
│                                                                │
│  Warmup ──▶ Capture Begin ──▶ Record Ops ──▶ Capture End     │
│    │              │                                │           │
│    ▼              ▼                                ▼           │
│  预热 kernel   创建 graph/stream           实例化 graph_exec  │
│  避免 lazy init 记录所有 MUSA/CUDA ops      生成可重放对象      │
│                                                                │
│  ┌──────────────────────────────────────────────────────┐     │
│  │  Replay（推理时）：                                     │     │
│  │  1. copy_ 输入到静态 buffer（保持 capture 时地址）      │     │
│  │  2. graph_exec.replay() — 一次性提交整图                │     │
│  │  3. 读取静态输出 buffer                                  │     │
│  └──────────────────────────────────────────────────────┘     │
│                                                                │
│  收益：消除 Python dispatch 开销、减少 kernel launch 次数       │
│  代价：不支持动态 shape、首次 capture 耗时、内存固定分配        │
└───────────────────────────────────────────────────────────────┘
```

### 3.2 torchada Graph Rotation（MUSA 驱动限制 workaround）

**问题**：MUSA 驱动限制单进程 ~2048 个 live `musaGraphExec_t`。vLLM/SGLang 的 piecewise CUDA graphs 为每层每 size 实例化一个 executable，>40 层模型即超限。

**解决方案**：Graph *template*（`musaGraph_t`）不计入限制，仅 LRU 管理 live *executable*：

| 文件:行 | 功能 |
|---------|------|
| `torchlorada/src/torchada/_graph_rotation.py:1` | 模块文档：解释 MUSA ~2048 驱动限制与 workaround 原理 |
| `torchlorada/src/torchada/_graph_rotation.py:44` | `_DEFAULT_CAP = 1900` — 默认 executable 上限（略低于 ~2043 驱动限制） |
| `torchlorada/src/torchada/_graph_rotation.py:79` | `_probe_live_exec_limit` — 独立子进程探测真实驱动限制（避免 RNG 状态污染） |
| `torchlorada/src/torchada/_graph_rotation.py:113` | `_resolve_cap()` — 优先级：显式 env > 自动探测 > 默认值 |
| `torchlorada/src/torchada/_graph_rotation.py:138` | `_Rotation` 类 — LRU rotation 核心实现 |
| `torchlorada/src/torchada/_graph_rotation.py:149` | `_live: OrderedDict[int, weakref]` — 弱引用 LRU 字典（不阻止 GC） |
| `torchlorada/src/torchada/_graph_rotation.py:162` | `_ensure_aux()` — JIT 加载 aux `.so` 扩展（free_exec / inst_exec） |
| `torchlorada/src/torchada/_graph_rotation.py:186` | `_evict_locked()` — LRU 驱逐：free_exec 销毁旧 executable，保留 template |
| `torchlorada/src/torchada/_graph_rotation.py:221` | `on_replay()` — 重放时若 exec 已驱逐则 re-instantiate（~0.3ms） |
| `torchlorada/src/torchada/_graph_rotation.py:261` | `install()` — monkey-patch `MUSAGraph.capture_end/replay` 启用 rotation |

**关键环境变量**：
| 变量 | 默认值 | 含义 |
|------|--------|------|
| `TORCHADA_GRAPH_ROTATION` | `"1"` | 0 = 禁用 rotation |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | 显式设置 live executable 上限 |
| `TORCHADA_GRAPH_AUTOPROBE` | `"0"` | 1 = 启动时探测真实驱动限制 |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | 探测时的安全余量 |

### 3.3 torch_musa Graph Capture 后端（续）

| 文件:行 | 功能 |
|---------|------|
| `torch_musa/torch_musa/musa_graph/graphs.py:45` | `MUSAGraph` 类 — 封装 `musaGraph_t` / `musaGraphExec_t` |
| `torch_musa/torch_musa/musa_graph/graphs.py:61` | `capture_begin()` — 开始捕获（支持 pool 共享、capture_error_mode） |
| `torch_musa/torch_musa/musa_graph/graphs.py:84` | `capture_end()` — 结束捕获，标记 `_has_capture = True` |
| `torch_musa/torch_musa/musa_graph/graphs.py:112` | `replay()` — 重放捕获的 MUSA 工作 |
| `torch_musa/torch_musa/musa_graph/graphs.py:238` | `make_graphed_callables()` — 将 function/nn.Module 转为 graphed 版本 |
| `torch_musa/torch_musa/musa_graph/graphs.py:373` | warmup 循环 — 避免 lazy init 进入 capture |
| `torch_musa/torch_musa/musa_graph/graphs.py:403` | forward graphs 捕获（与 backward graphs 逆序捕获共享 mempool） |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:54` | `kLargeBuffer = 20971520`（20 MiB）— 大块分配阈值 |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:59` | `kMinBlockSize = 512` — 最小 block 512 字节 |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:117` | `Block` 结构体 — 显存块元数据（device, stream, stream_uses） |
| `torch_musa/torch_musa/csrc/aten/musa/musagraph.cpp:48` | `MUSAGraph::MUSAGraph(bool keep_graph)` — C++ 构造函数 |
| `torch_musa/torch_musa/csrc/aten/musa/musagraph.cpp:65` | `capture_begin()` — 注册 generator state、检查非默认 stream |
| `torch_musa/torch_musa/csrc/aten/musa/musagraph.cpp:53` | `register_generator_state()` — 图安全 RNG 状态管理 |

### 3.4 vLLM/SGLang 中的 CUDA Graph（外部知识）

> ⚠️ **来源说明**：以下内容基于 vLLM/SGLang 公开文档与论文，**非本地仓库代码**。

- **vLLM piecewise CUDA graphs**：为每层、每个 capture size 单独实例化 graph executable，浅模型用 "splitting" 整图 capture，深模型用 per-layer capture
- **SGLang CUDA graphs**：支持 `cuda_graph_prefill` / `cuda_graph_decode` 分段 capture，decode 阶段默认启用
- **CUDA Graph Trees**（PyTorch Inductor）：处理动态 shape 的图版本管理（`torch._inductor.cudagraph_trees`）

---

## 4. 投机解码（Speculative Decoding）

### 4.1 概念原理

```
┌────────────────────────────────────────────────────────────────┐
│                   Speculative Decoding 流程                     │
│                                                                 │
│  Draft Model（小）:  ──▶ d₁ ──▶ d₂ ──▶ d₃ ──▶ d₄ ──▶ d₅     │
│                       （快速推测，单 token 成本低）               │
│                                                                 │
│  Target Model（大）: ──▶ verify(d₁..d₅) 一次并行验证            │
│                       （利用大 batch 并行能力）                   │
│                                                                 │
│  Acceptance:          ✓    ✓    ✗                                │
│                       ↓    ↓    ↓                                │
│  接受 d₁, d₂；在第 3 个位置从 target 分布重新采样                │
│                                                                 │
│  有效吞吐量 ≈ accepted_count / (1 + verification_overhead)      │
│  典型加速比：1.5× - 3×（取决于 draft/target 分布匹配度）        │
└────────────────────────────────────────────────────────────────┘
```

**关键变体**：
| 变体 | 特点 | 适用场景 |
|------|------|----------|
| 标准投机 | 独立 draft 模型（如 7B draft → 70B target） | 有合适小模型 |
| Medusa | 多头并行预测每层 token | 无需额外模型，训练成本高 |
| Lookahead | N-gram + 雅可比迭代验证 | 重复性/可复述文本 |
| Self-speculative | 浅层 exit（early exit）作为 draft | 通用，需模型支持 |
| MTP（Multi-Token Prediction） | 模型训练时即预测多 token | DeepSeek-V3 采用 |

### 4.2 Megatron-LM 投机解码实现

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:27` | `_rewind_kv_cache_kernel` — Triton kernel：投机失败后 KV cache 状态回滚 |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:55` | "Each program handles exactly one request" — 支持 CUDA graph 的 padding grid |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py:72` | `num_to_rewind = tl.where(prefill == 1, 0, NUM_SPEC_TOKENS - accepted)` — prefill 阶段不回滚 |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_pytorch.py` | PyTorch fallback 实现（非 Triton 路径） |
| `Megatron-LM/megatron/core/inference/sampling_params.py:43` | `do_kv_handoff: bool = False` — 投机 decode 时的 KV handoff 控制 |

---

## 5. MUSA 推理栈（torchlorada + torch_musa）

### 5.1 完整推理调用链

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MUSA 推理栈架构                                     │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │  用户代码（vLLM/SGLang 适配层）                                │    │
│  │   - 输入 tokenize → position_ids → attention_mask            │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │  torchlorada（MUSA 适配补丁层）                                │    │
│  │   ├─ _patch.py:1495 _patch_flash_attn — FA 接口重定向         │    │
│  │   ├─ _graph_rotation.py:261 install() — graph rotation 注入   │    │
│  │   ├─ triton/kernels/quant/fp8.py — Triton 量化 kernel          │    │
│  │   └─ triton/runtime/fused_moe/ — MoE 推理融合                 │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │  torch_musa（MUSA 后端运行时）                                 │    │
│  │   ├─ musa_graph/graphs.py:45 MUSAGraph — 图捕获封装           │    │
│  │   ├─ csrc/core/MUSACachingAllocator.cpp — 显存管理             │    │
│  │   └─ csrc/aten/musa/musagraph.cpp — C++ 图捕获后端            │    │
│  └──────────────────────────┬──────────────────────────────────┘    │
│                             │                                        │
│  ┌──────────────────────────▼──────────────────────────────────┐    │
│  │  MUSA Driver / Runtime                                        │    │
│  │   ├— musaGraphInstantiate → musaGraphExec_t                   │    │
│  │   ├— musaGraphLaunch                                         │    │
│  │   └— musaMalloc/musaFree                                     │    │
│  └─────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

### 5.2 Flash Attention on MUSA

| 文件:行 | 功能 |
|---------|------|
| `torchlorada/src/torchada/_patch.py:1495` | `_patch_flash_attn()` — 重定向 `sgl_kernel.flash_attn` → `flash_attn_interface` |
| `torchlorada/src/torchada/_patch.py:1510` | "On MUSA, the mate package provides an equivalent flash_attn_interface package" — 设计意图注释 |
| `torchlorada/src/torchada/_patch.py:1526` | `sgl_kernel.flash_attn = flash_attn_interface` — 模块注册到 sys.modules |
| `torchlorada/src/torchada/_patch.py:1541` | `_drop_only_qv()` — 处理 FA3 `only_qv` 参数在 MUSA 上的兼容性 |

### 5.3 MoE Inference（Fused MoE on MUSA）

| 文件:行 | 功能 |
|---------|------|
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py:331` | `fused_experts_impl()` — MoE 推理核心实现入口 |
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py:361` | `padded_size = 128` — FP8/INT8 非 block-quant 时的 padding 大小 |
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py:366` | `use_int4_w4a16` 约束检查：`hidden_states.shape[1] // 2 == w1.shape[2]` |
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py:383` | `_prepare_fused_moe_run()` — 准备 Triton kernel 配置 + token 排序 |
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py:435` | `fused_moe()` — 高层 MoE 接口（含 moe_runner_config） |
| `torchlorada/src/torchada/triton/runtime/fused_moe/config.py:56` | `get_moe_configs()` — 从 JSON 加载优化后的 Triton kernel 配置 |
| `torchlorada/src/torchada/triton/runtime/fused_moe/config.py:78` | `get_config_file_name()` — 根据 E/N/device/block_shape 生成配置文件名 |

### 5.4 C++ 扩展与显存管理

| 文件:行 | 功能 |
|---------|------|
| `torchlorada/src/torchada/_cpp_ops.py:73` | `load_cpp_ops()` — 加载 C++ 算子覆盖扩展 |
| `torchlorada/src/torchada/_cpp_ops.py:28` | `_detect_musa_arch()` — 通过 `musaInfo` 探测 GPU 架构（mp_21/mp_22/mp_31） |
| `torchlorada/src/torchada/_cpp_ops.py:58` | `arch = f"mp_{major}{minor}"` — 计算能力到架构名映射 |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:54` | `kLargeBuffer = 20971520` — 20 MiB 大块阈值 |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:59` | `kMinBlockSize = 512; kSmallSize = 1048576; kSmallBuffer = 2097152` — 三级分配策略 |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp:117` | `Block` 结构体定义（device, stream, stream_uses） |

---

## 6. Megatron-LM 推理引擎核心

### 6.1 推理调度器

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/scheduler.py:17` | `Scheduler` 类 — 推理请求调度器 |
| `Megatron-LM/megatron/core/inference/scheduler.py:28` | `__init__(max_batch_size)` — 初始化 active/waiting/completed 三个请求池 |
| `Megatron-LM/megatron/core/inference/scheduler.py:42` | `add_request()` — 添加新请求（active pool 或 waiting pool） |
| `Megatron-LM/megatron/core/inference/scheduler.py:72` | 状态判定：`ACTIVE_BUT_NOT_GENERATING_TOKENS` vs `WAITING_IN_QUEUE` |

### 6.2 Dynamic Inference Context

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/contexts/dynamic_context.py:1` | 模块文档（NVIDIA 2026）— 动态推理上下文 |
| `Megatron-LM/megatron/core/inference/contexts/dynamic_context.py:55` | 导入 `KVBlockAllocator`, `MambaSlotAllocator` — KV + SSM 双allocator |
| `Megatron-LM/megatron/core/inference/contexts/dynamic_context.py:63` | 导入 `triton_append_key_value_cache` — Triton KV append kernel |
| `Megatron-LM/megatron/core/inference/contexts/dynamic_context.py:17` | 导入 `CUDAGraphBatchDimensionBuilder`, `InferenceBatchDimensions` — CUDA graph batch 维度 |

### 6.3 GPT Inference Wrapper

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py:19` | `GPTInferenceWrapper` 类 — GPT 模型推理包装器 |
| `Megatron-LM/megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py:34` | `prep_inference_input()` — 准备推理输入（tokens, attention_mask, position_ids） |
| `Megatron-LM/megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py:54` | `_build_attention_mask_and_position_ids()` — 构建 attention mask + position ids |
| `Megatron-LM/megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py:70` | 根据 `AttnBackend`（local/flash/fused/unfused/auto）选择 mask 策略 |

### 6.4 Prefill-Decode 分离（Disaggregated Inference）

| 文件:行 | 功能 |
|---------|------|
| `Megatron-LM/megatron/core/inference/disaggregation/engine.py:11` | `DisaggDynamicInferenceEngine` — 分离式推理引擎 |
| `Megatron-LM/megatron/core/inference/disaggregation/inference_state_handoff.py:1` | "Engine-side lifecycle for disaggregated prefill/decode state handoff" — 状态 hand-off 生命周期 |
| `Megatron-LM/megatron/core/inference/disaggregation/inference_state_handoff.py:51` | `_PreparedHandoffMetadata` — 预填充完成后的 handoff 元数据 |
| `Megatron-LM/megatron/core/inference/disaggregation/kv_reshard.py:14` | `KVShardLayout` — TP/PP/EP/ETP KV 分片布局描述 |
| `Megatron-LM/megatron/core/inference/disaggregation/kv_reshard.py:47` | 约束检查：`num_heads % tp_size == 0` |
| `Megatron-LM/megatron/core/inference/disaggregation/kv_reshard.py:58` | PP 约束：`num_layers % pp_size == 0`（非均匀 split 需显式指定 window） |

```
┌────────────────────────────────────────────────────────────────┐
│           Disaggregated Inference 架构                          │
│                                                                 │
│   Prefill Cluster (计算优化)        Decode Cluster (访存优化)    │
│   ┌─────────────────────┐           ┌─────────────────────┐    │
│   │ GPU (HBM 少, 算力高) │─KV──▶     │ GPU (HBM 多, 带宽高) │    │
│   │ 大 batch prefill     │ Transfer  │ 多并发 decode        │    │
│   └─────────────────────┘           └─────────────────────┘    │
│                                                                 │
│   Transfer Backends: NCCL / NIXL（NVIDIA InferenceX Library）  │
│   Reshard: prefill KV (TP=a) → decode KV (TP=b) 跨集群重分布   │
└────────────────────────────────────────────────────────────────┘
```

---

## 7. 关键配置参数表

### 7.1 torchada Graph Rotation 配置

| 参数 | 默认值 | 含义 |
|------|--------|------|
| `TORCHADA_GRAPH_ROTATION` | `"1"` | 设为 `"0"` 禁用 rotation |
| `TORCHADA_GRAPH_EXEC_CAP` | `1900` | live executable 上限 |
| `TORCHADA_GRAPH_AUTOPROBE` | `"0"` | `"1"` = 启动时探测真实驱动限制 |
| `TORCHADA_GRAPH_EXEC_MARGIN` | `128` | 探测安全余量 |

### 7.2 Megatron-LM 推理配置

| 参数 | 类型 | 含义 |
|------|------|------|
| `max_batch_size` | int | Scheduler 单次推理最大 batch |
| `num_tokens_to_generate` | int | 每请求最大生成 token 数 |
| `temperature` | float | 采样温度（1.0=贪心，<1=保守，>1=多样） |
| `top_k` | int | Top-K 采样（0=禁用） |
| `top_p` | float | Nucleus 采样（0.0=禁用） |
| `do_kv_handoff` | bool | 启用 KV 块 pinning 以支持跨节点传输 |
| `streaming` | bool | 启用增量 token 流式返回 |
| `enable_prefix_caching` | bool | KV Block Allocator 启用前缀缓存 |
| `prefix_caching_eviction_policy` | enum | `REF_ZERO` / `LRU` 驱逐策略 |
| `paused_limit` | int | paused request block 保留上限 |

### 7.3 KV Cache 配置

| 参数 | 含义 | 典型值 |
|------|------|--------|
| `block_size` | PagedAttention block 大小（tokens） | 8-32 |
| `pool_size` | KV block 池总 block 数 | 1024-65536 |
| `kv_cache_dtype` | KV 数据类型 | fp16 / fp8 / int8 |
| `gpu_memory_utilization` | KV Cache 显存占比 | 0.8-0.95 |

### 7.4 MoE Inference 配置（torchlorada）

| 参数 | 含义 |
|------|------|
| `use_fp8_w8a8` | FP8 权重+激活量化 |
| `use_int8_w8a8` | INT8 权重+激活量化 |
| `use_int8_w8a16` | INT8 权重 + FP16 激活 |
| `use_int4_w4a16` | INT4 权重 + FP16 激活（W4A16） |
| `per_channel_quant` | 通道级量化（vs per-token） |
| `block_shape` | Block quantization shape（MXFP8 风格） |
| `filter_expert` | 过滤无 token 的专家计算 |

---

## 8. 推理优化全调用链（核心路径）

以下展示一个 MoE 模型在 MUSA 上的完整推理优化调用链：

```
用户请求（prompt tokens）
    │
    ▼
torchada._graph_rotation.install()          ← 启动时注入 graph rotation
    │  monkey-patch MUSAGraph.capture_end/replay
    │
    ▼
torch_musa.MUSAGraph.capture_begin()        ← 开始图捕获
    │  C++: musagraph.cpp:65 capture_begin()
    │  注册 generator state（图安全 RNG）
    │
    ▼
[Capture Phase] Model.forward()
    │
    ├─ flash_attn_varlen_func(...)          ← FA 计算
    │   └─ _patch.py:1495 重定向到 flash_attn_interface
    │
    ├─ per_token_group_quant_fp8(x, 128)    ← FP8 量化（如需）
    │   └─ fp8.py:55 → Triton kernel :12 _per_token_group_quant_8bit
    │
    └─ fused_moe(hidden_states, w1, w2, topk)
        └─ fused_moe.py:435
           ├─ moe_align_block_size(topk_ids, ...)   ← token-专家分配
           └─ fused_experts_impl(...)               ← :331 MoE 核心
              └─ Triton GEMM kernel (autotuned config)
    │
    ▼
MUSAGraph.capture_end()                      ← 图捕获结束
    │  _graph_rotation.py:284 → rot.register(graph)
    │  若 live > cap → _evict_locked() 驱逐 LRU executable
    │
    ▼
MUSAGraph.replay()                           ← 推理时重放
    │  _graph_rotation.py:292 → rot.on_replay(graph)
    │  若 exec 已被驱逐 → aux.inst_exec(graph) 重实例化
    │
    ▼
输出 tokens
```

---

## 9. 源码文件索引

### torchada 推理相关

| 文件路径 | 功能描述 |
|----------|----------|
| `torchlorada/src/torchada/_graph_rotation.py` | Graph Rotation LRU 管理（~2048 驱动限制 workaround） |
| `torchlorada/src/torchada/triton/kernels/quant/fp8.py` | FP8/INT8 per-token-group 量化 Triton kernel |
| `torchlorada/src/torchada/triton/runtime/fused_moe/fused_moe.py` | Fused MoE 推理核心（Triton GEMM） |
| `torchlorada/src/torchada/triton/runtime/fused_moe/config.py` | MoE kernel 配置 autotune 与加载 |
| `torchlorada/src/torchada/_patch.py` | MUSA 适配补丁（flash_attn 重定向等） |
| `torchlorada/src/torchada/_cpp_ops.py` | C++ 算子扩展加载 + MUSA 架构探测 |

### torch_musa 推理相关

| 文件路径 | 功能描述 |
|----------|----------|
| `torch_musa/torch_musa/musa_graph/graphs.py` | MUSAGraph 封装 + make_graphed_callables |
| `torch_musa/torch_musa/csrc/core/MUSACachingAllocator.cpp` | MUSA 显存分配器（Block 池 + expandable segment） |
| `torch_musa/torch_musa/csrc/aten/musa/musagraph.cpp` | MUSA Graph C++ 后端（capture/register/replay） |

### Megatron-LM 推理相关

| 文件路径 | 功能描述 |
|----------|----------|
| `Megatron-LM/megatron/core/inference/scheduler.py` | 推理请求调度器 |
| `Megatron-LM/megatron/core/inference/sampling_params.py` | 采样参数定义 |
| `Megatron-LM/megatron/core/inference/contexts/kv_block_allocator.py` | PagedAttention block allocator + prefix caching |
| `Megatron-LM/megatron/core/inference/contexts/dynamic_context.py` | 动态推理上下文（KV/Mamba/CUDA Graph） |
| `Megatron-LM/megatron/core/inference/contexts/fused_kv_append_kernel.py` | Triton KV append kernel（分页写入） |
| `Megatron-LM/megatron/core/inference/quantization/mxfp8_quantize.py` | MXFP8 量化 kernel + swizzled scale |
| `Megatron-LM/megatron/core/inference/quantization/utils.py` | MXFP8 后端选择（TE/FlashInfer/torch） |
| `Megatron-LM/megatron/core/inference/text_generation_controllers/mtp_utils_triton.py` | 投机解码 KV rewind kernel |
| `Megatron-LM/megatron/core/inference/model_inference_wrappers/gpt/gpt_inference_wrapper.py` | GPT 模型推理包装器 |
| `Megatron-LM/megatron/core/inference/disaggregation/engine.py` | 分离式推理引擎 |
| `Megatron-LM/megatron/core/inference/disaggregation/inference_state_handoff.py` | Prefill/Decode 状态 hand-off |
| `Megatron-LM/megatron/core/inference/disaggregation/kv_reshard.py` | TP/PP/EP KV 分片重分布 |

---

## 10. 面试高频问题与代码对应

| 面试问题 | 代码证据 |
|----------|----------|
| 如何解决 MUSA >40 层模型无法用 piecewise CUDA graph？ | `torchlorada/_graph_rotation.py:138` LRU rotation，仅驱逐 executable 保留 template |
| FP8 量化 kernel 如何实现 per-group scale？ | `torchlorada/triton/kernels/quant/fp8.py:46` `_absmax = tl.maximum(tl.max(tl.abs(y)), eps)` |
| PagedAttention block 分配与 prefix caching？ | `Megatron-LM/kv_block_allocator.py:69` block_hashes + hash_to_block_id |
| 投机解码失败后如何回滚 KV？ | `Megatron-LM/mtp_utils_triton.py:72` `num_to_rewind = tl.where(prefill == 1, 0, NUM_SPEC_TOKENS - accepted)` |
| MoE 推理时 token 如何分配到 expert？ | `torchlorada/fused_moe.py:29` `moe_align_block_size()` + Triton 排序 |
| Prefill-Decode 分离如何跨节点传输 KV？ | `Megatron-LM/disaggregation/kv_reshard.py:14` `KVShardLayout` + NCCL/NIXL backend |
| MUSA 驱动 ~2048 限制如何探测？ | `torchlorada/_graph_rotation.py:79` `_probe_live_exec_limit()` 独立子进程 |

---

> **编写说明**：本文档所有 `torchlorada` / `torch_musa` / `Megatron-LM/` 路径均为本地仓库真实文件，已通过代码阅读验证。vLLM/SGLang 相关内容仅在第 3.4 节作为外部知识引用并明确标注，未编造文件路径。