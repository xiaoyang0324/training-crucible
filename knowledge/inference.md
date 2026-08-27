# 推理优化 (Inference Optimization) 知识专家

## 1. 概述

推理优化目标是在有限硬件资源下，最大化推理吞吐量、最小化延迟、控制成本。

- **三大核心指标**：

| 指标 | 定义 | 优化目标 |
|------|------|----------|
| TTFT (Time To First Token) | 首 token 延迟 | 用户体验 |
| TPOT (Time Per Output Token) | 输出 token 间延迟 | 生成流畅度 |
| Throughput | 单位时间处理请求数 | 成本效率 |

- **推理瓶颈**：
  - **Prefill 阶段**：计算密集（处理 prompt），受 FLOPs 限制
  - **Decode 阶段**：访存密集（逐 token 生成），受显存带宽限制

```
┌─────────────────────────────────────────────────────┐
│                 Inference Pipeline                   │
│                                                      │
│  Prompt ──▶ [Prefill] ──▶ [Decode] ──▶ Output      │
│              (Compute)     (Memory)                  │
│              Bound        Bound                      │
└─────────────────────────────────────────────────────┘
```

## 2. KV Cache

### 2.1 原理

KV Cache 缓存已计算的 Key-Value 对，避免重复计算：

```
Without KV Cache: 每生成一个 token 需重新计算所有历史 KV
With KV Cache:    只计算新 token 的 KV，与历史 KV 拼接

Memory per token per layer = 2 × num_heads × head_dim × dtype_size
```

### 2.2 PagedAttention

借鉴操作系统虚拟内存，将 KV Cache 分页管理：

```
┌─────────────────────────────────────────────────────┐
│              Paged KV Cache Layout                   │
│                                                      │
│  Logical View (连续):                                │
│  [Seq1: K0 K1 K2 K3] [Seq2: K0 K1 K2  ]            │
│                                                      │
│  Physical View (分页):                               │
│  Block Table:                                        │
│  Seq1 → [P0][P1][P2][P3]                           │
│  Seq2 → [P4][P5][P6][FREE]                          │
│                                                      │
│  Physical Memory (4 blocks):                         │
│  [P0:S1][P1:S1][P2:S1][P3:S1]                       │
│  [P4:S2][P5:S2][P6:S2][P7:FREE]                     │
└─────────────────────────────────────────────────────┘
```

**优势**：消除 KV Cache 碎片，支持动态长度，显存利用率 >95%。

### 2.3 KV Cache 量化

| 量化方案 | 精度 | 显存节省 | 质量影响 |
|----------|------|----------|----------|
| FP16 KV | 基准 | 1× | 无 |
| INT8 KV | 8-bit | 2× | 极小 |
| INT4 KV | 4-bit | 4× | 可控 |
| FP8 KV | 8-bit | 2× | 极小 |

> 源码参考：Megatron-LM `megatron/core/quantization/quant_config.py` — 量化配置

## 3. 量化 (Quantization)

### 3.1 量化方案对比

| 方案 | 权重 | 激活 | 适用场景 | 工具链 |
|------|------|------|----------|--------|
| W4A16 | 4-bit | FP16 | 低内存部署 | AWQ/GPTQ |
| W8A8 | 8-bit | 8-bit | 通用加速 | SmoothQuant |
| FP8 | FP8 | FP8 | H100/H200 原生 | Transformer Engine |
| FP4 | FP4 | FP4 | 极致压缩 | Blackwell 原生 |

### 3.2 W4A16 (Weight-only Quantization)

```
原始: Y = X × W                    (W: FP16, 2 bytes/elem)
量化: Y = X × (W_int4 × scale)     (W_int4: 0.5 bytes/elem)

压缩比 ≈ 4× (考虑 scale 和 zero point 开销后约 3.5×)
```

> 源码参考：Megatron-LM `megatron/core/quantization/utils.py` — 量化工具函数

### 3.3 FP8 量化

H100/H200 GPU 原生支持 FP8 (E4M3/E5M2)：

```
FP8 E4M3: 1 sign + 4 exp + 3 mantissa  (适合前向激活)
FP8 E5M2: 1 sign + 5 exp + 2 mantissa  (适合梯度)

┌─────────────────────────────────────────┐
│         FP8 Quantization Flow            │
│                                          │
│  FP32/BF16 ──▶ Scale ──▶ FP8 ──▶ Compute│
│                 ▲                        │
│                 │                        │
│           amax history                  │
└─────────────────────────────────────────┘
```

> 源码参考：
> - Megatron-LM `megatron/core/fp8_utils.py` — FP8 量化/反量化工具
> - Megatron-LM `megatron/core/quantization/quant_config.py` — 量化配置

### 3.4 FP4 量化

Blackwell 架构原生支持 FP4，进一步压缩：

> 源码参考：Megatron-LM `megatron/core/fp4_utils.py` — FP4 量化工具

## 4. 投机解码 (Speculative Decoding)

### 4.1 原理

使用小模型 (draft model) 快速生成候选 token，大模型 (target model) 并行验证：

```
┌─────────────────────────────────────────────────────┐
│            Speculative Decoding Flow                 │
│                                                      │
│  Draft Model (小):  ─▶ t1 ─▶ t2 ─▶ t3 ─▶ t4       │
│                       (快速推测，可能错误)             │
│                                                      │
│  Target Model (大): ─▶ verify(t1,t2,t3,t4)          │
│                       (一次并行验证)                  │
│                                                      │
│  Result: 接受正确前缀，从第一个错误位置继续            │
│                                                      │
│  加速比 ≈ 1 / (1 - acceptance_rate × draft_len)     │
└─────────────────────────────────────────────────────┘
```

### 4.2 关键参数

- **Draft Model 选择**：小模型 / 模型早退 (early exit) / 自投机
- **推测长度 (γ)**：通常 3-8 tokens
- **接受率**：取决于 draft 与 target 的分布相似度

### 4.3 变体

| 变体 | 特点 | 适用场景 |
|------|------|----------|
| 标准投机 | 独立 draft 模型 | 有合适小模型 |
| Medusa | 多头并行预测 | 无需额外模型 |
| Lookahead | N-gram 推测 | 重复性文本 |
| Self-speculative | 模型浅层推测 | 通用场景 |

## 5. 推理服务

### 5.1 Continuous Batching (迭代级批处理)

```
传统 Static Batching:
[Req1: ████████      ]  等待最长序列完成
[Req2: ██            ]  空闲等待
[Req3: ██████        ]  空闲等待

Continuous Batching:
[Req1: ████████      ]
[Req2: ██][Req4: ████]  短请求完成后立即插入新请求
[Req3: ██████][Req5: █]  GPU 利用率最大化
```

### 5.2 SGLang 推理引擎

SGLang 是高性能推理框架，支持 RadixAttention、压缩状态管理：

> 源码参考：miles `miles/rollout/sglang_rollout.py` — SGLang 在 RL rollout 中的集成

### 5.3 Prefill-Decode 分离

```
┌─────────────────────────────────────────────────────┐
│           Disaggregated Inference                    │
│                                                      │
│  Prefill Cluster          Decode Cluster             │
│  (Compute-optimized)      (Memory-optimized)         │
│  ┌─────────────┐          ┌─────────────┐           │
│  │ GPU (HBM少) │──KV──▶   │ GPU (HBM多) │           │
│  │ 高算力       │  Transfer│ 高带宽       │           │
│  └─────────────┘          └─────────────┘           │
└─────────────────────────────────────────────────────┘
```

## 6. 关键配置参数表

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `max_num_seqs` | 最大并发序列数 | 32-256 | 吞吐量 |
| `max_num_batched_tokens` | 最大 batch token 数 | 2048-8192 | prefill 吞吐 |
| `gpu_memory_utilization` | KV Cache 显存占比 | 0.8-0.95 | 容量 vs 安全 |
| `max_model_len` | 最大序列长度 | 模型上限 | 支持上下文 |
| `block_size` | PagedAttention 块大小 | 8-32 | 显存碎片 |
| `kv_cache_dtype` | KV Cache 数据类型 | fp16/fp8/int4 | 显存与质量 |
| `quantization` | 量化方案 | w4a16/fp8 | 速度 vs 精度 |
| `speculative_num_draft_tokens` | 投机推测长度 | 3-8 | 加速比 |
| `enforce_eager` | 禁用 CUDA Graph | True/False | 动态 shape |
| `tensor_parallel_size` | 推理 TP 并行度 | 1-8 | 单请求延迟 |

## 7. 常见误区

### ❌ 误区 1："量化必然损失精度"
**正解**：
- W4A16 在多数任务上精度损失 <1%
- FP8 几乎无损（H100 原生支持）
- 量化感知训练 (QAT) 可进一步缩小差距
- 推理质量更多取决于模型本身而非量化

### ❌ 误区 2："KV Cache 越大越好"
**正解**：
- KV Cache 占用显存与 batch×seq 成正比
- 过大的 KV Cache 挤占模型权重空间
- 需平衡 `gpu_memory_utilization` 与并发数
- PagedAttention 已解决碎片问题

### ❌ 误区 3："投机解码总是能加速"
**正解**：投机解码加速条件：
- Draft model 接受率需足够高（>70%）
- 验证开销需小于直接生成
- 小 batch 下验证并行度不足，可能负优化
- 需要 draft 与 target 分布匹配

### ❌ 误区 4："Continuous Batching 只优化吞吐量"
**正解**：Continuous batching 同时优化：
- 吞吐量（GPU 利用率提升）
- 延迟（短请求无需等待）
- 公平性（避免队头阻塞）

### ❌ 误区 5："推理只需关注 Decode 阶段"
**正解**：
- Prefill 阶段在长 prompt 下成为瓶颈
- TTFT 是交互体验的关键
- Prefill-Decode 分离是前沿优化方向
- 混合 batch 中 prefill 会阻塞 decode

## 8. 源码文件索引表

| 文件路径 | 功能描述 |
|----------|----------|
| `Megatron-LM/megatron/core/quantization/quant_config.py` | 量化配置定义 |
| `Megatron-LM/megatron/core/quantization/utils.py` | 量化工具函数 |
| `Megatron-LM/megatron/core/fp8_utils.py` | FP8 量化/反量化实现 |
| `Megatron-LM/megatron/core/fp4_utils.py` | FP4 量化工具 |
| `Megatron-LM/megatron/core/inference/` | 推理引擎核心 |
| `Megatron-LM/megatron/core/inference_params.py` | 推理参数管理 |
| `miles/miles/rollout/sglang_rollout.py` | SGLang 推理集成 |
| `miles/miles/backends/sglang_utils/` | SGLang 后端工具 |
| `slime/slime/rollout/sglang_rollout.py` | SGLang rollout 实现 |
| `slime/slime/backends/sglang_utils/` | SGLang 后端集成 |
| `torchtitan/torchtitan/distributed/fsdp.py` | FSDP 推理支持 |

## 相关案例

- TICKET-20260827-005 — 美团推理 TND 布局下 dropout 无法入图
- TICKET-20260827-004 — PP+TP 混合并行下图捕获 stream 冲突
