# 预训练 (Pre-training) 知识专家

## 1. 概述

预训练是在大规模无标注语料上训练模型的基础阶段，目标是让模型学习通用的语言/视觉表征能力。

- **定义**：以自回归语言建模（next-token prediction）或掩码重建为目标，在万亿级 token 数据上训练。
- **目标**：获得强大的基础能力（推理、代码、数学、世界知识），为后训练提供高质量基座。
- **核心挑战 — 三墙问题**：
  - **内存墙 (Memory Wall)**：模型参数 + 优化器状态 + 梯度 + 激活值超出单卡显存。
  - **计算墙 (Compute Wall)**：FLOPs 需求巨大，需最大化 MFU (Model FLOPs Utilization)。
  - **通信墙 (Communication Wall)**：多卡并行时集合通信成为瓶颈，需 overlap 隐藏。

> 源码参考：Megatron-LM `megatron/core/config.py` 定义了训练全局配置；`megatron/core/model_parallel_config.py` 管理并行拓扑。

## 2. 并行策略

### 2.1 并行维度定义

| 缩写 | 全称 | 切分对象 | 通信类型 |
|------|------|----------|----------|
| DP  | Data Parallel | 数据 | AllReduce 梯度 |
| TP  | Tensor Parallel | 权重矩阵列/行 | AllReduce 激活 |
| PP  | Pipeline Parallel | 模型层 | P2P Send/Recv |
| CP  | Context Parallel | 序列维度 | AllGather/ReduceScatter |
| EP  | Expert Parallel | MoE 专家 | AlltoAll |

### 2.2 并行组合规则

```
┌─────────────────────────────────────────────────────┐
│                    Global Cluster                    │
│  ┌───────────────────────────────────────────────┐  │
│  │              Pipeline Parallel (PP)            │  │
│  │  ┌─────────┐   ┌─────────┐   ┌─────────┐     │  │
│  │  │ Stage 0 │──▶│ Stage 1 │──▶│ Stage 2 │     │  │
│  │  │ TP×DP×CP│   │ TP×DP×CP│   │ TP×DP×CP│     │  │
│  │  └─────────┘   └─────────┘   └─────────┘     │  │
│  └───────────────────────────────────────────────┘  │
│                                                      │
│  World Size = TP × PP × DP × CP                     │
└─────────────────────────────────────────────────────┘
```

**组合原则**：
- **TP** 优先在节点内（NVLink 高带宽），跨节点 TP 会严重降速。
- **PP** 用于突破单节点限制，层间切分引入 pipeline bubble。
- **CP** 处理超长序列（>32K），将序列分片到多卡。
- **EP** 仅 MoE 模型使用，专家分散到不同卡。

> 源码参考：
> - Megatron-LM `megatron/core/parallel_state.py` — 并行组初始化与管理
> - Megatron-LM `megatron/core/tensor_parallel/` — TP 实现（列并行/行并行）
> - Megatron-LM `megatron/core/pipeline_parallel/` — PP 调度（1F1B / interleaved）
> - Megatron-LM `megatron/core/context_parallel/` — CP 实现（Ulysses / Ring）
> - torchtitan `torchtitan/distributed/parallel_dims.py` — 并行维度配置
> - torchtitan `torchtitan/distributed/tensor_parallel.py` — TP 应用
> - torchtitan `torchtitan/distributed/pipeline_parallel.py` — PP 调度
> - torchtitan `torchtitan/distributed/context_parallel/` — CP 实现

### 2.3 Pipeline Bubble

```
1F1B Schedule (micro-batch = 4, PP = 2):

GPU0 (Stage0): [F0][F1][F2][F3][B0][B1][B2][B3]
GPU1 (Stage1): [ID][F0][F1][F2][F3][B0][B1][B2][B3]
                ↑ bubble = (PP-1) / (micro_batch + PP - 1)
```

Bubble 率 = (PP-1) / (micro_batch + PP - 1)，增大 micro-batch 可降低 bubble。

## 3. 内存优化

### 3.1 Activation Recompute (激活重算)

前向时不保存中间激活，反向时重新计算。以计算换内存。

- **Full Recompute**：重算整个 transformer block 的激活（最省内存，+30% 计算）。
- **Selective Recompute**：只重算昂贵的大激活（如 attention score）。

> 源码参考：Megatron-LM `megatron/core/recompute.py` — 重算策略实现

### 3.2 ZeRO 优化器状态分片

| Stage | 分片内容 | 通信量 | 显存节省 |
|-------|----------|--------|----------|
| ZeRO-1 | 优化器状态 | 1× | 4× (Adam) |
| ZeRO-2 | + 梯度 | 1× | 接近参数数×8 |
| ZeRO-3 | + 参数 | 1.5× | 线性扩展 |

> 源码参考：Megatron-LM `megatron/core/distributed/` — 分布式优化器实现

### 3.3 Gradient Checkpointing

与 Activation Recompute 同义，Megatron 中通过 `recompute_granularity` 控制。

### 3.4 CPU Offload

将优化器状态/参数卸载到 CPU 内存，适合超大模型。

> 源码参考：torchtitan `torchtitan/distributed/activation_checkpoint.py`

## 4. 关键配置参数表

| 参数 | 含义 | 典型值 | 影响 |
|------|------|--------|------|
| `micro_batch_size` | 单卡单次前向的 batch | 1-4 | 影响 bubble 和显存 |
| `global_batch_size` | 一次 optimizer step 的总样本 | 512-4096 | 影响收敛稳定性 |
| `gradient_accumulation_steps` | 梯度累积次数 | GBS/(MBS×DP) | 等效大 batch |
| `seq_length` | 序列长度 | 4K-128K | 影响 CP 需求 |
| `lr_warmup_steps` | 学习率预热步数 | 100-1000 | 训练稳定性 |
| `lr_decay_style` | 学习率衰减策略 | cosine/linear | 后期收敛质量 |
| `min_lr` | 最小学习率 | peak_lr×0.1 | 训练末期精度 |
| `weight_decay` | 权重衰减 | 0.01-0.1 | 正则化 |
| `clip_grad` | 梯度裁剪阈值 | 1.0 | 训练稳定性 |

**关系公式**：
```
global_batch_size = micro_batch_size × DP × gradient_accumulation_steps
```

## 5. 常见误区

### ❌ 误区 1："TP 越多越好"
**正解**：TP 受限于节点内互联带宽。8 卡 NVLink 内 TP=8 高效，跨节点 TP 带宽下降 10×+，此时应使用 PP。TP 的 AllReduce 通信量与 batch×seq 成正比，大 batch 下 TP 通信占比更高。

### ❌ 误区 2："PP bubble 无法避免，只能接受"
**正解**：Bubble 可通过以下方式降低：
- 增大 micro-batch size（降低 bubble 比例）
- Interleaved 1F1B（virtual pipeline，多 micro-batch 交错）
- Zero Bubble 方案（将反向拆分为 activation backward 和 weight backward）

> 源码参考：Megatron-LM `megatron/core/pipeline_parallel/combined_1f1b.py`

### ❌ 误区 3："Activation Recompute 总是值得开"
**正解**：Recompute 增加约 30% 计算量。当显存充足时（如小模型+大 TP），关闭 recompute 可提升吞吐。应通过 `recompute_granularity` 精细控制。

### ❌ 误区 4："global_batch_size 越大收敛越快"
**正解**：大 GBS 减少更新次数，但可能降低泛化性。需配合 LR scaling（如 Chinchilla scaling law），且 GBS 过大时需更多 warmup。

### ❌ 误区 5："ZeRO-3 总是比 ZeRO-2 好"
**正解**：ZeRO-3 引入额外 AllGather 通信（参数收集），在 TP 已切分参数的场景下，ZeRO-2 通常足够且通信更优。

## 6. 源码文件索引表

| 文件路径 | 功能描述 |
|----------|----------|
| `Megatron-LM/megatron/core/parallel_state.py` | 并行组初始化（TP/PP/DP/CP/EP 分组） |
| `Megatron-LM/megatron/core/tensor_parallel/` | 张量并行实现（列并行、行并行、词嵌入并行） |
| `Megatron-LM/megatron/core/pipeline_parallel/` | 流水线并行调度（1F1B、interleaved） |
| `Megatron-LM/megatron/core/context_parallel/` | 上下文并行（序列维度切分） |
| `Megatron-LM/megatron/core/recompute.py` | 激活重算策略 |
| `Megatron-LM/megatron/core/distributed/` | 分布式优化器与梯度同步 |
| `Megatron-LM/megatron/core/model_parallel_config.py` | 并行配置管理 |
| `torchtitan/torchtitan/distributed/parallel_dims.py` | 并行维度定义与计算 |
| `torchtitan/torchtitan/distributed/tensor_parallel.py` | TP 应用与通信 |
| `torchtitan/torchtitan/distributed/pipeline_parallel.py` | PP 调度实现 |
| `torchtitan/torchtitan/distributed/context_parallel/` | CP 实现 |
| `torchtitan/torchtitan/distributed/activation_checkpoint.py` | 激活检查点（重算） |
| `torchtitan/torchtitan/distributed/fsdp.py` | FSDP 数据并行实现 |
