# SOTA 优化技术目录 (State-of-the-Art Optimization Techniques)

> 本文档收录训练性能优化的 SOTA 技术，每个技术包含原理、适用场景、配置方法、预期收益和源码引用。
> 用于 Step 3 (Match) 和 Step 4 (Adapt) 匹配和适配优化方案。

---

## 1. 内存优化技术

### 1.1 Activation Recompute (激活重算)

**原理：**
- 前向传播时不保存中间激活值，反向传播时重新计算
- 用计算换内存，显著降低激活值内存占用

**适用场景：**
- 大模型训练，激活值占用大量内存
- 长序列训练（激活值随序列长度线性增长）
- 内存不足但计算资源有冗余

**配置方法：**
- 选择性 recompute：只重算注意力层等大型层的激活
- 全量 recompute：重算所有 Transformer 层的激活

**预期收益：**
- 内存节省：30-60%（取决于 recompute 比例）
- 计算开销增加：约 10-30%

**源码引用：**
- Megatron-LM: `megatron/core/recompute.py` — 激活重算核心实现
- Megatron-LM: `megatron/core/transformer/transformer_layer.py` — 层级别 recompute 控制
- torchtitan: `torchtitan/distributed/` — FSDP + recompute 集成

---

### 1.2 ZeRO-1/2/3 (Zero Redundancy Optimizer)

**原理：**
- ZeRO-1：分片优化器状态（shard optimizer states）
- ZeRO-2：分片优化器状态 + 梯度
- ZeRO-3：分片优化器状态 + 梯度 + 权重
- 通过分片减少每卡的内存冗余

**适用场景：**
- 大模型训练，优化器状态占用大量内存
- 多卡分布式训练

**配置方法：**
- ZeRO-1：适合中等规模模型
- ZeRO-2：适合大规模模型
- ZeRO-3：适合超大规模模型（通信开销增大）

**预期收益：**
- 内存节省：ZeRO-1 4x, ZeRO-2 8x, ZeRO-3 线性于 DP 并行度
- 通信开销：ZeRO-3 > ZeRO-2 > ZeRO-1

**源码引用：**
- DeepSpeed: `deepspeed/runtime/zero/` — ZeRO 核心实现
- torchtitan: `torchtitan/distributed/fsdp.py` — FSDP2 (ZeRO-3 等价)

---

### 1.3 CPU Offload

**原理：**
- 将优化器状态、梯度或权重卸载到 CPU 内存
- 减少 GPU 内存占用，利用 CPU 大内存

**适用场景：**
- GPU 内存不足但 CPU 内存充足
- 配合 ZeRO 使用进一步优化

**配置方法：**
- 卸载优化器状态到 CPU（最常见）
- 卸载梯度到 CPU
- 卸载权重到 CPU（通信开销大）

**预期收益：**
- 内存节省：取决于卸载内容
- 通信开销：CPU-GPU 数据传输增加

**源码引用：**
- DeepSpeed: `deepspeed/runtime/zero/offload.py` — CPU offload 实现
- Megatron-LM: `megatron/core/distributed/` — 分布式优化器 offload

---

### 1.4 Gradient Checkpointing

**原理：**
- 与 Activation Recompute 类似，但更细粒度
- 在 checkpoint 点保存激活值，中间值重算

**适用场景：**
- 通用内存优化
- 与 Activation Recompute 可叠加

**源码引用：**
- PyTorch: `torch.utils.checkpoint` — 标准实现
- Megatron-LM: `megatron/core/recompute.py` — 集成实现

---

## 2. 计算优化技术

### 2.1 Flash Attention 2/3

**原理：**
- 通过 tiling 和 kernel fusion 减少 HBM 访问
- 将 attention 计算分解为小块在 SRAM 中完成
- Flash Attention 3 针对 Hopper 架构优化

**适用场景：**
- 所有 Transformer 模型
- 长序列训练（序列越长收益越大）

**配置方法：**
- 替换标准 attention 实现为 Flash Attention
- 启用 causal mask（decoder-only 模型）

**预期收益：**
- Attention 计算加速：2-4x
- HBM 读写减少：O(N) vs O(N²)

**源码引用：**
- Megatron-LM: `megatron/core/transformer/dot_product_attention.py` — attention 实现
- Megatron-LM: `megatron/core/transformer/transformer_config.py` — attention 配置
- flash-attention 库: `flash_attn/flash_attn_interface.py` — Flash Attention 核心

---

### 2.2 Fused Operators (算子融合)

**原理：**
- 将多个小算子融合为一个大算子
- 减少 kernel launch 次数和中间值 HBM 读写

**适用场景：**
- 频繁出现的小算子组合（如 LayerNorm + Linear）
- 计算密集型模型

**常见融合：**
- LayerNorm + Linear
- Bias + Activation
- Adam optimizer 融合更新

**预期收益：**
- 整体加速：10-30%
- 内存读写减少

**源码引用：**
- Megatron-LM: `megatron/core/transformer/` — 融合算子实现
- torchtitan: `torchtitan/ops/` — 自定义融合 kernel

---

### 2.3 CUDA Graph

**原理：**
- 将多个 kernel 录制为图，一次性提交
- 消除 kernel launch 开销和 Python 调度开销

**适用场景：**
- 固定 shape 的训练循环
- 大量小 kernel 的场景

**配置方法：**
- 录制前向 + 反向 + 优化器步骤
- 在首次迭代录制，后续迭代重放

**预期收益：**
- 整体加速：10-20%
- Launch 开销几乎消除

**源码引用：**
- Megatron-LM: `megatron/core/` — CUDA Graph 集成
- PyTorch: `torch.cuda.CUDAGraph` — CUDA Graph API

---

### 2.4 FP8/FP4 训练

**原理：**
- 使用 FP8 (8-bit) 或 FP4 (4-bit) 进行矩阵乘法
- 利用 Hopper/Ada 架构的 Tensor Core 支持
- 配合 per-tensor/per-channel scaling 保持精度

**适用场景：**
- Hopper (H100) 或 Ada (RTX 4090) 架构
- 大模型训练（计算密集型）

**配置方法：**
- 启用 FP8 训练模式
- 配置 scaling 策略（per-tensor/per-channel）

**预期收益：**
- 计算吞吐提升：1.5-2x
- 内存占用减少（激活值/权重）

**源码引用：**
- Megatron-LM: `megatron/core/fp8_utils.py` — FP8 训练核心实现
- Megatron-LM: `megatron/core/fp4_utils.py` — FP4 训练核心实现
- Megatron-LM: `megatron/core/transformer/transformer_config.py` — `fp8` 配置

---

## 3. 通信优化技术

### 3.1 Communication-Overlap (通信重叠)

**原理：**
- 将通信操作（allreduce/allgather）与计算操作重叠执行
- 利用 NCCL 的异步通信能力

**适用场景：**
- 通信占比高的分布式训练
- 梯度同步、权重同步

**常见策略：**
- 梯度预取（gradient prefetching）
- 权重预取（weight prefetching）
- 计算-通信交错

**预期收益：**
- 通信隐藏：20-40%
- 整体加速：10-30%

**源码引用：**
- Megatron-LM: `megatron/core/pipeline_parallel/schedules.py` — 1F1B 调度中的通信重叠
- Megatron-LM: `megatron/core/parallel/tensor_parallel/` — TP 通信重叠
- DeepSpeed: `deepspeed/comm/` — 通信优化

---

### 3.2 Sequence Packing (序列打包)

**原理：**
- 将多个短序列打包到一个长序列中
- 减少 padding 带来的无效计算和通信

**适用场景：**
- 变长序列训练
- 数据长度差异大

**配置方法：**
- 设置 packing 策略（greedy/best-fit）
- 添加 attention mask 区分不同序列

**预期收益：**
- 计算效率提升：10-30%（取决于 padding 比例）
- 通信量减少

**源码引用：**
- Megatron-LM: `megatron/core/datasets/` — 数据 packing 实现
- torchtitan: `torchtitan/components/dataloader.py` — packing 支持

---

### 3.3 Async Pipeline (异步流水线)

**原理：**
- 将模型分为多个 stage，异步执行
- 减少流水线 bubble

**适用场景：**
- 大规模模型（需要 PP 并行）
- 多节点训练

**常见策略：**
- 1F1B (One Forward One Backward)
- Interleaved 1F1B
- Zero Bubble

**预期收益：**
- Bubble 减少：10-30%
- 内存节省（减少同时活跃的 micro-batch 数）

**源码引用：**
- Megatron-LM: `megatron/core/pipeline_parallel/schedules.py` — 1F1B / Interleaved 调度
- Megatron-LM: `megatron/core/pipeline_parallel/p2p_communication.py` — stage 间通信

---

### 3.4 MoE Dispatch (MoE 通信优化)

**原理：**
- 优化 MoE 模型的 All-to-All 通信
- 使用 DeepEP 等高效通信库

**适用场景：**
- MoE 模型训练
- Expert Parallelism

**配置方法：
- 选择 dispatcher（DeepEP / 原生 all-to-all）
- 配置 EP 并行度

**预期收益：**
- All-to-All 延迟降低
- MoE 训练加速

**源码引用：**
- Megatron-LM: `megatron/core/transformer/moe/` — MoE 实现与通信
- DeepEP 库 — 高效 MoE 通信

---

## 4. 并行策略

### 4.1 FSDP2 (Fully Sharded Data Parallelism)

**原理：**
- PyTorch 原生的 ZeRO-3 等价实现
- 权重、梯度、优化器状态全分片
- 支持 per-layer 控制

**适用场景：**
- PyTorch 原生训练
- 大规模模型训练

**配置方法：**
- 配置分片粒度（per-layer/per-model）
- 配置 reshard 策略

**预期收益：**
- 内存节省：线性于 DP 并行度
- 与 DTensor 集成

**源码引用：**
- torchtitan: `torchtitan/distributed/fsdp.py` — FSDP2 配置与使用
- PyTorch: `torch.distributed.fsdp` — FSDP2 API

---

### 4.2 Context Parallel (上下文并行)

**原理：**
- 将长序列切分到多卡上并行处理
- Ulysses 和 Ring 两种主要策略

**适用场景：**
- 超长序列训练（>32K tokens）
- 序列并行

**配置方法：**
- 选择 CP 策略（Ulysses/Ring）
- 配置 CP 并行度

**预期收益：**
- 长序列训练加速
- 内存分摊

**源码引用：**
- Megatron-LM: `megatron/core/tensor_parallel/` — CP 实现
- torchtitan: `torchtitan/distributed/` — CP 集成

---

### 4.3 Expert Parallel (专家并行)

**原理：**
- 将 MoE 的 expert 分布到不同 GPU
- 配合 All-to-All 通信

**适用场景：**
- MoE 模型训练
- Expert 数量多

**源码引用：**
- Megatron-LM: `megatron/core/transformer/moe/` — EP 实现

---

## 5. 速查表

| 类别 | 技术 | 原理 | 预期收益 | 源码仓 |
|------|------|------|---------|--------|
| **内存** | Activation Recompute | 重算换内存 | 30-60% 内存节省 | Megatron-LM `megatron/core/recompute.py` |
| **内存** | ZeRO-1/2/3 | 状态分片 | 线性内存节省 | DeepSpeed `deepspeed/runtime/zero/` |
| **内存** | CPU Offload | CPU 卸载 | 释放 GPU 内存 | DeepSpeed `deepspeed/runtime/zero/offload.py` |
| **计算** | Flash Attention 2/3 | Tiling + Fusion | 2-4x attention 加速 | Megatron-LM `megatron/core/transformer/dot_product_attention.py` |
| **计算** | Fused Operators | 算子融合 | 10-30% 整体加速 | Megatron-LM `megatron/core/transformer/` |
| **计算** | CUDA Graph | 图录制重放 | 10-20% launch 消除 | PyTorch `torch.cuda.CUDAGraph` |
| **计算** | FP8/FP4 训练 | 低精度 Tensor Core | 1.5-2x 计算吞吐 | Megatron-LM `megatron/core/fp8_utils.py` |
| **通信** | Comm Overlap | 计算通信重叠 | 20-40% 通信隐藏 | Megatron-LM `megatron/core/pipeline_parallel/schedules.py` |
| **通信** | Sequence Packing | 序列打包 | 10-30% 效率提升 | Megatron-LM `megatron/core/datasets/` |
| **通信** | Async Pipeline | 异步流水线 | 10-30% bubble 减少 | Megatron-LM `megatron/core/pipeline_parallel/schedules.py` |
| **通信** | MoE Dispatch | All-to-All 优化 | MoE 通信加速 | Megatron-LM `megatron/core/transformer/moe/` |
| **并行** | FSDP2 | 全分片 | 线性内存节省 | torchtitan `torchtitan/distributed/fsdp.py` |
| **并行** | Context Parallel | 序列切分 | 长序列加速 | Megatron-LM `megatron/core/tensor_parallel/` |
| **并行** | Expert Parallel | Expert 分布 | MoE 扩展 | Megatron-LM `megatron/core/transformer/moe/` |

---

## 引用

- `skill-performance/SKILL.md` — 5 步优化工作流
- `skill-performance/references/bottleneck-taxonomy.md` — 性能瓶颈分类体系
- `skill-tickets/` — 历史案例库
