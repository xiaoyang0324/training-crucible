# 性能瓶颈分类 (Performance Bottleneck Taxonomy)

> 本文档提供性能瓶颈的多维分类体系，用于 Step 2 (Identify) 快速识别瓶颈类型。

---

## 1. 按瓶颈类型分类

### 1.1 Compute-bound (计算瓶颈)

**特征：**
- GPU SM 利用率 > 80%
- 计算 FLOPS 接近硬件峰值
- 内存带宽尚有空间

**典型场景：**
- 大矩阵乘法（Attention、MLP）占主导
- 模型大、batch size 大
- 算子未充分优化

**诊断指标：**
- MFU > 50% 但吞吐仍不达标
- Nsight Compute 显示 compute pipeline 繁忙

**优化方向：**
- Flash Attention（减少 HBM 访问）
- Fused Operators（减少 kernel launch 和内存读写）
- FP8/FP4 训练（提升计算吞吐）

---

### 1.2 Memory-bound (内存瓶颈)

**特征：**
- GPU 内存接近上限
- OOM 或频繁 OOM
- 内存碎片化

**典型场景：**
- 大模型 + 大 batch size
- 激活值占用过多
- 权重 + 优化器状态占用过多

**诊断指标：**
- `torch.cuda.max_memory_allocated()` 接近 GPU 总内存
- 内存分配/释放频繁

**优化方向：**
- Activation Recompute（重算换内存）
- ZeRO-1/2/3（分片优化器状态/梯度/权重）
- CPU Offload（将状态卸载到 CPU）
- Gradient Checkpointing

---

### 1.3 Communication-bound (通信瓶颈)

**特征：**
- comm/compute ratio > 30%
- 多卡训练扩展效率差
- GPU 等待通信完成

**典型场景：**
- 大模型分布式训练（TP/PP/DP 通信）
- 梯度同步开销大
- All-to-All 通信（MoE）

**诊断指标：**
- Nsight Systems 显示通信 kernel 占比高
- 增加 GPU 后吞吐不线性增长

**优化方向：**
- Communication-Overlap（通信与计算重叠）
- Sequence Packing（减少 padding 通信）
- Async Pipeline（异步流水线）
- 优化并行策略（减少通信量）

---

### 1.4 I/O-bound (数据加载瓶颈)

**特征：**
- GPU 空闲等待数据
- GPU 利用率波动大
- 数据加载耗时 > 计算耗时

**典型场景：**
- 大规模数据集读取
- 数据预处理复杂
- dataloader worker 不足

**诊断指标：**
- torch profiler 显示 `DataLoader` 耗时占比高
- GPU util 间歇性下降

**优化方向：**
- 增加 dataloader worker 数量
- 数据预取（prefetch）
- 数据格式优化（如 mmap）
- 数据重排（减少随机读取）

---

### 1.5 Launch-bound (启动瓶颈)

**特征：**
- 大量小 kernel 执行
- kernel launch 开销占比高
- GPU 利用率低但无明显热点

**典型 scene：**
- 复杂计算图（大量小算子）
- 动态 shape 导致无法用 CUDA Graph
- Python 开销大

**诊断指标：**
- Nsight Systems 显示大量小 kernel
- kernel 间间隙大

**优化方向：**
- CUDA Graph（消除 kernel launch 开销）
- 算子融合（减少 kernel 数量）
- torch.compile（图优化）

---

## 2. 按训练阶段分类

### 2.1 预训练 (Pre-training)

| 常见瓶颈 | 典型表现 | 高发场景 |
|---------|---------|---------|
| **内存瓶颈** | OOM | 大模型 + 大 batch |
| **通信瓶颈** | 扩展效率差 | 大规模并行 (>100 GPU) |
| **计算瓶颈** | MFU 低 | 算子未优化 |
| **I/O 瓶颈** | GPU 空闲 | 大规模数据集 |

### 2.2 后训练 (Post-training: SFT/DPO/RLHF)

| 常见瓶颈 | 典型表现 | 高发场景 |
|---------|---------|---------|
| **内存瓶颈** | OOM | 长序列 + 大 batch |
| **计算瓶颈** | 推理慢 | 生成式推理 |
| **通信瓶颈** | 同步延迟 | 多卡 DPO 训练 |

### 2.3 强化学习 (RL: GRPO/PPO)

| 常见瓶颈 | 典型表现 | 高发场景 |
|---------|---------|---------|
| **I/O 瓶颈** | rollout 等待 | 生成式 rollout |
| **通信瓶颈** | 权重同步延迟 | 训练-推理分离架构 |
| **内存瓶颈** | OOM | 多模型共存（actor + critic + reward） |

---

## 3. 诊断方法

### 3.1 Profiling 工具

| 工具 | 用途 | 输出 |
|------|------|------|
| **torch.profiler** | Python 层性能分析 | kernel 耗时、调用栈、时间线 |
| **Nsight Systems** | 系统级性能分析 | CPU/GPU 时间线、通信 kernel |
| **Nsight Compute** | 单 kernel 详细分析 | SM 利用率、内存带宽、寄存器 |
| **nvidia-smi** | 实时监控 | GPU 利用率、内存占用、功耗 |

### 3.2 关键指标

| 指标 | 含义 | 健康范围 |
|------|------|---------|
| **MFU** | 模型 FLOPS 利用率 | > 40% 良好，> 50% 优秀 |
| **吞吐** | tokens/s | 与基线对比 |
| **步长** | 单步训练耗时 (ms) | 与基线对比 |
| **comm/compute** | 通信计算比 | < 20% 良好 |
| **GPU util** | GPU 利用率 | > 70% 良好 |
| **内存占用** | GPU 内存使用 | < 90% 总内存 |

### 3.3 诊断流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                        性能诊断流程                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. 收集基线指标 (吞吐/MFU/内存)                                     │
│           │                                                         │
│           ▼                                                         │
│  2. 运行 torch.profiler 获取 kernel 耗时                             │
│           │                                                         │
│           ▼                                                         │
│  3. 判断瓶颈类型:                                                    │
│     ├─ GPU util 高 + 内存有空间 → Compute-bound                     │
│     ├─ 内存接近上限 → Memory-bound                                  │
│     ├─ comm 占比高 → Communication-bound                            │
│     ├─ GPU 空闲等数据 → I/O-bound                                   │
│     └─ 大量小 kernel → Launch-bound                                 │
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

## 4. 瓶颈→技术→预期收益 速查表

| 瓶颈类型 | 推荐技术 | 预期加速/节省 | 引用 |
|---------|---------|--------------|------|
| **Compute-bound** | Flash Attention 2/3 | 2-4x attention 加速 | `sota-techniques.md` §2.1 |
| **Compute-bound** | Fused Operators | 10-30% 整体加速 | `sota-techniques.md` §2.2 |
| **Compute-bound** | CUDA Graph | 10-20% launch 开销消除 | `sota-techniques.md` §2.3 |
| **Compute-bound** | FP8/FP4 训练 | 1.5-2x 计算吞吐 | `sota-techniques.md` §2.4 |
| **Memory-bound** | Activation Recompute | 30-60% 内存节省 | `sota-techniques.md` §1.1 |
| **Memory-bound** | ZeRO-1/2/3 | 线性内存节省 | `sota-techniques.md` §1.2 |
| **Memory-bound** | CPU Offload | 释放 GPU 内存 | `sota-techniques.md` §1.3 |
| **Communication-bound** | Comm Overlap | 20-40% 通信隐藏 | `sota-techniques.md` §3.1 |
| **Communication-bound** | Sequence Packing | 减少 padding 通信 | `sota-techniques.md` §3.2 |
| **Communication-bound** | Async Pipeline | 减少 bubble | `sota-techniques.md` §3.3 |
| **Communication-bound** | MoE Dispatch | 优化 All-to-All | `sota-techniques.md` §3.4 |
| **I/O-bound** | 数据预取 + 多 worker | 隐藏 I/O 延迟 | — |
| **Launch-bound** | CUDA Graph | 消除 launch 开销 | `sota-techniques.md` §2.3 |
| **Launch-bound** | torch.compile | 图优化 + 融合 | — |

---

## 引用

- `performance/SKILL.md` — 5 步优化工作流
- `performance/references/sota-techniques.md` — SOTA 优化技术目录（含源码引用）
- `tickets/` — 历史案例库（按 `type: performance` 检索）
