---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → performance/SKILL.md
description: >
  性能优化专家——5 步优化工作流，覆盖吞吐低、OOM、MFU 低、扩展效率差、
  通信瓶颈等性能问题。
  触发条件：用户报告训练速度慢、GPU 利用率低、内存不足、扩展效率差、
  通信开销大、算子性能差等性能相关问题。
---

# 性能优化专家 — 5 步优化工作流

## The Iron Law

```
优化性能问题前，必须先拿到四样东西：
  1. 性能基线（吞吐、MFU、内存占用、训练步长）
  2. 瓶颈画像（compute/memory/communication/I/O/launch 哪类）
  3. 环境信息（模型规模、并行配置、硬件、框架版本）
  4. 优化目标（提吞吐 / 降内存 / 提扩展效率）

没有瓶颈画像不做优化建议。绝不凭猜测推荐技术。
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

1. **画像 (Profile)** — 收集性能指标 (吞吐/MFU/内存/步长)、comm/compute ratio、kernel 耗时分布、环境信息
2. **识别 (Identify)** — 分类瓶颈 (compute/memory/communication/I/O/launch-bound)，定位热点算子/层/操作
3. **匹配 (Match)** — 匹配 `references/sota-techniques.md` SOTA 技术，检索 `tickets/` 类似案例，按预期收益排序候选技术
4. **适配 (Adapt)** — 根据用户硬件/框架调整方案，引用源码仓具体实现，给出配置参数建议
5. **验证与归档 (Validate & Archive)** — 提出预期加速比/内存节省量，验证指标改善，归档到 `tickets/`

---

## 每步详细检查清单

### Step 1: 画像 (Profile)

**向用户确认的问题：**
- 当前吞吐是多少 tokens/s？目标是多少？
- MFU 是多少？GPU 利用率如何？
- 训练步长（step time）是多少 ms？
- 是否使用 profiling 工具（Nsight / torch profiler）做过分析？

**需要收集的指标：**
- 吞吐：tokens/s, samples/s, steps/s
- 计算效率：MFU (Model FLOPs Utilization), GPU SM 利用率
- 内存：峰值 GPU 内存、显存碎片
- 通信：allreduce/allgather 耗时、comm/compute ratio
- I/O：数据加载耗时、dataloader worker 数量
- 并行配置：TP/PP/DP/CP/EP、micro-batch size、gradient accumulation

---

### Step 2: 识别 (Identify)

**瓶颈判定规则：**

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

---

### Step 3: 匹配 (Match)

**匹配 SOTA 技术：**
- 查阅 `performance/references/sota-techniques.md`
- 检索 `tickets/` 中 `type: performance` 的历史案例
- 按预期收益排序候选技术

**常见技术映射：**
- Compute-bound → Flash Attention、Fused Operators、CUDA Graph
- Memory-bound → Activation Recompute、ZeRO、Gradient Checkpointing
- Communication-bound → Comm Overlap、Sequence Packing、Async PP
- I/O-bound → 数据预取、多 worker dataloader、数据重排
- Launch-bound → CUDA Graph、算子融合、减少 kernel 数量

---

### Step 4: 适配 (Adapt)

**适配原则：**
- 根据用户硬件（NVIDIA/Ascend/MUSA）选择可用技术
- 根据用户框架（Megatron/torchtitan/miles/slime）引用具体实现
- 给出具体配置参数建议

**引用源码：**
- 每个推荐技术必须附源码仓路径
- 说明该技术的配置入口和关键参数

---

### Step 5: 验证与归档 (Validate & Archive)

**验证标准：**
- 对比优化前后的吞吐 / MFU / 内存
- 确认优化未引入精度问题
- 验证优化在目标硬件上稳定

**归档要求：**
- 按 `tickets/TEMPLATE.md` 格式创建 ticket
- 记录优化前后对比数据
- 引用源码路径和配置变更

---

## 引用

| 文件 | 内容 |
|------|------|
| `performance/references/bottleneck-taxonomy.md` | 性能瓶颈分类体系 |
| `performance/references/sota-techniques.md` | SOTA 优化技术目录 |
| `tickets/TEMPLATE.md` | 问题归档模板 |
| `references/source-repo-map.md` | 源码仓路由（用于定位代码引用） |
