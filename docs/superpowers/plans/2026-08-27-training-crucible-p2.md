# training-crucible P2 (Expert Engines) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create the precision expert and performance expert modules — each with a SKILL.md (5-step analysis workflow engine) and references/ (failure taxonomy + known patterns).

**Architecture:** Two independent expert modules. Each has:
- `SKILL.md` — the workflow engine (5-step diagnostic/optimization process)
- `references/` — structured knowledge (taxonomy, known patterns, SOTA techniques)

**Tech Stack:** Markdown (Chinese skill docs), workflow diagrams, taxonomy tables.

---

## File Structure

```
training-crucible/
├── precision/
│   ├── SKILL.md                     # [CREATE] 5-step diagnostic workflow engine
│   └── references/
│       ├── failure-taxonomy.md      # [CREATE] Precision failure classification
│       └── known-patterns.md        # [CREATE] Known precision issues + solutions
└── performance/
    ├── SKILL.md                     # [CREATE] 5-step optimization workflow engine
    └── references/
        ├── bottleneck-taxonomy.md   # [CREATE] Performance bottleneck classification
        └── sota-techniques.md       # [CREATE] SOTA optimization techniques catalog
```

---

## Task 1: Write `precision/SKILL.md`

**Files:**
- Create: `training-crucible/precision/SKILL.md`

Write the precision expert workflow engine. Structure:

1. **触发条件** — 何时调用精度专家（loss NaN/spike/divergence, grad norm explosion, train-infer mismatch, accuracy regression）
2. **5 步诊断工作流**：
   - Step 1: **捕获 (Capture)** — 收集完整报错、训练曲线、环境信息
   - Step 2: **分类 (Classify)** — 将症状归类（loss NaN / loss spike / loss divergence / grad norm explosion / train-infer mismatch / accuracy regression）
   - Step 3: **定位 (Localize)** — 追踪到层级（数据 → 优化器 → 梯度 → 激活 → 权重 → loss 计算）
   - Step 4: **假设 (Hypothesize)** — 匹配 `references/known-patterns.md` 和 `tickets/`
   - Step 5: **解决与归档 (Resolve & Archive)** — 提出修复、验证、归档到 `tickets/`
3. **每步的详细检查清单** — 每步需要问用户的问题、需要查看的日志/指标
4. **工作流 ASCII 流程图**
5. **引用** — 链接到 `precision/references/`

Commit: `git commit -m "docs(precision): add 5-step diagnostic workflow engine"`

---

## Task 2: Write `precision/references/failure-taxonomy.md`

**Files:**
- Create: `training-crucible/precision/references/failure-taxonomy.md`

Write the precision failure classification taxonomy. Structure:

1. **按症状分类**：
   - Loss NaN — 数值溢出、除零、log(0)
   - Loss Spike — 学习率过大、数据异常、梯度爆炸
   - Loss Divergence — 模型崩溃、训练不稳定
   - Grad Norm Explosion — 梯度累积、loss scale 问题
   - Train-Infer Mismatch — 训练正常但推理异常
   - Accuracy Regression — 精度回退
2. **按训练阶段分类** — 预训练/后训练/RL 各阶段常见精度问题
3. **按层级分类** — 数据层/优化器层/梯度层/激活层/权重层/loss 层
4. **症状→根因→方案** 速查表

Commit: `git commit -m "docs(precision): add failure taxonomy reference"`

---

## Task 3: Write `precision/references/known-patterns.md`

**Files:**
- Create: `training-crucible/precision/references/known-patterns.md`

Write known precision issues and solutions. Structure:

1. **Loss NaN 模式**：
   - FP16 溢出 → 解决方案：loss scaling, BF16
   - 除零错误 → 解决方案：epsilon, 数值稳定实现
   - log(0) in cross-entropy → 解决方案：label smoothing, log_softmax 数值稳定
2. **Loss Spike 模式**：
   - 脏数据 → 解决方案：数据清洗, loss spike detection
   - 学习率过大 → 解决方案：warmup, LR 衰减
3. **Grad Norm 模式**：
   - 梯度爆炸 → 解决方案：grad clipping, loss scaling
   - 梯度消失 → 解决方案：residual scaling, initialization
4. **RL 特有模式**：
   - Reward hacking → 解决方案：KL 约束, reward normalization
   - Policy collapse → 解决方案：clip range, entropy bonus
5. **每个模式包含**：症状描述、根因分析、解决方案、源码引用

Commit: `git commit -m "docs(precision): add known patterns reference"`

---

## Task 4: Write `performance/SKILL.md`

**Files:**
- Create: `training-crucible/performance/SKILL.md`

Write the performance expert workflow engine. Structure:

1. **触发条件** — 何时调用性能专家（吞吐低、OOM、MFU 低、扩展效率差、通信瓶颈）
2. **5 步优化工作流**：
   - Step 1: **画像 (Profile)** — 收集 throughput, MFU, 内存, 通信/计算比
   - Step 2: **识别 (Identify)** — 分类瓶颈（compute-bound / memory-bound / communication-bound / I/O-bound / launch-bound）
   - Step 3: **匹配 (Match)** — 匹配 `references/sota-techniques.md` 中的 SOTA 方案
   - Step 4: **适配 (Adapt)** — 根据用户硬件/框架调整方案（引用源码仓）
   - Step 5: **验证与归档 (Validate & Archive)** — 提出预期加速比、验证、归档到 `tickets/`
3. **每步的详细检查清单**
4. **工作流 ASCII 流程图**
5. **引用** — 链接到 `performance/references/`

Commit: `git commit -m "docs(performance): add 5-step optimization workflow engine"`

---

## Task 5: Write `performance/references/bottleneck-taxonomy.md`

**Files:**
- Create: `training-crucible/performance/references/bottleneck-taxonomy.md`

Write the performance bottleneck classification taxonomy. Structure:

1. **按瓶颈类型分类**：
   - Compute-bound — 计算密集型（大矩阵乘、attention）
   - Memory-bound — 内存密集型（activation memory, KV cache）
   - Communication-bound — 通信密集型（all-reduce, all-to-all）
   - I/O-bound — IO 密集型（数据加载、checkpoint）
   - Launch-bound — 启动密集型（kernel launch overhead, CPU-GPU 同步）
2. **按训练阶段分类** — 预训练/后训练/RL 各阶段常见瓶颈
3. **诊断方法** — 如何用 profiling 工具识别瓶颈（Nsight, torch profiler）
4. **瓶颈→技术→预期收益** 速查表

Commit: `git commit -m "docs(performance): add bottleneck taxonomy reference"`

---

## Task 6: Write `performance/references/sota-techniques.md`

**Files:**
- Create: `training-crucible/performance/references/sota-techniques.md`

Write the SOTA optimization techniques catalog. Structure:

1. **内存优化技术**：
   - Activation Recompute (selective/full) — 引用 Megatron-LM `megatron/core/recompute.py`
   - ZeRO-1/2/3 — 引用 Megatron-LM `megatron/core/distributed/`
   - CPU Offload (SwapActivation, SwapOptimizer)
   - Gradient Checkpointing
2. **计算优化技术**：
   - Flash Attention 2/3
   - Fused Operators (RMSNorm + Residual, SwiGLU, RoPE)
   - CUDA Graph / NPU Graph
   - FP8/FP4 训练 — 引用 Megatron-LM `megatron/core/fp8_utils.py`, `megatron/core/fp4_utils.py`
3. **通信优化技术**：
   - Communication-Overlap (DP/PP/TP overlap)
   - Sequence Packing
   - Async Pipeline
   - MoE Dispatch Optimization (DeepEP, All-to-All)
4. **并行策略技术**：
   - FSDP2 — 引用 torchtitan `torchtitan/distributed/fsdp.py`
   - Context Parallel (Ulysses, Ring Attention)
   - Expert Parallel (MC2, Domino)
5. **每项技术包含**：原理、适用场景、配置方法、预期收益、源码引用

Commit: `git commit -m "docs(performance): add SOTA techniques catalog"`

---

## Task 7: Final verification

**Files:**
- Verify: all P2 files exist and have content

Steps:
1. `wc -l precision/SKILL.md precision/references/*.md performance/SKILL.md performance/references/*.md` — confirm all 6 files have content
2. `git log --oneline -8` — confirm clean commit history
3. `git push origin main` — push to remote

---

## Self-Review Notes

**Spec coverage:**
- ✅ precision/SKILL.md (5-step diagnostic workflow) — Task 1
- ✅ precision/references/failure-taxonomy.md — Task 2
- ✅ precision/references/known-patterns.md — Task 3
- ✅ performance/SKILL.md (5-step optimization workflow) — Task 4
- ✅ performance/references/bottleneck-taxonomy.md — Task 5
- ✅ performance/references/sota-techniques.md — Task 6

**Content standards:**
- Chinese language throughout
- 5-step workflow with ASCII diagrams
- Source-repo citations with file paths
- Taxonomy tables
- Each reference file 80-150 lines
