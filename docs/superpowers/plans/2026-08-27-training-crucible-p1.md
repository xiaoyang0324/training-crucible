# training-crucible P1 (Knowledge Core) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create the 4 knowledge expert files (pretraining, posttraining, rl, inference) with source-repo citations, architecture diagrams, configuration knobs, and common misconceptions.

**Architecture:** One markdown file per training stage. Each file is a self-contained knowledge module that SKILL.md's router dispatches to. Content cites local source repos (file:line) as primary evidence.

**Tech Stack:** Markdown (Chinese skill docs), ASCII diagrams, source-repo citations.

---

## File Structure

```
training-crucible/
├── knowledge/
│   ├── pretraining.md     # [CREATE] Pre-training: parallelism, scaling, memory
│   ├── posttraining.md    # [CREATE] SFT / DPO / RLHF alignment
│   ├── rl.md              # [CREATE] GRPO / PPO / RL data generation
│   └── inference.md       # [CREATE] Quant / KV Cache / speculative / serving
```

**Reference source repos (read-only):**
- `C:\y30062407\workspace\local\面试\train\Megatron-LM`
- `C:\y30062407\workspace\local\面试\train\torchtitan`
- `C:\y30062407\workspace\local\面试\train\miles`
- `C:\y30062407\workspace\local\面试\train\slime`
- `C:\y30062407\workspace\local\面试\train\torchada`
- `C:\y30062407\workspace\local\面试\train\torch_musa`

---

## Task 1: Write `knowledge/pretraining.md`

**Files:**
- Create: `training-crucible/knowledge/pretraining.md`

Write a knowledge file covering pre-training stage. Structure:

1. **概述** — 预训练的定义、目标、核心挑战（内存墙/计算墙/通信墙）
2. **并行策略** — TP / PP / DP / CP / EP 的定义、适用场景、组合规则
   - 引用: Megatron-LM `megatron/core/parallel_state.py`, `megatron/core/tensor_parallel/`, `megatron/core/pipeline_parallel/`, `megatron/core/context_parallel/`
   - 引用: torchtitan `torchtitan/distributed/` (fsdp.py, tensor_parallel.py, pipeline_parallel.py, context_parallel/)
3. **内存优化** — Activation Recompute, ZeRO, Offload, Gradient Checkpointing
   - 引用: Megatron-LM `megatron/core/recompute.py`, `megatron/core/distributed/`
4. **关键配置参数** — micro-batch, global-batch, gradient accumulation, learning rate schedule
5. **常见误区** — 如 "TP 越多越好"、"PP bubble 无法避免"
6. **相关源码文件索引** — 表格形式

Include ASCII architecture diagrams where helpful. Cite source repos with file paths.

Commit: `git commit -m "docs(knowledge): add pretraining stage knowledge file"`

---

## Task 2: Write `knowledge/posttraining.md`

**Files:**
- Create: `training-crucible/knowledge/posttraining.md`

Write a knowledge file covering post-training (SFT / DPO / RLHF). Structure:

1. **概述** — 后训练的目的、预训练 vs 后训练的区别
2. **SFT (Supervised Fine-Tuning)** — 数据格式、loss 计算、masking 策略
   - 引用: Megatron-LM `megatron/core/post_training/`
3. **DPO (Direct Preference Optimization)** — 原理、reference model、loss 函数
4. **RLHF** — reward model、PPO 在 RLHF 中的角色、KL 约束
5. **关键配置参数** — learning rate (比预训练小 1-2 数量级)、epoch 数、data mixing
6. **常见误区** — 如 "SFT epoch 越多越好"、"DPO 不需要 reward model"
7. **相关源码文件索引**

Commit: `git commit -m "docs(knowledge): add posttraining stage knowledge file"`

---

## Task 3: Write `knowledge/rl.md`

**Files:**
- Create: `training-crucible/knowledge/rl.md`

Write a knowledge file covering RL training (GRPO / PPO). Structure:

1. **概述** — LLM 强化学习的特点、RLHF vs RLHF-free (GRPO)
2. **GRPO (Group Relative Policy Optimization)** — 原理、group sampling、relative reward
   - 引用: miles `miles/true_on_policy/`, `miles/rollout/`
   - 引用: slime `slime/rollout/`, `slime/agent/`
   - 引用: torchtitan `torchtitan/experiments/rl/`
3. **PPO** — clipped surrogate、advantage estimation、value function
4. **Rollout 生成** — SGLang 集成、async rollout、data buffer
   - 引用: miles `miles/rollout/sglang_rollout.py`, `miles/rollout/fully_async_rollout.py`
   - 引用: slime `slime/rollout/sglang_rollout.py`, `slime/rollout/fully_async_rollout.py`
5. **训推一体 (Training-Inference Integration)** — weight sync、delta weight、P2P update
   - 引用: miles `miles/backends/megatron_utils/`, `miles/backends/sglang_utils/`
6. **关键配置参数** — group size、KL coefficient、clip range、rollout batch size
7. **常见误区** — 如 "GRPO 完全不需要 value model"、"async rollout 不影响 on-policy 性质"
8. **相关源码文件索引**

Commit: `git commit -m "docs(knowledge): add RL stage knowledge file"`

---

## Task 4: Write `knowledge/inference.md`

**Files:**
- Create: `training-crucible/knowledge/inference.md`

Write a knowledge file covering inference optimization. Structure:

1. **概述** — 推理优化的目标 (latency / throughput / cost)
2. **KV Cache** — 原理、PagedAttention、KV Cache 量化、GQA/MQA 对 Cache 的影响
3. **量化 (Quantization)** — W4A16, W8A8, GPTQ, AWQ, SmoothQuant, FP8
   - 引用: Megatron-LM `megatron/core/quantization/`, `megatron/core/fp8_utils.py`, `megatron/core/fp4_utils.py`
4. **投机解码 (Speculative Decoding)** — draft model、verification、acceptance rate
5. **推理服务 (Serving)** — continuous batching、request scheduling
   - 引用: miles `miles/rollout/sglang_rollout.py` (SGLang integration)
6. **关键配置参数** — max batch size, max seq length, KV cache memory fraction
7. **常见误区** — 如 "量化必然损失精度"、"投机解码总是加速"
8. **相关源码文件索引**

Commit: `git commit -m "docs(knowledge): add inference stage knowledge file"`

---

## Task 5: Final verification

**Files:**
- Verify: all 4 knowledge files exist and have content

Steps:
1. `wc -l knowledge/*.md` — confirm all 4 files have content (each 50+ lines)
2. `grep -c "megatron\|torchtitan\|miles\|slime" knowledge/*.md` — confirm source citations present
3. `git log --oneline -6` — confirm clean commit history
4. `git push origin main` — push to remote

---

## Self-Review Notes

**Spec coverage:**
- ✅ knowledge/pretraining.md — Task 1
- ✅ knowledge/posttraining.md — Task 2
- ✅ knowledge/rl.md — Task 3
- ✅ knowledge/inference.md — Task 4

**Content standards for each file:**
- Chinese language (matching interview docs voice)
- Source-repo citations with file paths
- ASCII architecture diagrams
- Configuration knob tables
- Common misconceptions section
- Source file index table
