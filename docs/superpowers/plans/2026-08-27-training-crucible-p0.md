# training-crucible P0 (Skeleton) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the minimal viable skeleton of training-crucible — entry router (SKILL.md), directory structure, ticket template, and source-repo map — so P1-P4 can accumulate organically.

**Architecture:** A Claude Code skill library where SKILL.md is the unified entry router that dispatches to knowledge/precision/performance sub-modules and a ticket archive. All content is Chinese markdown; the 6 local source repos (Megatron-LM, miles, slime, torchtitan, torchada, torcht_musa) are read-only knowledge sources.

**Tech Stack:** Markdown (skill docs), YAML frontmatter (ticket metadata), Claude Code skill conventions (SKILL.md + references/ + templates/).

---

## File Structure

```
training-crucible/
├── SKILL.md                              # [CREATE] Entry router — Iron Law + intent routing + sub-skill index
├── knowledge/                            # [CREATE DIR, P1 fills content]
│   └── .gitkeep
├── precision/                            # [CREATE DIR, P2 fills content]
│   └── .gitkeep
├── performance/                          # [CREATE DIR, P2 fills content]
│   └── .gitkeep
├── tickets/                              # [CREATE DIR + TEMPLATE]
│   └── TEMPLATE.md                       # [CREATE] Ticket frontmatter + body schema
└── references/                           # [CREATE DIR + source-repo-map]
    └── source-repo-map.md                # [CREATE] 6 repos → stage/feature mapping with key files
```

**Reference source repos (read-only, do not modify):**
- `C:\y30062407\workspace\local\面试\train\Megatron-LM`
- `C:\y30062407\workspace\local\面试\train\miles`
- `C:\y30062407\workspace\local\面试\train\slime`
- `C:\y30062407\workspace\local\面试\train\torchtitan`
- `C:\y30062407\workspace\local\面试\train\torchada`
- `C:\y30062407\workspace\local\面试\train\torch_musa`

---

## Task 1: Create directory structure

**Files:**
- Create: `training-crucible/knowledge/.gitkeep`
- Create: `training-crucible/precision/.gitkeep`
- Create: `training-crucible/performance/.gitkeep`
- Create: `training-crucible/tickets/.gitkeep`
- Create: `training-crucible/references/.gitkeep`

- [ ] **Step 1: Create directories and .gitkeep files**

Run:
```bash
cd C:\y30062407\workspace\training-crucible
mkdir -p knowledge precision performance tickets references
touch knowledge/.gitkeep precision/.gitkeep performance/.gitkeep tickets/.gitkeep references/.gitkeep
```

Expected: directories created, `.gitkeep` files exist (so git tracks empty dirs).

- [ ] **Step 2: Verify structure**

Run: `ls -la knowledge/ precision/ performance/ tickets/ references/`
Expected: each directory shows `.` `..` `.gitkeep`

- [ ] **Step 3: Commit**

```bash
cd C:\y30062407\workspace\training-crucible
git add -A
git commit -m "chore: create P0 directory structure (knowledge/precision/performance/tickets/references)"
```

---

## Task 2: Write `references/source-repo-map.md`

**Files:**
- Create: `training-crucible/references/source-repo-map.md`

- [ ] **Step 1: Write the source-repo map**

Create `references/source-repo-map.md` with the following content:

```markdown
# Source Repo → Stage/Feature Map

> This file maps the 6 local reference repos to the training stages and features they cover.
> When answering a question, the skill cites the **local source repo** (file path + line number)
> as primary evidence. External papers supplement, never replace, source evidence.

## Hardware Coverage Note

- **NVIDIA GPU** repos: Megatron-LM, torchtitan, miles, slime, torchada
- **Moore Threads MUSA GPU** repo: torcht_musa
- **Ascend NPU**: NOT covered by local repos. Ascend knowledge comes from external docs/papers
  (MindSpeed, torch_npu) and is marked as "external" — secondary to local source evidence.

---

## Repo 1: Megatron-LM

**Path:** `C:\y30062407\workspace\local\面试\train\Megatron-LM`

**Primary stages:** Pre-training, Post-training
**Secondary:** RL (training backbone for miles/slime)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `megatron/core/parallel/` | TP, PP, DP, EP, CP parallelism |
| `megatron/core/transformer/` | Transformer layer, attention, MLP |
| `megatron/core/pipeline_parallel/` | 1F1B, interleaved schedules |
| `megatron/core/distributed/` | Distributed optimizer, grad sync |
| `megatron/core/transformer/dot_product_attention.py` | Core attention implementation |
| `megatron/legacy/` | Legacy model implementations |

**When to cite:** Pre-training architecture, parallelism strategies, optimizer design, gradient pipeline.

---

## Repo 2: torchtitan

**Path:** `C:\y30062407\workspace\local\面试\train\torchtitan`

**Primary stage:** Pre-training
**Secondary:** RL (TitanRL experiment)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torchtitan/distributed/` | FSDP2, TP, PP, CP native PyTorch |
| `torchtitan/models/` | Llama 3, other model definitions |
| `torchtitan/experiments/rl/` | TitanRL — RL training stack |
| `torchtitan/ops/` | Custom kernels (flexattention, etc.) |
| `torchtitan/components/` | Loss, dataloader, metrics |

**When to cite:** PyTorch-native training patterns, FSDP2 usage, TitanRL RL approach.

---

## Repo 3: miles

**Path:** `C:\y30062407\workspace\local\面试\train\miles`

**Primary stage:** RL (GRPO/PPO)
**Secondary:** Inference (SGLang rollout integration)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `miles/rollout/` | Rollout generation, SGLang integration |
| `miles/backends/` | Megatron + FSDP2 backend adapters |
| `miles/true_on_policy/` | True on-policy training, numerics |
| `miles/router/` | Request routing for rollout |
| `miles/ray/` | Ray-based orchestration |
| `miles/dashboard/` | Observability, metrics |

**When to cite:** RL post-training loop, rollout strategies, on-policy numerics, weight sync.

---

## Repo 4: slime

**Path:** `C:\y30062407\workspace\local\面试\train\slime`

**Primary stage:** RL (GRPO/PPO)
**Secondary:** Inference (SGLang rollout), agentic RL

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `slime/rollout/` | Rollout generation, data buffer |
| `slime/backends/` | Megatron backend, SGLang pass-through |
| `slime/agent/` | Agent-based RL, multi-agent |
| `slime/ray/` | Ray orchestration |
| `slime/observability/` | Tracing, profiling, reproducibility |

**When to cite:** RL data generation workflows, agentic RL, train-infer integration.

---

## Repo 5: torchada

**Path:** `C:\y30062407\workspace\local\面试\train\torchada`

**Primary stage:** Pre-training (NVIDIA Ada / RTX consumer GPU)
**Secondary:** —

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torchada/` | PyTorch backend for Ada/RTX GPUs |
| `benchmarks/` | Performance benchmarks |

**When to cite:** Consumer-GPU training, Ada-specific optimizations, hardware-adaptation layer.

---

## Repo 6: torch_musa

**Path:** `C:\y30062407\workspace\local\面试\train\torch_musa`

**Primary stage:** Pre-training (Moore Threads MUSA GPU)
**Secondary:** —

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torch_musa/` | PyTorch backend for MUSA GPU |
| `benchmark/` | Performance benchmarks |

**When to cite:** MUSA GPU training, Moore Threads hardware adaptation, cross-platform porting.

---

## Quick Routing Index

| User asks about... | Primary repo(s) | Secondary repo(s) |
|--------------------|-----------------|-------------------|
| Pre-training at scale | Megatron-LM, torchtitan | torchada, torcht_musa |
| SFT / DPO / RLHF | Megatron-LM | torchtitan |
| GRPO / PPO RL | miles, slime | torchtitan (TitanRL) |
| Inference / rollout | miles, slime | — |
| FSDP / parallelism | torchtitan, Megatron-LM | — |
| MoE training | Megatron-LM | miles, slime |
| Consumer GPU (Ada) | torchada | — |
| MUSA GPU | torcht_musa | — |
```

- [ ] **Step 2: Verify file exists and is readable**

Run: `wc -l references/source-repo-map.md`
Expected: ~130+ lines, no errors.

- [ ] **Step 3: Commit**

```bash
cd C:\y30062407\workspace\training-crucible
git add references/source-repo-map.md
git commit -m "docs: add source-repo map (6 repos → stage/feature mapping)"
```

---

## Task 3: Write `tickets/TEMPLATE.md`

**Files:**
- Create: `training-crucible/tickets/TEMPLATE.md`

- [ ] **Step 1: Write the ticket template**

Create `tickets/TEMPLATE.md` with the following content:

```markdown
# Problem Ticket Template

> Copy this file, fill in the frontmatter and body, rename to `YYYY-MM-DD-<short-slug>.md`.

---

```markdown
---
id: TICKET-YYYYMMDD-NNN          # NNN = per-day sequence (001, 002, ...)
title: One-line summary of the problem
type: precision | performance | hybrid
stage: pretraining | posttraining | rl | inference | cross-stage
status: resolved | investigating | wontfix
severity: critical | major | minor
hardware:
  - NVIDIA-Ada                   # or MUSA, Ascend, CPU, agnostic
frameworks:
  - Megatron-LM                  # or torchtitan, miles, slime, torchada, torcht_musa
tags:
  - loss-nan                     # choose from tag vocabulary below
  - grad-explosion
  - throughput
  - oom
  - scaling-efficiency
  - train-infer-mismatch
  - accuracy-regression
created: YYYY-MM-DD
resolved: YYYY-MM-DD             # leave blank if status != resolved
related_tickets:
  - TICKET-YYYYMMDD-NNN
source_refs:                      # local repo files that informed the solution
  - Megatron-LM/megatron/core/parallel/tensor_parallel/lines.py:123
external_refs:                    # papers, docs, external links
  - https://arxiv.org/abs/...
---

## Symptom

What was observed. Include:
- Training curve snapshot (loss, grad norm, accuracy)
- Error message / traceback (if any)
- When it started (step number, after what change)

## Environment

| Item | Value |
|------|-------|
| Model | e.g., Llama-3 70B |
| Scale | e.g., 1024 GPUs, 64 nodes |
| Hardware | e.g., NVIDIA A100 80GB / Atlas 800 A3 |
| Framework | e.g., Megatron-LM commit abc123, PyTorch 2.4 |
| Parallel config | e.g., TP=8, PP=8, DP=16, CP=1 |
| Batch size | micro-batch × gradient accumulation |

## Analysis

Step-by-step root cause analysis. Cite source code with file paths.

1. **Initial observation:** ...
2. **Narrowed to:** ...
3. **Verified by:** ...
4. **Root cause confirmed:** ...

## Root Cause

One paragraph: what was actually wrong, at the code/algorithm level.

## Resolution

What fixed it. Be specific:
- Config changes (before → after)
- Code patches (file:line, diff format)
- Workflow adjustments

## Verification

How we confirmed the fix worked:
- Metrics before/after
- Regression tests run
- Duration of stable training after fix

## Lessons

Reusable insight. Cross-references:
- See `references/source-repo-map.md` for related repos
- See `precision/references/` for precision patterns (P2)
- See `performance/references/` for perf techniques (P2)

## References

- Papers: ...
- Docs: ...
- Related tickets: ...
```

- [ ] **Step 2: Verify template is complete**

Run: `grep -c "id:" tickets/TEMPLATE.md`
Expected: 1 (the frontmatter id field exists)

- [ ] **Step 3: Commit**

```bash
cd C:\y30062407\workspace\training-crucible
git add tickets/TEMPLATE.md
git commit -m "docs: add problem ticket template (frontmatter + body schema)"
```

---

## Task 4: Write `SKILL.md` (Entry Router)

**Files:**
- Create: `training-crucible/SKILL.md`

- [ ] **Step 1: Write the SKILL.md entry router**

Create `SKILL.md` with the following content:

```markdown
---
name: training-crucible
description: >
  AI 训练全栈专家技能——覆盖预训练、后训练（SFT/DPO/RLHF）、强化学习（GRPO/PPO）、
  推理优化四大阶段。提供知识问答、精度诊断、性能优化、问题归档四项核心能力。
  触发条件：用户问及 AI 训练、大模型训练、分布式训练、训练精度、训练性能、
  loss 异常、梯度异常、吞吐优化、Megatron、torchtitan、miles、slime、
  预训练、后训练、对齐、强化学习、GRPO、PPO、推理优化、量化、KV Cache 等
  任何训练全栈相关话题。
  源码参考仓：Megatron-LM, torchtitan, miles, slime, torchada, torcht_musa。
---

# training-crucible — AI 训练全栈专家

## The Iron Law

```
回答任何训练问题前，必须先确认两样东西：
  1. 训练阶段（预训练 / 后训练 / RL / 推理）
  2. 涉及的框架仓（Megatron-LM / torchtitan / miles / slime / 其他）

分析精度/性能问题时，还必须拿到：
  3. 症状描述（loss 曲线 / 报错 / 性能指标）+ 环境信息（模型规模、并行配置、硬件）

没有这些不开始分析。绝不凭空猜测根因。
```

```
所有代码引用和实现必须来自真实源码。
禁止虚构函数名、API、文件路径、行号、类名、变量名。
引用代码前必须先 grep/read 确认该代码真实存在。
不确定的 API 用 "需要确认" 标注，不编造。
优先引用本地源码仓（Megatron-LM, torchtitan, miles, slime, torchada, torcht_musa），
外部论文/文档作为补充，且必须标注为"外部知识"。
```

## 核心能力

| 能力 | 模块 | 说明 |
|------|------|------|
| **知识问答** | `knowledge/` | 按训练阶段路由，解释概念、架构、配置 |
| **精度诊断** | `precision/` | 5 步工作流诊断 loss/梯度/收敛问题 |
| **性能优化** | `performance/` | 5 步工作流优化吞吐/内存/扩展效率 |
| **问题归档** | `tickets/` | 结构化案例库，按症状/阶段/框架检索 |

---

## 意图路由表

> 根据用户问题的关键词，路由到对应模块。多个模块可串联（精度→性能→归档）。

### 路由规则

```
用户问题
    │
    ├─ "什么是 X" / "X 怎么工作" / "X 和 Y 区别" ──────────────► 知识问答
    │     │
    │     ├─ 含 pretrain / 预训练 / pre-training ────────► knowledge/pretraining.md (P1)
    │     ├─ 含 SFT / DPO / RLHF / alignment / 后训练 ──► knowledge/posttraining.md (P1)
    │     ├─ 含 GRPO / PPO / RL / 强化学习 ──────────────► knowledge/rl.md (P1)
    │     └─ 含 quant / KV cache / speculative / 推理 ───► knowledge/inference.md (P1)
    │
    ├─ loss NaN / loss spike / 梯度爆炸 / 精度异常 / ──────► 精度诊断
    │     train-infer mismatch / 不收敛 / 发散                    │
    │     └─ precision/SKILL.md (P2) + precision/references/
    │
    ├─ 慢 / OOM / 吞吐低 / 扩展效率 / 内存瓶颈 / ──────────► 性能优化
    │     MFU 低 / 通信瓶颈 / 算子慢                            │
    │     └─ performance/SKILL.md (P2) + performance/references/
    │
    ├─ "遇到过吗" / "历史案例" / "之前的问题" ──────────────► 问题归档
    │     └─ tickets/ — 按 type/stage/tags 检索
    │
    └─ 复杂问题（精度+性能+归档） ──────────────────────────► 串联：
         精度诊断 → 性能分析 → 归档到 tickets/
```

### 关键词触发表

| 关键词 | 路由目标 |
|--------|----------|
| 预训练, pretrain, pre-training, 预训练阶段 | `knowledge/pretraining.md` |
| 后训练, SFT, DPO, RLHF, alignment, 对齐 | `knowledge/posttraining.md` |
| 强化学习, GRPO, PPO, RL, reinforcement | `knowledge/rl.md` |
| 推理, quant, KV cache, speculative, serving, 量化 | `knowledge/inference.md` |
| 精度, loss, 梯度, grad, NaN, spike, divergence, 不收敛 | `precision/` |
| 性能, 吞吐, throughput, MFU, 内存, memory, OOM, 慢 | `performance/` |
| 案例, 问题单, ticket, 归档, 历史 | `tickets/` |
| Megatron, torchtitan, miles, slime, torchada, torch_musa | `references/source-repo-map.md` |

---

## 源码仓路由

> 根据用户提到的框架或训练阶段，定位到对应的本地源码仓。

| 用户提到 | 定位源码仓 | 关键路径 |
|----------|-----------|----------|
| Megatron-LM | `train/Megatron-LM` | `megatron/core/` |
| torchtitan | `train/torchtitan` | `torchtitan/distributed/`, `torchtitan/experiments/rl/` |
| miles | `train/miles` | `miles/rollout/`, `miles/backends/`, `miles/true_on_policy/` |
| slime | `train/slime` | `slime/rollout/`, `slime/backends/`, `slime/agent/` |
| torchada (Ada GPU) | `train/torchada` | `torchada/` |
| torch_musa (MUSA) | `train/torch_musa` | `torch_musa/` |

> 详细映射见 `references/source-repo-map.md`。

---

## 跨模块协作规则

### 精度 + 性能联合分析
当问题同时涉及精度和性能（如"开启 activation recompute 后 loss 异常"）：
1. 先走 `precision/` 诊断流程，确认精度问题根因
2. 再走 `performance/` 优化流程，评估性能影响
3. 最终结论归档到 `tickets/`

### 知识 + 归档联动
当知识问答中发现类似历史案例：
1. 在 `knowledge/` 回答后，主动检索 `tickets/` 中 `related_tickets`
2. 如有匹配，附上"类似案例：TICKET-..."

### 归档触发条件
以下情况必须归档到 `tickets/`：
- 精度问题已解决且根因明确
- 性能优化已完成且有量化收益
- 跨模块的复杂问题

---

## 子模块索引

| 模块 | 路径 | 状态 | 说明 |
|------|------|------|------|
| 知识专家 | `knowledge/` | P1 | 按训练阶段分 4 个文件 |
| 精度专家 | `precision/` | P2 | 5 步诊断工作流 + 精度故障分类 |
| 性能专家 | `performance/` | P2 | 5 步优化工作流 + SOTA 技术目录 |
| 问题归档 | `tickets/` | P3 | 结构化案例库 |
| 源码映射 | `references/source-repo-map.md` | ✅ | 6 仓 → 阶段映射 |
| 术语表 | `references/training-glossary.md` | P4 | 规范术语 |

---

## 回答规范

### 必须做
1. **先确认阶段和框架** — 回答前明确用户的训练阶段和涉及框架
2. **引用真实源码** — 技术主张必须附 `仓名/文件路径:行号`
3. **标注外部知识** — 来自论文/文档的知识标注 `[外部]`
4. **主动关联案例** — 发现匹配的历史问题单时主动附上

### 禁止做
1. **凭空猜测** — 没有源码证据不做根因判断
2. **虚构 API** — 不编造函数名、参数、文件路径
3. **跳过诊断流程** — 精度/性能问题必须走完整工作流
4. **重复归档** — 同一问题不创建多个 ticket

---

## 版本与阶段

- **当前阶段:** P0 (骨架) — 本文件 + 目录结构 + 模板 + 源码映射
- **下一阶段:** P1 (知识核心) — 填充 `knowledge/` 4 个文件
- **路线图:** 见 `docs/superpowers/specs/2026-08-27-training-crucible-design.md` §8
```

- [ ] **Step 2: Verify SKILL.md frontmatter is valid**

Run: `head -20 SKILL.md`
Expected: starts with `---`, contains `name: training-crucible`, ends with `---`.

- [ ] **Step 3: Verify routing table covers all 4 stages**

Run: `grep -c "knowledge/" SKILL.md`
Expected: ≥ 4 (pretraining, posttraining, rl, inference all mentioned)

- [ ] **Step 4: Commit**

```bash
cd C:\y30062407\workspace\training-crucible
git add SKILL.md
git commit -m "feat: add SKILL.md entry router (Iron Law + intent routing + sub-skill index)"
```

---

## Task 5: Final verification and commit

**Files:**
- Verify: all P0 files exist

- [ ] **Step 1: Verify complete P0 structure**

Run:
```bash
cd C:\y30062407\workspace\training-crucible
find . -not -path './.git/*' -not -path './.idea/*' -type f | sort
```

Expected output includes:
```
./SKILL.md
./README.md
./references/source-repo-map.md
./tickets/TEMPLATE.md
./knowledge/.gitkeep
./precision/.gitkeep
./performance/.gitkeep
./tickets/.gitkeep
./references/.gitkeep
```

- [ ] **Step 2: Verify SKILL.md links resolve**

Run:
```bash
cd C:\y30062407\workspace\training-crucible
grep -o '`references/[^`]*`' SKILL.md | sort
```

Expected: only `references/source-repo-map.md` (the only references/ file in P0).

- [ ] **Step 3: Final commit check**

Run: `git status`
Expected: working tree clean (all P0 files committed).

---

## Self-Review Notes

**Spec coverage:**
- ✅ SKILL.md entry router (Iron Law + intent routing + sub-skill index) — Task 4
- ✅ Directory structure (knowledge/precision/performance/tickets/references) — Task 1
- ✅ tickets/TEMPLATE.md (frontmatter + body schema) — Task 3
- ✅ references/source-repo-map.md (6 repos → stage mapping) — Task 2

**Placeholder scan:** No TBDs, no "implement later", all code blocks are complete file contents.

**Type consistency:** Ticket ID format `TICKET-YYYYMMDD-NNN` is consistent across Task 3 (template) and spec. Routing keywords are consistent across Task 4 (SKILL.md) and spec §4.2.