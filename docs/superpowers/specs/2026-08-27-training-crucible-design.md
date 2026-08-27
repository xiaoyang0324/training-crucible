# training-crucible Design Spec

> **Status:** Draft · **Date:** 2026-08-27 · **Author:** Yang Chenghan

## 1. Purpose

`training-crucible` is a **Claude Code skills library** that forges full-stack AI training expertise. It is an expert system embedded in the Claude Code harness — when the user asks a question about AI training, the skills route the query to the right domain expert, apply structured analysis workflows, and accumulate solved problems into a living case archive.

The "product" is **Chinese markdown skill documents + a structured problem-ticket archive**, not executable code. The reference source repos (Megatron-LM, miles, slime, torchtitan, torchada, torch_musa) are read-only knowledge sources.

## 2. Design Decisions (Approved)

| Dimension | Decision |
|-----------|----------|
| **Use case** | Personal-first, architecture reserves team extensibility |
| **Knowledge sources** | Reference repo source code + external papers/industry solutions + accumulated work problem tickets |
| **Knowledge expert routing** | Unified entry + sub-routing (graph-mode-expert style) |
| **Training stages** | Pre-training, Post-training (SFT/DPO/RLHF), RL (GRPO/PPO), Inference |
| **Module boundary** | Precision/Performance experts = analysis engines; Problem tickets = upper-case archive (references expert conclusions) |
| **Ticket format** | Structured markdown template + frontmatter metadata |
| **Expert work mode** | Analysis workflow engine (5-step diagnostic method) |

## 3. Architecture

### 3.1 Skill Topology

```
training-crucible/
├── SKILL.md                          # Iron Law + entry routing (the "crucible lid")
├── knowledge/                        # Knowledge Expert — unified entry, sub-routes by stage
│   ├── pretraining.md                # Pre-training stage knowledge
│   ├── posttraining.md               # SFT / DPO / RLHF
│   ├── rl.md                         # GRPO / PPO / RL data generation
│   └── inference.md                  # Quant / KV Cache / speculative decode / serving
├── precision/                        # Precision Expert — analysis workflow engine
│   ├── SKILL.md                      # Diagnostic workflow (5-step)
│   └── references/                   # Precision failure taxonomy, known patterns
├── performance/                      # Performance Expert — analysis workflow engine
│   ├── SKILL.md                      # Optimization workflow
│   └── references/                   # Perf bottleneck taxonomy, SOTA techniques
├── tickets/                          # Problem Ticket Archive
│   ├── TEMPLATE.md                   # Ticket frontmatter + body schema
│   └── ... .md                       # Accumulated cases (git-tracked)
└── references/                       # Shared cross-cutting references
    ├── source-repo-map.md            # Which repo covers which stage/feature
    └── training-glossary.md          # Canonical terminology
```

### 3.2 Routing Model

```
User query
    │
    ▼
SKILL.md (entry router)
    │
    ├─ "What is X?" / "How does Y work?" ──────────────► knowledge/ ──► sub-route by stage
    │
    ├─ "Loss NaN / spike / divergence / mismatch" ─────► precision/
    │
    ├─ "Slow / OOM / throughput / scaling efficiency" ──► performance/
    │
    ├─ "Have we seen this before?" / "past case" ──────► tickets/
    │
    └─ "Analyze this full problem" ─────────────────────► orchestrator:
                                                           precision → performance → tickets
```

### 3.3 Source Repo → Stage Mapping

| Repo | Primary Stage | Secondary | Key Files of Interest |
|------|--------------|-----------|----------------------|
| `Megatron-LM` | Pre-training, Post-training | RL (backbone) | `megatron/core/parallel/`, `megatron/core/transformer/`, `megatron/core/pipeline_parallel/` |
| `torchtitan` | Pre-training | RL (TitanRL experiment) | `torchtitan/distributed/`, `torchtitan/experiments/rl/` |
| `miles` | RL (GRPO/PPO) | Inference (SGLang rollout) | `miles/rollout/`, `miles/backends/`, `miles/true_on_policy/` |
| `slime` | RL (GRPO/PPO) | Inference (SGLang rollout) | `slime/rollout/`, `slime/backends/`, `slime/agent/` |
| `torchada` | Pre-training (Ada GPU) | — | Hardware-adaptation layer |
| `torcht_musa` | Pre-training (MUSA GPU) | — | Hardware-adaptation layer |

> **Rule:** When answering a question, the skill cites the **local source repo** (file path + line number) as primary evidence. External papers supplement, never replace, source evidence.
>
> **Hardware coverage note:** The 6 local source repos cover **NVIDIA GPU** (Megatron-LM, torchtitan, miles, slime, torchada) and **Moore Threads MUSA GPU** (torcht_musa). **Ascend NPU** knowledge is sourced from external docs/papers (e.g., MindSpeed, torch_npu) — these citations are clearly marked as "external" and are secondary to local source evidence.

## 4. Module Detail

### 4.1 SKILL.md — The Crucible Lid (Entry Router)

**Purpose:** Single entry point. Reads the user's intent and routes to the right sub-skill.

**Frontmatter triggers:** `training`, `预训练`, `后训练`, `SFT`, `DPO`, `RLHF`, `GRPO`, `PPO`, `精度`, `loss`, `梯度`, `性能`, `吞吐`, `推理`, `量化`, `Megatron`, `torchtitan`, `miles`, `slime`, and all training-stage keywords.

**Content:**
1. Iron Law (must-have before any analysis)
2. Intent classification rules (keyword → sub-skill mapping)
3. Sub-skill index with one-line descriptions
4. Cross-module orchestration rules (when to chain experts)

### 4.2 Knowledge Expert (`knowledge/`)

**Purpose:** Answer "what is X" and "how does Y work" for each training stage.

**Structure:** One markdown file per stage. Each file contains:
- Core concepts (with source-repo citations)
- Architecture diagrams (ASCII)
- Key configuration knobs and their effects
- Common misconceptions
- Links to relevant tickets and precision/performance patterns

**Sub-routing logic (inside SKILL.md):**
```
"pretrain / 预训练 / pre-training / 预训练阶段" → knowledge/pretraining.md
"SFT / DPO / RLHF / alignment / 后训练 / 对齐" → knowledge/posttraining.md
"GRPO / PPO / RL / reinforcement / 强化学习" → knowledge/rl.md
"quant / KV cache / speculative / serving / 推理 / 量化" → knowledge/inference.md
```

### 4.3 Precision Expert (`precision/`)

**Purpose:** Diagnose and resolve precision failures through a structured workflow.

**Diagnostic Workflow (5-Step):**
1. **Capture** — collect full error message, training curve snapshot, environment (model size, parallel config, hardware, framework version)
2. **Classify** — categorize the symptom (loss NaN / loss spike / loss divergence / grad norm explosion / train-infer mismatch / accuracy regression)
3. **Localize** — trace the symptom to a layer (data → optimizer → gradient → activation → weight → loss computation)
4. **Hypothesize** — match against known patterns from `references/` and `tickets/`
5. **Resolve & Archive** — propose fix, verify, then archive to `tickets/`

**References contain:**
- Precision failure taxonomy (by symptom × by stage)
- Known patterns from source repos (e.g., Megatron's `grad_scaler`, miles's `true_on_policy/` numerics path)
- External paper techniques (mixed precision, loss scaling, grad clipping strategies)

### 4.4 Performance Expert (`performance/`)

**Purpose:** Optimize training performance against SOTA techniques.

**Optimization Workflow (5-Step):**
1. **Profile** — collect throughput, MFU, memory, communication/compute ratio, bottleneck indicator
2. **Identify** — classify bottleneck (compute-bound / memory-bound / communication-bound / I/O-bound / launch-bound)
3. **Match** — match against SOTA techniques in `references/` (e.g., FSDP2, context parallel, MoE dispatch, activation recompute, CUDA graph, async pipeline)
4. **Adapt** — adapt the technique to the user's specific hardware/framework (cite source repo)
5. **Validate & Archive** — propose expected speedup, verify, archive to `tickets/`

**References contain:**
- Performance bottleneck taxonomy
- SOTA technique catalog (with source-repo citations)
- Hardware-specific tuning guides (NVIDIA Ada / Moore Threads MUSA); Ascend NPU tuning is covered via external knowledge, marked as such

### 4.5 Problem Ticket Archive (`tickets/`)

**Purpose:** Accumulate solved problems as a living, queryable case library.

**Ticket Schema (frontmatter + body):**

```markdown
---
id: TICKET-YYYYMMDD-NNN   # NNN = per-day sequence (001, 002, ...)
title: One-line summary
type: precision | performance | hybrid
stage: pretraining | posttraining | rl | inference | cross-stage
status: resolved | investigating | wontfix
severity: critical | major | minor
hardware: [NVIDIA-Ada, MUSA, Ascend, CPU, agnostic]
frameworks: [Megatron-LM, torchtitan, miles, slime, torchada, torcht_musa]
tags: [loss-nan, grad-explosion, throughput, oom, ...]
created: YYYY-MM-DD
resolved: YYYY-MM-DD
related_tickets: [TICKET-...]
source_refs:  # file paths in local repos that informed the solution
  - Megatron-LM/megatron/core/...:line
---

## Symptom
What was observed (with logs/metrics if available).

## Environment
Model, scale, hardware, framework versions, parallel config.

## Analysis
Step-by-step root cause analysis. Cite source code.

## Root Cause
One paragraph: what was actually wrong.

## Resolution
What fixed it. Config changes, code patches, workflow adjustments.

## Verification
How we confirmed the fix worked.

## Lessons
Reusable insight. Cross-references to precision/performance references.

## References
Papers, docs, external links.
```

**Indexing:** Tickets are queryable by frontmatter fields. SKILL.md's router can search tickets by `type`, `stage`, `tags`, `frameworks`.

## 5. Iron Law (Top-Level Constraint)

```
Before ANY analysis, the skill MUST:
  1. Confirm the user's training stage (pretrain / posttrain / RL / inference)
  2. Confirm the framework(s) involved (Megatron / torchtitan / miles / slime / other)
  3. For precision/performance issues: require symptom description + environment

The skill MUST NOT:
  - Guess root cause without source-code evidence
  - Cite APIs/functions that were not verified against local repos
  - Propose fixes that contradict known source-repo behavior
```

## 6. Out of Scope (v1)

- **Automated code fixes** — this is a knowledge/analysis system, not a code-modification agent
- **Real-time training monitoring** — no live metrics ingestion
- **Multi-user collaboration features** — no auth, no assignment, no review workflow (reserved for team extensibility)
- **Non-training topics** — data engineering, MLOps deployment, model architecture design per se

## 7. Success Criteria

1. **Routing accuracy** — a question about GRPO RL training routes to `knowledge/rl.md` + `miles/` + `slime/` sources, not to pretraining
2. **Source-grounded answers** — every technical claim cites a local source-repo file path
3. **Workflow completeness** — a precision issue gets the full 5-step diagnostic, not an ad-hoc guess
4. **Ticket reusability** — a resolved problem is archived and retrievable by symptom/tags within 30 seconds
5. **Extensibility** — adding a new source repo (e.g., a new RL framework) requires only a new knowledge file + mapping entry, not a rewrite

## 8. Implementation Phases

| Phase | Deliverable | Scope |
|-------|-------------|-------|
| **P0** | Skeleton | `SKILL.md` (router), directory structure, `tickets/TEMPLATE.md`, `references/source-repo-map.md` |
| **P1** | Knowledge core | `knowledge/` 4 files (pretrain/posttrain/rl/inference) with source citations |
| **P2** | Expert engines | `precision/SKILL.md` + `performance/SKILL.md` with workflows |
| **P3** | Ticket seed | 3-5 seed tickets from existing interview/work notes |
| **P4** | References | Precision failure taxonomy + SOTA perf technique catalog |

> P0 is the minimal viable "crucible" — once it exists, P1-P4 accumulate organically.
