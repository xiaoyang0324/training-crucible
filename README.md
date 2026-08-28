[中文](README_zh.md) | [English](README.md)

<h1 align="center">training-crucible</h1>

<p align="center">
  <em>Forging full-stack AI training expertise under fire.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v1.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  <img src="https://img.shields.io/badge/modules-4-orange" alt="modules">
  <img src="https://img.shields.io/badge/lines-13312-lightgrey" alt="lines">
</p>

<p align="center">
  <b>Pre-training</b> · <b>Post-training</b> · <b>Reinforcement Learning</b> · <b>Inference</b>
</p>

---

## The Problem

Training large-scale AI models is systems engineering at its hardest. You're combining parallelism strategies across hundreds of GPUs, diagnosing why loss suddenly went NaN at step 500, or figuring out why your 64-GPU cluster only delivers 40% MFU.

The knowledge exists — scattered across 8 open-source repos, hundreds of thousands of lines of code, and the tribal memory of engineers who've fought these battles.

**training-crucible brings it together and puts it on call inside your Claude Code assistant.**

---

## What It Does

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "My loss is NaN at step 500, how do I debug?"                 │
│                                                                 │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   SKILL.md Router   │                            │
│              │  (Intent Detection) │                            │
│              └──────────┬──────────┘                            │
│                         │                                       │
│           ┌─────────────┼─────────────┐                         │
│           ▼             ▼             ▼                         │
│    ┌────────────┐ ┌──────────┐ ┌──────────┐                     │
│    │  Knowledge │ │ Precision│ │Performance│                    │
│    │  Expert    │ │ Expert   │ │ Expert   │                     │
│    └────────────┘ └──────────┘ └──────────┘                     │
│           │             │             │                         │
│           └─────────────┼─────────────┘                         │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  Source-Grounded    │                            │
│              │  Answer + Workflow  │                            │
│              └─────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

It's not a search engine. It's an AI Infra expert that:
- **Answers** "what is X" and "how does Y work" with code-level citations
- **Diagnoses** precision anomalies via a 5-step workflow traced to `file:line`
- **Optimizes** performance by matching bottlenecks to 14 SOTA techniques
- **Archives** every solved problem into a structured, searchable case library

---

## See It In Action

**You ask:**
> "What's the difference between TP and PP?"

**training-crucible answers:**

```
┌────────────────────────────────────────────────────────────┐
│ 📚 Tensor Parallelism (TP) vs Pipeline Parallelism (PP)   │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  TP: splits ONE layer across devices                       │
│  PP: splits LAYERS across devices                          │
│                                                            │
│  ┌─────────────┐      ┌─────────────┐                      │
│  │ Layer 1-4   │ ───► │ Device 0    │  ← PP               │
│  │ Layer 5-8   │ ───► │ Device 1    │                      │
│  └─────────────┘      └─────────────┘                      │
│                                                            │
│  ┌─────────────┐      ┌──────┬──────┐                      │
│  │ Layer 1     │ ───► │ D0   │ D1   │  ← TP               │
│  │  (split)    │      │ W_a  │ W_b  │                      │
│  └─────────────┘      └──────┴──────┘                      │
│                                                            │
│  Source-grounded:                                          │
│  • Megatron-LM: layers.py:986 ColumnParallelLinear          │
│  • Megatron-LM: schedules.py:2147 1F1B schedule             │
│  • torchtitan: parallel_dims.py:132 ParallelDims           │
│                                                            │
│  Trade-off:                                                │
│  • TP → high comm-per-layer, best within one node          │
│  • PP → low comm-per-layer, required across nodes          │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

Every claim cites real source code. No hallucinated APIs. No vague descriptions.

---

## Four Capabilities

| | Module | What It Does | When to Use |
|---|--------|--------------|-------------|
| 📚 | `skill-knowledge/` | Knowledge Q&A routed by training stage (8 files, 9257 lines) | "What is X" / "How does Y work" |
| 🔬 | `skill-precision/` | 5-step diagnostic workflow (Capture → Classify → Localize → Hypothesize → Resolve) | Loss NaN / Gradient explosion / Accuracy regression |
| ⚡ | `skill-performance/` | 5-step optimization workflow (Profile → Identify → Match → Adapt → Validate) | Low throughput / OOM / Poor scaling efficiency |
| 📋 | `skill-tickets/` | Structured case archive with YAML metadata + 8-section body | "Have we seen this before" / "Past cases" |

---

## How Routing Works

```
"Your question"
      │
      ├─ Concept / How-it-works ──────► skill-knowledge/ (sub-routed by stage)
      ├─ Precision anomaly ───────────► skill-precision/ (5-step diagnosis → archive)
      ├─ Performance bottleneck ──────► skill-performance/ (5-step optimization → archive)
      ├─ Historical cases ────────────► skill-tickets/ (filter by tags)
      └─ Complex problem ─────────────► precision → performance → archive (chained)
```

---

## Source Repos Analyzed

Every technical claim traces back to one of these repos:

| Repo | Stage | Hardware | Coverage |
|------|-------|----------|----------|
| PyTorch | Base framework | CUDA GPU | nn/autograd/optim/distributed/FSDP2/DTensor |
| Megatron-LM | Pre-training, Post-training, MoE | NVIDIA GPU | TP/PP/CP/EP/MoE/GRPO |
| torchtitan | Pre-training, RL | NVIDIA GPU | FSDP/Float8/DAPO |
| DeepSpeed | Pre-training optimization | NVIDIA GPU | ZeRO-1/2/3, MoE, PP, Autotuning |
| miles | RL (GRPO/PPO) | NVIDIA GPU | Rollout, Reward Hub, async training |
| slime | RL (GRPO/PPO) | NVIDIA GPU | TIS/OPSM/OPD off-policy correction |
| torchada | Hardware adapter | NVIDIA Ada GPU | CUDA→MUSA shim, Graph Rotation |
| torch_musa | Hardware backend | Moore Threads MUSA GPU | Device/Memory/Stream/Graph/MCCL |

---

## Content Structure

```
training-crucible/
├── SKILL.md                              # Entry router — Iron Law + intent recognition
├── skill-knowledge/                      # Knowledge expert (9257 lines)
│   ├── pretraining.md                    #   Parallelism strategies, memory optimization
│   ├── post-training.md                  #   SFT · DPO · RLHF
│   ├── rl.md                             #   GRPO · PPO · rollout generation
│   ├── inference.md                      #   KV Cache · Quantization · Speculative decoding
│   ├── hardware-adapter.md               #   Hardware adapter layer (torchada + torch_musa)
│   ├── moe.md                            #   MoE cross-repo deep analysis
│   ├── deepspeed.md                      #   DeepSpeed deep analysis
│   └── pytorch.md                        #   PyTorch core features analysis
├── skill-precision/                      # Precision expert (887 lines)
│   ├── SKILL.md                          #   Capture → Classify → Localize → Hypothesize → Resolve
│   └── references/                       #   Failure taxonomy · Known patterns
├── skill-performance/                    # Performance expert (1225 lines)
│   ├── SKILL.md                          #   Profile → Identify → Match → Adapt → Validate
│   └── references/                       #   Bottleneck taxonomy · SOTA techniques
├── skill-tickets/                        # Problem archive (583 lines)
│   ├── SKILL.md                          #   Search & archive workflow
│   ├── TEMPLATE.md                       #   YAML frontmatter + 8 body sections
│   └── *.md                              #   Seed cases from real project experience
└── skill-references/                     # Shared references (1362 lines)
    ├── source-repo-map.md                #   8 repos → stage mapping
    ├── training-glossary.md              #   90 canonical terms
    └── answer-conventions.md             #   Answer conventions + output templates
```

---

## Project Stats

| Module | Files | Lines | Core |
|--------|-------|-------|------|
| Knowledge | 8 | ~9257 | 8-file code-level knowledge Q&A |
| Precision | 3 | ~887 | 5-step diagnosis + failure taxonomy |
| Performance | 3 | ~1225 | 5-step optimization + SOTA techniques |
| Tickets | 7 | ~583 | Template + 5 seed cases |
| References | 3 | ~1362 | Source repo map + glossary + conventions |
| **Total** | **24** | **~13312** | **4 capabilities, 8 source repos** |

---

## Key Differentiators

| | | |
|---|---|---|
| **Source-grounded** | Every technical claim cites a local source repo (file path + line number) as primary evidence | No hallucinated APIs |
| **Code-level** | Complete call chains with `file:line`, ASCII architecture diagrams, real code snippets | Not just concept descriptions |
| **Cross-repo** | Side-by-side comparison across 8 source repos | See how each framework solves the same problem |
| **Workflow-driven** | 5-step diagnostic + 5-step optimization workflows, not just Q&A | Structured problem-solving |

---

## Quick Start

```bash
# 1. Clone into your workspace
git clone git@github.com:xiaoyang0324/training-crucible.git

# 2. Claude Code auto-loads SKILL.md — just ask:
#    "What is the difference between TP and PP?"
#    "My loss is NaN at step 500, how to debug?"
#    "How to optimize throughput for 7B model on 64 GPUs?"
```

Or symlink to `~/.claude/skills/training-crucible/`.

---

## Roadmap

```
v1.0 (Current)  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
  Skeleton · Knowledge · Experts · Tickets · References
  Structure hardening · Naming conventions · CONTRIBUTING

v1.1 (Next)     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  Citation audit · More seed tickets · Coverage gaps

v2.0 (Future)   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  New hardware repos (Ascend/MUSA) · Team workflows
```

---

## License

MIT
