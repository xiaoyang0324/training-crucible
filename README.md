[中文](README_zh.md) | [English](README.md)

<h1 align="center">training-crucible</h1>

<p align="center">
  <em>Forging full-stack AI training expertise under fire.</em>
</p>

<p align="center">
  <b>Pre-training</b> · <b>Post-training</b> · <b>Reinforcement Learning</b> · <b>Inference</b>
</p>

---

## Why this exists

Training large-scale AI models is systems engineering at its hardest — combining parallelism strategies, diagnosing precision failures, and optimizing performance bottlenecks demands deep domain knowledge and battle-tested experience.

**training-crucible structures that expertise and embeds it into your Claude Code assistant.** It's not a search engine. It's an AI Infra expert on call: you describe the problem, it runs a structured workflow, cites real source code, and delivers verifiable solutions.

---

## Four capabilities

| | Module | What it does | When to use |
|---|--------|--------------|-------------|
| 📚 | `knowledge/` | Knowledge Q&A routed by training stage | "What is X" / "How does Y work" |
| 🔬 | `precision/` | 5-step diagnostic workflow | Loss NaN / Gradient explosion / Accuracy regression |
| ⚡ | `performance/` | 5-step optimization workflow | Low throughput / OOM / Poor scaling efficiency |
| 📋 | `tickets/` | Structured case archive retrieval | "Have we seen this before" / "Past cases" |

## Scope

This library covers:
- ✅ Training parallelism strategies (TP/PP/DP/CP/EP/FSDP/ZeRO)
- ✅ Precision issue diagnosis (loss NaN/spike/divergence, gradient explosion, etc.)
- ✅ Performance optimization (throughput, memory, scaling efficiency)
- ✅ RL training (GRPO/PPO, rollout, training-inference integration)
- ✅ Inference optimization (KV Cache, quantization, speculative decoding)

This library does **NOT** cover:
- ❌ Model architecture design (MoE architecture selection, layer/dimension tuning)
- ❌ Data engineering (data pipeline bugs, data quality assessment)
- ❌ Hardware-level debugging (NPU/GPU kernel bugs, driver issues)
- ❌ Deployment infrastructure (Kubernetes, networking, storage)
- ❌ Multi-modal training specifics (vision encoders, cross-modal fusion)

---

## Architecture

```
training-crucible/
├── SKILL.md                        # Entry router — Iron Law + intent recognition
├── knowledge/                      # Knowledge expert
│   ├── pretraining.md              #   Parallelism strategies, memory optimization
│   ├── posttraining.md             #   SFT · DPO · RLHF
│   ├── rl.md                       #   GRPO · PPO · rollout generation
│   └── inference.md                #   KV Cache · Quantization · Speculative decoding
├── precision/                      # Precision expert
│   ├── SKILL.md                    #   Capture → Classify → Localize → Hypothesize → Resolve
│   └── references/                 #   Failure taxonomy · Known patterns
├── performance/                    # Performance expert
│   ├── SKILL.md                    #   Profile → Identify → Match → Adapt → Validate
│   └── references/                 #   Bottleneck taxonomy · SOTA techniques
├── tickets/                        # Problem archive
│   ├── SKILL.md                    #   Search & archive workflow
│   ├── TEMPLATE.md                 #   YAML frontmatter + 8 body sections
│   └── *.md                        #   Seed cases from real project experience
└── references/                     # Shared references
    ├── source-repo-map.md          #   9 repos → stage mapping
    ├── training-glossary.md        #   90 canonical terms
    └── answer-conventions.md       #   Answer conventions + output templates
```

---

## How routing works

```
"Your question"
      │
      ├─ Concept / How-it-works ──────► knowledge/ (sub-routed by stage)
      ├─ Precision anomaly ───────────► precision/ (5-step diagnosis → archive)
      ├─ Performance bottleneck ──────► performance/ (5-step optimization → archive)
      ├─ Historical cases ────────────► tickets/ (filter by tags)
      └─ Complex problem ─────────────► precision → performance → archive (chained)
```

---

## Source-grounded. Not hallucinated.

Every technical claim cites a **local source repo** (file path + line number) as primary evidence. External papers supplement — never replace.

| Repo | Stage | Hardware |
|------|-------|----------|
| Megatron-LM | Pre-training, Post-training | NVIDIA GPU |
| torchtitan | Pre-training, RL | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | Pre-training | NVIDIA Ada GPU |
| torch_musa | Pre-training | Moore Threads MUSA GPU |

---

## At a glance

| Module | Files | Lines | Core |
|--------|-------|-------|------|
| Knowledge | 4 | 853 | 4-stage knowledge Q&A |
| Precision | 3 | 567 | 5-step diagnosis + failure taxonomy + known patterns |
| Performance | 3 | 822 | 5-step optimization + bottleneck taxonomy + SOTA techniques |
| Tickets | 7 | ~490 | Template + SKILL.md + 5 seed cases |
| References | 3 | ~391 | Source repo map + glossary + answer conventions |

---

## Roadmap

```
P0 Skeleton     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P1 Knowledge    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P2 Experts      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P3 Tickets      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P4 References   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P5 New repos    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
P6 Practice     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
P7 Team scale   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
```

---

## Usage

1. Place this repo in your workspace — Claude Code auto-loads `SKILL.md`
2. Or symlink to `~/.claude/skills/training-crucible/`
3. Ask any training question — Claude routes to the right expert

---

## License

MIT
