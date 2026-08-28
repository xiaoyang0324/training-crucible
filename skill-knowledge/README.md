[← Back to Root](../../README.md) | [中文](README_zh.md) | [English](README.md)

# Knowledge Expert

> Answers "what is X" and "how does Y work" — knowledge Q&A routed by training stage.

## What It Does

This module is the **concept & mechanism layer** of training-crucible. When the user asks a knowledge question ("What is TP?", "How does GRPO work?"), the entry router (`SKILL.md`) dispatches to the corresponding stage file. Each file delivers: core concepts, source-grounded citations (`file:line`), ASCII architecture diagrams, config parameter tables, and common misconceptions.

## Stages Covered

| File | Stage | Core Content | Lines |
|------|-------|--------------|-------|
| `pretraining.md` | Pre-training | Parallelism (TP/PP/DP/CP/EP), memory optimization, config knobs | ~1590 |
| `post-training.md` | Post-training | SFT / DPO / RLHF principles and configuration | ~715 |
| `rl.md` | Reinforcement Learning | GRPO / PPO, rollout generation, training-inference integration | ~1421 |
| `inference.md` | Inference | KV Cache, quantization, speculative decoding, serving | ~961 |
| `hardware-adapter.md` | Hardware Adapter | torchada (CUDA→MUSA shim) + torch_musa (MUSA backend) | ~1098 |
| `moe.md` | MoE Deep-Dive | Router, load balancing, token dispatch — cross-repo analysis | ~1080 |
| `deepspeed.md` | DeepSpeed Deep-Dive | ZeRO-1/2/3, MoE, Pipeline, Autotuning | ~1227 |
| `pytorch.md` | PyTorch Internals | nn.Module, autograd, FSDP2, DTensor, compile | ~1165 |

**Total: ~9257 lines**

## How Routing Works

```
"What is X?" / "How does Y work?"
      │
      ├─ Pre-training topic ──────────────► pretraining.md
      ├─ Post-training topic ─────────────► post-training.md
      ├─ RL topic ────────────────────────► rl.md
      ├─ Inference topic ─────────────────► inference.md
      ├─ Hardware adapter topic ──────────► hardware-adapter.md
      ├─ MoE-specific topic ──────────────► moe.md
      ├─ DeepSpeed-specific topic ────────► deepspeed.md
      ├─ PyTorch internals topic ─────────► pytorch.md
      └─ Cross-cutting topic ─────────────► multiple files (chained)
```

## Content Standards

- **Chinese language** (English terms in parentheses)
- **Source-grounded**: every technical claim cites a local source repo (`file:line`)
- **Structured format** per file:
  - §0 Panoramic diagram (ASCII art)
  - §1–N Core modules (concept + call chain + cross-repo comparison)
  - Config parameter table (exact arg names)
  - Call chain summary diagram
  - Appendix A: Source file index
  - Appendix B: Practical quick-reference
  - Appendix C: Common pitfalls & solutions

## Cross-References

| Need | Go To |
|------|-------|
| Loss NaN / gradient explosion | `skill-precision/` |
| Low throughput / OOM | `skill-performance/` |
| Historical cases | `skill-tickets/` |
| Source repo map | `skill-references/source-repo-map.md` |
| Term definitions | `skill-references/training-glossary.md` |
