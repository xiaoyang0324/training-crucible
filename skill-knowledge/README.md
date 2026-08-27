[中文](README_zh.md) | [English](README.md)

# Knowledge Expert

> Answers "what is X" and "how does Y work" — knowledge Q&A module routed by training stage.

## Stages Covered

| File | Stage | Core Content |
|------|-------|--------------|
| `pretraining.md` | Pre-training | Parallelism (TP/PP/DP/CP/EP), memory optimization, config knobs |
| `post-training.md` | Post-training | SFT / DPO / RLHF principles and configuration |
| `rl.md` | Reinforcement Learning | GRPO / PPO, rollout generation, training-inference integration |
| `inference.md` | Inference | KV Cache, quantization, speculative decoding, serving |

## How It Works

When the user asks "what is X" or "how does Y work", SKILL.md routes to the corresponding stage file.
Answers include: core concepts, source citations, architecture diagrams, config tables, common misconceptions.

## Content Standards

- Chinese language (English terms in parentheses)
- Every technical claim cites a local source repo (file path)
- Includes ASCII architecture diagrams, config parameter tables, common misconceptions
- Ends with a source file index table

## Source References

See `skill-references/source-repo-map.md` for the full source repo mapping.
