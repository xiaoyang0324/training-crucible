[中文](README_zh.md) | [English](README.md)

# Precision Expert

> 5-step diagnostic workflow — resolving loss anomalies, gradient explosions, accuracy regressions.

## Workflow

```
Capture → Classify → Localize → Hypothesize → Resolve & Archive
```

## File Structure

| File | Purpose |
|------|---------|
| `SKILL.md` | 5-step diagnostic workflow engine (entry point) |
| `references/failure-taxonomy.md` | Multi-dimensional precision failure classification |
| `references/known-patterns.md` | 9 known precision issue patterns + solutions |

## Trigger Conditions

| Symptom | Urgency |
|---------|---------|
| Loss NaN | 🔴 Critical |
| Loss Spike | 🔴 Critical |
| Loss Divergence | 🟠 High |
| Grad Norm Explosion | 🔴 Critical |
| Train-Infer Mismatch | 🟠 High |
| Accuracy Regression | 🟡 Medium |

## Diagnostic Principles

1. **Symptoms first, then judgment** — no analysis without full error + environment info
2. **Layer-by-layer localization** — data → optimizer → gradient → activation → weight → loss
3. **Match known patterns** — check `known-patterns.md` and `skill-tickets/` first
4. **Archive after fix** — resolved issues must be archived to `skill-tickets/`

## Patterns Covered

- Loss NaN: FP16 overflow, division by zero, log(0)
- Loss Spike: dirty data, excessive learning rate
- Grad Norm: gradient explosion, gradient vanishing
- RL-specific: Reward hacking, policy collapse
