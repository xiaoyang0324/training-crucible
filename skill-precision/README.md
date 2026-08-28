[← Back to Root](../../README.md) | [中文](README_zh.md) | [English](README.md)

# Precision Expert

> 5-step diagnostic workflow — resolving loss anomalies, gradient explosions, accuracy regressions.

## What It Does

This module is the **precision troubleshooting engine** of training-crucible. When the user reports a precision anomaly (loss NaN, gradient explosion, accuracy regression), it executes a structured 5-step diagnostic workflow. Every diagnosis is source-grounded — symptoms are traced to specific code locations (`file:line`), matched against known patterns, and after resolution, archived to `skill-tickets/`.

## Workflow

```
Capture → Classify → Localize → Hypothesize → Resolve & Archive
```

| Step | Action | Output |
|------|--------|--------|
| **Capture** | Collect full error message, environment, training state | Symptom snapshot |
| **Classify** | Categorize by symptom type + training stage | Failure category |
| **Localize** | Trace layer-by-layer: data → optimizer → gradient → activation → weight → loss | Code location (`file:line`) |
| **Hypothesize** | Match against `known-patterns.md` + `skill-tickets/` | Ranked hypotheses |
| **Resolve** | Apply fix, verify, archive to `skill-tickets/` | Resolved ticket |

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `SKILL.md` | 5-step diagnostic workflow engine (entry point) | ~380 |
| `references/failure-taxonomy.md` | Multi-dimensional precision failure classification (symptom × stage × layer) | ~260 |
| `references/known-patterns.md` | 9 known precision issue patterns + solutions | ~247 |

**Total: ~887 lines**

## Trigger Conditions

| Symptom | Urgency | Typical Stage |
|---------|---------|---------------|
| Loss NaN | 🔴 Critical | Pre-training, RL |
| Loss Spike | 🔴 Critical | Pre-training, Post-training |
| Loss Divergence | 🟠 High | Pre-training |
| Grad Norm Explosion | 🔴 Critical | Pre-training, RL |
| Train-Infer Mismatch | 🟠 High | RL (GRPO/PPO) |
| Accuracy Regression | 🟡 Medium | Post-training (SFT/DPO) |

## Diagnostic Principles

1. **Symptoms first, then judgment** — no analysis without full error + environment info
2. **Layer-by-layer localization** — data → optimizer → gradient → activation → weight → loss
3. **Match known patterns** — check `known-patterns.md` and `skill-tickets/` first
4. **Archive after fix** — resolved issues must be archived to `skill-tickets/`

## Patterns Covered

- **Loss NaN**: FP16 overflow, division by zero, log(0)
- **Loss Spike**: dirty data, excessive learning rate
- **Grad Norm**: gradient explosion, gradient vanishing
- **RL-specific**: Reward hacking, policy collapse

## Cross-References

| Need | Go To |
|------|-------|
| Concept explanation of a failure mode | `skill-knowledge/` |
| Performance impact of a precision fix | `skill-performance/` |
| Similar historical cases | `skill-tickets/` |
| Source code for diagnosis | `skill-references/source-repo-map.md` |
