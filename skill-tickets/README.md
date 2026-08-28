[← Back to Root](../../README.md) | [中文](README_zh.md) | [English](README.md)

# Problem Ticket Archive

> Structured case library — every solved problem becomes reusable knowledge.

## What It Does

This module is the **institutional memory** of training-crucible. When a precision or performance issue is resolved (by `skill-precision/` or `skill-performance/`), it gets archived here as a structured ticket. Future similar problems can be retrieved instantly by filtering on frontmatter fields (`type`, `stage`, `tags`, `frameworks`).

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `SKILL.md` | Search & archive workflow engine | ~160 |
| `TEMPLATE.md` | Ticket template (YAML frontmatter + 8 body sections) | ~80 |
| `2026-08-27-*.md` | Seed tickets (from real project experience) | ~343 |

**Total: ~583 lines**

## Ticket Structure

### Frontmatter (YAML Metadata)

```yaml
id: TICKET-YYYYMMDD-NNN       # Unique identifier
title: One-line summary
type: precision | performance | hybrid
stage: pretraining | posttraining | rl | inference
status: resolved | investigating | wontfix
severity: critical | major | minor
hardware: [Ascend, NVIDIA-Ada, MUSA, CPU]
frameworks: [Megatron-LM, torchtitan, miles, slime]
tags: [loss-nan, throughput, oom, ...]
```

### Body (8 Sections)

1. **Symptom** — What was observed (with metrics)
2. **Environment** — Model, scale, hardware, framework
3. **Analysis** — Step-by-step root cause analysis
4. **Root Cause** — Root cause conclusion
5. **Resolution** — What fixed it
6. **Verification** — How the fix was confirmed
7. **Lessons** — Reusable insight
8. **References** — Source references

## Archive Rules

- Precision/performance issues resolved with clear root cause → **must archive**
- Cross-module complex problems → **archive**
- Same problem does **not** create multiple tickets (link instead)

## Search & Retrieval

Filter by frontmatter fields:

| Field | Example Values |
|-------|---------------|
| `type` | precision, performance, hybrid |
| `stage` | pretraining, posttraining, rl, inference |
| `tags` | loss-nan, throughput, oom, grad-explosion |
| `frameworks` | Megatron-LM, torchtitan, miles, slime |
| `hardware` | Ascend, NVIDIA-Ada, MUSA, CPU |
| `severity` | critical, major, minor |

## Cross-References

| Need | Go To |
|------|-------|
| Diagnose a new precision issue | `skill-precision/` |
| Optimize a performance bottleneck | `skill-performance/` |
| Understand the concept behind a case | `skill-knowledge/` |
| Source code referenced in a ticket | `skill-references/source-repo-map.md` |
