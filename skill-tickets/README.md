[中文](README_zh.md) | [English](README.md)

# Problem Ticket Archive

> Structured case library — every solved problem becomes reusable knowledge.

## File Structure

| File | Purpose |
|------|---------|
| `TEMPLATE.md` | Ticket template (YAML frontmatter + 8 body sections) |
| `2026-08-27-*.md` | Seed tickets (from real project experience) |

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

- Precision/performance issues resolved with clear root cause → must archive
- Cross-module complex problems → archive
- Same problem does not create multiple tickets

## Search

Filter by frontmatter fields: `type`, `stage`, `tags`, `frameworks`
