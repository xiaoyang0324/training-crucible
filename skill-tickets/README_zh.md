[中文](README_zh.md) | [English](README.md)

# 问题归档库 (Problem Ticket Archive)

> 结构化的案例库——每个已解决的问题都是一份可复用的知识。

## 文件结构

| 文件 | 用途 |
|------|------|
| `TEMPLATE.md` | 问题单模板（frontmatter + 8 段正文） |
| `2026-08-27-*.md` | 种子 ticket（来自真实项目经验） |

## Ticket 结构

### Frontmatter (YAML 元数据)

```yaml
id: TICKET-YYYYMMDD-NNN       # 唯一标识
title: 一句话摘要
type: precision | performance | hybrid
stage: pretraining | posttraining | rl | inference
status: resolved | investigating | wontfix
severity: critical | major | minor
hardware: [Ascend, NVIDIA-Ada, MUSA, CPU]
frameworks: [Megatron-LM, torchtitan, miles, slime]
tags: [loss-nan, throughput, oom, ...]
```

### Body (8 段正文)

1. **Symptom** — 症状描述（含指标）
2. **Environment** — 环境信息（模型、规模、硬件、框架）
3. **Analysis** — 根因分析步骤
4. **Root Cause** — 根因结论
5. **Resolution** — 解决方案
6. **Verification** — 验证方法
7. **Lessons** — 可复用经验
8. **References** — 引用来源

## 归档规则

- 精度/性能问题已解决且根因明确 → 必须归档
- 跨模块复杂问题 → 归档
- 同一问题不创建多个 ticket

## 检索方式

通过 frontmatter 字段筛选：`type`, `stage`, `tags`, `frameworks`
