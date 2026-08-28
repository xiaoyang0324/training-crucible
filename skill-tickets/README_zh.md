[← 返回根目录](../../README.md) | [中文](README_zh.md) | [English](README.md)

# 问题归档库 (Problem Ticket Archive)

> 结构化的案例库——每个已解决的问题都是一份可复用的知识。

## 解决什么问题

本模块是 training-crucible 的**组织记忆**。当精度或性能问题被解决后（由 `skill-precision/` 或 `skill-performance/` 完成），以结构化 ticket 归档于此。未来遇到类似问题可通过 frontmatter 字段（`type`、`stage`、`tags`、`frameworks`）即时检索。

## 文件结构

| 文件 | 用途 | 行数 |
|------|------|------|
| `SKILL.md` | 检索 & 归档工作流引擎 | ~160 |
| `TEMPLATE.md` | 问题单模板（YAML frontmatter + 8 段正文） | ~80 |
| `2026-08-27-*.md` | 种子 ticket（来自真实项目经验） | ~343 |

**合计：~583 行**

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

- 精度/性能问题已解决且根因明确 → **必须归档**
- 跨模块复杂问题 → **归档**
- 同一问题**不**创建多个 ticket（改为链接引用）

## 检索方式

通过 frontmatter 字段筛选：

| 字段 | 示例值 |
|------|--------|
| `type` | precision, performance, hybrid |
| `stage` | pretraining, posttraining, rl, inference |
| `tags` | loss-nan, throughput, oom, grad-explosion |
| `frameworks` | Megatron-LM, torchtitan, miles, slime |
| `hardware` | Ascend, NVIDIA-Ada, MUSA, CPU |
| `severity` | critical, major, minor |

## 交叉引用

| 需求 | 跳转 |
|------|------|
| 诊断新的精度问题 | `skill-precision/` |
| 优化性能瓶颈 | `skill-performance/` |
| 理解案例背后的概念 | `skill-knowledge/` |
| ticket 引用的源码位置 | `skill-references/source-repo-map.md` |
