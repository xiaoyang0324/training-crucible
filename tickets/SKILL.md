---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → tickets/SKILL.md
description: >
  问题归档与检索模块——按症状/阶段/框架检索历史案例，
  或将已解决的精度/性能问题归档为新 ticket。
---

# 问题归档模块

## 检索工作流

1. 读取用户问题的关键词（症状、阶段、框架）
2. 扫描 tickets/ 下所有 *.md 的 frontmatter（tags, type, stage）
3. 按匹配度排序返回，格式：`TICKET-ID — title（匹配标签: xxx）`

## 归档工作流

触发条件：
- 精度/性能问题已解决且根因明确
- 跨模块复杂问题
- 用户明确要求归档

步骤：
1. 复制 TEMPLATE.md → tickets/YYYY-MM-DD-<slug>.md
2. 填写 frontmatter（id, type, stage, tags, source_refs）
3. 填写 8 段正文（Symptom, Environment, Analysis, Root Cause, Resolution, Verification, Lessons, References）
4. 检查 related_tickets 是否需要更新

## 检索命令示例

- "遇到过 loss 发散吗" → 按 `tags: loss-divergence` 检索
- "torch_npu 相关" → 按 `frameworks: torch_npu` 检索
- "掉速 / 快慢卡问题" → 按 `tags: slow-card` / `tags: straggler` 检索
