[← 返回根目录](../../README.md) | [中文](README_zh.md) | [English](README.md)

# 精度诊断专家 (Precision Expert)

> 5 步诊断工作流——解决 loss 异常、梯度爆炸、精度回退等问题。

## 解决什么问题

本模块是 training-crucible 的**精度排查引擎**。当用户报告精度异常（loss NaN、梯度爆炸、精度回退），执行结构化的 5 步诊断工作流。每次诊断都源码为凭——症状追溯到具体代码位置（`file:line`），匹配已知模式，解决后归档到 `skill-tickets/`。

## 工作流

```
捕获 (Capture) → 分类 (Classify) → 定位 (Localize) → 假设 (Hypothesize) → 解决与归档 (Resolve & Archive)
```

| 步骤 | 动作 | 输出 |
|------|------|------|
| **捕获 (Capture)** | 收集完整报错、环境、训练状态 | 症状快照 |
| **分类 (Classify)** | 按症状类型 + 训练阶段分类 | 故障类别 |
| **定位 (Localize)** | 逐层追溯：数据 → 优化器 → 梯度 → 激活 → 权重 → loss | 代码位置 (`file:line`) |
| **假设 (Hypothesize)** | 匹配 `known-patterns.md` + `skill-tickets/` | 排序后的假设列表 |
| **解决 (Resolve)** | 应用修复、验证、归档到 `skill-tickets/` | 已解决的 ticket |

## 文件结构

| 文件 | 用途 | 行数 |
|------|------|------|
| `SKILL.md` | 5 步诊断工作流引擎（入口） | ~380 |
| `references/failure-taxonomy.md` | 精度故障多维分类（症状 × 阶段 × 层级） | ~260 |
| `references/known-patterns.md` | 9 种已知精度问题模式 + 解决方案 | ~247 |

**合计：~887 行**

## 触发条件

| 症状 | 紧急度 | 典型阶段 |
|------|--------|----------|
| Loss NaN | 🔴 紧急 | 预训练、RL |
| Loss Spike | 🔴 紧急 | 预训练、后训练 |
| Loss Divergence | 🟠 高 | 预训练 |
| Grad Norm 爆炸 | 🔴 紧急 | 预训练、RL |
| Train-Infer Mismatch | 🟠 高 | RL (GRPO/PPO) |
| Accuracy Regression | 🟡 中 | 后训练 (SFT/DPO) |

## 诊断原则

1. **先拿症状，再做判断** — 没有完整报错和环境信息不开始分析
2. **按层级定位** — 数据 → 优化器 → 梯度 → 激活 → 权重 → loss
3. **匹配已知模式** — 优先查 `known-patterns.md` 和 `skill-tickets/`
4. **修复后归档** — 已解决的问题必须归档到 `skill-tickets/`

## 覆盖模式

- **Loss NaN**: FP16 溢出、除零、log(0)
- **Loss Spike**: 脏数据、学习率过大
- **Grad Norm**: 梯度爆炸、梯度消失
- **RL 特有**: Reward hacking、Policy collapse

## 交叉引用

| 需求 | 跳转 |
|------|------|
| 某故障模式的概念解释 | `skill-knowledge/` |
| 精度修复的性能影响 | `skill-performance/` |
| 类似历史案例 | `skill-tickets/` |
| 诊断用源码位置 | `skill-references/source-repo-map.md` |
