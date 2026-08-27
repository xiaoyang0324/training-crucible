[中文](README_zh.md) | [English](README.md)

# 精度诊断专家 (Precision Expert)

> 5 步诊断工作流——解决 loss 异常、梯度爆炸、精度回退等问题。

## 工作流

```
捕获 (Capture) → 分类 (Classify) → 定位 (Localize) → 假设 (Hypothesize) → 解决与归档 (Resolve & Archive)
```

## 文件结构

| 文件 | 用途 |
|------|------|
| `SKILL.md` | 5 步诊断工作流引擎（入口） |
| `references/failure-taxonomy.md` | 精度故障多维分类（按症状/阶段/层级） |
| `references/known-patterns.md` | 9 种已知精度问题模式 + 解决方案 |

## 触发条件

| 症状 | 紧急度 |
|------|--------|
| Loss NaN | 🔴 紧急 |
| Loss Spike | 🔴 紧急 |
| Loss Divergence | 🟠 高 |
| Grad Norm 爆炸 | 🔴 紧急 |
| Train-Infer Mismatch | 🟠 高 |
| Accuracy Regression | 🟡 中 |

## 诊断原则

1. **先拿症状，再做判断** — 没有完整报错和环境信息不开始分析
2. **按层级定位** — 数据 → 优化器 → 梯度 → 激活 → 权重 → loss
3. **匹配已知模式** — 优先查 `known-patterns.md` 和 `tickets/`
4. **修复后归档** — 已解决的问题必须归档到 `tickets/`

## 覆盖模式

- Loss NaN: FP16 溢出、除零、log(0)
- Loss Spike: 脏数据、学习率过大
- Grad Norm: 梯度爆炸、梯度消失
- RL 特有: Reward hacking、Policy collapse
