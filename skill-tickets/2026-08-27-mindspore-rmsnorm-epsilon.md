---
id: TICKET-20260827-001
title: MindSpore 版本迁移导致 RMSNorm epsilon 静默偏差
type: precision
stage: posttraining
status: resolved
severity: critical
hardware:
  - Ascend
frameworks:
  - MindSpore
  - MindSpeed
tags:
  - silent-precision-drift
  - rmsnorm
  - framework-migration
  - epsilon-mismatch
created: 2026-08-27
resolved: 2026-08-27
related_tickets: []
source_refs: []
  # 无本地源码仓对应——本 ticket 涉及 MindSpore 框架，非本地 6 仓范围
  # 详见 external_refs 中的来源笔记
external_refs:
  - 小艺5000卡集群_疑难杂症攻坚手册.md
---

## Symptom

小艺大模型从 MindSpore 1.x 迁移到 2.x 后，训练 loss 曲线在中后期出现 **0.5-1% 的静默偏差**。无 NaN/Inf 报错，不对比两条曲线根本发现不到。偏差在训练约 2000 step 后开始显现，随 step 增加逐步扩大。

- Loss 曲线：前 2000 step 与基线完全重合，之后缓慢偏离
- 无显式报错，训练正常完成
- 最终 checkpoint 精度评测下降约 0.8%

## Environment

| Item | Value |
|------|-------|
| Model | 小艺大模型（数十亿参数级） |
| Scale | 5000 卡昇腾集群 |
| Hardware | Atlas 800 A3 (Ascend 910B) |
| Framework | MindSpore 1.x → MindSpore 2.x, MindSpeed |
| Parallel config | TP + PP + DP 混合并行 |
| Batch size | 千卡级全局 batch |

## Analysis

1. **初始观察：** 迁移后 loss 曲线中后期偏离基线，无显式报错，属于典型的静默精度问题。
2. **二分法逐层比对：** 在相同输入下，逐层比对 MindSpore 1.x 和 2.x 的中间激活值（activation diff），定位到 **RMSNorm 算子** 的输出开始出现偏差。
3. **根因定位：** 检查 RMSNorm 算子配置，发现 1.x 使用 `epsilon=1e-6`，2.x 使用 `epsilon=1e-5`。进一步追溯配置项语义——1.x 读 `layernorm_epsilon`，2.x 读 `rms_norm_eps`，两个版本同一配置项的**默认值语义不一致**。
4. **影响量化：** 5000 卡 × 每层 RMSNorm，epsilon 差一个数量级（1e-6 vs 1e-5），在深层网络中累计放大，最终导致不可忽略的精度偏差。

## Root Cause

MindSpore 1.x 到 2.x 的版本升级中，RMSNorm 算子的 epsilon 默认值从 `1e-6` 变为 `1e-5`，且配置项名称从 `layernorm_epsilon` 改为 `rms_norm_eps`。迁移时未显式指定 `rms_norm_eps`，导致框架使用了新版本的默认值 1e-5。epsilon 差异在深层网络逐层累积，最终造成 loss 曲线中后期 0.5-1% 的静默偏差。

## Resolution

- **短期止血：** 在配置中显式设置 `rms_norm_eps=1e-6`，覆盖 MindSpore 2.x 的默认值，恢复与 1.x 一致的 epsilon 语义。
- **长期根治：** 建立版本兼容性测试套件——关键算子（RMSNorm、LayerNorm、Softmax 等）在版本升级前后做 activation diff 自动化比对，阈值设为 `1e-7`，超过阈值即告警。

## Verification

- 显式设置 `rms_norm_eps=1e-6` 后，loss 曲线与 MindSpore 1.x 基线完全重合，0.5-1% 偏差消失。
- 版本兼容性测试套件上线后，同类问题发现时间从**数天缩短到数小时**。
- 连续训练 10000+ step 无精度回退。

## Lessons

1. **框架迁移必须逐算子比对 activation，不能只看最终 loss。** 静默精度偏差无显式报错，只有自动化 diff 才能发现。
2. **配置项的"同名不同义"是迁移陷阱。** `layernorm_epsilon` 和 `rms_norm_eps` 名字不同但语义相同，版本升级时容易遗漏。
3. **epsilon 这种"小参数"影响巨大。** 1e-6 vs 1e-5 看似微小，但乘以网络深度和集群规模后，累积偏差不可忽略。
4. **建立版本升级回归体系是长期投资。** 一次搭建，每次版本升级自动跑，避免同类问题反复出现。

## References

- 来源笔记：`小艺5000卡集群_疑难杂症攻坚手册.md` — 问题四：MindSpore 兼容性导致的静默精度偏差
- 相关概念：RMSNorm epsilon 敏感性、框架迁移精度对齐
