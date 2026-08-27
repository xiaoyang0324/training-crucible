---
id: TICKET-20260827-002
title: 5000 卡集群快慢卡导致周期性掉速 30-50%
type: performance
stage: pretraining
status: resolved
severity: critical
hardware:
  - Ascend
frameworks:
  - MindSpeed
  - torch_npu
tags:
  - slow-card
  - straggler
  - throughput-degradation
  - cluster-reliability
  - hccs-link
created: 2026-08-27
resolved: 2026-08-27
related_tickets: []
source_refs: []
  # 无本地源码仓对应——本 ticket 根因为 HCCL 链路/散热硬件问题
  # 详见 external_refs 中的攻坚手册笔记
external_refs:
  - 小艺5000卡集群_疑难杂症攻坚手册.md
---

## Symptom

小艺大模型在 5000 卡昇腾集群训练时，出现**周期性抖动**——每隔几小时吞吐下降 30-50%，偶发训练 hang。具体表现：

- 正常时单步耗时 100ms，慢卡出现后单步耗时 300ms+
- 集群实际吞吐 = 最慢那张卡的速度 × N
- 无规律出现，人工排查耗时数小时
- 5000 卡集群 MTBF ≈ 10 小时，故障是统计必然

## Environment

| Item | Value |
|------|-------|
| Model | 小艺大模型（数十亿参数级） |
| Scale | 5000 卡昇腾集群 |
| Hardware | Atlas 800 A3 (Ascend 910B), HCCS 互联 392 GB/s |
| Framework | MindSpeed + torch_npu |
| Parallel config | TP + PP + DP 混合并行，AllReduce 集合通信 |
| Batch size | 千卡级全局 batch |

## Analysis

1. **规模感分析：** 单卡 MTBF ≈ 5 万小时，集群 MTBF = 5 万 / 5000 ≈ 10 小时。每 10 小时就有一张卡出问题，慢卡不是偶发，是统计必然。
2. **现象定位：** 通过 HCCL heartbeat 日志分析，发现特定 rank 的 AllReduce 耗时是集群中位数的 5-10 倍（P50=142ms vs 集群 P50=20.8ms）。
3. **三层排查流水线：**
   - 第一层（统计层）：HCCL Heartbeat 时序分析，筛出嫌疑 rank（P50 超出集群中位数 3x）
   - 第二层（硬件层）：HCCS 链路带宽测试 + NPU 热历史匹配，确定根因类别
   - 第三层（决策层）：分类为散热降频 / HCCS 链路降级 / 芯片体质差异
4. **根因分布：** 散热降频 ~50%，HCCS 链路硬件故障 ~30%，芯片体质差异 ~20%。

## Root Cause

5000 卡集群中，部分 NPU 卡因散热降频、HCCS 链路降级或芯片体质差异导致计算性能显著低于集群中位数。集合通信（AllReduce）需要所有 rank 同步完成，最慢的单卡决定了整步耗时。由于集群规模大，慢卡故障是每约 10 小时必然发生的统计现象，若无自动化检测手段，每次故障都需要人工翻日志排查，耗时数小时。

## Resolution

开发自研工具 `slow_card_detector.py`，实现三层自动化流水线：

1. **统计层：** 解析 HCCL heartbeat 日志，按 `{step: {rank: [duration]}}` 组织，用 P50 比例 + 连续 5 步确认机制筛出嫌疑 rank（排除瞬态 GC/OS 抖动）。
2. **硬件层：** 对嫌疑卡逐条 HCCS 链路做带宽测试（标称 392 GB/s，实际 < 50% 判定降级），匹配 NPU 温度/降频历史。
3. **决策层：** 按优先级分类——thermal → 散热降频（可恢复），HCCS degraded → 链路故障（可恢复），compute deficit → 芯片体质（不可恢复，永久隔离）。

检测到慢卡后，MindSpeed 通信组管理层动态重编排 HCCL 通信组，剔除异常卡，训练按新拓扑继续。

## Verification

- 故障恢复时间：**小时级 → 分钟级**
- 三层排查全链路自动化：**10 分钟内**产出根因 + 处理建议
- 与业界对比：百度百舸万卡集群平均 3 分钟自愈，同一量级
- 有效训练时长占比提升至 95%+

## Lessons

1. **大集群的慢卡是统计必然，不是偶发。** MTBF / N 是集群可靠性的基本公式，5000 卡意味着每 10 小时出一次问题。
2. **工具化的核心是把隐性经验显性化。** 新人拿着自动化工具也能完成之前只有老手能做的事。
3. **P50 + 3x + 连续 5 步是经验证的检测参数。** P99 噪声大，2x 假阳性多，5x 发现太迟；连续确认排除瞬态抖动。
4. **统计规则 >92% 准确率，可解释性远好于黑箱。** 慢 5-10x 不需要深度学习，简单规则即可。
5. **动态隔离优于全局重启。** 5000 卡重启代价 30+ 分钟，动态剔除异常卡让健康卡无缝接管。

## References

- 来源笔记：`小艺5000卡集群_疑难杂症攻坚手册.md` — 问题一：千卡级快慢卡检测与自愈
- 业界对比：百度百舸万卡集群 3 分钟自愈、字节 42.5% 任务受 straggler 影响
