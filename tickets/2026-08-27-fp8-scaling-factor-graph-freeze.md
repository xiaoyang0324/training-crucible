---
id: TICKET-20260827-003
title: FP8 模型入图后 loss 发散——scaling factor 被冻住
type: hybrid
stage: pretraining
status: resolved
severity: major
hardware:
  - Ascend
frameworks:
  - torch_npu
tags:
  - fp8
  - scaling-factor
  - amax
  - graph-capture
  - update-specs
  - loss-divergence
created: 2026-08-27
resolved: 2026-08-27
related_tickets: []
source_refs:
  - torch_npu/npu/_npugraph_handlers/npugraph_handler.py
  - torch_npu/npu/graphs.py
external_refs:
  - NPU图模式泛化_项目详析.md
---

## Symptom

FP8 模型在 eager 模式下训练 loss 正常，开启 aclgraph 图模式捕获后 **loss 快速发散**。具体表现：

- Eager 模式：loss 曲线正常下降，grad norm 稳定
- 图模式：前几步 loss 与 eager 一致，约 10-20 步后 loss 开始飙升，随后 NaN
- 不同 batch 的数据分布差异越大，发散越快
- 小 batch（数据分布变化慢）能多撑几十步，大 batch 迅速发散

## Environment

| Item | Value |
|------|-------|
| Model | FP8 混合精度训练模型（LLM 规模） |
| Scale | 多卡昇腾集群 |
| Hardware | Atlas 800 A3 (Ascend 910B) |
| Framework | torch_npu + aclgraph 图模式 |
| Parallel config | TP + DP，FP8 训练 |
| Batch size | 标准训练 batch |

## Analysis

1. **初始观察：** Eager 正常 + 图模式发散，说明问题不在模型/数据，而在图捕获机制。
2. **对比 eager vs 图模式差异：** Eager 模式下每个算子的 FP8 scaling factor（amax）是每步动态计算的；图模式下所有参数在捕获时被"固化"。
3. **根因定位：** FP8 的 scaling factor 依赖输入数据的绝对值最大值（amax），每步随数据分布变化。图捕获时 scaling factor 被冻住为捕获时刻的值，后续 step 的数据分布变化导致精度不够——amax 过小则溢出，amax 过大则精度损失。
4. **验证：** 在图模式下手动每步重捕获（而非 replay），loss 恢复正常，确认是 scaling factor 冻结导致。

## Root Cause

FP8 混合精度训练中，scaling factor（amax）是每步根据输入数据动态计算的。aclgraph 图捕获把"这一刻"的状态固化为"所有时刻"的状态——捕获时的 scaling factor 被冻住，后续 step 的数据分布变化使得冻住的 scaling factor 不再适用，导致 FP8 量化误差累积，最终 loss 发散。这揭示了图模式的本质限制：任何依赖运行时动态计算的值都需要通过 update 机制注入。

## Resolution

将 FP8 scaling factor 注册为图的可更新参数：

1. 在算子 handler 的 `UPDATE_SPECS` 中声明 scaling factor 参数位置
2. 训练每步通过 `graph_task_update_begin/end` 更新 scaling factor
3. 图的其余部分（算子拓扑、权重地址、workspace 分配）保持不变

```python
# 伪代码：FP8 scaling factor 注册为可更新参数
class FP8Handler(NpuGraphOpHandler):
    UPDATE_SPECS = {
        'scale': ARG_INDEX_SCALE,  # 声明 scale 参数位置
    }
```

## Verification

- 图模式下 FP8 训练 loss 曲线与 eager 完全对齐
- 不同 batch size / 数据分布下均无发散
- 图模式加速比保持预期（省去重捕获的秒级开销）
- 连续训练 5000+ step 无精度回退

## Lessons

1. **图模式的本质限制：捕获固化状态，运行时动态值需 update。** FP8 scale、dropout mask、shape 推导值都在此列。
2. **UPDATE_SPECS 是图模式的"生命线"。** 注册的动态参数越多，图模式能覆盖的场景越广。
3. **FP8 入图比 BF16 入图复杂一个数量级。** BF16 无动态 scaling，图捕获天然安全；FP8 必须处理 scaling factor 的动态性。
4. **验证图模式精度时，不能用"前几步正常"作为通过标准。** 必须跑足够多的 step 覆盖数据分布变化。

## References

- 来源笔记：`NPU图模式泛化_项目详析.md` — 问题 3：FP8 模型入图精度丢失
- 源码：`torch_npu/npu/_npugraph_handlers/npugraph_handler.py` — UPDATE_SPECS 机制
- 源码：`torch_npu/npu/graphs.py` — graph_task_update 流程
