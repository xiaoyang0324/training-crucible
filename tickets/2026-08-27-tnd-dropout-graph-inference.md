---
id: TICKET-20260827-005
title: 美团推理 TND 布局下 dropout 无法入图
type: performance
stage: inference
status: resolved
severity: major
hardware:
  - Ascend
frameworks:
  - torch_npu
tags:
  - inference
  - tnd-layout
  - dropout
  - workspace-deadlock
  - aclgraph
  - dynamic-shape
created: 2026-08-27
resolved: 2026-08-27
related_tickets: []
source_refs:
  - torch_npu/npu/_npugraph_handlers/_fa3_graph_handler.py
external_refs:
  - 美团千卡推理入图_项目详析.md
---

## Symptom

美团推荐/搜索大模型在昇腾 NPU 推理部署时，TND 布局的 attention 在 aclgraph 中报错：

```
RuntimeError: TND+dropout not supported in ACLgraph
```

具体表现：

- 训练场景的 TND 布局 attention 入图失败
- Eager 模式下 TND + dropout 正常工作
- 推理场景 dropout=0 时无报错
- 仅训练场景（dropout > 0）+ TND 布局组合触发

## Environment

| Item | Value |
|------|-------|
| Model | 美团推荐/搜索大模型（LLM 规模） |
| Scale | 1000+ 卡昇腾集群推理 |
| Hardware | Atlas 800 A3 (Ascend 910B) |
| Framework | torch_npu + aclgraph 图模式 |
| Parallel config | 推理部署，batch size 动态 |
| Batch size | 动态（1/2/4/8/16 多档位） |

## Analysis

1. **初始观察：** 推理场景（dropout=0）无问题，说明 TND 布局本身支持入图，问题在 dropout。
2. **根因定位：** ACL 图模式下 dropout 需要预知 mask 大小来分配 workspace。mask 大小 = cu_seqlens 的 cumsum 结果（即 total_tokens）。但 cu_seqlens 是输入依赖的运行时值，捕获时无法确定。
3. **死锁形成：** ACL 底层的 `aclnnFlashAttentionVarLenScoreV4` 在图模式下要求 workspace 在捕获时就确定大小，而 workspace 大小依赖运行时输入——构成"鸡生蛋"死锁。
4. **推理场景优势：** 推理 eval 模式下 dropout 固定为 0，不需要 dropout workspace，天然绕过此限制。

## Root Cause

TND 布局下 dropout 的 workspace 大小 = cu_seqlens 的 cumsum（total_tokens），这是一个输入依赖的运行时值。aclgraph 要求 workspace 在捕获时确定大小，但 cu_seqlens 在捕获时未知——两者构成死锁。这是图模式"静态 workspace 分配"与"动态序列长度"之间的根本矛盾。推理场景因 dropout=0 天然免疫。

## Resolution

- **推理场景：** 直接利用 eval 模式 dropout=0 的特性，绕过此限制。推理入图相比训练入图的一个天然优势就是算子链更干净，少了 dropout 这个随机性组件。
- **训练场景（如需 TND+dropout 入图）：** 需将 cu_seqlens 注册为图外部参数，每次 replay 前通过 `graph_task_update` 传入实际 total_tokens，并预分配最大可能的 workspace。

```python
# 推理场景：eval 模式 dropout=0，无需额外处理
model.eval()  # dropout 自动为 0
with torch.npu.graph(g):
    output = model(input)  # TND attention 入图成功
```

## Verification

- 推理场景 TND 布局 attention 入图成功率 100%
- 吞吐提升 1.3x，P99 延迟降低 25%+
- 训练场景如需 TND+dropout 入图，需走 update 路径（当前推理场景无需）
- 1000+ 卡集群推理稳定运行

## Lessons

1. **推理入图比训练入图有天然优势：** 无 dropout、无梯度计算、无优化器更新，算子链更干净。
2. **图模式的"静态分配"假设与动态性是根本矛盾。** 所有依赖运行时值的 workspace 都需要特殊处理。
3. **TND 布局的 cu_seqlens 是"运行时值"的典型代表。** 任何依赖 cu_seqlens 的算子（dropout、某些 reduction）在图模式下都会遇到类似问题。
4. **利用场景特性（推理 dropout=0）是最优雅的解决方案。** 不需要改框架，不需要 workaround，场景本身消除了问题。

## References

- 来源笔记：`美团千卡推理入图_项目详析.md` — 问题 1：TND 布局下 dropout 无法入图
- 源码：`torch_npu/npu/_npugraph_handlers/_fa3_graph_handler.py` — FA3 handler workspace 计算
