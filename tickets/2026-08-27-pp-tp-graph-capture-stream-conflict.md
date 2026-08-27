---
id: TICKET-20260827-004
title: PP+TP 混合并行下图捕获 stream 冲突导致失败
type: performance
stage: pretraining
status: resolved
severity: major
hardware:
  - Ascend
frameworks:
  - torch_npu
tags:
  - pipeline-parallel
  - tensor-parallel
  - graph-capture
  - stream-conflict
  - pp-tp-compatibility
created: 2026-08-27
resolved: 2026-08-27
related_tickets: []
source_refs:
  - torch_npu/csrc/core/npu/NPUGraph.cpp
external_refs:
  - 字节千卡训练图模式_项目详析.md
---

## Symptom

字节 Seed 系列视频生成模型在 1000+ 卡昇腾集群上开启 aclgraph 图模式时，`AclmdlRICaptureBegin` 报错 `stream != capture_stream_`，图捕获失败。具体表现：

- 单 PP stage 独立捕获成功，多 stage 联合捕获失败
- 报错发生在 PP stage 边界切换时
- 纯 TP 场景无此问题，PP+TP 混合时必现
- 错误信息：`stream != capture_stream_`（`NPUGraph.cpp:263-264`）

## Environment

| Item | Value |
|------|-------|
| Model | Seedance 2.0/2.5 视频生成、Seedream 5.0 Large |
| Scale | 1000+ 卡昇腾集群 |
| Hardware | Atlas 800 A3 (Ascend 910B) |
| Framework | 字节自研训练框架 + torch_npu aclgraph |
| Parallel config | PP（多 stage）+ TP 混合并行 |
| Batch size | 视频生成模型，3D 卷积 + 时空 attention |

## Analysis

1. **初始观察：** 单 stage 捕获成功，多 stage 失败，说明问题不在算子本身，而在 stage 间交互。
2. **根因定位：** PP 每个 stage 有独立的计算流（stream），但 aclgraph 要求 capture 必须在**同一个 non-default stream** 上进行（`NPUGraph.cpp:192-195`）。PP 的多流模型与 aclgraph 的单流捕获模型直接冲突。
3. **验证：** 在 `capture_begin` 前打印当前 stream，发现 PP stage 切换时 stream 上下文已变为新 stage 的 stream，与 capture 起始时的 stream 不一致，触发 `NPUGraph.cpp:263-264` 的校验失败。
4. **设计权衡：** 如果把所有 PP stage 捕获为一张大图，需要全局统一 stream，跨 1000+ 卡的物理边界不现实（跨节点 stream 同步开销太大）。

## Root Cause

Pipeline Parallel 的每个 stage 有独立的计算流，而 aclgraph 的 `capture_begin/capture_end` 必须在同一条非 default 流上进行。PP 多 stage 切换时 stream 上下文改变，导致 `capture_end` 校验 `stream == capture_stream_` 失败。这是 PP 的多流调度模型与 aclgraph 的单流捕获模型之间的架构冲突。

## Resolution

采用"每 PP stage 独立子图 + event 同步"方案：

1. 每个 PP stage 创建独立的 `NPUGraph` 实例 + 独立 capture stream
2. Stage 之间的 P2P 通信不捕获（标记为 graph external）
3. Replay 时通过 `Event.record/wait` 在 stage 之间同步

```python
# 每个 PP stage 独立捕获
for stage in pp_stages:
    graph = torch.npu.NPUGraph()
    stream = torch.npu.Stream()
    # 在 stage 的独立 stream 上捕获
    with torch.npu.stream(stream):
        with torch.npu.graph(graph):
            stage.forward()
    graphs.append(graph)

# Replay 时 event 同步
for i, graph in enumerate(graphs):
    if i > 0:
        event.wait(stream)  # 等待前一个 stage
    graph.replay()
    event.record(stream)
```

## Verification

- PP+TP 混合并行下图捕获成功率 100%
- 1000+ 卡集群稳定训练，无 stream 冲突报错
- 图模式加速比达到预期 2-3%
- Stage 间 event 同步开销 < 1ms，可忽略

## Lessons

1. **图捕获是"单流假设"，多流架构必须拆子图。** PP、任何多流调度器都需要独立子图方案。
2. **图粒度是工程权衡：** 图太大则捕获复杂度和跨节点同步开销爆炸，图太小则 replay 的 overhead 优势消失。
3. **P2P 通信应标记为 graph external。** 通信与计算的 overlap 由调度器管理，不需要入图。
4. **event 同步是连接子图的"胶水"。** 简单、高效、跨 stream 安全。

## References

- 来源笔记：`字节千卡训练图模式_项目详析.md` — 问题 1：PP+TP 混合并行下图捕获边界兼容
- 源码：`torch_npu/csrc/core/npu/NPUGraph.cpp:192-195` — 非 default 流校验
- 源码：`torch_npu/csrc/core/npu/NPUGraph.cpp:263-264` — stream == capture_stream_ 校验
