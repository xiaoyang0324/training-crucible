[← 返回根目录](../../README.md) | [中文](README_zh.md) | [English](README.md)

# 性能优化专家 (Performance Expert)

> 5 步优化工作流——解决吞吐低、OOM、扩展效率差等问题。

## 解决什么问题

本模块是 training-crucible 的**性能优化引擎**。当用户报告性能瓶颈（吞吐低、OOM、扩展效率差），执行结构化的 5 步优化工作流。瓶颈分为 5 大类，匹配 14 种 SOTA 技术，解决后归档到 `skill-tickets/`。

## 工作流

```
画像 (Profile) → 识别 (Identify) → 匹配 (Match) → 适配 (Adapt) → 验证与归档 (Validate & Archive)
```

| 步骤 | 动作 | 输出 |
|------|------|------|
| **画像 (Profile)** | 收集指标：吞吐、MFU、内存、通信占比 | 性能快照 |
| **识别 (Identify)** | 分类瓶颈类型（计算/内存/通信/IO/启动） | 瓶颈类别 |
| **匹配 (Match)** | 映射到 `sota-techniques.md` 中的 SOTA 技术 | 候选优化方案 |
| **适配 (Adapt)** | 应用配置变更、代码修改 | 优化后的配置/代码 |
| **验证 (Validate)** | 测量提升、归档到 `skill-tickets/` | 已验证的 ticket |

## 文件结构

| 文件 | 用途 | 行数 |
|------|------|------|
| `SKILL.md` | 5 步优化工作流引擎（入口） | ~480 |
| `references/bottleneck-taxonomy.md` | 5 类瓶颈 + 诊断方法 | ~310 |
| `references/sota-techniques.md` | 14 种 SOTA 优化技术目录 | ~435 |

**合计：~1225 行**

## 触发条件

- 吞吐低 / MFU 低
- OOM / 内存瓶颈
- 扩展效率差 (scaling efficiency)
- 通信瓶颈
- 算子慢 / kernel launch 瓶颈

## 瓶颈类型

| 类型 | 特征 | 典型优化 |
|------|------|----------|
| Compute-bound | GPU 利用率低 | Flash Attention, 融合算子 |
| Memory-bound | OOM / 内存不足 | Activation Recompute, Offload |
| Communication-bound | 通信占比高 | Comm Overlap, Sequence Packing |
| I/O-bound | 数据加载慢 | 数据预取, 多 worker |
| Launch-bound | kernel 启动开销 | CUDA Graph, 算子融合 |

## SOTA 技术覆盖

- **内存优化**: Activation Recompute, ZeRO, CPU Offload, Gradient Checkpointing
- **计算优化**: Flash Attention, Fused Operators, CUDA Graph, FP8/FP4
- **通信优化**: Comm Overlap, Sequence Packing, Async Pipeline, MoE Dispatch
- **并行策略**: FSDP2, Context Parallel, Expert Parallel

## 交叉引用

| 需求 | 跳转 |
|------|------|
| 某技术的概念解释 | `skill-knowledge/` |
| 性能修复的精度影响 | `skill-precision/` |
| 类似历史案例 | `skill-tickets/` |
| 优化用源码位置 | `skill-references/source-repo-map.md` |
