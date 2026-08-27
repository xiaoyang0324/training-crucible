# 性能优化专家 (Performance Expert)

> 5 步优化工作流——解决吞吐低、OOM、扩展效率差等问题。

## 工作流

```
画像 (Profile) → 识别 (Identify) → 匹配 (Match) → 适配 (Adapt) → 验证与归档 (Validate & Archive)
```

## 文件结构

| 文件 | 用途 |
|------|------|
| `SKILL.md` | 5 步优化工作流引擎（入口） |
| `references/bottleneck-taxonomy.md` | 瓶颈分类（5 类瓶颈 + 诊断方法） |
| `references/sota-techniques.md` | 14 种 SOTA 优化技术目录 |

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
