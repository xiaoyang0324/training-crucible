[中文](README_zh.md) | [English](README.md)

# Performance Expert

> 5-step optimization workflow — resolving low throughput, OOM, poor scaling efficiency.

## Workflow

```
Profile → Identify → Match → Adapt → Validate & Archive
```

## File Structure

| File | Purpose |
|------|---------|
| `SKILL.md` | 5-step optimization workflow engine (entry point) |
| `references/bottleneck-taxonomy.md` | Bottleneck classification (5 types + diagnostic methods) |
| `references/sota-techniques.md` | 14 SOTA optimization techniques catalog |

## Trigger Conditions

- Low throughput / Low MFU
- OOM / Memory bottleneck
- Poor scaling efficiency
- Communication bottleneck
- Slow kernels / Launch overhead

## Bottleneck Types

| Type | Characteristic | Typical Optimization |
|------|---------------|---------------------|
| Compute-bound | Low GPU utilization | Flash Attention, fused operators |
| Memory-bound | OOM / Insufficient memory | Activation Recompute, Offload |
| Communication-bound | High communication ratio | Comm Overlap, Sequence Packing |
| I/O-bound | Slow data loading | Data prefetch, multi-worker |
| Launch-bound | Kernel launch overhead | CUDA Graph, operator fusion |

## SOTA Techniques Covered

- **Memory**: Activation Recompute, ZeRO, CPU Offload, Gradient Checkpointing
- **Compute**: Flash Attention, Fused Operators, CUDA Graph, FP8/FP4
- **Communication**: Comm Overlap, Sequence Packing, Async Pipeline, MoE Dispatch
- **Parallelism**: FSDP2, Context Parallel, Expert Parallel
