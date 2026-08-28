[← Back to Root](../../README.md) | [中文](README_zh.md) | [English](README.md)

# Performance Expert

> 5-step optimization workflow — resolving low throughput, OOM, poor scaling efficiency.

## What It Does

This module is the **performance optimization engine** of training-crucible. When the user reports a performance bottleneck (low throughput, OOM, poor scaling), it executes a structured 5-step optimization workflow. Bottlenecks are classified into 5 types, matched against 14 SOTA techniques, and after resolution, archived to `skill-tickets/`.

## Workflow

```
Profile → Identify → Match → Adapt → Validate & Archive
```

| Step | Action | Output |
|------|--------|--------|
| **Profile** | Collect metrics: throughput, MFU, memory, communication ratio | Performance snapshot |
| **Identify** | Classify bottleneck type (compute/memory/comm/IO/launch) | Bottleneck category |
| **Match** | Map to SOTA techniques from `sota-techniques.md` | Candidate optimizations |
| **Adapt** | Apply config changes, code modifications | Optimized config/code |
| **Validate** | Measure improvement, archive to `skill-tickets/` | Validated ticket |

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `SKILL.md` | 5-step optimization workflow engine (entry point) | ~480 |
| `references/bottleneck-taxonomy.md` | 5 bottleneck types + diagnostic methods | ~310 |
| `references/sota-techniques.md` | 14 SOTA optimization techniques catalog | ~435 |

**Total: ~1225 lines**

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

## Cross-References

| Need | Go To |
|------|-------|
| Concept explanation of a technique | `skill-knowledge/` |
| Precision impact of a performance fix | `skill-precision/` |
| Similar historical cases | `skill-tickets/` |
| Source code for optimization | `skill-references/source-repo-map.md` |
