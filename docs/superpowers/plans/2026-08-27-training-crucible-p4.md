# training-crucible P4 (References Finalization) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create the training glossary (`references/training-glossary.md`) — canonical terminology for the entire skills library. This is the final P0-P4 phase.

**Architecture:** A single reference file defining all key terms used across knowledge/precision/performance/tickets modules. Ensures consistent language throughout.

---

## File Structure

```
training-crucible/
└── references/
    ├── source-repo-map.md            # (exists from P0)
    └── training-glossary.md          # [CREATE] Canonical terminology
```

---

## Task 1: Write `references/training-glossary.md`

**Files:**
- Create: `training-crucible/references/training-glossary.md`

Write the training glossary in Chinese (with English terms). Structure:

1. **并行计算术语** — TP, PP, DP, CP, EP, FSDP, ZeRO, AllReduce, AllGather, ReduceScatter, All-to-All, etc.
2. **训练阶段术语** — Pre-training, Post-training, SFT, DPO, RLHF, GRPO, PPO, Rollout, On-policy, Off-policy
3. **精度术语** — Loss NaN, Loss Spike, Grad Norm, Mixed Precision, FP16/BF16/FP8/FP4, Loss Scaling, Gradient Clipping
4. **性能术语** — Throughput, MFU, MFU, Activation Recompute, KV Cache, PagedAttention, CUDA Graph, NPU Graph, Speculative Decoding
5. **推理术语** — Quantization (W4A16, W8A8), KV Cache, Continuous Batching, Prefix Caching
6. **硬件术语** — Ascend, Atlas, HCCL, NCCL, NVLink, RoCE, HCCS
7. **每个术语包含**：英文缩写、中文译名、一句话定义、相关术语链接

Quality standards:
- Chinese language (English terms in parentheses)
- 100-150 lines
- Alphabetical or category-grouped
- Each term: `**TERM** (中文) — definition`

Commit: `git commit -m "docs(references): add training glossary"`

---

## Task 2: Final verification and push

Steps:
1. `wc -l references/training-glossary.md` — confirm 100+ lines
2. `git log --oneline -5` — confirm clean history
3. `git push origin main`
4. Report final P0-P4 completion status
