<!-- Copy this file, fill in the frontmatter and body, rename to YYYY-MM-DD-<short-slug>.md -->

---
id: TICKET-YYYYMMDD-NNN          # NNN = per-day sequence (001, 002, ...)
title: One-line summary of the problem
type: precision | performance | hybrid
stage: pretraining | posttraining | rl | inference | cross-stage
status: resolved | investigating | wontfix
severity: critical | major | minor
hardware:
  - NVIDIA-Ada                   # or MUSA, Ascend, CPU, agnostic
frameworks:
  - Megatron-LM                  # or torchtitan, miles, slime, torchada, torch_musa
tags:
  - loss-nan                     # examples: grad-explosion, throughput, oom, scaling-efficiency, train-infer-mismatch, accuracy-regression
created: YYYY-MM-DD
resolved: YYYY-MM-DD             # leave blank if status != resolved
related_tickets:
  - TICKET-YYYYMMDD-NNN
source_refs:                      # local repo files that informed the solution
  - Megatron-LM/megatron/core/parallel/tensor_parallel/lines.py:123
external_refs:                    # papers, docs, external links
  - https://arxiv.org/abs/...
---

## Symptom

What was observed. Include:
- Training curve snapshot (loss, grad norm, accuracy)
- Error message / traceback (if any)
- When it started (step number, after what change)

## Environment

| Item | Value |
|------|-------|
| Model | e.g., Llama-3 70B |
| Scale | e.g., 1024 GPUs, 64 nodes |
| Hardware | e.g., NVIDIA A100 80GB / Atlas 800 A3 |
| Framework | e.g., Megatron-LM commit abc123, PyTorch 2.4 |
| Parallel config | e.g., TP=8, PP=8, DP=16, CP=1 |
| Batch size | micro-batch × gradient accumulation |

## Analysis

Step-by-step root cause analysis. Cite source code with file paths.

1. **Initial observation:** ...
2. **Narrowed to:** ...
3. **Verified by:** ...
4. **Root cause confirmed:** ...

## Root Cause

One paragraph: what was actually wrong, at the code/algorithm level.

## Resolution

What fixed it. Be specific:
- Config changes (before → after)
- Code patches (file:line, diff format)
- Workflow adjustments

## Verification

How we confirmed the fix worked:
- Metrics before/after
- Regression tests run
- Duration of stable training after fix

## Lessons

Reusable insight. Cross-references:
- See `references/source-repo-map.md` for related repos
- See `precision/references/` for precision patterns (coming in P2)
- See `performance/references/` for perf techniques (coming in P2)

## References

- Papers: ...
- Docs: ...
- Related tickets: ...
