# Source Repo → Stage/Feature Map

> This file maps the 6 local reference repos to the training stages and features they cover.
> When answering a question, the skill cites the **local source repo** (file path + line number)
> as primary evidence. External papers supplement, never replace, source evidence.

## Hardware Coverage Note

- **NVIDIA GPU** repos: Megatron-LM, torchtitan, miles, slime, torchada
- **Moore Threads MUSA GPU** repo: torcht_musa
- **Ascend NPU**: NOT covered by local repos. Ascend knowledge comes from external docs/papers
  (MindSpeed, torch_npu) and is marked as "external" — secondary to local source evidence.

---

## Repo 1: Megatron-LM

**Path:** `C:\y30062407\workspace\local\面试\train\Megatron-LM`

**Primary stages:** Pre-training, Post-training
**Secondary:** RL (training backbone for miles/slime)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `megatron/core/parallel/` | TP, PP, DP, EP, CP parallelism |
| `megatron/core/transformer/` | Transformer layer, attention, MLP |
| `megatron/core/pipeline_parallel/` | 1F1B, interleaved schedules |
| `megatron/core/distributed/` | Distributed optimizer, grad sync |
| `megatron/core/transformer/dot_product_attention.py` | Core attention implementation |
| `megatron/legacy/` | Legacy model implementations |

**When to cite:** Pre-training architecture, parallelism strategies, optimizer design, gradient pipeline.

---

## Repo 2: torchtitan

**Path:** `C:\y30062407\workspace\local\面试\train\torchtitan`

**Primary stage:** Pre-training
**Secondary:** RL (TitanRL experiment)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torchtitan/distributed/` | FSDP2, TP, PP, CP native PyTorch |
| `torchtitan/models/` | Llama 3, other model definitions |
| `torchtitan/experiments/rl/` | TitanRL — RL training stack |
| `torchtitan/ops/` | Custom kernels (flexattention, etc.) |
| `torchtitan/components/` | Loss, dataloader, metrics |

**When to cite:** PyTorch-native training patterns, FSDP2 usage, TitanRL RL approach.

---

## Repo 3: miles

**Path:** `C:\y30062407\workspace\local\面试\train\miles`

**Primary stage:** RL (GRPO/PPO)
**Secondary:** Inference (SGLang rollout integration)

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `miles/rollout/` | Rollout generation, SGLang integration |
| `miles/backends/` | Megatron + FSDP2 backend adapters |
| `miles/true_on_policy/` | True on-policy training, numerics |
| `miles/router/` | Request routing for rollout |
| `miles/ray/` | Ray-based orchestration |
| `miles/dashboard/` | Observability, metrics |

**When to cite:** RL post-training loop, rollout strategies, on-policy numerics, weight sync.

---

## Repo 4: slime

**Path:** `C:\y30062407\workspace\local\面试\train\slime`

**Primary stage:** RL (GRPO/PPO)
**Secondary:** Inference (SGLang rollout), agentic RL

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `slime/rollout/` | Rollout generation, data buffer |
| `slime/backends/` | Megatron backend, SGLang pass-through |
| `slime/agent/` | Agent-based RL, multi-agent |
| `slime/ray/` | Ray orchestration |
| `slime/observability/` | Tracing, profiling, reproducibility |

**When to cite:** RL data generation workflows, agentic RL, train-infer integration.

---

## Repo 5: torchada

**Path:** `C:\y30062407\workspace\local\面试\train\torchada`

**Primary stage:** Pre-training (NVIDIA Ada / RTX consumer GPU)
**Secondary:** —

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torchada/` | PyTorch backend for Ada/RTX GPUs |
| `benchmarks/` | Performance benchmarks |

**When to cite:** Consumer-GPU training, Ada-specific optimizations, hardware-adaptation layer.

---

## Repo 6: torch_musa

**Path:** `C:\y30062407\workspace\local\面试\train\torch_musa`

**Primary stage:** Pre-training (Moore Threads MUSA GPU)
**Secondary:** —

**Key directories:**
| Directory | What it covers |
|-----------|---------------|
| `torch_musa/` | PyTorch backend for MUSA GPU |
| `benchmark/` | Performance benchmarks |

**When to cite:** MUSA GPU training, Moore Threads hardware adaptation, cross-platform porting.

---

## Quick Routing Index

| User asks about... | Primary repo(s) | Secondary repo(s) |
|--------------------|-----------------|-------------------|
| Pre-training at scale | Megatron-LM, torchtitan | torchada, torcht_musa |
| SFT / DPO / RLHF | Megatron-LM | torchtitan |
| GRPO / PPO RL | miles, slime | torchtitan (TitanRL) |
| Inference / rollout | miles, slime | — |
| FSDP / parallelism | torchtitan, Megatron-LM | — |
| MoE training | Megatron-LM | miles, slime |
| Consumer GPU (Ada) | torchada | — |
| MUSA GPU | torcht_musa | — |
