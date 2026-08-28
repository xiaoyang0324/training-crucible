[← Back to Root](../../README.md) | [中文](README_zh.md) | [English](README.md)

# Shared References

> Cross-module reference docs — source repo map, glossary, and answer conventions.

## What It Does

This module is the **shared foundation** of training-crucible. It provides the reference data that all other modules depend on: which source repos cover which stages, what terms mean exactly, and how answers should be structured. It is read by the entry router (`SKILL.md`) and by every sub-module when producing source-grounded answers.

## File Structure

| File | Purpose | Lines |
|------|---------|-------|
| `source-repo-map.md` | 8 source repos → training stage/feature mapping (full code map) | ~480 |
| `training-glossary.md` | 90 canonical terms (7 categories) | ~520 |
| `answer-conventions.md` | Answer conventions + output templates | ~362 |

**Total: ~1362 lines**

## source-repo-map.md

Defines coverage of 8 local source repos — the **code map** for the entire library:

| Repo | Primary Stage | Hardware | Coverage |
|------|--------------|----------|----------|
| PyTorch | Base framework | CUDA GPU | nn/autograd/optim/distributed/FSDP2/DTensor |
| Megatron-LM | Pre-training, Post-training, MoE | NVIDIA GPU | TP/PP/CP/EP/MoE/GRPO |
| torchtitan | Pre-training, RL | NVIDIA GPU | FSDP/Float8/DAPO |
| DeepSpeed | Pre-training optimization | NVIDIA GPU | ZeRO-1/2/3, MoE, PP, Autotuning |
| miles | RL (GRPO/PPO) | NVIDIA GPU | Rollout, Reward Hub, async training |
| slime | RL (GRPO/PPO) | NVIDIA GPU | TIS/OPSM/OPD off-policy correction |
| torchada | Hardware adapter | NVIDIA Ada GPU | CUDA→MUSA shim, Graph Rotation |
| torch_musa | Hardware backend | Moore Threads MUSA GPU | Device/Memory/Stream/Graph/MCCL |

> **Hardware coverage**: Local repos cover NVIDIA GPU and MUSA GPU. Ascend NPU knowledge comes from external docs, marked as secondary.

## training-glossary.md

Canonical terminology ensuring consistent language across the library. 7 categories:

1. Parallel Computing (TP/PP/DP/CP/EP/FSDP/ZeRO)
2. Training Stages (Pre-training/SFT/DPO/RLHF/GRPO/PPO)
3. Precision & Numerics (Loss NaN/FP16/FP8/Loss Scaling)
4. Performance Optimization (MFU/KV Cache/CUDA Graph)
5. Inference (Quantization/Speculative Decoding)
6. Hardware (Ascend/HCCL/NCCL/NVLink)
7. Frameworks (Megatron-LM/torchtitan/miles/slime)

## answer-conventions.md

Defines how every answer in training-crucible should be structured:
- Output templates (concept answer, diagnostic report, optimization plan)
- Citation format (`file:line` standard)
- ASCII diagram conventions
- Cross-reference format

## Cross-References

| Need | Go To |
|------|-------|
| Knowledge Q&A | `skill-knowledge/` |
| Precision diagnosis | `skill-precision/` |
| Performance optimization | `skill-performance/` |
| Historical cases | `skill-tickets/` |
