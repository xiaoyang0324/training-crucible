[中文](README_zh.md) | [English](README.md)

# Shared References

> Cross-module reference docs — source repo map and glossary.

## File Structure

| File | Purpose |
|------|---------|
| `source-repo-map.md` | 6 source repos → training stage/feature mapping |
| `training-glossary.md` | 90 canonical terms (7 categories) |

## source-repo-map.md

Defines coverage of 6 local source repos:

| Repo | Primary Stage | Hardware |
|------|--------------|----------|
| Megatron-LM | Pre-training, Post-training | NVIDIA GPU |
| torchtitan | Pre-training, RL | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | Pre-training | NVIDIA Ada GPU |
| torch_musa | Pre-training | Moore Threads MUSA GPU |

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
