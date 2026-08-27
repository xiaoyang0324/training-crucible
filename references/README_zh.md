[中文](README_zh.md) | [English](README.md)

# 参考资料 (Shared References)

> 跨模块共享的参考文档——源码映射和术语表。

## 文件结构

| 文件 | 用途 |
|------|------|
| `source-repo-map.md` | 6 个源码仓 → 训练阶段/特性映射 |
| `training-glossary.md` | 90 个规范术语（7 大分类） |

## source-repo-map.md

定义 6 个本地源码仓的覆盖范围：

| 源码仓 | 主要阶段 | 硬件 |
|--------|----------|------|
| Megatron-LM | 预训练、后训练 | NVIDIA GPU |
| torchtitan | 预训练、RL | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | 预训练 | NVIDIA Ada GPU |
| torch_musa | 预训练 | Moore Threads MUSA |

> **硬件覆盖说明**: 本地仓覆盖 NVIDIA GPU 和 MUSA GPU。Ascend NPU 知识来自外部文档，标注为 secondary。

## training-glossary.md

规范术语表，确保全仓语言一致。覆盖 7 大分类：

1. 并行计算 (TP/PP/DP/CP/EP/FSDP/ZeRO)
2. 训练阶段 (Pre-training/SFT/DPO/RLHF/GRPO/PPO)
3. 精度与数值 (Loss NaN/FP16/FP8/Loss Scaling)
4. 性能优化 (MFU/KV Cache/CUDA Graph)
5. 推理 (Quantization/Speculative Decoding)
6. 硬件 (Ascend/HCCL/NCCL/NVLink)
7. 框架 (Megatron-LM/torchtitan/miles/slime)
