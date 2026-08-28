[← 返回根目录](../../README.md) | [中文](README_zh.md) | [English](README.md)

# 参考资料 (Shared References)

> 跨模块共享的参考文档——源码映射、术语表、回答规范。

## 解决什么问题

本模块是 training-crucible 的**共享基础**。提供所有其他模块依赖的参考数据：哪些源码仓覆盖哪些阶段、术语的精确含义、回答的结构规范。被入口路由（`SKILL.md`）和每个子模块在生成源码引用回答时读取。

## 文件结构

| 文件 | 用途 | 行数 |
|------|------|------|
| `source-repo-map.md` | 8 个源码仓 → 训练阶段/特性映射（全仓库代码地图） | ~480 |
| `training-glossary.md` | 90 个规范术语（7 大分类） | ~520 |
| `answer-conventions.md` | 回答规范 + 输出模板 | ~362 |

**合计：~1362 行**

## source-repo-map.md

定义 8 个本地源码仓的覆盖范围——全库的**代码地图**：

| 源码仓 | 主要阶段 | 硬件 | 覆盖范围 |
|--------|----------|------|----------|
| PyTorch | 基础框架 | CUDA GPU | nn/autograd/optim/distributed/FSDP2/DTensor |
| Megatron-LM | 预训练、后训练、MoE | NVIDIA GPU | TP/PP/CP/EP/MoE/GRPO |
| torchtitan | 预训练、RL | NVIDIA GPU | FSDP/Float8/DAPO |
| DeepSpeed | 预训练优化 | NVIDIA GPU | ZeRO-1/2/3、MoE、PP、Autotuning |
| miles | RL (GRPO/PPO) | NVIDIA GPU | Rollout、Reward Hub、异步训练 |
| slime | RL (GRPO/PPO) | NVIDIA GPU | TIS/OPSM/OPD off-policy 校正 |
| torchada | 硬件适配 | NVIDIA Ada GPU | CUDA→MUSA shim、Graph Rotation |
| torch_musa | 硬件后端 | Moore Threads MUSA GPU | Device/Memory/Stream/Graph/MCCL |

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

## answer-conventions.md

定义 training-crucible 中每个回答的结构规范：
- 输出模板（概念回答、诊断报告、优化方案）
- 引用格式（`file:line` 标准）
- ASCII 图规范
- 交叉引用格式

## 交叉引用

| 需求 | 跳转 |
|------|------|
| 知识问答 | `skill-knowledge/` |
| 精度诊断 | `skill-precision/` |
| 性能优化 | `skill-performance/` |
| 历史案例 | `skill-tickets/` |
