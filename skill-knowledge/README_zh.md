[中文](README_zh.md) | [English](README.md)

# 知识专家 (Knowledge Expert)

> 回答"是什么"和"怎么工作"——按训练阶段路由的知识问答模块。

## 覆盖阶段

| 文件 | 阶段 | 核心内容 |
|------|------|----------|
| `pretraining.md` | 预训练 | 并行策略 (TP/PP/DP/CP/EP)、内存优化、配置参数 |
| `post-training.md` | 后训练 | SFT / DPO / RLHF 原理与配置 |
| `rl.md` | 强化学习 | GRPO / PPO、rollout 生成、训推一体 |
| `inference.md` | 推理优化 | KV Cache、量化、投机解码、推理服务 |

## 使用方式

当用户问"是什么"或"怎么工作"类问题时，SKILL.md 路由到对应阶段的知识文件。
回答包含：核心概念、源码引用、架构图、配置参数表、常见误区。

## 内容规范

- 中文撰写，英文术语在括号中标注
- 每个技术主张必须引用本地源码仓（文件路径）
- 包含 ASCII 架构图、配置参数表、常见误区
- 文末附源码文件索引表

## 源码参考

详见 `skill-references/source-repo-map.md`。
