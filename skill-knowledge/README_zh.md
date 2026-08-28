[← 返回根目录](../../README.md) | [中文](README_zh.md) | [English](README.md)

# 知识专家 (Knowledge Expert)

> 回答"是什么"和"怎么工作"——按训练阶段路由的知识问答模块。

## 解决什么问题

本模块是 training-crucible 的**概念与机制层**。当用户提出知识性问题（"TP 是什么？"、"GRPO 怎么工作？"），入口路由（`SKILL.md`）分发到对应阶段的知识文件。每个文件交付：核心概念、源码引用（`file:line`）、ASCII 架构图、配置参数表、常见误区。

## 覆盖阶段

| 文件 | 阶段 | 核心内容 | 行数 |
|------|------|----------|------|
| `pretraining.md` | 预训练 | 并行策略 (TP/PP/DP/CP/EP)、内存优化、配置参数 | ~1590 |
| `post-training.md` | 后训练 | SFT / DPO / RLHF 原理与配置 | ~715 |
| `rl.md` | 强化学习 | GRPO / PPO、rollout 生成、训推一体 | ~1421 |
| `inference.md` | 推理优化 | KV Cache、量化、投机解码、推理服务 | ~961 |
| `hardware-adapter.md` | 硬件适配层 | torchada (CUDA→MUSA shim) + torch_musa (MUSA 后端) | ~1098 |
| `moe.md` | MoE 深度分析 | Router、负载均衡、token dispatch——跨仓库对比 | ~1080 |
| `deepspeed.md` | DeepSpeed 深度分析 | ZeRO-1/2/3、MoE、Pipeline、Autotuning | ~1227 |
| `pytorch.md` | PyTorch 内核 | nn.Module、autograd、FSDP2、DTensor、compile | ~1165 |

**合计：~9257 行**

## 路由逻辑

```
"X 是什么？" / "Y 怎么工作？"
      │
      ├─ 预训练相关 ──────────────► pretraining.md
      ├─ 后训练相关 ──────────────► post-training.md
      ├─ RL 相关 ─────────────────► rl.md
      ├─ 推理相关 ────────────────► inference.md
      ├─ 硬件适配相关 ─────────────► hardware-adapter.md
      ├─ MoE 专题 ────────────────► moe.md
      ├─ DeepSpeed 专题 ───────────► deepspeed.md
      ├─ PyTorch 内核 ─────────────► pytorch.md
      └─ 跨领域问题 ───────────────► 多文件串联
```

## 内容规范

- **中文撰写**，英文术语在括号中标注
- **源码为凭**：每个技术主张必须引用本地源码仓（`file:line`）
- **统一结构**（每个文件）：
  - §0 全景图（ASCII 架构图）
  - §1–N 核心模块（概念 + 调用链 + 跨仓库对比）
  - 配置参数表（exact arg names）
  - 调用链总图
  - 附录 A：源码文件索引
  - 附录 B：工作实战要点速查
  - 附录 C：常见坑与解决方案

## 交叉引用

| 需求 | 跳转 |
|------|------|
| Loss NaN / 梯度爆炸 | `skill-precision/` |
| 吞吐低 / OOM | `skill-performance/` |
| 历史案例 | `skill-tickets/` |
| 源码仓映射 | `skill-references/source-repo-map.md` |
| 术语定义 | `skill-references/training-glossary.md` |
