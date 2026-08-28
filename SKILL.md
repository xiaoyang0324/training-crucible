---
name: training-crucible
description: >
  AI 训练全栈专家技能——覆盖预训练、后训练（SFT/DPO/RLHF）、强化学习（GRPO/PPO）、
  推理优化四大阶段。提供知识问答、精度诊断、性能优化、问题归档四项核心能力。
  触发条件：用户问及 AI 训练、大模型训练、分布式训练、训练精度、训练性能、
  loss 异常、梯度异常、吞吐优化、Megatron、torchtitan、miles、slime、
  预训练、后训练、对齐、强化学习、GRPO、PPO、推理优化、量化、KV Cache 等
  任何训练全栈相关话题。
  源码参考仓：Megatron-LM, torchtitan, miles, slime, torchada, torch_musa。
---

# training-crucible — AI 训练全栈专家

## The Iron Law

```
回答任何训练问题前，必须先确认两样东西：
  1. 训练阶段（预训练 / 后训练 / RL / 推理）
  2. 涉及的框架仓（Megatron-LM / torchtitan / miles / slime / 其他）

分析精度/性能问题时，还必须拿到：
  3. 症状描述（loss 曲线 / 报错 / 性能指标）+ 环境信息（模型规模、并行配置、硬件）

没有这些不开始分析。绝不凭空猜测根因。
```

```
所有代码引用和实现必须来自真实源码。
禁止虚构函数名、API、文件路径、行号、类名、变量名。
引用代码前必须先 grep/read 确认该代码真实存在。
不确定的 API 用 "需要确认" 标注，不编造。
优先引用本地源码仓（Megatron-LM, torchtitan, miles, slime, torchada, torch_musa），
外部论文/文档作为补充，且必须标注为"外部知识"。
```

## 核心能力

| 能力 | 模块 | 说明 |
|------|------|------|
| **知识问答** | `skill-knowledge/` | 按训练阶段路由，解释概念、架构、配置 |
| **硬件适配** | `skill-knowledge/` | 按训练阶段 + 硬件适配层路由，覆盖 Megatron/torchtitan/miles/slime/torchada/torch_musa 六仓代码级分析 |
| **MoE 深度分析** | `skill-knowledge/moe.md` | MoE 跨仓库代码级分析（Router/Load Balancing/Token Dispatch/EP） |
| **DeepSpeed 深度分析** | `skill-knowledge/deepspeed.md` | ZeRO-1/2/3、MoE、Pipeline、Autotuning 代码级分析 |
| **精度诊断** | `skill-precision/` | 5 步工作流诊断 loss/梯度/收敛问题 |
| **性能优化** | `skill-performance/` | 5 步工作流优化吞吐/内存/扩展效率 |
| **问题归档** | `skill-tickets/` | 结构化案例库，按症状/阶段/框架检索 |

---

## 意图路由表

> 根据用户问题的关键词，路由到对应模块。多个模块可串联（精度→性能→归档）。

> **路由优先级：意图优先于框架。** 框架关键词（Megatron/torchtitan/miles 等）不单独路由，
> 仅用于在确定意图后定位源码仓。先按意图确定模块，再在模块内用框架关键词定位具体代码。

### 路由规则

```
用户问题
    │
    ├─ "什么是 X" / "X 怎么工作" / "X 和 Y 区别" ──────────────► 知识问答
    │     │
    │     ├─ 含 pretrain / 预训练 / pre-training ────────► skill-knowledge/pretraining.md
    │     ├─ 含 SFT / DPO / RLHF / alignment / 后训练 ──► skill-knowledge/post-training.md
    │     ├─ 含 GRPO / PPO / RL / 强化学习 ──────────────► skill-knowledge/rl.md
    │     ├─ 含 quant / KV cache / speculative / 推理 ───► skill-knowledge/inference.md
    │     ├─ 含 torchada / torch_musa / MUSA / 硬件适配 ──► skill-knowledge/hardware-adapter.md
    │     ├─ 含 MoE / mixture of expert / router / expert ► skill-knowledge/moe.md
    │     └─ 含 DeepSpeed / ZeRO / offload / CPU Adam ────► skill-knowledge/deepspeed.md
    │
    ├─ loss NaN / loss spike / 梯度爆炸 / 精度异常 / ──────► 精度诊断
    │     train-infer mismatch / 不收敛 / 发散                    │
    │     └─ skill-precision/SKILL.md + skill-precision/references/
    │
    ├─ 慢 / OOM / 吞吐低 / 扩展效率 / 内存瓶颈 / ──────────► 性能优化
    │     MFU 低 / 通信瓶颈 / 算子慢                            │
    │     └─ skill-performance/SKILL.md + skill-performance/references/
    │
    ├─ "遇到过吗" / "历史案例" / "之前的问题" ──────────────► 问题归档
    │     └─ skill-tickets/ — 按 type/stage/tags 检索
    │
    └─ 复杂问题（精度+性能+归档） ──────────────────────────► 串联：
         精度诊断 → 性能分析 → 归档到 skill-tickets/
```

### 关键词触发表

| 关键词 | 路由目标 |
|--------|----------|
| 预训练, pretrain, pre-training, 预训练阶段 | `skill-knowledge/pretraining.md` |
| 后训练, SFT, DPO, RLHF, alignment, 对齐 | `skill-knowledge/post-training.md` |
| 强化学习, GRPO, PPO, RL, reinforcement | `skill-knowledge/rl.md` |
| 推理, quant, KV cache, speculative, serving, 量化 | `skill-knowledge/inference.md` |
| 精度, loss, 梯度, grad, NaN, spike, divergence, 不收敛 | `skill-precision/` |
| 性能, 吞吐, throughput, MFU, 内存, memory, OOM, 慢 | `skill-performance/` |
| 案例, 问题单, ticket, 归档, 历史 | `skill-tickets/` |
| 训练不稳定, 模型训不动, loss 震荡, 不收敛 | `skill-precision/` |
| 显存不够用, 内存不够, 爆显存, 卡OOM | `skill-performance/` |
| 训练太慢, 速度上不去, GPU闲置 | `skill-performance/` |
| torchada, torch_musa, MUSA, 摩尔线程, Moore Threads, 硬件适配, 兼容层 | `skill-knowledge/hardware-adapter.md` |
| graph rotation, patch engine, CUDA Graph 轮换 | `skill-knowledge/hardware-adapter.md` |
| MoE, mixture of experts, router, TopK, expert parallel, token dispatch, load balancing | `skill-knowledge/moe.md` |
| DeepSpeed, ZeRO, ZeRO-1/2/3, offload, CPU Adam, DeepSpeedEngine, autotuning | `skill-knowledge/deepspeed.md` |

> **注：** 框架关键词（Megatron/torchtitan/miles/slime/torchada/torch_musa）不单独路由。
> 先按意图确定模块，再在模块内用框架关键词定位具体代码。详见 `skill-references/source-repo-map.md`。

> **兜底规则：** 如果用户问题不含上述关键词，根据问题语义判断意图。无法判断时主动询问用户。

---

## 源码仓路由 & 回答规范

> 详见：
> - `skill-references/source-repo-map.md` — 源码仓 → 阶段映射
> - `skill-references/answer-conventions.md` — 回答规范与输出格式模板
> - `skill-references/training-glossary.md` — 规范术语表

---

## 子模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| 知识专家 | `skill-knowledge/` | 按训练阶段分 5 个文件 |
| 硬件适配 | `skill-knowledge/hardware-adapter.md` | torchada CUDA→MUSA 兼容层 + torch_musa MUSA 后端，含 patch 引擎、graph rotation、算子调度 |
| MoE 深度分析 | `skill-knowledge/moe.md` | MoE 跨仓库代码级分析：Router/Load Balancing/Token Dispatch/EP，覆盖 Megatron/DeepSpeed/torchtitan/torchada |
| DeepSpeed 深度分析 | `skill-knowledge/deepspeed.md` | ZeRO-1/2/3、MoE、Pipeline、Autotuning、Inference 代码级分析 |
| 精度专家 | `skill-precision/` | 5 步诊断工作流 + 精度故障分类 |
| 性能专家 | `skill-performance/` | 5 步优化工作流 + SOTA 技术目录 |
| 问题归档 | `skill-tickets/` | 检索 & 归档工作流 |
| 源码映射 | `skill-references/source-repo-map.md` | 外部仓 → 阶段映射 |
| 术语表 | `skill-references/training-glossary.md` | 规范术语 |
| 回答规范 | `skill-references/answer-conventions.md` | 输出格式模板 + 引用规则 |
