[中文](README_zh.md) | [English](README.md)

<h1 align="center">training-crucible</h1>

<p align="center">
  <em>在熔炉中锻造 AI 训练全栈精度。</em><br>
  <em>Forging full-stack AI training expertise under fire.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v1.1.0-blue" alt="version">
</p>

<p align="center">
  <b>预训练</b> · <b>后训练</b> · <b>强化学习</b> · <b>推理优化</b>
</p>

---

## 解决什么问题

大模型训练是一项复杂的系统工程——并行策略的组合、精度问题的排查、性能瓶颈的优化，每一项都需要深厚的领域知识和实战经验。

**training-crucible 将这些知识结构化，嵌入你的 Claude Code 助手。** 它不是搜索引擎，而是一位随时待命的 AI Infra 专家：你描述问题，它按工作流诊断、引用真实源码、给出可验证的方案。

---

## 四大能力

| | 模块 | 能力 | 触发场景 |
|---|------|------|----------|
| 📚 | `skill-knowledge/` | 按训练阶段路由的知识问答 (5 文件) | "是什么" / "怎么工作" |
| 🔬 | `skill-precision/` | 5 步诊断工作流 | loss NaN / 梯度爆炸 / 精度回退 |
| ⚡ | `skill-performance/` | 5 步优化工作流 | 吞吐低 / OOM / 扩展效率差 |
| 📋 | `skill-tickets/` | 结构化案例库检索 | "遇到过吗" / "历史案例" |

## 范围

本库覆盖：
- ✅ 训练并行策略（TP/PP/DP/CP/EP/FSDP/ZeRO）
- ✅ 精度问题诊断（loss NaN/spike/divergence、梯度爆炸等）
- ✅ 性能优化（吞吐、内存、扩展效率）
- ✅ RL 训练（GRPO/PPO、rollout、训推一体）
- ✅ 推理优化（KV Cache、量化、投机解码）

本库**不覆盖**：
- ❌ 模型架构设计（MoE 架构选择、层数/维度调优）
- ❌ 数据工程（数据管道 bug、数据质量评估）
- ❌ 硬件级调试（NPU/GPU kernel bug、驱动问题）
- ❌ 部署基础设施（Kubernetes、网络配置、存储）
- ❌ 多模态训练细节（视觉编码器、跨模态融合）

---

## 架构

```
training-crucible/
├── SKILL.md                        # 入口路由 — Iron Law + 意图识别
├── skill-knowledge/                      # 知识专家
│   ├── pretraining.md              #   并行策略、内存优化
│   ├── post-training.md             #   SFT · DPO · RLHF
│   ├── rl.md                       #   GRPO · PPO · rollout
│   ├── inference.md                #   KV Cache · 量化 · 投机解码
│   ├── hardware-adapter.md         #   硬件适配层 (torchada + torch_musa)
│   ├── moe.md                      #   MoE 跨仓库深度分析
│   ├── deepspeed.md                #   DeepSpeed 深度分析
│   └── pytorch.md                  #   PyTorch 核心特性分析
├── skill-precision/                      # 精度专家
│   ├── SKILL.md                    #   捕获→分类→定位→假设→解决
│   └── references/                 #   故障分类 · 已知模式
├── skill-performance/                    # 性能专家
│   ├── SKILL.md                    #   画像→识别→匹配→适配→验证
│   └── references/                 #   瓶颈分类 · SOTA 技术
├── skill-tickets/                        # 问题归档
│   ├── SKILL.md                    #   检索 & 归档工作流
│   ├── TEMPLATE.md                 #   YAML frontmatter + 8 段正文
│   └── *.md                        #   来自真实项目的种子案例
└── skill-references/                     # 共享参考
    ├── source-repo-map.md          #   8 仓 → 阶段映射
    ├── training-glossary.md        #   90 个规范术语
    └── answer-conventions.md       #   回答规范 + 输出模板
```

---

## 路由逻辑

```
"你的问题"
      │
      ├─ 概念原理 ──────────────► skill-knowledge/ (按阶段子路由)
      ├─ 精度异常 ──────────────► skill-precision/ (5 步诊断 → 归档)
      ├─ 性能瓶颈 ──────────────► skill-performance/ (5 步优化 → 归档)
      ├─ 历史案例 ──────────────► skill-tickets/ (按标签检索)
      └─ 复杂问题 ──────────────► 精度 → 性能 → 归档 (串联)
```

---

## 源码为凭，拒绝幻觉

所有技术主张优先引用 **本地源码仓**（文件路径 + 行号）作为主要证据。外部论文/文档仅作补充，不可替代源码引用。

| 源码仓 | 阶段 | 硬件 |
|--------|------|------|
| PyTorch | 基础框架 (nn/autograd/optim/distributed/FSDP2/DTensor) | CUDA GPU |
| Megatron-LM | 预训练、后训练、MoE | NVIDIA GPU |
| torchtitan | 预训练、RL | NVIDIA GPU |
| DeepSpeed | 预训练优化 (ZeRO/MoE/PP) | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | 预训练、硬件适配 | NVIDIA Ada GPU |
| torch_musa | 预训练、硬件后端 | Moore Threads MUSA GPU |

---

## 内容概览

| 模块 | 文件数 | 行数 | 核心能力 |
|------|--------|------|----------|
| 知识专家 | 8 | ~9074 | 8 文件知识问答 (预训练 + RL + 后训练 + 推理 + 硬件适配 + MoE + DeepSpeed + PyTorch) |
| 精度专家 | 3 | ~887 | 5 步诊断 + 故障分类 + 已知模式 |
| 性能专家 | 3 | ~1225 | 5 步优化 + 瓶颈分类 + SOTA 技术 |
| 问题归档 | 7 | ~681 | 模板 + SKILL.md + 5 个种子 ticket |
| 共享参考 | 3 | ~1362 | 源码映射 + 术语表 + 回答规范 |

---

## 路线图

```
v1.0 (当前)     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
  骨架 · 知识 · 专家 · 问题 · 参考资料
  结构加固 · 命名规范 · 贡献指南

v1.1 (下一步)   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  引用审计 · 更多种子 ticket · 覆盖盲区

v2.0 (未来)     ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
  新硬件仓 (Ascend/MUSA) · 团队化工作流
```

---

## 使用方式

1. 将本仓库放在工作区，Claude Code 自动加载 `SKILL.md`
2. 或软链到 `~/.claude/skills/training-crucible/`
3. 问任何训练相关问题，Claude 自动路由到对应专家模块

---

## License

MIT
