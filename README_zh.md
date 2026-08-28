[中文](README_zh.md) | [English](README.md)

<h1 align="center">training-crucible</h1>

<p align="center">
  <em>在熔炉中锻造 AI 训练全栈精度。</em><br>
  <em>Forging full-stack AI training expertise under fire.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-v1.1.0-blue" alt="version">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license">
  <img src="https://img.shields.io/badge/modules-4-orange" alt="modules">
  <img src="https://img.shields.io/badge/lines-13312-lightgrey" alt="lines">
</p>

<p align="center">
  <b>预训练</b> · <b>后训练</b> · <b>强化学习</b> · <b>推理优化</b>
</p>

---

## 解决什么问题

大模型训练是一项复杂的系统工程——你在数百张 GPU 上组合并行策略，排查 step 500 时 loss 突然 NaN 的原因，或者搞明白为什么 64 卡集群只能跑出 40% 的 MFU。

这些知识确实存在——散落在 8 个开源仓库、几十万行代码里，以及经历过这些战役的工程师的肌肉记忆中。

**training-crucible 把它们整合起来，嵌入你的 Claude Code 助手，随时待命。**

---

## 它做了什么

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   "step 500 时 loss 出现 NaN，怎么排查？"                         │
│                                                                 │
│                         │                                       │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   SKILL.md 路由     │                            │
│              │  (意图识别)         │                            │
│              └──────────┬──────────┘                            │
│                         │                                       │
│           ┌─────────────┼─────────────┐                         │
│           ▼             ▼             ▼                         │
│    ┌────────────┐ ┌──────────┐ ┌──────────┐                     │
│    │  知识专家   │ │ 精度专家  │ │ 性能专家  │                    │
│    └────────────┘ └──────────┘ └──────────┘                     │
│           │             │             │                         │
│           └─────────────┼─────────────┘                         │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │  源码为凭的回答      │                            │
│              │  + 结构化工作流      │                            │
│              └─────────────────────┘                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

它不是搜索引擎，而是一位随时待命的 AI Infra 专家：
- **回答**"是什么"和"怎么工作"——带代码级引用
- **诊断**精度异常——5 步工作流追溯到 `file:line`
- **优化**性能——将瓶颈匹配到 14 种 SOTA 技术
- **归档**每个已解决的问题——结构化、可检索的案例库

---

## 看看真实输出

**你问：**
> "TP 和 PP 有什么区别？"

**training-crucible 回答：**

```
┌────────────────────────────────────────────────────────────┐
│ 📚 张量并行 (TP) vs 流水线并行 (PP)                        │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  TP: 把一层拆到多个设备上                                   │
│  PP: 把多层拆到多个设备上                                   │
│                                                            │
│  ┌─────────────┐      ┌─────────────┐                      │
│  │ Layer 1-4   │ ───► │ Device 0    │  ← PP               │
│  │ Layer 5-8   │ ───► │ Device 1    │                      │
│  └─────────────┘      └─────────────┘                      │
│                                                            │
│  ┌─────────────┐      ┌──────┬──────┐                      │
│  │ Layer 1     │ ───► │ D0   │ D1   │  ← TP               │
│  │  (split)    │      │ W_a  │ W_b  │                      │
│  └─────────────┘      └──────┴──────┘                      │
│                                                            │
│  源码为凭：                                                 │
│  • Megatron-LM: layers.py:986 ColumnParallelLinear          │
│  • Megatron-LM: schedules.py:2147 1F1B schedule             │
│  • torchtitan: parallel_dims.py:132 ParallelDims           │
│                                                            │
│  核心 trade-off:                                           │
│  • TP → 每层通信量大，适合机内                               │
│  • PP → 每层通信量小，跨机必需                               │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

每个技术主张都引用真实源码。没有幻觉 API，没有模糊描述。

---

## 四大能力

| | 模块 | 能力 | 触发场景 |
|---|------|------|----------|
| 📚 | `skill-knowledge/` | 按训练阶段路由的知识问答（8 文件，9257 行） | "是什么" / "怎么工作" |
| 🔬 | `skill-precision/` | 5 步诊断工作流（捕获→分类→定位→假设→解决） | loss NaN / 梯度爆炸 / 精度回退 |
| ⚡ | `skill-performance/` | 5 步优化工作流（画像→识别→匹配→适配→验证） | 吞吐低 / OOM / 扩展效率差 |
| 📋 | `skill-tickets/` | 结构化案例库（YAML 元数据 + 8 段正文） | "遇到过吗" / "历史案例" |

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

## 源码仓分析

每个技术主张都能追溯到以下源码仓：

| 源码仓 | 阶段 | 硬件 | 覆盖范围 |
|--------|------|------|----------|
| PyTorch | 基础框架 | CUDA GPU | nn/autograd/optim/distributed/FSDP2/DTensor |
| Megatron-LM | 预训练、后训练、MoE | NVIDIA GPU | TP/PP/CP/EP/MoE/GRPO |
| torchtitan | 预训练、RL | NVIDIA GPU | FSDP/Float8/DAPO |
| DeepSpeed | 预训练优化 | NVIDIA GPU | ZeRO-1/2/3、MoE、PP、Autotuning |
| miles | RL (GRPO/PPO) | NVIDIA GPU | Rollout、Reward Hub、异步训练 |
| slime | RL (GRPO/PPO) | NVIDIA GPU | TIS/OPSM/OPD off-policy 校正 |
| torchada | 硬件适配 | NVIDIA Ada GPU | CUDA→MUSA shim、Graph Rotation |
| torch_musa | 硬件后端 | Moore Threads MUSA GPU | Device/Memory/Stream/Graph/MCCL |

---

## 内容结构

```
training-crucible/
├── SKILL.md                              # 入口路由 — Iron Law + 意图识别
├── skill-knowledge/                      # 知识专家（9257 行）
│   ├── pretraining.md                    #   并行策略、内存优化
│   ├── post-training.md                  #   SFT · DPO · RLHF
│   ├── rl.md                             #   GRPO · PPO · rollout
│   ├── inference.md                      #   KV Cache · 量化 · 投机解码
│   ├── hardware-adapter.md               #   硬件适配层（torchada + torch_musa）
│   ├── moe.md                            #   MoE 跨仓库深度分析
│   ├── deepspeed.md                      #   DeepSpeed 深度分析
│   └── pytorch.md                        #   PyTorch 核心特性分析
├── skill-precision/                      # 精度专家（887 行）
│   ├── SKILL.md                          #   捕获→分类→定位→假设→解决
│   └── references/                       #   故障分类 · 已知模式
├── skill-performance/                    # 性能专家（1225 行）
│   ├── SKILL.md                          #   画像→识别→匹配→适配→验证
│   └── references/                       #   瓶颈分类 · SOTA 技术
├── skill-tickets/                        # 问题归档（583 行）
│   ├── SKILL.md                          #   检索 & 归档工作流
│   ├── TEMPLATE.md                       #   YAML frontmatter + 8 段正文
│   └── *.md                              #   来自真实项目的种子案例
└── skill-references/                     # 共享参考（1362 行）
    ├── source-repo-map.md                #   8 仓 → 阶段映射
    ├── training-glossary.md              #   90 个规范术语
    └── answer-conventions.md             #   回答规范 + 输出模板
```

---

## 项目统计

| 模块 | 文件数 | 行数 | 核心能力 |
|------|--------|------|----------|
| 知识专家 | 8 | ~9257 | 8 文件代码级知识问答 |
| 精度专家 | 3 | ~887 | 5 步诊断 + 故障分类 |
| 性能专家 | 3 | ~1225 | 5 步优化 + SOTA 技术 |
| 问题归档 | 7 | ~583 | 模板 + 5 个种子 ticket |
| 共享参考 | 3 | ~1362 | 源码映射 + 术语表 + 回答规范 |
| **合计** | **24** | **~13312** | **4 大能力、8 个源码仓** |

---

## 核心优势

| | | |
|---|---|---|
| **源码为凭** | 每个技术主张优先引用本地源码仓（文件路径 + 行号）作为主要证据 | 拒绝幻觉 |
| **代码级** | 完整调用链（带 file:line）、ASCII 架构图、真实代码片段 | 不止于概念描述 |
| **跨仓库** | 8 个源码仓横向对比 | 看每个框架如何解决同一问题 |
| **工作流驱动** | 5 步诊断 + 5 步优化工作流，不止于问答 | 结构化解决问题 |

---

## 快速开始

```bash
# 1. 克隆到工作区
git clone git@github.com:xiaoyang0324/training-crucible.git

# 2. Claude Code 自动加载 SKILL.md——直接提问：
#    "TP 和 PP 有什么区别？"
#    "step 500 时 loss 出现 NaN，怎么排查？"
#    "7B 模型在 64 卡上如何优化吞吐？"
```

或软链到 `~/.claude/skills/training-crucible/`。

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

## License

MIT
