[中文](README_zh.md) | [English](README.md)

<h1 align="center">training-crucible</h1>

<p align="center">
  <em>Forging full-stack AI training expertise under fire.</em><br>
  <em>在熔炉中锻造 AI 训练全栈精度。</em>
</p>

<p align  ="center">
  <b>预训练</b> · <b>后训练</b> · <b>强化学习</b> · <b>推理优化</b>
</p>

---

## Why this exists

大模型训练是一项复杂的系统工程——并行策略的组合、精度问题的排查、性能瓶颈的优化，每一项都需要深厚的领域知识和实战经验。

**training-crucible 将这些知识结构化，嵌入你的 Claude Code 助手。** 它不是搜索引擎，而是一位随时待命的 AI Infra 专家：你描述问题，它按工作流诊断、引用真实源码、给出可验证的方案。

---

## Four capabilities

| | 模块 | 能力 | 触发 |
|---|------|------|------|
| 📚 | `knowledge/` | 按训练阶段路由的知识问答 | "是什么" / "怎么工作" |
| 🔬 | `precision/` | 5 步诊断工作流 | loss NaN / 梯度爆炸 / 精度回退 |
| ⚡ | `performance/` | 5 步优化工作流 | 吞吐低 / OOM / 扩展效率差 |
| 📋 | `tickets/` | 结构化案例库检索 | "遇到过吗" / "历史案例" |

---

## Architecture

```
training-crucible/
├── SKILL.md                        # 入口路由 — Iron Law + 意图识别
├── knowledge/                      # 知识专家
│   ├── pretraining.md              #   并行策略、内存优化
│   ├── posttraining.md             #   SFT · DPO · RLHF
│   ├── rl.md                       #   GRPO · PPO · rollout
│   └── inference.md                #   KV Cache · 量化 · 投机解码
├── precision/                      # 精度专家
│   ├── SKILL.md                    #   捕获→分类→定位→假设→解决
│   └── references/                 #   故障分类 · 已知模式
├── performance/                    # 性能专家
│   ├── SKILL.md                    #   画像→识别→匹配→适配→验证
│   └── references/                 #   瓶颈分类 · SOTA 技术
├── tickets/                        # 问题归档
│   ├── TEMPLATE.md                 #   YAML frontmatter + 8 段正文
│   └── *.md                        #   来自真实项目的种子案例
└── references/                     # 共享参考
    ├── source-repo-map.md          #   6 仓 → 阶段映射
    └── training-glossary.md        #   90 个规范术语
```

---

## How routing works

```
"Your question"
      │
      ├─ 概念原理 ──────────────► knowledge/ (按阶段子路由)
      ├─ 精度异常 ──────────────► precision/ (5 步诊断 → 归档)
      ├─ 性能瓶颈 ──────────────► performance/ (5 步优化 → 归档)
      ├─ 历史案例 ──────────────► tickets/ (按标签检索)
      └─ 复杂问题 ──────────────► 精度 → 性能 → 归档 (串联)
```

---

## Source-grounded. Not hallucinated.

Every technical claim cites a **local source repo** (file path + line number) as primary evidence. External papers supplement — never replace.

| Repo | Stage | Hardware |
|------|-------|----------|
| Megatron-LM | Pre-training, Post-training | NVIDIA GPU |
| torchtitan | Pre-training, RL | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | Pre-training | NVIDIA Ada GPU |
| torch_musa | Pre-training | Moore Threads MUSA GPU |

---

## At a glance

| Module | Files | Lines | Core |
|--------|-------|-------|------|
| 知识专家 | 4 | 853 | 4 阶段知识问答 |
| 精度专家 | 3 | 567 | 5 步诊断 + 故障分类 + 已知模式 |
| 性能专家 | 3 | 822 | 5 步优化 + 瓶颈分类 + SOTA 技术 |
| 问题归档 | 6 | 457 | 模板 + 5 个种子 ticket |
| 共享参考 | 2 | 276 | 源码映射 + 术语表 |

---

## Roadmap

```
P0 骨架      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P1 知识核心  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P2 专家引擎  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P3 问题种子  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P4 参考资料  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ ✅
P5 扩展新仓  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
P6 实战积累  ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
P7 团队化    ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─
```

---

## Usage

1. Place this repo in your workspace — Claude Code auto-loads `SKILL.md`
2. Or symlink to `~/.claude/skills/training-crucible/`
3. Ask any training question — Claude routes to the right expert

---

## License

MIT
