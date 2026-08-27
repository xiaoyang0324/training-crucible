[中文](README_zh.md) | [English](README.md)

# training-crucible

> 在熔炉中锻造 AI 训练全栈精度。
> Forging full-stack AI training expertise under fire.

**精度诊断、性能优化、问题归档——覆盖预训练、后训练、强化学习、推理优化四大阶段。**

Precision diagnostics, performance optimization, and a living problem archive — spanning pre-training, post-training, RL, and inference.

---

## 这是什么

training-crucible 是一个 **Claude Code 技能库**（skills library），面向 AI 基础设施工程师。它不是可执行代码，而是一套结构化的中文知识文档 + 问题案例库，嵌入 Claude Code 助手，在以下场景提供专家级支持：

- **知识问答** — "GRPO 和 PPO 有什么区别？" "TP 和 PP 怎么组合？"
- **精度诊断** — "训练 loss 突然 NaN 了怎么排查？"
- **性能优化** — "千卡训练 MFU 只有 30% 怎么提升？"
- **问题归档** — "之前有没有遇到过类似问题？"

## 架构

```
training-crucible/
├── SKILL.md                    # 🛡️ 入口路由 — Iron Law + 意图路由 + 子模块索引
├── knowledge/                  # 📚 知识专家 — 按训练阶段路由的知识问答
│   ├── pretraining.md          #   预训练：并行策略、内存优化
│   ├── posttraining.md         #   后训练：SFT / DPO / RLHF
│   ├── rl.md                   #   强化学习：GRPO / PPO、rollout
│   └── inference.md            #   推理优化：KV Cache、量化、投机解码
├── precision/                  # 🔬 精度专家 — 5 步诊断工作流
│   ├── SKILL.md                #   工作流引擎：捕获→分类→定位→假设→解决
│   └── references/             #   故障分类 + 已知模式
├── performance/                # ⚡ 性能专家 — 5 步优化工作流
│   ├── SKILL.md                #   工作流引擎：画像→识别→匹配→适配→验证
│   └── references/             #   瓶颈分类 + SOTA 技术目录
├── tickets/                    # 📋 问题归档 — 结构化案例库
│   ├── TEMPLATE.md             #   问题单模板（YAML frontmatter + 8 段正文）
│   └── 2026-08-27-*.md         #   种子 ticket（来自真实项目经验）
└── references/                 # 📖 共享参考 — 源码映射 + 术语表
    ├── source-repo-map.md      #   6 仓 → 阶段映射
    └── training-glossary.md    #   90 个规范术语
```

## 路由逻辑

```
用户问题
    │
    ├─ "是什么" / "怎么工作" ──────────────► knowledge/ (按阶段子路由)
    ├─ loss NaN / 梯度爆炸 / 精度异常 ─────► precision/ (5 步诊断)
    ├─ 慢 / OOM / 吞吐低 ─────────────────► performance/ (5 步优化)
    ├─ "遇到过吗" / "历史案例" ────────────► tickets/ (按标签检索)
    └─ 复杂问题 ──────────────────────────► 精度 → 性能 → 归档 (串联)
```

## 源码参考仓

| 仓库 | 阶段 | 硬件 |
|------|------|------|
| Megatron-LM | 预训练、后训练 | NVIDIA GPU |
| torchtitan | 预训练、RL | NVIDIA GPU |
| miles | RL (GRPO/PPO) | NVIDIA GPU |
| slime | RL (GRPO/PPO) | NVIDIA GPU |
| torchada | 预训练 | NVIDIA Ada GPU |
| torch_musa | 预训练 | Moore Threads MUSA GPU |

> 所有技术主张优先引用本地源码仓（文件路径 + 行号），外部论文/文档作为补充。

## 内容统计

| 模块 | 文件数 | 行数 | 核心能力 |
|------|--------|------|----------|
| 知识专家 | 4 | 853 | 4 阶段知识问答 |
| 精度专家 | 3 | 567 | 5 步诊断 + 故障分类 + 已知模式 |
| 性能专家 | 3 | 822 | 5 步优化 + 瓶颈分类 + SOTA 技术 |
| 问题归档 | 6 | 457 | 模板 + 5 个种子 ticket |
| 共享参考 | 2 | 276 | 源码映射 + 术语表 |
| **合计** | **20** | **~4,568** | |

## 使用方式

本仓库作为 Claude Code 的 skills 目录使用：

1. **作为项目技能**: 将本仓库放在工作区，Claude Code 自动加载 `SKILL.md`
2. **作为个人技能**: 软链到 `~/.claude/skills/training-crucible/`
3. **触发**: 问任何训练相关问题，Claude 自动路由到对应专家模块

## 路线图

- [x] **P0** 骨架（入口路由 + 目录 + 模板 + 源码映射）
- [x] **P1** 知识核心（4 阶段知识文件）
- [x] **P2** 精度/性能专家（5 步工作流 + 参考资料）
- [x] **P3** 问题单种子（5 个真实项目 ticket）
- [x] **P4** 参考资料（术语表）
- [ ] **P5** 扩展新源码仓（DeepSpeed、op-plugin 等）
- [ ] **P6** 实战积累（持续归档真实问题）
- [ ] **P7** 团队化（问题单生命周期、review 流程）

## License

MIT
