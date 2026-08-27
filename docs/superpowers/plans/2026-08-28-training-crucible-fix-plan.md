# training-crucible 修复方案

> 基于专业审查识别的 10 项问题，按优先级分三阶段修复。

**审查日期:** 2026-08-28
**问题总数:** 10（🔴 架构 3 / 🟡 结构 2 / 🟢 细节 5，其中 1 项降级豁免）

---

## 问题清单

| # | 类别 | 问题 | 影响 |
|---|------|------|------|
| 1 | 🔴 架构 | SKILL.md 违反单一职责（路由 + 规则 + 索引 + 源码表混在一起） | 后续维护易回归 |
| 2 | 🔴 架构 | precision/performance 子 SKILL.md 的 frontmatter `name:` 造成"子技能独立注册"幻觉 | 用户/助手误以为它们是独立 skill |
| 3 | 🔴 架构 | tickets/ 缺少 SKILL.md（无检索/归档工作流定义） | 归档模块无使用指南 |
| 4 | 🟡 结构 | docs/ 目录不应入库 | ~~污染发布内容~~ → **豁免** |
| 5 | 🟡 结构 | source-repo-map 缺少 DeepSpeed / vLLM / SGLang 等外部框架 | 引用链断裂 |
| 6 | 🟡 结构 | knowledge/ 文件偏大需拆分 | ~~可读性差~~ → **降级豁免** |
| 7 | 🟢 细节 | 命名不一致：`posttraining.md` vs spec 的 `post-training` | 搜索/引用混淆 |
| 8 | 🟢 细节 | 缺少 .gitignore | 可能误提交 OS/编辑器垃圾文件 |
| 9 | 🟢 细节 | 回答输出格式缺少可复用模板 | 回答质量不稳定 |
| 10 | 🟢 细节 | 4/5 ticket 的 `source_refs: []` 为空 | 违反"源码为凭"原则 |

---

## 豁免项说明

### ~~Issue #4: docs/ 不应入库~~ → 豁免

`docs/superpowers/specs/` 和 `docs/superpowers/plans/` 是 superpowers 工作流的副产物，属于项目历史记录。设计 spec 本身就是有价值的参考文档。**保留入库，不做改动。**

### ~~Issue #6: knowledge/ 文件需拆分~~ → 降级豁免

实际行数核查：pretraining.md 161 行、posttraining.md 170 行、rl.md 256 行、inference.md 266 行。最大文件 266 行，结构清晰（概述→原理→配置→误区→索引），无拆分必要。**不做改动。**

---

## 修复阶段

```
Phase A (架构)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  #3 → #2 → #1
Phase B (结构)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  #5
Phase C (细节)   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  #7 → #8 → #9 → #10
```

---

## Phase A：架构修复（🔴 3 项）

### Task A1: 创建 `tickets/SKILL.md`

**问题:** tickets/ 只有 TEMPLATE.md 和种子案例，缺少检索/归档工作流定义。

**创建:** `tickets/SKILL.md`

**内容结构:**
```markdown
---
name: ticket-archive
description: >
  问题归档与检索模块——按症状/阶段/框架检索历史案例，
  或将已解决的精度/性能问题归档为新 ticket。
---

# 问题归档模块

## 检索工作流
1. 读取用户问题的关键词（症状、阶段、框架）
2. 扫描 tickets/ 下所有 *.md 的 frontmatter（tags, type, stage）
3. 按匹配度排序返回，格式：`TICKET-ID — title（匹配标签: xxx）`

## 归档工作流
触发条件：
- 精度/性能问题已解决且根因明确
- 跨模块复杂问题
- 用户明确要求归档

步骤：
1. 复制 TEMPLATE.md → tickets/YYYY-MM-DD-<slug>.md
2. 填写 frontmatter（id, type, stage, tags, source_refs）
3. 填写 8 段正文
4. 检查 related_tickets 是否需要更新

## 检索命令示例
- "遇到过 loss NaN 吗" → 按 `tags: loss-nan` 检索
- "预训练性能问题" → 按 `stage: pretraining` + `type: performance` 检索
- "Megatron 相关" → 按 `frameworks: Megatron-LM` 检索
```

**Commit:** `feat(tickets): add SKILL.md with search & archive workflow`

---

### Task A2: 修正 precision/performance 子 SKILL.md frontmatter

**问题:** 子模块 SKILL.md 的 `name: precision-expert` / `name: performance-expert` 暗示它们是独立注册的 skill，造成路由混乱。

**修改:**

**`precision/SKILL.md`** 第 1-8 行:
```markdown
---
name: precision-expert
description: >
  精度诊断专家——5 步诊断工作流...
---
```
→
```markdown
---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → precision/SKILL.md
description: >
  精度诊断专家——5 步诊断工作流...
---
```

**`performance/SKILL.md`** 第 1-7 行:
```markdown
---
name: performance-expert
description: >
  性能优化专家——5 步优化工作流...
---
```
→
```markdown
---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → performance/SKILL.md
description: >
  性能优化专家——5 步优化工作流...
---
```

**Commit:** `fix(precision,performance): clarify sub-module frontmatter is not standalone skill registration`

---

### Task A3: 拆分 SKILL.md — 提取"回答规范 + 源码仓路由"到独立引用文件

**问题:** SKILL.md 168 行，混合了：入口路由、Iron Law、核心能力表、意图路由表、关键词触发表、源码仓路由表、跨模块协作规则、子模块索引、回答规范、版本信息。

**方案:** SKILL.md 保留路由核心（Iron Law + 意图路由 + 关键词触发），提取"回答规范"和"源码仓路由"到 `references/answer-conventions.md`。

**修改 `SKILL.md`:**

删除第 97-168 行（从"## 源码仓路由"到末尾），替换为：

```markdown
---

## 源码仓路由 & 回答规范

> 详见：
> - `references/source-repo-map.md` — 源码仓 → 阶段映射
> - `references/answer-conventions.md` — 回答规范与输出格式模板
> - `references/training-glossary.md` — 规范术语表

---

## 子模块索引

| 模块 | 路径 | 说明 |
|------|------|------|
| 知识专家 | `knowledge/` | 按训练阶段分 4 个文件 |
| 精度专家 | `precision/` | 5 步诊断工作流 + 精度故障分类 |
| 性能专家 | `performance/` | 5 步优化工作流 + SOTA 技术目录 |
| 问题归档 | `tickets/` | 检索 & 归档工作流 |
| 源码映射 | `references/source-repo-map.md` | 外部仓 → 阶段映射 |
| 术语表 | `references/training-glossary.md` | 规范术语 |
| 回答规范 | `references/answer-conventions.md` | 输出格式模板 + 引用规则 |
```

**创建 `references/answer-conventions.md`:**

```markdown
# 回答规范与输出格式

> 本文件定义 training-crucible 所有模块的回答规范。SKILL.md 路由到子模块后，子模块的回答必须遵循本规范。

---

## 源码仓路由表

> 根据用户提到的框架或训练阶段，定位到对应的本地源码仓。

| 用户提到 | 定位源码仓 | 关键路径 |
|----------|-----------|----------|
| Megatron-LM | `train/Megatron-LM` | `megatron/core/` |
| torchtitan | `train/torchtitan` | `torchtitan/distributed/`, `torchtitan/experiments/rl/` |
| miles | `train/miles` | `miles/rollout/`, `miles/backends/`, `miles/true_on_policy/` |
| slime | `train/slime` | `slime/rollout/`, `slime/backends/`, `slime/agent/` |
| torchada (Ada GPU) | `train/torchtada` | `torchada/` |
| torch_musa (MUSA) | `train/torcht_musa` | `torch_musa/` |
| DeepSpeed | `train/DeepSpeed` | `deepspeed/` |
| vLLM | 外部知识 | 标注 `[外部]` |
| SGLang | 外部知识 | 标注 `[外部]` |

> 详细映射见 `references/source-repo-map.md`。

---

## 跨模块协作规则

### 精度 + 性能联合分析
当问题同时涉及精度和性能（如"开启 activation recompute 后 loss 异常"）：
1. 先走 `precision/` 诊断流程，确认精度问题根因
2. 再走 `performance/` 优化流程，评估性能影响
3. 最终结论归档到 `tickets/`

### 知识 + 归档联动
当知识问答中发现类似历史案例：
1. 在 `knowledge/` 回答后，主动检索 `tickets/` 中 `related_tickets`
2. 如有匹配，附上"类似案例：TICKET-..."

### 归档触发条件
以下情况必须归档到 `tickets/`：
- 精度问题已解决且根因明确
- 性能优化已完成且有量化收益
- 跨模块的复杂问题

---

## 回答规范

### 必须做
1. **先确认阶段和框架** — 回答前明确用户的训练阶段和涉及框架
2. **引用真实源码** — 技术主张必须附 `仓名/文件路径:行号`
3. **标注外部知识** — 来自论文/文档的知识标注 `[外部]`
4. **主动关联案例** — 发现匹配的历史问题单时主动附上

### 禁止做
1. **凭空猜测** — 没有源码证据不做根因判断
2. **虚构 API** — 不编造函数名、参数、文件路径
3. **跳过诊断流程** — 精度/性能问题必须走完整工作流
4. **重复归档** — 同一问题不创建多个 ticket

---

## 输出格式模板

### 知识问答类
```
【确认】训练阶段：xxx ｜ 涉及框架：xxx

【回答正文】
...（含源码引用 `仓名/文件路径:行号`）...

【关联案例】（如有）
- TICKET-xxx — xxx
```

### 精度诊断类
```
【症状确认】
- 症状：xxx
- 阶段：xxx
- 环境：xxx

【诊断步骤】
1. Capture — ...
2. Classify — ...
3. Localize — ...
4. Hypothesize — ...
5. Resolve — ...

【建议方案】
...

【归档】→ 创建 ticket TICKET-xxx
```

### 性能优化类
```
【基线确认】
- 当前吞吐：xxx
- 瓶颈类别：compute/memory/communication/I/O/launch
- 环境：xxx

【优化步骤】
1. Profile — ...
2. Identify — ...
3. Match — ...
4. Adapt — ...
5. Validate — ...

【预期收益】
...

【归档】→ 创建 ticket TICKET-xxx
```
```

**Commit:** `refactor(SKILL): extract answer conventions & source routing to references/answer-conventions.md`

---

## Phase B：结构修复（🟡 1 项）

### Task B1: 扩展 source-repo-map.md — 添加外部框架

**问题:** 当前只覆盖 6 个本地仓，但实际工作涉及 DeepSpeed、vLLM、SGLang 等。

**修改 `references/source-repo-map.md`:**

在 "Hardware Coverage Note" 之后、"Repo 1: Megatron-LM" 之前，添加：

```markdown
## External Frameworks (外部知识，标注 `[外部]`)

以下框架无本地源码仓，回答中引用时**必须标注 `[外部]`**，且不能作为主要证据。

### DeepSpeed
- **Source:** `C:\y30062407\workspace\local\面试\train\DeepSpeed`（如已克隆）
- **Primary stages:** Pre-training, Post-training
- **Key features:** ZeRO-1/2/3, ZeRO-Offload, DeepSpeed-MoE, pipeline parallelism
- **When to cite:** ZeRO optimizer states partitioning, memory optimization techniques

### vLLM
- **Source:** 外部（未本地克隆）
- **Primary stage:** Inference
- **Key features:** PagedAttention, Continuous Batching, Tensor Parallelism serving
- **When to cite:** Inference serving architecture, KV Cache management

### SGLang
- **Source:** 外部（未本地克隆，但 miles/slime 仓内有集成代码）
- **Primary stage:** Inference (rollout generation)
- **Key features:** RadixAttention, structured generation, RL rollout backend
- **When to cite:** RL rollout generation, inference engine integration
```

在 "Quick Routing Index" 表格中添加：

```markdown
| ZeRO / memory optimization | DeepSpeed | Megatron-LM, torchtitan |
| Inference serving | vLLM | SGLang (via miles/slime) |
| RL rollout backend | SGLang (via miles/slime) | vLLM |
```

**Commit:** `docs(references): extend source-repo-map with external frameworks (DeepSpeed, vLLM, SGLang)`

---

## Phase C：细节修复（🟢 4 项）

### Task C1: 统一命名 — `posttraining.md` → `post-training.md`

**问题:** 文件名 `posttraining.md` 与 spec 中的 `post-training` 不一致。

**操作:**
1. `git mv knowledge/posttraining.md knowledge/post-training.md`
2. 更新 `SKILL.md` 路由表中的引用：`knowledge/posttraining.md` → `knowledge/post-training.md`
3. 更新 `references/training-glossary.md` 中如有引用

**Commit:** `style(knowledge): rename posttraining.md to post-training.md for consistency`

---

### Task C2: 添加 .gitignore

**问题:** 缺少 .gitignore，可能误提交 OS/编辑器垃圾文件。

**创建:** `.gitignore`

```gitignore
# OS files
.DS_Store
Thumbs.db
desktop.ini

# Editor files
*.swp
*.swo
*~
.vscode/
.idea/
*.sublime-*

# Obsidian
.obsidian/workspace*
.obsidian/cache/

# Python (for users who extend with scripts)
__pycache__/
*.pyc
*.pyo
.venv/
venv/

# Build artifacts
*.egg-info/
dist/
build/

# Logs
*.log
```

**Commit:** `chore: add .gitignore`

---

### Task C3: 回答规范模板（已在 Phase A3 中完成）

**说明:** Issue #9（输出格式模板）已在 Task A3 中通过 `references/answer-conventions.md` 解决，无需单独 commit。

---

### Task C4: 补全 tickets 的 source_refs

**问题:** 5 个种子 ticket 中，2 个 `source_refs: []` 为空（slow-card-detection、mindspore-rmsnorm-epsilon），违反"源码为凭"原则。

**分析:**
- `slow-card-detection-5000-npu.md` — 华为昇腾集群问题，根因是 HCCL 链路/散热，无本地源码仓对应
- `mindspore-rmsnorm-epsilon.md` — MindSpore 框架问题，无本地源码仓对应

**方案:** 对于无本地源码仓的 ticket，在 `external_refs` 中补充来源笔记路径，并在 `source_refs` 位置添加注释说明。

**修改 `tickets/2026-08-27-slow-card-detection-5000-npu.md`:**

```yaml
source_refs: []
  # 无本地源码仓对应——本 ticket 根因为 HCCL 链路/散热硬件问题
  # 详见 external_refs 中的攻坚手册笔记
```

**修改 `tickets/2026-08-27-mindspore-rmsnorm-epsilon.md`:**

```yaml
source_refs: []
  # 无本地源码仓对应——本 ticket 涉及 MindSpore 框架，非本地 6 仓范围
  # 详见 external_refs 中的来源笔记
```

**Commit:** `docs(tickets): annotate empty source_refs with reason`

---

## 修复顺序与依赖

```
A1 (tickets/SKILL.md)  ─┐
A2 (frontmatter 修正)  ─┼─ Phase A → Phase B → Phase C
A3 (SKILL.md 拆分)    ─┘
        │
B1 (source-repo-map 扩展)
        │
C1 (重命名) ─┐
C2 (gitignore) ─┼─ Phase C（可并行）
C4 (source_refs) ─┘
```

**建议执行顺序:** A1 → A2 → A3 → B1 → C1 → C2 → C4（串行，每步一个 commit）

---

## 预期修复后状态

| 指标 | 修复前 | 修复后 |
|------|--------|--------|
| SKILL.md 行数 | 168 | ~80 |
| tickets/ 有工作流 | ❌ | ✅ |
| 子模块 frontmatter 歧义 | ✅ | ❌ |
| source-repo-map 覆盖仓数 | 6 | 9 |
| .gitignore | ❌ | ✅ |
| 空 source_refs 未注释 | 2 | 0 |
| 命名一致性 | posttraining | post-training |

---

## 不在本次范围

- **Issue #4 (docs/ 入库):** 豁免，设计 spec 是有价值的项目文档
- **Issue #6 (knowledge/ 拆分):** 豁免，最大文件 266 行结构清晰
- **新增功能:** 本次仅做结构修复，不新增知识内容
