# 贡献指南

> 本指南说明如何扩展 training-crucible 的内容。

---

## 添加新 Ticket

1. 复制 `skill-tickets/TEMPLATE.md` → `skill-tickets/YYYY-MM-DD-<slug>.md`
2. 分配 id: `TICKET-YYYYMMDD-NNN`（检查当天已有 ticket，递增 NNN）
3. 填写 frontmatter：
   - `type`: precision | performance | hybrid
   - `stage`: pretraining | posttraining | rl | inference | cross-stage
   - `severity`: critical（训练中断）| major（功能受损）| minor（轻微影响）
   - `hardware`: NVIDIA-Ada | MUSA | Ascend | CPU | agnostic
   - `frameworks`: Megatron-LM | torchtitan | miles | slime | torchada | torch_musa | torch_npu | MindSpeed
   - `tags`: 优先使用已有 tags，新增 tags 参考 `skill-references/training-glossary.md`
4. 填写 8 段正文（Symptom / Environment / Analysis / Root Cause / Resolution / Verification / Lessons / References）
5. 检查已有 ticket 是否需要更新 `related_tickets`
6. Commit: `docs(tickets): add ticket for <short-description>`

## 添加新 Source Repo

1. 在 `skill-references/source-repo-map.md` 中添加新 repo 章节
2. 包含：Path、Primary stages、Key directories、When to cite
3. 在 Quick Routing Index 表格中添加相关条目
4. 同步更新 `skill-references/answer-conventions.md` 的源码仓路由表
5. Commit: `docs(references): add <repo-name> to source repo map`

## 添加新术语到 Glossary

1. 在 `skill-references/training-glossary.md` 对应类别中添加术语
2. 格式：`**ENG** (中文) — 定义。相关: [term1] [term2]。`
3. Commit: `docs(glossary): add <term>`

## 命名规范

### 文件命名

| 类型 | 格式 | 示例 |
|------|------|------|
| Ticket | `YYYY-MM-DD-<slug>.md` | `2026-08-27-pp-tp-stream-conflict.md` |
| Knowledge | `<stage>.md` | `post-training.md` |
| References | `<topic>.md` | `training-glossary.md` |
| 通用规则 | 全小写，横线分隔，无空格无下划线 | ✅ `answer-conventions.md` ❌ `answer_conventions.md` |

### Ticket Frontmatter

| 字段 | 格式 | 示例 |
|------|------|------|
| `id` | `TICKET-YYYYMMDD-NNN` | `TICKET-20260827-004` |
| `title` | 中文一行摘要，无句点 | `PP+TP 混合并行下图捕获 stream 冲突` |
| `type` | `precision` \| `performance` | `performance` |
| `stage` | `pretraining` \| `posttraining` \| `rl` \| `inference` \| `cross-stage` | `pretraining` |
| `severity` | `critical` \| `major` \| `minor` | `major` |
| `hardware` | 首字母大写产品名 | `Ascend`, `NVIDIA-Ada`, `MUSA` |
| `frameworks` | 官方拼写 | `Megatron-LM`, `torchtitan`, `torch_npu` |
| `tags` | 全小写，横线分隔，复数名词优先 | `loss-divergence`, `slow-card`, `graph-capture` |

### 术语命名（Glossary）

| 规则 | 说明 | 示例 |
|------|------|------|
| 英文缩写 | 首次出现给全称 | `TP (Tensor Parallelism, 张量并行)` |
| 中文译名 | 紧跟英文，括号内 | `TP (Tensor Parallelism, 张量并行)` |
| 定义 | 一句话，以相关术语结尾 | `— 将权重矩阵切分到多卡。相关: [[#AllReduce|AllReduce]]` |
| 相关术语 | Obsidian 内链格式 | `[[#PP|PP]]` `[[#AllReduce|AllReduce]]` |

### Tag 命名

- 全小写，横线分隔
- 优先使用已有 tags，避免近义词重复（用 `loss-divergence` 而非 `loss-diverge`）
- 新增 tags 先在 `skill-references/training-glossary.md` 中添加定义
- 常见 tag 类别：
  - 症状类: `loss-nan`, `loss-spike`, `grad-explosion`, `oom`
  - 技术类: `fp8`, `graph-capture`, `activation-recompute`
  - 硬件类: `ascend`, `nvidia-ada`, `musa`
  - 并行类: `tensor-parallel`, `pipeline-parallel`, `context-parallel`

## 引用规范

- 所有源码引用必须先用 grep/read 确认文件存在
- 格式：`仓名/文件路径:行号`
- 外部知识（论文/文档）标注 `[外部]`
