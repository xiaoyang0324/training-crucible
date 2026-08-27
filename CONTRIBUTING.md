# 贡献指南

> 本指南说明如何扩展 training-crucible 的内容。

---

## 添加新 Ticket

1. 复制 `tickets/TEMPLATE.md` → `tickets/YYYY-MM-DD-<slug>.md`
2. 分配 id: `TICKET-YYYYMMDD-NNN`（检查当天已有 ticket，递增 NNN）
3. 填写 frontmatter：
   - `type`: precision | performance | hybrid
   - `stage`: pretraining | posttraining | rl | inference | cross-stage
   - `severity`: critical（训练中断）| major（功能受损）| minor（轻微影响）
   - `hardware`: NVIDIA-Ada | MUSA | Ascend | CPU | agnostic
   - `frameworks`: Megatron-LM | torchtitan | miles | slime | torchada | torch_musa | torch_npu | MindSpeed
   - `tags`: 优先使用已有 tags，新增 tags 参考 `references/training-glossary.md`
4. 填写 8 段正文（Symptom / Environment / Analysis / Root Cause / Resolution / Verification / Lessons / References）
5. 检查已有 ticket 是否需要更新 `related_tickets`
6. Commit: `docs(tickets): add ticket for <short-description>`

## 添加新 Source Repo

1. 在 `references/source-repo-map.md` 中添加新 repo 章节
2. 包含：Path、Primary stages、Key directories、When to cite
3. 在 Quick Routing Index 表格中添加相关条目
4. 同步更新 `references/answer-conventions.md` 的源码仓路由表
5. Commit: `docs(references): add <repo-name> to source repo map`

## 添加新术语到 Glossary

1. 在 `references/training-glossary.md` 对应类别中添加术语
2. 格式：`**ENG** (中文) — 定义。相关: [term1] [term2]。`
3. Commit: `docs(glossary): add <term>`

## 文件命名约定

- Ticket: `YYYY-MM-DD-<slug>.md`（日期 + 短横线分隔的英文/拼音 slug）
- Knowledge: `<stage>.md`（全小写，用横线分隔）
- References: `<topic>.md`（全小写，用横线分隔）

## 引用规范

- 所有源码引用必须先用 grep/read 确认文件存在
- 格式：`仓名/文件路径:行号`
- 外部知识（论文/文档）标注 `[外部]`
