# training-crucible P3 (Ticket Seed) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Create 3-5 seed problem tickets extracted from the user's existing interview/work notes. These demonstrate the ticket schema in action and provide immediately reusable cases.

**Architecture:** Each ticket is a markdown file in `tickets/` following the `TEMPLATE.md` schema (frontmatter + 8 body sections). Tickets are based on real problems from the user's project experience.

---

## Source Materials (read-only)

The user's existing notes contain real training problems:
- `C:\y30062407\workspace\local\面试\小艺5000卡集群_疑难杂症攻坚手册.md` — 5000-card cluster troubleshooting
- `C:\y30062407\workspace\local\面试\NPU图模式泛化_项目详析.md` — NPU graph mode project
- `C:\y30062407\workspace\local\面试\字节千卡训练图模式_项目详析.md` — ByteDance 1000-card graph mode
- `C:\y30062407\workspace\local\面试\美团千卡推理入图_项目详析.md` — Meituan 1000-card inference graph mode
- `C:\y30062407\workspace\local\面试\ACLGraph图模式_完整技术文档.md` — ACL graph mode technical doc

---

## File Structure

```
training-crucible/
└── tickets/
    ├── TEMPLATE.md              # (exists from P0)
    ├── 2026-08-27-graph-oom.md  # [CREATE] Graph mode OOM problem
    ├── 2026-08-27-precision-nan.md  # [CREATE] Loss NaN in large-scale training
    ├── 2026-08-27-scaling-efficiency.md  # [CREATE] Scaling efficiency issue
    ├── 2026-08-27-rl-reward-hacking.md  # [CREATE] RL reward hacking
    └── 2026-08-27-inference-latency.md  # [CREATE] Inference latency regression
```

---

## Task 1: Read source materials

**Files:**
- Read: the 5 source materials listed above

Steps:
1. Read each file to extract real training problems
2. For each problem, note: symptom, environment, root cause, resolution
3. Select 4-5 diverse problems covering: precision, performance, scaling, RL, inference

---

## Task 2: Write seed tickets

**Files:**
- Create: 4-5 ticket files in `tickets/`

For each selected problem, create a ticket following TEMPLATE.md:
- Frontmatter: id (TICKET-20260827-NNN), title, type, stage, status (all resolved), severity, hardware, frameworks, tags, dates
- Body: Symptom, Environment, Analysis, Root Cause, Resolution, Verification, Lessons, References

Quality standards:
- Based on REAL problems from source notes (not fabricated)
- Specific: include actual model sizes, hardware counts, metrics
- Cite source repos where applicable
- Each ticket 60-100 lines

Commit pattern:
```bash
git add tickets/2026-08-27-*.md
git commit -m "docs(tickets): add seed tickets from project experience"
```

---

## Task 3: Final verification

Steps:
1. `wc -l tickets/2026-08-27-*.md` — confirm all tickets have content
2. `grep -c "id:" tickets/2026-08-27-*.md` — confirm frontmatter present
3. `git log --oneline -5` — confirm clean history
4. `git push origin main`
