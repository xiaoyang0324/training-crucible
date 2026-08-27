---
# 子模块 frontmatter 仅作说明，不注册为独立 skill
# 入口：SKILL.md 的意图路由表 → precision/SKILL.md
description: >
  精度诊断专家——5 步诊断工作流，覆盖 loss NaN/spike/divergence、梯度爆炸、
  train-infer 不一致、精度回退等精度问题。
  触发条件：用户报告 loss 异常、梯度异常、训练不收敛、推理结果不一致、
  精度回退、数值溢出、NaN、Inf 等精度相关问题。
---

# 精度诊断专家 — 5 步诊断工作流

## The Iron Law

```
诊断精度问题前，必须先拿到四样东西：
  1. 症状描述（loss 曲线 / 报错信息 / 异常步数）
  2. 训练阶段（预训练 / 后训练 / RL）
  3. 环境信息（模型规模、并行配置、硬件、框架版本）
  4. 变更历史（问题出现前改了什么：数据 / 配置 / 代码）

没有这些不做根因判断。绝不凭经验跳过定位步骤。
```

---

## 触发条件

| 症状 | 典型表现 | 紧急度 |
|------|---------|--------|
| **Loss NaN** | loss 突然变为 NaN，后续步持续 NaN | 🔴 紧急 |
| **Loss Spike** | loss 突增后恢复或持续升高 | 🔴 紧急 |
| **Loss Divergence** | loss 单调上升不收敛 | 🟠 高 |
| **Grad Norm 爆炸** | grad norm 突增 >10x 正常值 | 🔴 紧急 |
| **Train-Infer Mismatch** | 训练 loss 正常但推理结果异常 | 🟠 高 |
| **Accuracy Regression** | 同配置下精度相比基线下降 | 🟡 中 |

---

## 5 步诊断工作流

1. **捕获 (Capture)** — 收集错误消息、训练曲线快照 (loss/grad norm/lr)、环境信息 (模型/并行/硬件/框架)、变更历史
2. **分类 (Classify)** — 归类症状 (NaN/Spike/Divergence/GradExplosion/TrainInferMismatch/AccuracyRegression)，判断首次出现步数 & 可复现性
3. **定位 (Localize)** — 逐层排查：数据 → 优化器 → 梯度 → 激活 → 权重 → loss 计算，对比正常/异常步的中间值
4. **假设 (Hypothesize)** — 匹配 `references/known-patterns.md` 已知模式，检索 `tickets/` 类似案例，按概率排序列出候选根因
5. **解决与归档 (Resolve & Archive)** — 提出修复方案 (配置调整/代码patch/流程变更)，验证指标恢复正常，归档到 `tickets/`

---

## 每步详细检查清单

### Step 1: 捕获 (Capture)

**向用户确认的问题：**
- 问题首次出现的步数？之前跑了多少步正常？
- 是偶发（单步）还是持续（后续步都异常）？
- 问题出现前做了什么变更？（数据 / 配置 / 代码 / 硬件）
- 是否在多卡上都出现？还是只有个别 rank？

**需要收集的日志/指标：**
- 完整错误消息 / traceback（如有）
- 训练曲线：loss, grad norm, learning rate, 最近 100 步
- 环境信息：模型规模、并行配置 (TP/PP/DP/CP/EP)、硬件、框架 commit
- 数据批次信息：异常步使用的数据 shard / index

---

### Step 2: 分类 (Classify)

**症状判定规则：**

| 观察到的现象 | 分类 |
|-------------|------|
| loss = NaN 或 Inf | Loss NaN |
| loss 突增 >3x 后恢复 | Loss Spike (瞬时) |
| loss 突增后持续不降 | Loss Spike (持续) |
| loss 单调上升 | Loss Divergence |
| grad norm 突增 >10x | Grad Norm Explosion |
| 训练 loss 正常但推理结果异常 | Train-Infer Mismatch |
| 同配置精度低于基线 | Accuracy Regression |

**关键判断：**
- 异常是**首次出现**还是**一直存在**？
- 异常是**确定性复现**还是**随机出现**？
- 异常前是否有**配置/数据/代码变更**？

---

### Step 3: 定位 (Localize)

**逐层排查顺序：**

```
数据层 ──► 优化器层 ──► 梯度层 ──► 激活层 ──► 权重层 ──► loss 层
  │           │           │          │          │          │
  │           │           │          │          │          └─ loss 计算溢出?
  │           │           │          │          └─ 权重含 NaN/Inf?
  │           │           │          └─ 激活值异常 (ReLU/Softmax)?
  │           │           └─ 梯度含 NaN/Inf? 梯度同步是否正确?
  │           └─ lr 是否正确? 状态是否损坏?
  └─ 数据是否含 NaN/异常值? label 是否正确?
```

**定位方法：**
- 对比正常步和异常步的中间激活值 / 梯度
- 逐层打印 tensor 的 min/max/mean，定位首次出现 NaN 的层
- 检查数据 pipeline：异常步的数据是否正常加载

---

### Step 4: 假设 (Hypothesize)

**匹配已知模式：**
- 查阅 `precision/references/known-patterns.md`，匹配症状
- 检索 `tickets/` 中 `type: precision` 的历史案例
- 按概率排序列出候选根因

**常见候选根因：**
- 数值精度不足（FP16 溢出 → 需 loss scaling / BF16）
- 数据异常（脏数据 / 错误 label → 需数据清洗）
- 学习率过大（→ 需 warmup / 降低 LR）
- 梯度同步错误（→ 检查 allreduce / grad norm 计算）
- 算子数值不稳定（→ 检查 softmax / layer norm 实现）

---

### Step 5: 解决与归档 (Resolve & Archive)

**修复方案类型：**
- **配置调整**：loss scale、LR、batch size、grad clip
- **代码 patch**：修复算子实现、添加数值保护
- **流程变更**：数据清洗流程、checkpoint 回滚

**验证标准：**
- 修复后连续跑 N 步（N ≥ 100），loss / grad norm 恢复正常
- 对比修复前后指标曲线

**归档要求：**
- 按 `tickets/TEMPLATE.md` 格式创建 ticket
- 填写完整：Symptom / Environment / Analysis / Root Cause / Resolution / Verification
- 引用真实源码路径作为 `source_refs`

---

## 引用

| 文件 | 内容 |
|------|------|
| `precision/references/failure-taxonomy.md` | 精度故障分类（按症状 / 阶段 / 层级） |
| `precision/references/known-patterns.md` | 已知精度问题模式库 |
| `tickets/TEMPLATE.md` | 问题归档模板 |
| `references/source-repo-map.md` | 源码仓路由（用于定位代码引用） |
