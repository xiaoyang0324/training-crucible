# 精度故障分类 (Precision Failure Taxonomy)

> 本文档提供精度故障的多维分类体系，用于 Step 2 (Classify) 和 Step 3 (Localize) 快速定位问题类型。

---

## 1. 按症状分类

### 1.1 Loss NaN

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **数值溢出** | FP16 上溢 (max=65504) | 大 loss scale、大梯度 |
| **除零** | 分母为零 (grad norm=0) | 空 batch、全 padding |
| **log(0)** | CE loss 中 log(0) | 无 label smoothing、极端 logits |
| **Inf 传播** | 中间值 Inf 向后传播 | 未做数值裁剪 |

### 1.2 Loss Spike

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **LR 过大** | 学习率突增 / warmup 不足 | 自定义 schedule、恢复训练 |
| **数据异常** | 脏数据 / 错误 label | 新数据 shard、数据 pipeline bug |
| **梯度突变** | 某层梯度异常 | 新数据分布、权重损坏 |

### 1.3 Loss Divergence

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **LR 过大** | 学习率始终过高 | 配置错误 |
| **Loss scale 不足** | 动态 loss scale 频繁下调 | FP16 训练 |
| **模型损坏** | 权重出现 NaN/Inf | 上游数值问题未处理 |

### 1.4 Grad Norm Explosion

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **梯度爆炸** | 深层网络梯度累积 | 无 grad clip、残差连接异常 |
| **Loss scale 过大** | 动态 loss scale 过高 | FP16 训练初期 |
| **并行同步错误** | allreduce 异常 | 分布式配置错误 |

### 1.5 Train-Infer Mismatch

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **模式差异** | train/eval 模式未切换 | Dropout / BatchNorm |
| **权重同步延迟** | 推理使用旧权重 | RL rollout 权重同步 |
| **数值精度差异** | 训练 FP16 / 推理 FP32 | 精度敏感模型 |

### 1.6 Accuracy Regression

| 子类 | 典型根因 | 常见场景 |
|------|---------|---------|
| **数据分布变化** | 训练数据分布偏移 | 新数据集、数据混合比例变化 |
| **超参漂移** | 配置被意外修改 | 多人协作、配置管理不当 |
| **框架版本差异** | 算子实现变化 | 框架升级 |

---

## 2. 按训练阶段分类

### 2.1 预训练 (Pre-training)

| 常见问题 | 典型表现 | 高发阶段 |
|---------|---------|---------|
| FP16 溢出 | Loss NaN | 训练初期 |
| 数据加载异常 | Loss Spike | 切换数据 shard |
| 学习率 warmup 不足 | Loss Spike / Divergence | 训练前几百步 |
| 梯度爆炸 | Grad Norm 爆炸 | 深层模型 (>100 层) |

### 2.2 后训练 (Post-training: SFT/DPO/RLHF)

| 常见问题 | 典型表现 | 高发阶段 |
|---------|---------|---------|
| 数据质量差 | Loss Spike | 新数据混入 |
| 过拟合 | Train loss ↓ / Eval loss ↑ | 训练后期 |
| DPO 数值不稳定 | Loss NaN | 极端偏好对 |
| RLHF reward 异常 | 精度回退 | reward model 偏差 |

### 2.3 强化学习 (RL: GRPO/PPO)

| 常见问题 | 典型表现 | 高发阶段 |
|---------|---------|---------|
| Reward hacking | reward ↑ / 真实精度 ↓ | 训练中期 |
| Policy collapse | entropy → 0, 输出单一 | 训练后期 |
| KL 约束失效 | 策略偏离过大 | KL 系数过小 |
| Rollout 数据偏差 | 训练-推理分布不一致 | 权重同步延迟 |

---

## 3. 按层级分类

```
┌─────────────────────────────────────────────────────────────────────┐
│                        精度故障层级模型                               │
├─────────────────────────────────────────────────────────────────────┤
│  Layer 6: Loss 计算层    ── loss 函数、label smoothing、loss scale   │
│  Layer 5: 权重层        ── 权重初始化、权重更新、权重同步             │
│  Layer 4: 激活层        ── 激活函数、softmax、layer norm、残差连接   │
│  Layer 3: 梯度层        ── 反向传播、梯度裁剪、梯度同步 (allreduce)  │
│  Layer 2: 优化器层      ── Adam/LR schedule、状态 (m/v)、权重衰减    │
│  Layer 1: 数据层        ── 数据加载、预处理、tokenization、label     │
└─────────────────────────────────────────────────────────────────────┘
```

**层级排查原则：自下而上排查，自上而下验证。**
- 排查：先查数据层（最常见），再逐层向上
- 验证：修复后从 loss 层向下确认每层正常

---

## 4. 症状→根因→方案 速查表

| 症状 | 最可能根因 | 首选方案 | 引用 |
|------|-----------|---------|------|
| **Loss NaN (FP16)** | 数值溢出 | 启用/增大 loss scale，或切 BF16 | `known-patterns.md` §1.1 |
| **Loss NaN (除零)** | grad norm=0 或空 batch | 添加 epsilon、检查数据 | `known-patterns.md` §1.1 |
| **Loss NaN (log(0))** | CE loss 中 log(0) | 启用 label smoothing | `known-patterns.md` §1.1 |
| **Loss Spike (脏数据)** | 异常数据批次 | 数据清洗、跳过异常 batch | `known-patterns.md` §2.1 |
| **Loss Spike (LR)** | 学习率过大 | 增加 warmup、降低 peak LR | `known-patterns.md` §2.2 |
| **Loss Divergence** | LR 过大 / loss scale 不足 | 调整 LR schedule、检查 loss scale | `known-patterns.md` §1 |
| **Grad Norm 爆炸** | 梯度累积 / loss scale 过大 | 启用 grad clipping、降低 loss scale | `known-patterns.md` §3.1 |
| **Grad Norm 消失** | 残差连接异常 / 深层网络 | 检查残差缩放、调整初始化 | `known-patterns.md` §3.2 |
| **Train-Infer Mismatch** | 模式未切换 / 权重同步延迟 | 检查 train/eval 模式、确认权重同步 | `known-patterns.md` |
| **Accuracy Regression** | 数据分布变化 / 超参漂移 | 对比数据分布、回滚配置 | `known-patterns.md` |
| **RL: Reward Hacking** | reward 被策略利用 | 增强 KL 约束、检查 reward 设计 | `known-patterns.md` §4.1 |
| **RL: Policy Collapse** | 策略熵降至 0 | 调整 clip range、添加 entropy bonus | `known-patterns.md` §4.2 |

---

## 引用

- `skill-precision/SKILL.md` — 5 步诊断工作流
- `skill-precision/references/known-patterns.md` — 已知精度问题模式库（含源码引用）
- `skill-tickets/` — 历史案例库（按 `type: precision` 检索）
