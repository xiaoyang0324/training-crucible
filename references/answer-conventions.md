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
| torchada (Ada GPU) | `train/torchada` | `torchada/` |
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
5. **验证源码引用** — 引用文件前先用 grep/read 确认该文件存在，不确定的标注"需要确认"

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

【归档】（如问题已解决且根因明确）→ 创建 ticket TICKET-xxx
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

【归档】（如问题已解决且根因明确）→ 创建 ticket TICKET-xxx
```
