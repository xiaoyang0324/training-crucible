# Changelog

> 所有显著变更记录在此。

---

## [v1.0.0] - 2026-08-28

### 架构
- 拆分 SKILL.md：提取回答规范到 `skill-references/answer-conventions.md`
- 添加 `skill-tickets/SKILL.md`：定义检索与归档工作流
- 扩展 `skill-references/source-repo-map.md`：覆盖 9 个仓（+DeepSpeed/vLLM/SGLang）

### 内容
- 统一命名：`post-training.md` → `post-training.md`
- 添加 `.gitignore`
- 添加 `CONTRIBUTING.md`
- 添加范围声明

### 修复
- 修正子模块 frontmatter `name` 字段不一致
- 更新 README 文件计数和架构树
- 输出模板归档行改为条件式
- knowledge 文件添加 tickets 交叉引用
- TEMPLATE.md 添加 severity 定义和 tag 规范
- skill-tickets/SKILL.md 扩充（id 分配、双向回填）

### 命名规范
- 统一模块文件夹命名：添加 `skill-` 前缀（`knowledge/` → `skill-knowledge/` 等）
- 更新所有内部引用路径（SKILL.md、README、子模块 README、引用文件、TEMPLATE）

### 文档
- 添加版本号 v1.0.0
- 添加 CHANGELOG.md
- 添加范围声明
