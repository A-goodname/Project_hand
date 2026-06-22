# Active Context — Project_hand

## 当前焦点

重构 Cline 规则架构: 从 `.clinerules` 单文件迁移到官方标准 `.clinerules/` + `Documents\Cline\Rules\` 分层体系.

## 最近变更

- 最终架构已就绪:
  - **项目级规则**: `.clinerules/project.md` — Memory Bank + Git + 代码风格 + 模块 + 环境
  - **全局级规则**: `C:\Users\荒\Documents\Cline\Rules\context.md` — New Task 提醒策略 (300K/15轮)
  - **上下文持久化**: `memory-bank/` — 6 个核心文件
- 清理遗留配置:
  - 删除旧根目录 `.clinerules` 文件
  - 清理 `.cline/` 临时目录
  - 清理 `settings.json` 中的 `cline.customInstructions` (已迁移至 `Documents\Cline\Rules\context.md`)

## 下一步待办

1. 验证 `.clinerules/project.md` 被 Cline 正确识别
2. 验证 `Documents\Cline\Rules\context.md` 被 Cline 正确加载
3. Git 提交本次迁移

## 活跃决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 项目规则位置 | `.clinerules/` 目录 (官方标准) | Cline 自动检测, 支持多文件 |
| 全局规则位置 | `Documents\Cline\Rules\` (Windows 官方路径) | 所有项目生效, 独立于 VS Code 设置 |
| Memory Bank 位置 | `memory-bank/` 项目根目录 | 随 git 版本控制, 新任务自动加载 |
| New Task 阈值 | 300K token 或 15 轮对话 | 适配 1M 上下文窗口 |