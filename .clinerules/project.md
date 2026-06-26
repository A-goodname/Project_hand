<!-- ============================================================================= -->
<!-- Cline 项目规则 — Project_hand                                                -->
<!-- ============================================================================= -->
<!-- 本文件为项目级规则, 每次新任务 Cline 自动读取.                               -->
<!-- 全局级补充规则见 Documents\Cline\Rules\context.md.                           -->
<!-- ============================================================================= -->

# Project_hand 项目规则

## 1. 任务启动流程 (新任务入口)

Cline 接受新任务后, 按以下顺序执行:

1. 读取 `memory-bank/` 目录下的全部核心文件, 恢复项目上下文:
   - `memory-bank/projectbrief.md`
   - `memory-bank/productContext.md`
   - `memory-bank/systemPatterns.md`
   - `memory-bank/techContext.md`
   - `memory-bank/activeContext.md`
   - `memory-bank/progress.md`
2. 阅读 `README.md` 了解项目整体结构与模块划分
3. 根据任务涉及的模块查阅 `doc/Module_Rules.md` 了解对应模块的开发规则

若 `memory-bank/` 目录或核心文件不存在, 则应主动创建并填充内容.

## 2. Memory Bank 机制

### 2.1 文件结构

所有文件位于 `memory-bank/` 目录, 使用 Markdown 格式, 文件间存在依赖层级:

```text
projectbrief.md          ← 基础文档, 定义核心需求与项目范围
    ├── productContext.md  ← 项目动机、解决的问题、用户体验目标
    ├── systemPatterns.md  ← 系统架构、技术决策、设计模式、组件关系
    └── techContext.md     ← 技术栈、开发环境、依赖、工具链
            ├── activeContext.md  ← 当前焦点、最近变更、下一步、活跃决策
            └── progress.md       ← 已完成/待完成、当前状态、已知问题
```

### 2.2 更新触发条件

Memory Bank 在以下情况必须更新:

1. 发现新的项目模式或关键架构决策
2. 完成有意义的代码修改并验证通过后 (与 Git 提交同步)
3. 用户明确要求 "update memory bank"
4. 上下文窗口接近阈值, 准备生成 New Task 摘要时
5. 需要澄清上下文或记录关键洞察时

更新时需审阅所有核心文件, 重点确保 `activeContext.md` 和 `progress.md` 反映最新状态.

### 2.3 摘要生成规范

生成 memory-bank 摘要时, 需包含以下关键信息:

- `activeContext.md`: 当前进行到哪一步、最近决策、下一步待办
- `progress.md`: 已实现功能清单、已知问题、遗留决策
- 确保新任务读取后能无缝衔接, 无需用户额外解释

## 3. Git 工作流

1. 本仓库使用 Git 进行纯本地版本管理 (无远程仓库)
2. Cline 每次完成有意义的代码修改且验证通过后, 经用户同意后执行:

   ```bash
   git add -A && git commit -m "描述性的中文提交信息"
   ```

3. 提交信息应包含标题行 + 详细 body (由于 Windows cmd 的 -m 不支持换行, 需用 Python 写临时文件方式):

   ```python
   # 1) 用 Python 生成提交信息文件
   python -c "msg='''标题行\n\n1. 第1点...\n2. 第2点...'''; open('_commit_msg.txt','w',encoding='utf-8').write(msg)"
   # 2) 从文件读取提交信息
   git commit --amend -F _commit_msg.txt
   # 或首次提交时:
   git add -A && git commit -F _commit_msg.txt
   # 3) 清理临时文件
   del _commit_msg.txt
   ```

4. 提交信息示例:
   - 标题: "修复 YOLO 检测中的坐标偏移 bug"
   - 标题: "添加 Orbbec 相机自动曝光功能"
   - 标题: "重构 RealSense_R 模块: 提取基类 + 配置后移 + 位姿坐标修复"
   - body 详细列举各改动点 (编号列表)
5. git 提交应积累一定工作量, 且实际成功运行后再提交
6. 若要追加说明到上一次提交, 使用 `git commit --amend -F _commit_msg.txt` (不改变已暂存的文件)

## 4. 代码风格

严格遵循 `doc/python_script_template.yaml` 和 `doc/python_script_template.py` 中定义的规范:

- 注释使用**中文文字 + 英文标点**: 句号用 `.` 不用 `。`, 逗号用 `,` 不用 `，`
- docstring 中参数/返回描述使用英文冒号 `:` 而非中文冒号 `：`
- 日志标签使用 `[INFO]` `[WARN]` `[ERROR]` 等英文标签
- 章节分隔符使用 `# =====` (长等号线), 段落使用 `# ----` (短横线)
- 模块级配置加载在文件顶部完成, 使用 `_cfg` 或全大写常量

## 5. 模块规则

各硬件模块详细开发规则见 `doc/Module_Rules.md`.

## 6. 项目环境

使用 Conda 虚拟环境, 地址为: D:\Environment\miniconda\envs\Proj\python.exe
