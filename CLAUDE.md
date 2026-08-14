# Project_hand — 项目规范

实机侧机器人项目（ESTUN + OmniHand O10 + D435i + UGREEN×2）。本机角色 = 实机侧；训练侧 nickwang@192.168.112.82。

## Memory-bank 使用规范（必须遵守）

`memory-bank/` 是项目知识库（Cline 风格 6 文件），**每次改动代码/配置后必须同步**：

| 文件 | 内容 | 何时更新 |
|------|------|----------|
| `activeContext.md` | 当前焦点 + 最近变更 + 下一步 | **每次变更后**：新的工作项/焦点/重要结论必须追加"最近变更"；会话开始先读它 |
| `progress.md` | 已完成/未完成清单（[x]/[ ]） | **每次完成一项**：勾选或追加；未完成的留在列表 |
| `techContext.md` | 环境、依赖、关键约束、排坑 | 环境/依赖/约束变化时 |
| `projectbrief.md` / `productContext.md` / `systemPatterns.md` | 项目背景/架构 | 架构性变更时才动 |

**规则**：
1. 变更记录格式：`### <主题> (<日期>)` + 编号列表，保留排坑细节（原因+解决）
2. 会话结束前确认 memory-bank 与实况一致，**变更随代码一起提交**
3. 当前任务权威文档：`doc/p2p1-dual-confirm-handoff-2026-08-14.md`（P2/P1 双确认适配）

## 工具与 Skill

- **CRP/ESTUN/OmniHand 开发**：先读 `.claude/skills/estun-omnihand-robotics/SKILL.md`（CRP 修改规则、陷阱、helper 脚本），禁止凭记忆合成 CRP schema
- CRP 分析/修改：`python .claude/skills/estun-omnihand-robotics/scripts/estun_crp_helpers.py`

## 环境速查（详情见全局记忆 `~/.claude/CLAUDE.md`）

- **Python**：conda `Proj`（Python 3.11, torch 2.13.0+cu126）
- **串口**：`/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00` @ 460800（未重登会话用 `sg dialout -c` 包裹）
- **相机**：Camera B 固定 = USB 口 4.2 的 UGREEN（`camera_path: usb-0:4.2`）；**禁硬编码 /dev/videoN，按 sysfs 名称枚举**；D435i 独立控制器 2e:00.3
- **机械臂**：192.168.101.100 连本机 `192.168.101.96:8080`（本机 IP 为 DHCP 动态）
- **git**：身份 `qaq <huangzhiwuming@163.com>`；`core.autocrlf input`；**本地提交不 push**（交接方式待定）

## 安全与协作限制

- **D0 门禁**（训练侧要求）：标定失败必须阻止启动（禁 `np.eye(4)`/默认 Tsai 回退）；PREGRIP ≠ 位姿确认；P2/P1 一次性双确认（GRASP_AUTH/GRIP_AUTH）
- **实机动作必须训练侧显式批准**：只做已批准的只读/自检（如灵巧手 RESET 自检）
- 回传材料：inbox 目录 + sha256、序列号脱敏、不含私钥/凭证/日志/模型
