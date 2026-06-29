# Active Context — Project_hand

## 当前焦点

`.crp` / `estun_handle_cup_run.py` / `estun_handle_cup_code.txt` 全面审查修复:
Cup 阶段 RELEASE/GRIP 补全, H_P3 单位修正, flag条件循环, MovL Home 双位置.

## 最近变更

1. **机械臂端 .crp / 参考代码 / PC端 全面审查与修复** (2026-06-29):
   - **`.crp` (handlepluscup-Down-826.crp)**:
     - 修复 H_P3 TranStrToCpos 单位 `mm` → `m`
     - 补全 Cup 阶段: 3 个 TranStrToCpos (C_P1/C_P2/C_P3) + 14 个 Cup 运动控件 (RELEASE/GRIP/RESET 序列)
     - While 循环改用 `strflag` 条件循环替代 `break` (无 break 控件)
     - 两个 While 尾部均补 Wait(500ms)
     - MovL Home 双位置: Handle 末尾 (安全过渡 H_P2→C_P2) + Cup 末尾 (回原点)
   - **`estun_handle_cup_run.py`**: Phase C 新增 RELEASE/GRIP 指令处理 (`release_object()`/`grip_object()`)
   - **`estun_handle_cup_code.txt`**: v3 同步 .crp, flag条件循环/Home双位置/Cup完整流程/移除末尾Return
   - **全局设置**: `~/.claude/settings.json` attribution.commit/pr 置空, 移除所有历史提交 Co-Authored-By

2. **灵巧手使能可靠性修复** (2026-06-26):
   - **问题**: `OmniHandGesture.enable()` 返回 True 仅表示 ACK 收到, 不代表硬件真正使能.
   - **`Estun/utils_shared.py` — `init_hand()`**: `enable()` 后 `is_enabled()` 验证 + 自动重试 (10 次/0.5s)
   - **`is_hand_available()`**: 新函数替代 `_hand_available = True`

3. **通信协议与流程重构** (2026-06-25):
   - PC端: `fmt_pose()`/`pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程
   - 机械臂端: `estun_handle_cup_code.txt` v2 — 线性流程 + P4 预倒水点

4. **back 模式 P3 示教验证通过**
5. **B/C 角防跳变修正** (2026-06-23)

## 下一步

- P4 预倒水点: 机械臂端代码已写好, 待在实际示教器上测试
- left 模式示教 + t_cup_offset_left 微调 (待完成)
- 双相机体系 (眼在手上 + 眼在手外) — 见 doc/Future_work.md
- 全面自动化测试
