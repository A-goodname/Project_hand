# Active Context — Project_hand

## 当前焦点

灵巧手失能后再使能可靠性修复: **`init_hand()` 添加 `query_enable()` 验证 + 自动重试 (最多 10 次)**,
`is_hand_available()` 替代 run 脚本不可靠的 `_hand_available = True` 乐观标志.

## 最近变更

1. **灵巧手使能可靠性修复** (2026-06-26):
   - **问题**: `OmniHandGesture.enable()` 返回 True 仅表示 ACK 收到, 不代表硬件真正使能.
     失能后重新使能经常失败, 需多次重试才能成功.
   - **`Estun/utils_shared.py` — `init_hand()`**:
     - `enable()` 后通过 `is_enabled()` → `query_enable()` 读取硬件实际状态确认
     - 未确认则自动重试 (默认 10 次, 间隔 0.5s)
     - 全部失败自动 fallback 到模拟模式 (不影响机械臂通信)
     - 新增参数 `enable_retries` / `enable_retry_delay`
   - **`Estun/utils_shared.py` — `is_hand_available()`**: 新函数, 检查 `_hand is not None`
   - **`Estun/estun_handle_cup_run.py` / `estun_handle_run.py`**:
     - Import `is_hand_available`
     - `_hand_available = True` → `_hand_available = is_hand_available()`
     - init 失败时打印 WARN 而非静默假装成功

2. **通信协议与流程重构** (2026-06-25):
   - PC端: `fmt_pose()` / `pack_three()` 抽函数, 清零时机改为断开时, 自动重稳定流程
   - 机械臂端: `estun_handle_cup_code.txt` v2 — 线性流程 + P4 预倒水点 + 独立 while 循环
   - 经 code-reviewer skill 审阅通过

3. **back 模式 P3 示教验证通过**: 实际倒水精度验证
4. **B/C 角防跳变修正** (2026-06-23): Handle P1 / Cup P1 vertical 模式守卫
5. **t_handle_offset / t_cup_offset 微调完成**

## 下一步

- P4 预倒水点: 机械臂端代码已写好, 待在实际示教器上测试
- left 模式示教 + t_cup_offset_left 微调 (待完成)
- 双相机体系 (眼在手上 + 眼在手外) — 见 doc/Future_work.md
- 全面自动化测试
