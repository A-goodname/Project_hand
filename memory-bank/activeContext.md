# Active Context — Project_hand

## 当前焦点

back 模式 P3 示教验证通过 + 抓取微调完成 + B/C 角防跳变修正 (仅 vertical 模式).
**通信协议与流程已重构** (2026-06-25): 线性流程 + P4 预倒水点 + 断开时清零位姿.

## 最近变更

1. **通信协议与流程重构** (2026-06-25):
   - **PC端 `estun_handle_cup_run.py`**:
     - 抽 `fmt_pose()` / `pack_three()` 替代重复格式串 (同时修正 Cup P1 括号 `[...]`→`(...)` 的 bug)
     - 位姿清零时机从"连接时"改为"断开时"(`handle_client` wrapper finally), 避免机械臂 2s 超时等 PC 重新稳定 ~10s 的竞态
     - 断开后自动进入重稳定流程: `_poses_cleared` 标志 → reset stabilizer → 双检测器均稳定后打印"可以启动机械臂"
     - 机械臂提前连接也可容忍: Phase B 轮询 `last_known_pose`, 稳定后即发
     - 经 code-reviewer skill 审阅通过, 修复两个 minor issues (过时注释 + 通知改为双稳定检查)
   - **机械臂端 `doc/estun_handle_cup_code.txt` v2**:
     - `SocketSendStr`移到 while 外, 两个 `SocketReadStr` 各自独立 while 循环
     - 新增 P4 预倒水点: `CposOffset(H_P1, [0, 0, H_P3.z - H_P1.z, 0, 0, 0])`
     - 路径: P2 → P1 → **P4** → P3 → **P4** → P1 → P2
   - `doc/estun_handle_code.txt` 未改 (handle-only 暂维持原样)

2. **back 模式 P3 示教验证通过**: 实际倒水精度验证, 运行 demo/run 测试确认可行
3. **t_handle_offset / t_cup_offset 微调完成**: 用户自行在 `estun_config.yaml` 中调整
4. **B/C 角防跳变修正** (2026-06-23): Handle P1 / Cup P1 vertical 模式守卫
5. **OpenCV 窗口标题去中文**

## 下一步

- P4 预倒水点: 机械臂端代码已写好, 待在实际示教器上测试
- left 模式示教 + t_cup_offset_left 微调 (待完成)
- Orbbec 相机与主入口集成
- RealSense (新版, rs.align) 模块与主入口集成
- 双相机体系 (眼在手上 + 眼在手外) — 见 doc/Future_work.md
- 全面自动化测试
