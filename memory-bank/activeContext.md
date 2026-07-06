# Active Context — Project_hand

## 当前焦点

**机器人端代码 v4 建立完成** — 伪代码 + CRP 示教器文件 + PC 端响应格式同步更新。
**HandInCup 检测器完成** — YOLOv8m + YOLOv8n-Pose 双模型，可独立测试。

## 最近变更

1. **机器人端代码 v4** (2026-07-03):
   - `doc/estun_handle_cup_code_v4.txt`: 完整伪代码，含 READY/CONTINUE/ABORT 协议 + 双分支
   - `doc/handlepluscup-v4.crp`: 示教器项目文件 (由 `update_crp_v4.py` 从 v3 CRP 自动生成)
   - `doc/update_crp_v4.py`: CRP 自动修改脚本 (可复现)
   - `Orchestrator/state_machine.py`: PC 端 CONTINUE/ABORT 响应格式改为 `<0,0,0,0,0,0>` / `[0,0,0,0,0,0]` (与机器人端 TranStrToCpos 标记配对)
   - 新增变量: str7="READY", socketstr5, strflag_cont, strflag_abort
   - 新增位姿点: DCp2 (DCPOS), C_P4 (CPOS), DummyCpos (CPOS)
   - Cup P4 安全退避算法: CposOffset(C_P1, DCp2), DCp2.dz = C_P3.z - C_P1.z

2. **Orchestrator 状态机重构** (2026-07-03):
   - `Orchestrator/state_machine.py`: 6 状态 → 2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR)
   - 单 TCP Session 完整协议:
     - Phase A: Robot → PC 初始位姿
     - Phase B: PC → Robot Handle `<P1>[P2]{P3}`
     - Phase C: PREGRIP → GRIPDONE → RESET → CUP → PC 回复 `<Cup_P1>[Cup_P2]{P3}`
     - Phase D: READY → RELEASE → HandInCup 判定 → CONTINUE/ABORT
   - Cup P3 的 X 来自 Camera B 人体检测 + EyeToHand 标定, YZ/ABC 来自 config 预示教
   - 新增指令: READY (Robot→PC), CONTINUE (PC→Robot), ABORT (PC→Robot)
   - Camera A 修正为 `RealSense/` (rs.align, 内参完好的新相机)

2. **HandInCup 检测器** (2026-07-03, 实测更新 2026-07-06):
   - `EyeToHand/hand_in_cup.py`: YOLOv8m (杯 bbox) + YOLOv8n-Pose (手腕关键点) 双模型
   - 2D 判定: 连续 N 帧手腕在杯 bbox 扩展区 → is_hand_on_cup = True
   - **无遮挡实测**: 人手侧握杯把时手腕在 ROI 外, 引入 `dist_factor` (距离阈值系数) 和 `depth_tolerance` (深度容差 5cm) 修正; 修正参数未实地验证, 仍为初始猜测值
   - **灵巧手遮挡**: 尚未测试
   - 模型文件 `model/yolov8m-pose.pt` + `yolov8n-pose.pt` 均已存在
   - 配置段: `EyeToHand/config.yaml` → `hand_in_cup` 段

3. **文档更新** (2026-07-03): memory-bank 更新至当前状态.

## 下一步 — 按优先级

| 优先级 | 任务 | 需要硬件? |
|:---:|------|:---:|
| 🔴 P0 | **实操标定** `EyeToHand/calibration/calibrate.py` | ✅ 机械臂 |
| 🔴 P0 | **填入递送参数** `delivery.Y_TEACH/Z_TEACH/ABC` 预示教值 | ✅ 机械臂 |
| 🟡 P1 | **HandInCup 无遮挡重测**: 调优 `dist_factor`/`depth_tolerance` | ✅ Camera B |
| 🟡 P1 | **HandInCup 有遮挡实测**: 灵巧手遮挡杯把场景 | ✅ Camera B + 灵巧手 |
| 🟡 P2 | **机器人 .crp 适配**: 新协议 (READY/CONTINUE/ABORT 指令) | ✅ 机械臂 |
| 🟢 P3 | **Orchestrator 全流程联调**: 双相机 + 机器人 + 灵巧手 | ✅ 全部硬件 |
