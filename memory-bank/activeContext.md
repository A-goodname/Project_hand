# Active Context — Project_hand

## 当前焦点

**Orchestrator 状态机重构完成** — 2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR)，单 TCP Session 全流程。
**HandInCup 检测器完成** — YOLOv8m + YOLOv8n-Pose 双模型，可独立测试。

## 最近变更

1. **Orchestrator 状态机重构** (2026-07-03):
   - `Orchestrator/state_machine.py`: 6 状态 → 2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR)
   - 单 TCP Session 完整协议:
     - Phase A: Robot → PC 初始位姿
     - Phase B: PC → Robot Handle `<P1>[P2]{P3}`
     - Phase C: PREGRIP → GRIPDONE → RESET → CUP → PC 回复 `<Cup_P1>[Cup_P2]{P3}`
     - Phase D: READY → RELEASE → HandInCup 判定 → CONTINUE/ABORT
   - Cup P3 的 X 来自 Camera B 人体检测 + EyeToHand 标定, YZ/ABC 来自 config 预示教
   - 新增指令: READY (Robot→PC), CONTINUE (PC→Robot), ABORT (PC→Robot)
   - Camera A 修正为 `RealSense/` (rs.align, 内参完好的新相机)

2. **HandInCup 检测器** (2026-07-03):
   - `EyeToHand/hand_in_cup.py`: YOLOv8m (杯 bbox) + YOLOv8n-Pose (手腕关键点) 双模型
   - 2D 判定: 连续 N 帧手腕在杯 bbox 扩展区 → is_hand_on_cup = True
   - 可独立测试: `python EyeToHand/hand_in_cup.py`
   - 配置段: `EyeToHand/config.yaml` → `hand_in_cup` 段

3. **文档更新** (2026-07-03): memory-bank 更新至当前状态.

## 下一步 — 按优先级

| 优先级 | 任务 | 需要硬件? |
|:---:|------|:---:|
| 🔴 P0 | **实操标定** `EyeToHand/calibration/calibrate.py` | ✅ 机械臂 |
| 🔴 P0 | **填入递送参数** `delivery.Y_TEACH/Z_TEACH/ABC` 预示教值 | ✅ 机械臂 |
| 🟡 P1 | **测试 HandInCup 检测器** `python EyeToHand/hand_in_cup.py` | ✅ Camera B |
| 🟡 P2 | **机器人 .crp 适配**: 新协议 (READY/CONTINUE/ABORT 指令) | ✅ 机械臂 |
| 🟢 P3 | **Orchestrator 全流程联调**: 双相机 + 机器人 + 灵巧手 | ✅ 全部硬件 |
