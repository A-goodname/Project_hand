# Active Context — Project_hand

## 当前焦点

**代码改动完成，待实机测试** — 挥手回归 `do_gesture`，CRP 速度/点位/label 全部对齐。
greeting (单相机) 和状态机 (双相机) 均需实机验证.

## 最近变更

1. **挥手重构 + CRP 全面修复** (2026-07-13, 工作区):
   - 挥手回归 `do_gesture` 一步到位 (移除 KinematicsConverter / wave_speed / _speed_cfg)
   - `greeting_handlepluscup.crp`: 4 MovJ→MovL, 速度/label 与 greeting CRP 对齐
   - `handlepluscup-orchestrator.crp`:
     - Phase 0: label/speed 对齐 greeting CRP (全部 V100, "MovL")
     - Phase A-C: H_P3/H_P4→sudu, C_P3→sudu+MovJ→MovL
     - Phase D: 5 个 widget 点位修正 (GH_ZHAOSHOU1→正确 C_P4/Home/C_P1/C_P2), ABORT C_P1→sudu, 其余 V100
   - `state_machine.py` GreetingHandler: `_execute_wave(ctx)` do_gesture + stop_event
   - `estun_greeting_handle_cup_run.py`: 同回归 do_gesture
   - `estun_config.yaml`: wave_speed 清除

2. **CRP 运动优化 + 压力配置化 + B 角防跳变** (2026-07-13, `c324bd2`):
   - P3 倒水偏移 Z 轴统一上移: back +2cm, left +3cm
   - 压力检测硬编码全部提取到 `estun_config.yaml` greeting 段 (8 参数)
   - Handle P1 B 角: <90→89, >90→91 (4 run + Demo + state_machine)

3. **RELEASE 语义修复 + CRP/TXT 同步** (2026-07-10, 工作区):
   - `state_machine.py`: RELEASE handler 从 2 分支改为 3 分支, 用 `hand_in_cup_active` 标志区分:
     - Phase C (READY 之前) → `release_object()` 松把手
     - Phase D (READY 之后) → HandInCup 判定 → CONTINUE/ABORT
     - ABORT 路径第二次 RELEASE → `release_object()` 放回后松杯
   - `handlepluscup-orchestrator.crp`: Phase D Wait 5000ms → 2000ms
   - `estun_handle_cup_code_orchestrator.txt`: 更新协议文档, 补充 RELEASE 双语义说明
   - README.md: 2状态→3状态, 更新流程图, 补全 Camera B 职责, 修正 HandInCup 检测器描述

2. **人就位检测** (2026-07-10, `58c18be`):
   - `person_detector.py`: 新增 `calibrate_chair()` (YOLO chair class=56, 中心偏好 + 稳定窗口)
   - `person_detector.py`: 新增 `detect_person_seated()` (person∩chair/chair_area ≥ 0.4 → 累积10帧)
   - `EyeToHand/config.yaml`: 新增 `chair` 配置段 (conf/stabilize/overlap/seated_stable)
   - 独立测试模式 `run()` 已更新: 椅子标定进度 + 就位状态可视化

3. **3 状态状态机** (2026-07-10, `9182139`):
   - IDLE → GREETING → HANDLE_GRASP_AND_POUR → IDLE 闭环
   - GreetingHandler: Phase 0 打招呼 (TCP 8600, 4步协议 + YOLO 人检测 + 挥手握手)
   - IdleHandler: 椅子标定 → 人就位检测 → 触发 GREETING
   - HandleGraspAndPourHandler: Phase A-D 全流程 (TCP 8080)

4. **HandInCup 状态机联调修复** (2026-07-10, `e571a3f`):
   - 修复双实例 bug (统一 `ctx.hand_in_cup_detector` 共享)
   - 修复 reset() 失效 + 可视化集成 + 清理路径修复 + 配置化

## 未解决问题

- **实机测试待进行**: greeting (单相机) + state_machine (双相机) 均需验证

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **greeting 单相机测试** | `estun_greeting_handle_cup_run.py` + `greeting_handlepluscup.crp` |
| 🔴 P0 | **状态机双相机测试** | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
| 🟢 P2 | **EyeToHand 标定接入** | 3D→1D 人体标定替换 delivery.X_TEACH |
