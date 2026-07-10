# Active Context — Project_hand

## 当前焦点

**TCP 协议一致性验证完成** — CRP/TXT/状态机三方对比, 修复 RELEASE 语义冲突 bug, Wait 5s→2s.
3 状态状态机代码就绪, 等待实机联调.

## 最近变更

1. **RELEASE 语义修复 + CRP/TXT 同步** (2026-07-10, 工作区):
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

- **打招呼联调**: CRP+PC 全流程待验证

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **打招呼全流程联调** | CRP+PC 从头到尾验证 |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
| 🟢 P2 | **EyeToHand 标定接入** | 3D→1D 人体标定替换 delivery.X_TEACH |
