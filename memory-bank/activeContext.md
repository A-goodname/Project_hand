# Active Context — Project_hand

## 当前焦点

**人就位检测开发中** — `person_detector.py` 新增椅子标定 + 人就位检测, 为 3 状态状态机做准备.

## 最近变更

1. **人就位检测** (2026-07-10, 工作区):
   - `person_detector.py`: 新增 `calibrate_chair()` (YOLO chair class=56, 中心偏好 + 稳定窗口)
   - `person_detector.py`: 新增 `detect_person_seated()` (person∩chair/chair_area ≥ 0.4 → 累积10帧)
   - `EyeToHand/config.yaml`: 新增 `chair` 配置段 (conf/stabilize/overlap/seated_stable)
   - 独立测试模式 `run()` 已更新: 椅子标定进度 + 就位状态可视化
   - 下一步: 测试通过后集成到 Orchestrator 状态机

2. **HandInCup 状态机联调修复** (2026-07-10, `e571a3f`):
   - **修复双实例 bug**: main.py 和 state_machine.py 各自创建 `HandInCupDetector` → 统一为 `ctx.hand_in_cup_detector` 共享实例
   - **修复 reset() 失效**: TCP READY 时的 `reset()` 现在作用于 main loop `update()` 的同一实例
   - **可视化集成**: Camera B 激活时使用 `draw_visualization()` (MediaPipe landmarks + cup bbox + 计分)
   - **清理路径修复**: 嵌套 `_cleanup()` 函数, 9 处退出路径 (6 early return + 3 assert) 全部执行清理
   - **配置化**: `phase_b_timeout` 入 config; CONTINUE/ABORT 移除末尾 `\n`

2. **HandInCup MediaPipe 重构** (2026-07-10):
   - **手检测**: YOLOv8n-Pose → MediaPipe Hands (21 landmarks, 帧间跟踪)
   - **判定**: landmarks-in-bbox 计分制 (wrist×2 + MCP×1 + TIP×1 ≥ 2), bbox_margin=20px
   - **curled gate 改**: wrist 在 bbox 内 → 跳过握拳检测; wrist 不在 → curled ≥ 2
   - **深度**: YOLO→画面采样; fallback→cup_depth_expected 预标定值
   - **杯 ROI 回退**: YOLO 失败 → cup_roi_fallback; --calibrate-roi 标定工具 (同时写入深度)
   - **递送位姿 P3**: 已填入 config (0.2295/0.4427/0.3837m)

3. **打招呼 Phase 0 修复** (2026-07-10, `e1cafbd`):
   - 位姿单位 mm; PAPER→RESET; TCP 8080 提前启动

## 未解决问题

- **打招呼联调**: CRP+PC 全流程待验证

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **打招呼全流程联调** | 修复后 CRP+PC 从头到尾验证 |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
| 🟢 P2 | **EyeToHand 标定接入** | 3D→1D 人体标定替换 delivery.X_TEACH |
