# Active Context — Project_hand

## 当前焦点

**HandInCup 状态机联调完成** — 双实例 bug 已修复 + 可视化集成.

## 最近变更

1. **HandInCup 状态机联调修复** (2026-07-10):
   - **修复双实例 bug**: main.py 和 state_machine.py 各自创建 `HandInCupDetector` → 统一为 `ctx.hand_in_cup_detector` 共享实例
   - **修复 reset() 失效**: TCP READY 时的 `reset()` 现在作用于 main loop `update()` 的同一实例
   - **可视化集成**: Camera B 激活时使用 `draw_visualization()` (MediaPipe landmarks + cup bbox + 计分)
   - **TODO 清理**: 移除 `# TODO: 实现完整的 MediaPipe Hand-in-cup 检测` (已在上次提交完成)

2. **HandInCup MediaPipe 重构** (2026-07-10):
   - **手检测**: YOLOv8n-Pose → MediaPipe Hands (21 landmarks, 帧间跟踪)
   - **判定**: landmarks-in-bbox 计分制 (wrist×2 + MCP×1 + TIP×1 ≥ 2), bbox_margin=20px
   - **curled gate 改**: wrist 在 bbox 内 → 跳过握拳检测; wrist 不在 → curled ≥ 2
   - **深度**: YOLO→画面采样; fallback→cup_depth_expected 预标定值
   - **杯 ROI 回退**: YOLO 失败 → cup_roi_fallback; --calibrate-roi 标定工具 (同时写入深度)
   - **递送位姿 P3**: 已填入 config (0.2295/0.4427/0.3837m)

3. **打招呼 Phase 0 修复** (2026-07-09, 工作区待提交):
   - 位姿单位 mm; PAPER→RESET; TCP 8080 提前启动

## 未解决问题

- **打招呼联调**: CRP+PC 全流程待验证

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **打招呼全流程联调** | 修复后 CRP+PC 从头到尾验证 |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
| 🟢 P2 | **EyeToHand 标定接入** | 3D→1D 人体标定替换 delivery.X_TEACH |
