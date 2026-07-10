# Active Context — Project_hand

## 当前焦点

**HandInCup MediaPipe 重构完成** — 深度逻辑 + curled gate 已修复, 待全流程联调.

## 最近变更

1. **HandInCup MediaPipe 重构** (2026-07-10):
   - **手检测**: YOLOv8n-Pose → MediaPipe Hands (21 landmarks, 帧间跟踪)
   - **判定**: landmarks-in-bbox 计分制 (wrist×2 + MCP×1 + TIP×1 ≥ 2), bbox_margin=20px
   - **curled gate 改**: wrist 在 bbox 内 → 跳过握拳检测; wrist 不在 → curled ≥ 2
   - **深度**: YOLO→画面采样; fallback→cup_depth_expected 预标定值
   - **杯 ROI 回退**: YOLO 失败 → cup_roi_fallback; --calibrate-roi 标定工具 (同时写入深度)
   - **递送位姿 P3**: 已填入 config (0.2295/0.4427/0.3837m)

2. **打招呼 Phase 0 修复** (2026-07-09, 工作区待提交):
   - 位姿单位 mm; PAPER→RESET; TCP 8080 提前启动

## 未解决问题

- **HandInCup 联调**: Orchestrator 全流程待实测
- **打招呼联调**: CRP+PC 全流程待验证

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **HandInCup 联调** | MediaPipe 判定在状态机中从头到尾验证 |
| 🔴 P0 | **打招呼全流程联调** | 修复后 CRP+PC 从头到尾验证 |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
