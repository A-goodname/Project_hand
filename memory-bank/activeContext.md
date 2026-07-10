# Active Context — Project_hand

## 当前焦点

**HandInCup MediaPipe 重构完成** — landmarks-in-bbox 计分制 + 握拳检测 gate, 待联调验证.

## 最近变更

1. **HandInCup MediaPipe 重构** (2026-07-10):
   - **手检测**: YOLOv8n-Pose (2 手腕点) → MediaPipe Hands (21 landmarks, 帧间跟踪)
   - **新判定**: landmarks-in-bbox 计分制 (wrist×2 + MCP×1 + TIP×1 ≥ 2)
   - **握拳 gate**: MCP→TIP 距离 < 25px 判定手指弯曲 (curled ≥ 2 才判握杯), 防止开掌划过误触发
   - **杯 ROI 回退**: YOLO 检测不到杯 → 固定 ROI (cup_roi_fallback), 解决灵巧手遮挡
   - **递送位姿**: P3 `[229.5,442.7,383.7]mm` + ABC `[148.8,82.8,-113.6]°` 已填入 config
   - **工具**: `--calibrate-roi` 鼠标框选杯ROI标定

2. **打招呼 Phase 0 修复** (2026-07-09, 工作区待提交):
   - 位姿单位: PC 直接发 mm; PAPER→RESET; TCP 8080 提前启动
   - Phase 0 + CRP 重写 (`9b06269`)

## 未解决问题

- **HandInCup 联调**: 新 MediaPipe 判定在 Orchestrator 全流程中待实测
- **打招呼联调**: CRP+PC 全流程待验证
- **EyeToHand 标定暂废弃**: 递送全示教

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **HandInCup 联调** | MediaPipe 判定在状态机中从头到尾验证 |
| 🔴 P0 | **打招呼全流程联调** | 修复后 CRP+PC 从头到尾验证 |
| 🟡 P1 | **offset 微调** | 三个打招呼位姿偏移实测 |
| 🟢 P2 | EyeToHand 标定 | 3D→1D 人体标定实操 |
