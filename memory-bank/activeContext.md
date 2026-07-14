# Active Context — Project_hand

## 当前焦点

**Camera B 从 D435i → UGREEN Camera 720P 替换完成 + HandInCup 纯 RGB 调试完成。**
椅子实时检测 + 非对称阈值 + baseline 锚定，人就位检测可通过。
HandInCup curled gate 修复：score≥5 绕过握拳约束。
待实机双相机全流程测试。

## 最近变更

1. **HandInCup curled gate 修复** (2026-07-14):
   - 手腕在框外 + score≥5（多手指在框内）→ 直接判定 hit，跳过 curl 检测
   - `curled_threshold: 25→35`（适配 720p）
   - `bbox_margin: 20→40`（适配 720p）
   - config 标注 `[分辨率相关]` 注释

2. **椅子实时检测 + 非对称阈值 + baseline 锚定** (2026-07-14):
   - 删除 `calibrate_chair()` 锁定逻辑 → 每帧实时检测椅子 (`_detect_chair_live`)
   - 非对称阈值：进 10 帧 / 出 5 帧立刻 False
   - `_chair_baseline` 锚定：仅 not person_near 时更新，person_near 时面积差 ≤30% 才更新 `_chair_ref`
   - `is_chair_calibrated` → `has_chair`
   - `PersonDetector` 加 `start_camera` 参数/方法

3. **Camera B 生命周期调整** (2026-07-14):
   - IDLE: Camera B 关闭
   - GREETING enter: 开 Camera B + 启 seated detection（不等人，直接进）
   - HANDLE: seated detection 继续 → READY 判定

4. **Camera B D435i → UGREEN 替换** (2026-07-13):
   - 新增 `EyeToHand/camera.py`, 纯 RGB 无深度
   - HandInCup 移除深度校验

## 未解决问题

- **实机双相机全流程测试待进行**

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **双相机全流程测试** | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 P1 | **HandInCup 纯 RGB 精度** | 确认 score≥5 绕过 curl 后的误判率 |
| 🟢 P2 | **UGREEN 720p 帧率确认** | YOLO+MediaPipe 在 1280×720 上的推理延迟 |
