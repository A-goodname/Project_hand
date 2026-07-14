# Active Context — Project_hand

## 当前焦点

**Camera A Handle+Cup 检测器轻量化改造完成。待实机全流程测试。**

## 最近变更

1. **HandleCupDetector 轻量化改造** (2026-07-14):
   - 新建 `Orchestrator/handle_cup_detector.py`: 独立检测器, 不依赖 RealSense_R
   - Handle: YOLO kettle → 白色把手 HSV+RGB → 红色胶带(R通道优势) → 3D
   - Cup: YOLO cup → bbox 裁剪(left/right) + 采样 → 3D
   - 去掉 SAM2 (~3s/帧) + PCA 6D, skip_frames 90→5
   - 固定 ABC 模式: 基系 ABC 来自 `Orchestrator/config.yaml`, P1/P2/P3 直用
   - 白色把手+红色胶带 HSV 参数移植自同事 visionpro.py
   - `state_machine.py` Phase B 适配: fixed 模式 Handle+Cup 统一使用基系 ABC
   - 新建 `Orchestrator/config.yaml`: 独立配置 (模型/深度/ABC/白把手/红胶带)

2. **pyrealsense2 pyd 兼容** (2026-07-14):
   - sys.path 提前到 import pyrealsense2 之前
   - 去掉 rs 类型标注, 用 assert guard 替代 # type: ignore

3. **Cup left/right Bbox 裁剪** (2026-07-14):
   - 移植自 test_cup_6D.py, 排除把手侧背景干扰
   - 裁剪后更新锚点深度 + 可视化 bbox

## 未解决问题

- **实机双相机全流程测试待进行**

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **双相机全流程测试** | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 | Camera B 杯 ROI 标定 | UGREEN 替换后首次 |
| 🟡 | skip_frames 实机微调 | 当前=5, 根据实际帧率调整 |
