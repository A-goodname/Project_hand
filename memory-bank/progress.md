# Progress — Project_hand

## 已实现功能

### 双相机体系
- [x] Camera A (342222071364, RealSense_R) + Camera B (UGREEN 1080P, OpenCV)
- [x] Camera A pipeline 全生命周期共享 (GREETING→HANDLE, 只开一次)
- [x] Camera B 预加载 (GREETING 启动, 椅子+人就位全程运行)
- [x] Orchestrator 3 状态: IDLE → GREETING → HANDLE (单次, exit(0))
- [x] CRP/TXT/状态机 TCP 协议一致性验证 + 修复
- [x] CRP SocketClose 修复 (Socket1→Socket0)
- [x] CRP 重复 Wait 删除

### HandleCupDetector 轻量化
- [x] 独立检测器 Orchestrator/handle_cup_detector.py + config.yaml
- [x] Handle: YOLO kettle → 白色把手 HSV+RGB → 红色胶带(简化) → 3D
- [x] Cup: YOLO cup → bbox 裁剪(left/right) + 采样 → 3D
- [x] 去 SAM2+PCA 6D, skip_frames 90→5
- [x] 固定 ABC 模式: 基系 ABC 来自 config.yaml
- [x] 红色胶带检测对齐 RealSense_R/detect_red_tape (HSV+最大连通域)
- [x] 白色把手 HSV 阈值适配 (min_value:80, min_rgb:90)

### Phase B fixed 模式
- [x] compute_grasp_rotation 仅非 fixed 分支调用
- [x] 稳定器+标定在 ctx.handle_cup_detector 赋值前创建 (竞态修复)
- [x] Pipeline 不在 daemon 线程 close (对齐 estun_greeting)

### 打招呼 Phase 0
- [x] Phase0Server 类: 4 步单连接协议
- [x] Camera A 人检测复用 HandleCupDetector pipeline
- [x] CRP orchestrator 段完整
- [x] 实机测试: Phase 0 通过 ✅

### 通信协议
- [x] pack_three / READY/CONTINUE/ABORT
- [x] RELEASE 双语义
- [x] Phase D 判定逻辑

### HandInCup MediaPipe 重构
- [x] YOLOv8m + MediaPipe Hands (21 landmarks)
- [x] landmarks-in-bbox 计分制 + curled gate
- [x] 纯 RGB 判定
- [x] 杯 ROI 回退 + 标定工具

### 人就位检测 (Camera B)
- [x] 椅子实时检测 (每帧 YOLO, 中心偏好)
- [x] 非对称阈值 (进10/出30)
- [x] has_chair 门控修复 (去掉鸡生蛋逻辑)

### 日志系统
- [x] Orchestrator/logger.py — 终端+文件双输出
- [x] state_machine 关键事件日志 (Phase A/B/C/D)
- [x] HandInCup 超时去重

### 修复
- [x] MediaPipe clearcut 遥测禁用 (hand_in_cup.py)
- [x] Camera B D435i serial 死代码删除
- [x] HandInCup 超时刷屏修复

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 Phase C/D 全流程测试 | PREGRIP/GRIPDONE/CUP/READY/RELEASE 完整链路 |
| 🟡 人就位检测改进 | bbox 稳定性, 防 flicker |
| 🟡 EyeToHand 标定接入 | 3D→1D 人体标定 (暂用 delivery.X_TEACH) |
| 🟡 skip_frames 实机微调 | 当前=5 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯 RGB, 无深度数据
- EyeToHand result.yaml 标定暂未接入
- 人就位 YOLO 漏检 → leave_frames=30 临时缓解
