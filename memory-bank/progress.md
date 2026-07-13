# Progress — Project_hand

## 已实现功能

### 瓶身抓取
- [x] RealSense 实时版 + EpicEye 单次触发版
- [x] 位姿转换链路验证 Demo + 手眼标定工具

### 把手抓取
- [x] RealSense Handle 6D 版 (微调 YOLOv8n + SAM2 + 扫描线分离)
- [x] 红色胶带辅助检测 / PoseStabilizer
- [x] Handle offset fine_tune: [0.048, -0.037, 0.074]

### 水杯抓取
- [x] 水杯 6D 检测 + 双检测器共享 Pipeline
- [x] Cup offset fine_tune: back=[0.051,-0.008,0.104], left=[0.048,-0.050,0.082]

### Handle+Cup 融合
- [x] estun_handle_cup_run.py 全流程
- [x] back/left 双模式 + P3 多角度示教 + P4 Y后退 CRP
- [x] P3 倒水偏移 Z+2~3cm 运动优化 (10°/15°/20°/30°)
- [x] Cup 检测修复 (PoseStabilizer 代际同步)
- [x] Handle P1 B 角防跳变: <90→89, >90→91 (4 run + Demo + state_machine)

### 灵巧手控制
- [x] USB 串口 / 手势 / 速度控制 / 使能
- [x] 手势扩展: HAND_SHAKE(19) / HELLO_1(20) / HELLO_2(21)
- [x] 压力检测 v1: get_pressure / wait_for_pressure / wait_for_pressure_release
- [x] 压力检测 v2: 移植同事基线+增量双阈值

### 打招呼 Phase 0
- [x] Phase0Server 类: 4 步单连接协议
- [x] YOLOv8n 人检测 (独立 Camera pipeline)
- [x] 挥手: Hello_1↔Hello_2 ×3 → RESET (do_gesture 一步到位)
- [x] 握手: RESET→压力→HAND_SHAKE→松手→RESET
- [x] 压力检测配置化: 全部阈值从 estun_config.yaml 读取
- [x] CRP greeting 段: 全 MovL, speed/label 对齐
- [x] CRP orchestrator 段:
  - Phase 0 speed/label 对齐 greeting CRP
  - H_P3/H_P4→sudu, C_P3→sudu+MovJ→MovL
  - Phase D 5 点位修正 (GH_ZHAOSHOU1→C_P4/Home/C_P1/C_P2/C_P2)
  - ABORT C_P1→sudu, 其余 V100
- [x] `state_machine.py` GreetingHandler: `_execute_wave(ctx)` do_gesture + stop_event + 压力配置化
- [x] `estun_greeting_handle_cup_run.py`: 同回归 do_gesture
- [x] `estun_config.yaml`: wave_speed 清除, wave_interval=0.15
- [x] Phase 0 协议 PC 端验证通过
- [ ] **实机测试**: greeting 单相机 + 状态机双相机

### 通信协议
- [x] pack_three / Orchestrator CRP / 同事 CRP
- [x] READY/CONTINUE/ABORT 协议 + TranStrToCpos 括号约定
- [x] RELEASE 双语义修复: hand_in_cup_active 标志区分 Phase C 松把手 vs Phase D 判定

### 双相机体系
- [x] Serial 绑定: Camera A (342222071364, RealSense_R) + Camera B (819612071851, RealSense)
- [x] EyeToHand: PersonDetector + HandInCup + 3D→1D 标定工具
- [x] Orchestrator 3 状态状态机: IDLE → GREETING → HANDLE_GRASP_AND_POUR → IDLE
- [x] CRP/TXT/状态机 TCP 协议一致性验证 + 修复

### HandInCup MediaPipe 重构
- [x] YOLOv8n-Pose → MediaPipe Hands (21 landmarks, 帧间跟踪)
- [x] landmarks-in-bbox 计分制: wrist(×2) + MCP(×1) + TIP(×1) ≥ 2, bbox_margin=20px
- [x] curled gate: wrist在bbox内→跳过; wrist不在→curled≥2 (防开掌划过)
- [x] 深度: YOLO→画面采样; fallback→cup_depth_expected 预标定值
- [x] 杯 ROI 回退: YOLO 失败 → cup_roi_fallback; --calibrate-roi 标定工具 (含深度)
- [x] 递送位姿 P3 已填入 config
- [x] 独立测试通过 (hand_in_cup.py)
- [x] 状态机联调: 统一共享实例 + reset() 修复 + draw_visualization() 集成

### 人就位检测 (Camera B)
- [x] `calibrate_chair()`: YOLO chair class=56 → 中心偏好 + 稳定窗口
- [x] `detect_person_seated()`: person∩chair/chair_area ≥ 0.4 → 累积10帧
- [x] `EyeToHand/config.yaml`: 新增 `chair` 配置段
- [x] 独立测试模式 `run()`: 椅子标定 + 就位状态可视化
- [x] 实机测试: 椅子标定 + 人就位判定效果 ✅ (2026-07-10)

### EyeToHand 标定
- [x] calibrate.py 重写: 3D→1D 人体标定 (SVD)
- [x] result.yaml: {a,b,c,d} + detector.py: person_to_base_x()
- [ ] 暂未接入 (递送 X 用 delivery.X_TEACH)

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 greeting 单相机测试 | `estun_greeting_handle_cup_run.py` + `greeting_handlepluscup.crp` |
| 🔴 状态机双相机测试 | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 EyeToHand 标定实操 | 3D→1D 人体标定 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- EyeToHand result.yaml a,b,c,d 默认占位
- P4 Y 后退 15/20° 线性插值未实测验证
- 机械臂端 MovJ widget ZONE datavar 无法正确解析 → 已全部改为 MovL
- greeting + 状态机 代码改动完成, 待实机验证
