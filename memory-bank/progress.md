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
- [x] **CRP 一致性修复** (2026-07-14): orchestrator CRP 补全 SocketCreate(8080)/If/GetCurCPos + Phase D socket Socket1→Socket0
- [x] **TXT 伪代码同步** (2026-07-14): v8 changelog + Phase 0 段 + 双 Socket 架构 + socket 引用统一
- [x] **CRP Phase D 逻辑修复** (2026-07-14): strflag_cont/abort 变量 + TranStrToCpos 标志分离 + If 条件 == + CONTINUE C_P4 计算 + 标签规范化
- [x] **Python RELEASE 分支修正** (2026-07-14): abort_mode 优先判断 + 人未就位走正常判定 + hand_in_cup_active 生命周期
- [x] **Cup 检测 intrinsics 补全** (2026-07-14): depth_intrinsics / c2d_extr / d2c_extr
- [x] **Camera B CPU 解耦** (2026-07-14): PersonDetector YOLO gpu→cpu
- [x] **退出加速** (2026-07-14): disable_hand skip_reset + HandInCup close daemon 超时

### 双相机体系
- [x] Serial 绑定: Camera A (342222071364, RealSense_R) + Camera B (819612071851, RealSense)
- [x] EyeToHand: PersonDetector + HandInCup + 3D→1D 标定工具
- [x] Orchestrator 3 状态: IDLE → GREETING → HANDLE (单次, exit(0))
- [x] Camera A 按需开闭: GREETING 独立 pipeline, HANDLE Phase B 开→发送后关
- [x] Camera B 全生命周期共享: PersonDetector pipeline, 人就位+HandInCup 复用帧
- [x] 人就位门控 HandInCup: READY 时查 is_person_seated, 未就位→ABORT
- [x] HandInCup ≤5s 超时检测, TRUE 提前结束
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
- [x] 回退 `do_gesture` (移除 KinematicsConverter / wave_speed)
- [ ] **实机测试**: greeting 单相机 + 状态机双相机

### 通信协议
- [x] pack_three / Orchestrator CRP / 同事 CRP
- [x] READY/CONTINUE/ABORT 协议 + TranStrToCpos 括号约定
- [x] RELEASE 双语义修复: hand_in_cup_active 标志区分 Phase C 松把手 vs Phase D 判定

### 双相机体系
- [x] Serial 绑定: Camera A (342222071364, RealSense_R) + Camera B (UGREEN 1080P, OpenCV)
- [x] Camera B D435i → UGREEN 替换: camera.py / person_detector 重写 / hand_in_cup 纯 RGB
- [x] EyeToHand: PersonDetector + HandInCup + 3D→1D 标定工具
- [x] Orchestrator 3 状态: IDLE → GREETING → HANDLE (单次, exit(0))
- [x] Camera A 按需开闭: GREETING 独立 pipeline, HANDLE Phase B 开→发送后关
- [x] Camera B 生命周期: IDLE 关 → GREETING enter 开 → HANDLE 继续 → exit 关
- [x] 人就位门控 HandInCup: READY 时查 is_person_seated, 未就位→ABORT
- [x] HandInCup ≤5s 超时检测, TRUE 提前结束
- [x] CRP/TXT/状态机 TCP 协议一致性验证 + 修复

### HandInCup MediaPipe 重构
- [x] YOLOv8n-Pose → MediaPipe Hands (21 landmarks, 帧间跟踪)
- [x] landmarks-in-bbox 计分制: wrist(×2) + MCP(×1) + TIP(×1) ≥ 2, bbox_margin=20px
- [x] curled gate: wrist在bbox内→跳过; wrist不在→curled≥2 (防开掌划过)
- [x] ~~深度辅助~~ → 纯 RGB 判定 (Camera B UGREEN 替换后移除)
- [x] 杯 ROI 回退: YOLO 失败 → cup_roi_fallback; --calibrate-roi 标定工具
- [x] 递送位姿 P3 已填入 config
- [x] 独立测试通过 (hand_in_cup.py)
- [x] 状态机联调: 统一共享实例 + reset() 修复 + draw_visualization() 集成

### 人就位检测 (Camera B)
- [x] `calibrate_chair()` 废弃 → `_detect_chair_live()` 每帧实时检测 (不锁定)
- [x] `detect_person_seated()`: 非对称阈值 (进10帧/出5帧) + `_chair_baseline` 锚定 (面积差≤30%更新)
- [x] `EyeToHand/config.yaml`: `chair` 段 (overlap_threshold / seated_stable_frames / leave_frames)
- [x] 独立测试模式 `run()`: 椅子实时 + 就位状态可视化
- [x] `is_chair_calibrated` → `has_chair`; `start_camera` 参数/方法

### HandInCup 纯 RGB 精度评估
- [x] 纯 RGB 判定精度验证通过 ✅ (2026-07-14)
- [x] score≥5 绕过 curl 后误判率可接受 ✅ (2026-07-14)

### UGREEN 720p 帧率确认
- [x] YOLO+MediaPipe 在 1280×720 上的推理延迟可接受 ✅ (2026-07-14)

### EyeToHand 标定
- [x] calibrate.py 重写: 3D→1D 人体标定 (SVD)
- [x] result.yaml: {a,b,c,d} + detector.py: person_to_base_x()
- [ ] 暂未接入 (递送 X 用 delivery.X_TEACH)

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 UGREEN Camera B 杯 ROI 标定 | `python EyeToHand/hand_in_cup.py --calibrate-roi` |
| 🔴 双相机全流程测试 | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
| 🟡 EyeToHand 标定实操 | 3D→1D 人体标定 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- ~~Camera B D435i 频繁断联~~ → 已替换为 UGREEN Camera 1080P
- EyeToHand result.yaml a,b,c,d 默认占位
- P4 Y 后退 15/20° 线性插值未实测验证
- 机械臂端 MovJ widget ZONE datavar 无法正确解析 → 已全部改为 MovL
- UGREEN Camera B 纯 RGB，无深度数据 (HandInCup 依赖 curled gate 防误判)
