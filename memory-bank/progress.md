# Progress — Project_hand

## 已实现功能

### Orchestrator 4 阶段调度
- [x] greeting → handle_detect → seated_check → hand_in_cup 线性流程
- [x] 删除 StateMachine/StateHandler 状态机框架
- [x] SharedContext 简化 + Event-based phase 过渡
- [x] Camera B 在 handle_detect 阶段暂停 (零 YOLO 竞争)
- [x] 删除跳帧补丁 (_pd_skip, _hic_skip)
- [x] _camera_a_to_close 修复 (Phase B 后 main 线程安全关闭)

### Phase D 联调
- [x] CONTINUE 分支全流程测试通过 ✅
- [x] ABORT 分支全流程测试通过 ✅
- [x] CRP CposOffset pointa/pointb 交换修复
- [x] CONTINUE/ABORT 信号发送可达位姿 (Phase A 机器人初始位姿)
- [x] SocketReadStr 超时 5s→15s
- [x] ABORT P3↔P4 / 倒水 H_P3↔H_P4 使用 sudu 慢速

### 双相机体系
- [x] Camera A (342222071364, RealSense_R) + Camera B (UGREEN 1080P, OpenCV)
- [x] Camera A pipeline 全生命周期共享 (GREETING→HANDLE, 只开一次)
- [x] Camera B 预加载 (GREETING 启动, 椅子+人就位全程运行)
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
- [x] 白色把手找不到时不回退到壶身搜红色

### Phase B fixed 模式
- [x] compute_grasp_rotation 仅非 fixed 分支调用
- [x] 稳定器+标定在 ctx.handle_cup_detector 赋值前创建 (竞态修复)
- [x] Pipeline 不在 daemon 线程 close (对齐 estun_greeting)
- [x] Handle P2 可达性修复 (pre_offset.t_offset: `[0,0.10,-0.05]`)

### 打招呼 Phase 0
- [x] GreetingHandler 4 步单连接协议
- [x] Camera A 人检测复用 HandleCupDetector pipeline
- [x] CRP orchestrator 段完整
- [x] 实机测试: Phase 0 通过 ✅

### 通信协议
- [x] pack_three / READY/CONTINUE/ABORT
- [x] RELEASE 双语义 (hand_on_cup is None → Phase C, is not None → Phase D)
- [x] Phase D 判定逻辑

### CRP v11 优化
- [x] 删 PREGRIP: P2→P1→GRIPDONE 直接夹紧, RESET 移到 P1
- [x] while 循环删 Wait 500ms (防丢包)
- [x] 倒水延时 5s→3s
- [x] DCp2.dz 固定 100mm (C_P4 安全退避)
- [x] C_P5 安全高度点 (Z+10cm 防碰撞)

### HandInCup 重构+调通
- [x] YOLOv8n + MediaPipe Hands (21 landmarks)
- [x] landmarks-in-bbox 计分制 + curled gate
- [x] 纯 RGB 判定, cup_roi_fallback 回退
- [x] device 加入 config (`hand_in_cup.device: cuda`)
- [x] 超时 5s→10s, 诊断日志
- [x] 独立测试通过 ✅

### 人就位检测 (Camera B)
- [x] 椅子实时检测 (每帧 YOLO, 中心偏好)
- [x] 非对称阈值 (进10/出5)
- [x] has_chair 门控修复 (去掉鸡生蛋逻辑)

### 日志系统
- [x] Orchestrator/logger.py — 终端+文件双输出
- [x] state_machine 关键事件日志 (Phase A/B/C/D)

### 修复
- [x] MediaPipe clearcut 遥测增强 (GLOG_stderrthreshold=4, GRPC_VERBOSITY=NONE)
- [x] Camera B D435i serial 死代码删除
- [x] CRP data-as-string 修复 (socketclose inside if blocks)
- [x] Skill 更新 (CRP 修改规则 + 陷阱 + Phase D 协议 + 4 阶段架构)

## 待完成

| 任务 | 说明 |
|------|------|
| 🟢 完整流程稳定性测试 | 多次连续运行 |
| 🟢 MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 日志泄漏 |
| 🟢 EyeToHand 标定接入 | 3D→1D 人体标定 (暂用 delivery.X_TEACH) |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯 RGB, 无深度数据
- EyeToHand result.yaml 标定暂未接入
- MediaPipe clearcut shutdown ERROR 无法完全抑制
