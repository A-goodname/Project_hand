# Progress — Project_hand

### 主流程优化 + 语音控制修复 (2026-07-17)
- [x] P3/HandInCup 超时双向延长: CRP SocketReadStr 15s→20s + Python 10s→25s (config-driven)
- [x] 手动 CONTINUE 按键: 空格/'c' 最高权限覆盖, Phase D 随时可用
- [x] 人就位简化: `only_person_detect: true` → `_run_person_check()` 纯人检测, 10帧稳定, 确认后退出YOLO
- [x] Camera B 自动探测: `camera_id: -1` → `UGreenCamera` 按 `camera_name` 自动搜索 (pygrabber)
- [x] `PersonDetector.detect_person_only()` — 仅 YOLO class 0
- [x] `EyeToHand/camera.py` 新增 `find_camera_id()` — DSHOW 设备名→索引
- [x] 语音控制防误触: `match_command()` 关键词占比检查 (text>6字 & 占比<30% → 拒绝)
- [x] 实机测试: 4 阶段流程通过 ✅ (2026-07-17)

### 语音控制基础设施 (v14, 2026-07-16)
- [x] ASR 选型: sherpa-onnx SenseVoice int8 (227MB, 零 PyTorch)
- [x] funasr 废弃: torch 2.12 输出乱码, 已卸载 + 清缓存
- [x] `Omnihand/voice_omnihand_control.py` 完成 (22手势+序列+速度)
- [x] VAD 能量阈值方案测试通过 ("嗯"/"你好" 正确识别)
- [x] `doc/voice_control_design.md` 完整方案文档
- [x] voice_omnihand_control 单手势测试通过 ✅ (2026-07-17)
- [ ] voice_omnihand_control 序列+速度 测试
- [ ] 完整语音控制接入 Orchestrator

### 人就位 GPU 优化 + Cup P3 提速 (v13, 2026-07-16)
- [x] PersonDetector GPU 化: `device: cpu → gpu`, 合并 person+chair 双推理为单次
- [x] `leave_frames: 30 → 45` (配合 GPU 高帧率 ~30FPS)
- [x] CRP 新增 sudu2 变量 (sper=55, stcp=150, sori=65)
- [x] Cup 递送 4 处 sudu→sudu2: C_P5↔C_P3 + ABORT C_P5→C_P1
- [x] Skill 更新: CRP 3 步分析 + 编码陷阱 + 变量多容器遍历
- [x] CLAUDE.md 全局: 搜索与推理纪律 (第 8 节)
- [x] 备份统一 `.bak` 后缀 (匹配 .gitignore)

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
| 🔴 实机验证 v15 改动 | 超时同步 + 按键覆盖 + 人就位简化 + Camera B 自动探测 |
| 🟡 HandInCup 继续优化 | landmarks-in-bbox 调参，遮挡场景未覆盖 |
| 🟡 voice_omnihand_control 序列+速度 实机测 | 连灵巧手测试语音手势流程 |
| 🟡 加水实测 | 完整流程加水倒水, 验证防溅 + HandInCup 判定 |
| 🟡 抓取姿态微调 | 末端抓取姿态 + 灵巧手手势微调优化抓取成功率 |
| 🟢 实机验证 sudu2 | CRP 导入示教器, 验证 sudu2 速度合理性 |
| 🟢 完整流程稳定性测试 | 多次连续运行 |
| 🟢 完整语音控制接入 | voice_control.py + 决策点 CRP |
| 🟢 MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 日志泄漏 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯 RGB 无深度 — 3D→1D 人体标定不可行 (已废弃, P3 X 全示教)
- EyeToHand result.yaml 标定已废弃 (Camera B 无深度无法获得 Z)
- MediaPipe clearcut shutdown ERROR 无法完全抑制
