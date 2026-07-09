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
- [x] Cup offset fine_tune: back=[0.048,-0.009,0.098], left=[0.040,-0.060,0.080]

### Handle+Cup 融合
- [x] estun_handle_cup_run.py 全流程
- [x] back/left 双模式 + P3 多角度示教 + P4 Y后退 CRP
- [x] Cup 检测修复 (PoseStabilizer 代际同步)

### 灵巧手控制
- [x] USB 串口 / 手势 / 速度控制 / 使能
- [x] 手势扩展: HAND_SHAKE(19) / HELLO_1(20) / HELLO_2(21)
- [x] 压力检测 v1: get_pressure / wait_for_pressure / wait_for_pressure_release
- [x] 压力检测 v2: 移植同事基线+增量双阈值

### 打招呼 Phase 0 (未提交)
- [x] Phase0Server 类: 4 步单连接协议 (替代旧的 7 步每步 Accept)
- [x] YOLOv8n 人检测 (Camera A, class=0)
- [x] 挥手: Hello_1↔Hello_2 ×3 (do_gesture)
- [x] 握手: PAPER→等压力→HAND_SHAKE→松手→PAPER
- [x] CRP greeting 段重建: 30 widget 单 Socket 持久连接 (全部 MovL)
- [x] Phase A-C 完整保留 (与 estun_handle_cup_run.py 对齐)
- [ ] CRP+PC 联调实测

### 通信协议
- [x] pack_three / Orchestrator CRP / 同事 CRP

### 双相机体系
- [x] Serial 绑定: Camera A (342222071364, RealSense_R) + Camera B (819612071851, RealSense)
- [x] EyeToHand: PersonDetector + HandInCup + 3D→1D 标定工具
- [x] Orchestrator 状态机 + HandInCup 无遮挡实测

### EyeToHand 标定
- [x] calibrate.py 重写: 3D→1D 人体标定 (SVD)
- [x] result.yaml: {a,b,c,d} + detector.py: person_to_base_x()
- [ ] 暂未接入 (递送 X 用 delivery.X_TEACH)

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 打招呼联调 | CRP + PC 全流程实测 |
| 🔴 HandInCup 有遮挡实测 | 灵巧手遮挡杯把 |
| 🟡 EyeToHand 标定实操 | 3D→1D 人体标定 |
| 🟡 递送参数示教 | delivery 全字段 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- EyeToHand result.yaml a,b,c,d 默认占位
- HandInCup 有遮挡未测试
- P4 Y 后退 15/20° 线性插值未实测验证
- 新打招呼 CRP 待联调验证
