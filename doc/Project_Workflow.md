# 机械臂+灵巧手抓取系统 — 项目流程文档

## 上次更新: 2026-07-08

---

## 1. 系统概述

双 RealSense D435i 相机 + ESTUN 机械臂 + OmniHand O10 灵巧手协同工作的自动化抓取系统。

**核心任务**: 检测人进门 → 打招呼握手 → 抓水壶倒水 → 抓水杯递送 → HandInCup 判定松手。

### 1.1 硬件清单

| 设备 | 型号/标识 | 角色 |
|------|----------|------|
| Camera A | RealSense D435i, serial `342222071364` | **眼在手上**, 装在机械臂末端法兰 |
| Camera B | RealSense D435i, serial `819612071851` | **眼在手外**, 固定安装 (上方/侧方) |
| 机械臂 | ESTUN | TCP 通信, 执行 .crp 程序 |
| 灵巧手 | OmniHand O10 | USB 串口 (460800 8N1), 10 主动自由度, 7 触觉传感器 |

### 1.2 软件模块

```
Project_hand/
├── Orchestrator/          ← 系统入口, 双相机状态机
│   ├── main.py            ← 主循环 + Person/HandInCup 切换
│   └── state_machine.py   ← 2 状态状态机 (IDLE ↔ HANDLE_GRASP_AND_POUR)
├── RealSense_R/           ← Camera A 检测 (手动像素映射, 内参损坏)
│   ├── test_handle_6D.py  ← 水壶把手 6D 检测
│   ├── test_cup_6D.py     ← 水杯 6D 检测
│   └── realsense_shared.py← BaseRealSenseDetector 基类
├── RealSense/             ← Camera B 基类 (rs.align, 内参完好)
│   └── realsense_shared.py← BaseRealSenseDetector 基类
├── EyeToHand/             ← 眼在手外模块
│   ├── person_detector.py ← 人体位置检测 (YOLOv8m COCO person)
│   ├── hand_in_cup.py     ← 人手-杯把手重合检测 (YOLOv8m + YOLOv8n-Pose)
│   ├── detector.py        ← load_calibration / person_to_base_x
│   ├── config.yaml        ← Person + Delivery(全示教) + HandInCup 配置
│   └── calibration/
│       ├── calibrate.py   ← 3D→1D 人体标定工具
│       └── result.yaml    ← a,b,c,d 标定参数
├── Estun/                 ← 机器人通信 + 灵巧手控制
│   ├── estun_config.yaml  ← 主配置 (IP/端口/偏移/模式/P3示教)
│   ├── utils_shared.py    ← 坐标变换 + 灵巧手控制函数
│   └── estun_handle_cup_demo.py ← 位姿转换验证 (无硬件)
├── Omnihand/              ← 灵巧手 USB 串口模块
│   ├── omnihand_usb_serial.py  ← 核心通信库 (20+ 命令码)
│   ├── omnihand_gesture.py     ← 手势控制 + 压力检测
│   ├── omnihand_speed_control.py ← 软件插值速度控制
│   └── actlist.json       ← 上位机导出的手势编码器位置
├── calib_data/            ← 眼在手上标定结果
│   └── calibration_result.yaml ← T_cam_ee (相机→末端法兰)
└── doc/                   ← 文档 + .crp 程序文件
```

---

## 2. 双相机体系

### 2.1 Camera A — 眼在手上 (Eye-in-Hand)

| 属性 | 值 |
|------|-----|
| 型号 | Intel RealSense D435i |
| Serial | `342222071364` |
| 内参状态 | **损坏 (NaN)** |
| 深度对齐方式 | `rs2_project_color_pixel_to_depth_pixel` 手动像素映射 |
| 代码模块 | `RealSense_R/` |
| 内参文件 | `RealSense_R/D435i_info.yaml` |
| 检测配置 | `RealSense_R/config.yaml` |

**检测功能**:
- 水壶把手 6D 位姿: 微调 YOLOv8n + SAM2 + 扫描线分离 → PCA + PointNet++
- 水杯 6D 位姿: 标准 YOLOv8m COCO cup (class_id=41) → PCA + PointNet++
- 输出: Handle P1/P2/P3, Cup P1/P2/P3 (6D 位姿, 末端坐标系)

**初始化流程**:
```python
config = rs.config()
config.enable_device("342222071364")  # 绑定指定 serial
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
# ⚠️ 不使用 rs.align — 内参损坏, 需手动像素映射
```

**关键参数**: 相机内参从 `RealSense_R/D435i_info.yaml` 读取; 支持三种互斥检测模式 (标准 YOLOv8 / YOLO-World / 微调 YOLOv8n); 把手检测依赖 SAM2 (`sam2_b.pt`); 红色胶带辅助检测通过 HSV 双阈值实现.

### 2.2 Camera B — 眼在手外 (Eye-to-Hand)

| 属性 | 值 |
|------|-----|
| 型号 | Intel RealSense D435i |
| Serial | `819612071851` |
| 内参状态 | **完好** |
| 深度对齐方式 | `rs.align(rs.stream.color)` |
| 代码模块 | `RealSense/` (基类) + `EyeToHand/` (检测逻辑) |
| 内参文件 | `RealSense/D435i_info.yaml` / `EyeToHand/D435i_info.yaml` |
| 检测配置 | `EyeToHand/config.yaml` |

**检测功能**:
- 人体位置: YOLOv8m COCO person (class_id=0) → bbox 中心深度 → 相机系 3D → T_cam_base → 基系 X 坐标
- HandInCup: YOLOv8m (杯 bbox class=41) + YOLOv8n-Pose (手腕关键点 9/10) → 2D+深度判定
- 人体稳定: PersonStabilizer (window=10, pos_thresh=0.03m)

**初始化流程**:
```python
config = rs.config()
config.enable_device("819612071851")  # 绑定指定 serial
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
pipeline = rs.pipeline()
profile = pipeline.start(config)
# 使用 rs.align 自动对齐 — 内参完好的新相机
align = rs.align(rs.stream.color)
```

**PersonDetector / HandInCup 基类**: 继承 `RealSense/BaseRealSenseDetector`, 使用 `rs.align` 深度对齐.

### 2.3 双相机 Serial 绑定

两台同型号 D435i 插在同一台 PC, 必须通过 `config.enable_device(serial)` 区分:

```python
# Camera A: RealSense_R/realsense_shared.py
config.enable_device("342222071364")

# Camera B: RealSense/realsense_shared.py
config.enable_device("819612071851")
```

Serial 从各自的 `D435i_info.yaml` 读取, 由 `load_camera_serial()` 工具函数加载.

---

## 3. 完整工作流程

### 3.0 阶段 0 — 打招呼 (Greeting) [新增, 待实现]

```
状态: 双相机启动, 机械臂在 Home_1 (观察位姿)

Camera B 检测到人进门
  → 机械臂移动到打招呼位姿 (预示教)
  → 灵巧手: Hello_1 → Hello_2 挥手 ×2~3 次
  → 机械臂移动到握手位姿 (预示教)
  → 灵巧手: Paper (伸手)
  → 等待手背压力 > 100 → Handshake (握住)
  → 保持 2s
  → 等待手背压力 < 20 → Paper (松手)
  → 机械臂回 Home_2 (抓取就绪位姿)
  → 进入阶段 A

实现分工:
  - 机械臂运动 (.crp) 由同事完成
  - PC 侧: 手势控制 + 压力检测, 通过协议信号与 .crp 交互
  - Camera B 持续检测人是否在椅子上 → 触发后续流程
```

**灵巧手接口 (已就绪)**:
```python
from Omnihand.omnihand_gesture import OmniHandGesture, Gesture

hand = OmniHandGesture(port="COM5")
hand.do_gesture(Gesture.HELLO_1, delay=0.6)     # 挥手阶段1
hand.do_gesture(Gesture.HELLO_2, delay=0.6)     # 挥手阶段2
hand.do_gesture(Gesture.PAPER, delay=0.3)        # 伸手
hand.wait_for_pressure(threshold=100,             # 等待抓握 (手背)
                       sensor_filter=["back"])
hand.do_gesture(Gesture.HAND_SHAKE, delay=2.0)   # 握住
hand.wait_for_pressure_release(threshold=20,      # 等待松手
                               sensor_filter=["back"])
```

### 3.1 阶段 A — 初始位姿接收

```
机械臂运动到抓取就绪位姿
  → Robot 通过 TCP 发送当前末端位姿 [x,y,z,A,B,C]
  → PC 接收并记录 robot_xyz, robot_ABC
  → 用于后续位姿转换 (末端系 → 基系)
  → 5s 超时机制
```

### 3.2 阶段 B — Handle 抓取 + 倒水

```
Camera A (眼在手上) 双检测器共享 pipeline:

  Handle 检测 (微调 YOLOv8n + SAM2 + 扫描线):
    1. YOLOv8n 检测水壶 bbox
    2. SAM2 分割水壶 mask
    3. 水平扫描线分离把手 (按 handle_side 取最外段)
    4. (可选) 红色胶带辅助定位把手 3D 位置
    5. 深度投影 → 把手 3D 点云
    6. PCA + PointNet++ → 把手 6D 位姿 (相机系)
    7. PoseStabilizer 多帧稳定
    8. cam→ee (手眼标定 T_cam_ee)

  Cup 检测 (YOLOv8m, start_pipeline=False, inject_frames):
    1. YOLOv8m 检测水杯 bbox (COCO class 41)
    2. 深度投影 → 杯体 3D 点云
    3. PCA → 杯主轴方向 (指数平滑 AXIS_SMOOTH_ALPHA=0.7)
    4. 杯口定位 (像素 Y 坐标判定, Y 小=杯口)

  发送位姿 (TCP pack_three 格式):
    <Handle_P1>[Handle_P2]{Handle_P3}
    P1: 抓取点 (C 角 clamp [-100,-80], 手指垂直把手)
    P2: 预抓取点 (P1 + pre_offset, grasp ee frame 下偏移)
    P3: 倒水目标 (由 PC 根据 P3_teach_offset 计算)

机械臂端 CRP 执行:
  MoveP2 → PREGRIP → MoveP1 → GRIPDONE
  → MoveP4(预倒水安全点) → MoveP3(B轴旋转倒水) → wait 1s
  → MoveP4 → MoveP1 → PREGRIP → MoveP2 → RESET → MoveHome
```

**P3 倒水多角度示教**: `estun_config.yaml` 中 `p3_teach_offset.{side}` 按 B 角度数 key (10/15/20/30) 索引, `pour_angle` 控制当前使用角度.

**P4 预倒水安全点**: P4 = [P3.x, P3.y - Y_RETREAT, P3.z, P1.a, P1.b, P1.c] (Y 后退), 倒水为纯 B 轴旋转. Y_RETREAT 按角度映射: 10°→200mm, 15°→175mm, 20°→150mm, 30°→100mm.

### 3.3 阶段 C — Cup 抓取 + 指令循环

```
TCP 指令循环:
  PREGRIP → GRIPDONE → RESET → CUP → RELEASE → GRIP

CUP 指令触发:
  1. Cup P3 递送位姿 XYZ/ABC 全部来自 EyeToHand/config.yaml delivery 段 (全示教固定值)
     (标定暂废弃, X/Y/Z/A/B/C 均由示教器确定)
  2. PC 回复: <Cup_P1>[Cup_P2]{Cup_P3}
     P1: 抓取点 (back/left 分模式)
     P2: 预抓取点
     P3: 递送点 (全示教固定)
```

### 3.4 阶段 D — HandInCup 判定 + 递送/放回

```
机械臂到达 P3 → Robot 发送 "READY"
  → PC 等待 2s (期间 HandInCup 检测运行)
  → RELEASE 触发:
      PC 判定 hand_on_cup:
        TRUE  → CONTINUE → release_object() → P4 退避 → Home_2
        FALSE → ABORT    → 放回 P1 → P2 退避 → Home_2

HandInCup 检测逻辑:
  1. YOLOv8m 检测杯 bbox (COCO class 41)
  2. YOLOv8n-Pose 检测手腕关键点 (kpt 9=左腕, 10=右腕)
  3. 判定: 连续 5 帧手腕在杯 bbox 扩展区内 → True
  4. 距离阈值 = bbox 对角线 × dist_factor (当前 1.2)
  5. (可选) 深度一致性: |手腕深度 - 杯心深度| < depth_tolerance (0.08m)
```

---

## 4. 坐标变换链路

### 4.1 完整变换链

```
Camera A (眼在手上):
  相机检测 → P_obj_cam
    → 手眼标定 T_cam_ee
    → P_obj_ee
    → 工具偏移补偿 (t_tool_ee 旋转到 C=-90°方向)
    → P_grasp_ee
    → T_ee_base (机械臂当前位置)
    → P_grasp_base (TCP 发送)

Camera B (眼在手外):
  相机检测 → bbox 中心深度 → person_pos_cam (相机系 3D)
    → 人体检测触发信号 (检测到人 → 启动抓取流程)
    → 递送位姿 XYZ/ABC 全示教固定 (暂废弃标定)
```

### 4.2 物体坐标系约定

PCA 输出后处理:
- Z 轴强制指向相机 (target_z[2] > 0 时翻转) — `test_6D.py` 内部
- Y 轴统一向下 (右手抓取) 或向上 (左手抓取) — `compute_grasp_rotation` 内部

### 4.3 工具偏移

灵巧手实际安装角度 C=-90° (掌心向下). 标准偏移 `t_offset_std` 旋转到 C=-90° 方向:
```
t_effective = R_z(-90°) @ t_offset_std
```

当前实测偏移值 (`estun_config.yaml`):
| 偏移 | RealSense_R 值 | 说明 |
|------|:---:|------|
| t_handle_offset | [0.048, -0.037, 0.074] | 把手抓取, Camera A |
| t_cup_offset (back) | [0.048, -0.009, 0.098] | 水杯 back 模式 |
| t_cup_offset_left | [0.040, -0.060, 0.080] | 水杯 left 模式 |

---

## 5. 通信协议

### 5.1 机器人 ↔ PC (TCP)

| 方向 | 格式/指令 | 说明 |
|------|----------|------|
| Robot→PC | `x,y,z,A,B,C` | 当前末端位姿 (米, 度) |
| PC→Robot | `<P1>[P2]{P3}` | pack_three 三组位姿 |
| Robot→PC | `PREGRIP` | 执行预抓取 |
| Robot→PC | `GRIPDONE` | 执行完全夹紧 |
| Robot→PC | `RESET` | 执行复位 |
| Robot→PC | `CUP` | 请求 Cup 位姿 |
| Robot→PC | `RELEASE` | 松手前判定 HandInCup |
| Robot→PC | `GRIP` | 执行抓取 |
| Robot→PC | `READY` | 到达 P3, 等待 HandInCup |
| PC→Robot | `CONTINUE` | HandInCup=True, 退避 |
| PC→Robot | `ABORT` | HandInCup=False, 放回 |

### 5.2 灵巧手 ↔ PC (USB 串口)

- 协议: 460800 8N1, 二进制帧
- 帧格式: HEADER(EE AA) + NODE_ID(2B LE) + LEN(1B) + CMD(1B) + DATA(0~62B) + CRC16(2B LE)
- CRC16-CCITT: poly=0x1021, init=0x0000, MSB-first
- 命令码: 见 `Omnihand/omnihand_usb_serial.py` 常量定义

---

## 6. 关键配置

### 6.1 estun_config.yaml (机器人 + 灵巧手)

```yaml
estun:
  host: "192.168.1.100"        # PC 本机 IP
  port: 30000                  # TCP 端口
  orientation_mode: "vertical" # free|vertical|horizontal|fixed
  grasp_direction: "right"     # right|left
  # 瓶身偏移
  gripper_offset:
    t_offset: [0.06, -0.04, 0.07]
    r_offset: [0, 0, -90]
  # 把手偏移
  handle_grasp:
    gripper_offset:
      t_handle_offset: [0.048, -0.037, 0.074]
      r_handle_offset: [0, 0, -90]
    pre_offset: [0, 0, 0.05]
    target_offset: [0, 0, 0.15]
    # P3 倒水示教 (按角度 + 模式索引)
    p3_teach_offset:
      back:
        10: { t_offset: [0.15, -0.30, 0.05], r_abc: [180, 40, -90] }
        15: { ... }
        20: { ... }
        30: { ... }
      left: { ... }
    pour_angle: 10             # 当前使用角度
    fine_tune_mode: true
  # 水杯偏移
  cup_grasp:
    gripper_offset:
      t_cup_offset: [0.048, -0.009, 0.098]       # back 模式
      t_cup_offset_left: [0.040, -0.060, 0.080]  # left 模式
      r_cup_offset: [0, 0, -90]
    pre_offset: [0, 0, 0.05]
    target_offset: [0, 0, 0.15]
  # 灵巧手
  hand:
    port: "COM5"
  # 位姿稳定
  stabilizer:
    window: 5
    pos_thresh: 0.005
    angle_thresh: 5.0
```

### 6.2 RealSense/config.yaml (Camera A 检测)

```yaml
# 水壶模型
kettle_model:
  model: "model/kettle_yolov8n_finetune.pt"
# 把手检测
handle:
  skip_frames: 90
  handle_side: "right"        # right|left
  roi_expand_ratio: 0.2
  center_deviation_ratio: 0.3
# 水杯检测
cup:
  height: 0.100               # 杯高 (m)
  rim_radius: 0.035           # 杯口半径 (m)
  rim_offset_ratio: 0.15
  handle_side: "back"         # back|left
# 红色胶带辅助
red_tape:
  enabled: false
  min_area: 100
# SAM
sam:
  model: "model/sam2_b.pt"
```

### 6.3 EyeToHand/config.yaml (Camera B 检测)

```yaml
model:
  name: "./model/yolov8m"
  target_class_id: 0           # COCO person
stabilizer:
  window: 10
  pos_thresh: 0.03             # 3cm 稳定阈值
delivery:
  X_TEACH: 0.0                 # [必填] 递送 X
  Y_TEACH: 0.0                 # [必填] 递送 Y
  Z_TEACH: 0.0                 # [必填] 递送 Z
  A_TEACH: 0.0                 # 递送 A
  B_TEACH: 0.0                 # 递送 B
  C_TEACH: 0.0                 # 递送 C
# 注: 标定暂废弃, 递送位姿全部由示教确定
hand_in_cup:
  cup_model: "model/yolov8m.pt"
  pose_model: "model/yolov8n-pose.pt"
  required_stable: 5
  dist_factor: 1.2             # 手腕-杯心距离因子
  depth_tolerance: 0.08        # 深度容差 (m)
```

---

## 7. 标定流程

### 7.1 眼在手上标定 (Camera A)

**工具**: `calib_data/` 目录

1. 采集: 机械臂移动到多组位姿, 拍摄棋盘格, 记录 `poses.csv`
2. 标定: 运行标定脚本, 求解 Tsai/Park/Horaud/Andreff/Daniilidis 五种方法
3. 选最优: 用 Board-in-base 一致性评分, 输出 `calibration_result.yaml`
4. 文件:
   - `calib_data/calib_config.yaml` — 棋盘格参数
   - `calib_data/calibration_result.yaml` — 含 T_cam_ee 4x4 矩阵
   - `calib_data/board_in_base.yaml` — 仅最佳方法的标定板位姿

### 7.2 眼在手外标定 (Camera B)

**工具**: `EyeToHand/calibration/calibrate.py` — 3D→1D 人体标定

**原理**: YZ/ABC 示教固定, 只需标定 person_3d_cam → X_base.
拟合模型: `X_base = a·x_cam + b·y_cam + c·z_cam + d`

```
采集:
  1. 人站在桌边位置 A (如左边)
  2. 机械臂示教 Cup P3 递送位姿, 记录 X_base
  3. 按 'c' → 终端输入 X_base → 自动配对 person_3d_cam
  4. 换位置 B, C... 重复 ≥3 次 (建议 3-5 点)

拟合 (SVD 最小二乘):
  X_base = a·x + b·y + c·z + d

运行时:
  person_3d = PersonDetector 检测
  X_person = a·x + b·y + c·z + d
```

**操作**: 运行 `python EyeToHand/calibration/calibrate.py` → 'c' 采集 → 'f' 拟合 → 's' 保存.
**注意**: 当前递送 X 使用 `delivery.X_TEACH` 示教固定值, 标定暂未接入 Orchestrator.

---

## 8. 模型文件

| 模型 | 路径 | 大小 | 用途 |
|------|------|------|------|
| YOLOv8n | `model/yolov8n.pt` | 6.5 MB | 水壶检测 (微调基模型) |
| YOLOv8m | `model/yolov8m.pt` | 52 MB | 杯检测 (COCO class 41), 人体检测 |
| YOLOv8n-Pose | `model/yolov8n-pose.pt` | 6.8 MB | 手腕关键点 (HandInCup) |
| YOLOv8m-Pose | `model/yolov8m-pose.pt` | 53 MB | 备用 (HandInCup 高精度) |
| 微调水壶 | `model/kettle_yolov8n_finetune.pt` | 6.2 MB | 水壶检测 (Camera A) |
| SAM2 | `model/sam2_b.pt` | 162 MB | 把手分割 |
| YOLO-World v2 | `model/yolov8s-worldv2.pt` | 26 MB | 零样本检测 (备用) |

---

## 9. 状态机 (Orchestrator)

```
       ┌─────────────────────────────────────────────┐
       │                                             │
       ▼                                             │
     IDLE ──(Camera B 检测到人)──→ HANDLE_GRASP_AND_POUR
       ▲                              │
       │                   单 TCP Session 全流程:
       │                   Phase A: 收初始位姿
       │                   Phase B: 发 Handle <P1>[P2]{P3}
       │                   Phase C: PREGRIP/GRIPDONE/RESET/CUP
       │                           发 Cup <P1>[P2]{P3}
       │                   Phase D: READY/RELEASE/HandInCup
       │                            CONTINUE or ABORT
       │                              │
       └──────────────────────────────┘
                      (SocketClose)
```

**状态转换**:
- IDLE → HANDLE_GRASP_AND_POUR: PersonDetector 连续检测到人
- HANDLE_GRASP_AND_POUR → IDLE: TCP SocketClose (正常结束或异常)

---

## 10. 已知问题 & 待完成

| 状态 | 问题 |
|:---:|------|
| ⚠️ | **递送参数未填入**: `delivery.X_TEACH/Y_TEACH/Z_TEACH/A_TEACH/B_TEACH/C_TEACH` 需示教 |
| ℹ️ | **EyeToHand 标定暂废弃**: 递送 XYZ/ABC 全部改用示教固定值, 不再动态检测 X |
| ⚠️ | **HandInCup 有遮挡未实测**: 灵巧手握住杯子后遮挡手腕+杯 bbox |
| ⚠️ | **打招呼 .crp 接口未对接**: 等同事完成 .crp 后接入 PC 端手势+压力检测 |
| ⚠️ | **P4 Y 后退 15°/20° 为线性插值**: 待实测验证 |
| ℹ️ | RealSense_R 旧相机的 rs.align 不可用 (硬件内参损坏) |
| ℹ️ | `estun_handle_run.py` 通信协议未同步 handle-cup 版 v2 改进 (暂缓) |
