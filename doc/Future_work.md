# 未来工作规划 — 双相机体系

## 上次更新: 2026-06-30 (v3: 简化递送策略 — 定时松手 + YZ固定X动态)

---

## 1. 硬件布局

```text
两台 Intel RealSense D435i, 通过 USB 连接到同一台 PC:

  Camera A (眼在手上)                     Camera B (眼在手外)
  ───────────────────                     ───────────────────
  Serial: 819612071851                    Serial: 342222071364
  内参: 完好                              内参: 损坏 (NaN)
  位置: 机械臂末端法兰                     位置: 固定安装 (上方/侧方)
  模块: RealSense/ (rs.align)             模块: RealSense_R/ (手动像素映射)
  配置: RealSense/D435i_info.yaml         配置: RealSense_R/D435i_info.yaml
  角色: 检测水壶/水杯 → 抓取+倒水          角色: 检测人/手势 → 递杯+松手
```

---

## 2. 总体架构

```text
┌─────────────────────────────────────────────────────────────────────┐
│                       双相机协作流程                                  │
│                                                                     │
│  ┌──────┐  检测水壶/水杯   ┌──────────┐  抓取把手   ┌──────────┐   │
│  │ 眼在  │ ─────────────→ │ 机械臂 + │ ────────→ │ 倒水动作 │   │
│  │ 手上  │   ←─────────── │ 灵巧手   │ ←──────── │          │   │
│  │(已实  │   6D 位姿反馈   └──────────┘   完成    └──────────┘   │
│  │ 现)   │                                                        │
│  └──────┘                                                        │
│       │                                                          │
│       │ 倒水完成 + 已抓取水杯                                       │
│       ▼                                                          │
│  ┌──────┐  检测人位置    ┌──────────┐  递送水杯   ┌──────────┐   │
│  │ 眼在  │ ────────────→ │ 机械臂   │ ────────→ │ 等待手   │   │
│  │ 手外  │   ←────────── │ (防碰撞) │ ←──────── │ 势判断   │   │
│  │(待实  │  检测手势     └──────────┘   松手/    └──────────┘   │
│  │ 现)   │   判断是否松手              继续握持                  │
│  └──────┘                                                        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 眼在手上 (Eye-in-Hand) — 当前已实现

- 相机: RealSense D435i (安装在机械臂末端), Serial `819612071851`
- 模块: `RealSense/` (rs.align 版本)
- 功能:
  - 检测水壶把手位置, 计算 6D 位姿
  - 控制机械臂 + 灵巧手完成抓取把手 → 倒水动作
  - 检测水杯位置, 抓取水杯
- 入口: `Estun/estun_handle_cup_run.py`
- 状态: ✅ 把手抓取 + 倒水 + 水杯抓取流程已完成 (back 模式已验证)

### 2.2 眼在手外 (Eye-to-Hand) — 待实现

- 相机: RealSense D435i (固定安装), Serial `342222071364`
- 模块: `RealSense_R/` (手动像素映射版本, 内参损坏)
- 功能:
  - 检测人的位置 (人体检测)
  - 识别人的手势, 判断是否松手 (open palm = 松手)
  - 控制机械臂将水杯递送到人手中
  - 作为整个系统的启动触发器 (检测到人 → 开启全流程)
- 状态: ❌ 未开始

---

## 3. 关键改造: Serial 绑定

### 3.1 问题

两台同型号 D435i 插在同一台 PC, 当前代码 `rs.pipeline()` 不指定设备, 取第一个枚举到的相机, 结果不可控.

### 3.2 解决方案

在 `_init_realsense()` 中通过 `config.enable_device(serial)` 绑定指定设备. 两个模块各自从自己的 `D435i_info.yaml` 读取 `camera.serial`:

```text
RealSense/realsense_shared.py   → RealSense/D435i_info.yaml   → serial "819612071851"
RealSense_R/realsense_shared.py → RealSense_R/D435i_info.yaml → serial "342222071364"
```

### 3.3 改动量

两个 `_init_realsense()` 方法各加 3 行 (enable_device 调用), 新增一个 `load_camera_serial()` 工具函数.

librealsense2 API:

```python
config.enable_device("819612071851")   # 绑定指定 serial 的相机
```

---

## 4. 眼在手外模块设计 (hand_to_eye/)

### 4.1 目录结构

```text
Project_hand/
├── hand_to_eye/                    # [新建] 眼在手外模块
│   ├── __init__.py
│   ├── config.yaml                 # 外置相机配置 + 标定参数 + 预示教递送位置
│   ├── detector.py                 # 检测器基类 (继承 RealSense_R 基类)
│   ├── person_detector.py          # 人体位置检测 (YOLOv8 COCO person class)
│   └── calibration/                # 外置相机→基坐标系的标定
│       └── calibrate.py
│
├── orchestrator/                   # [新建] 双相机流程编排
│   ├── __init__.py
│   ├── state_machine.py            # 状态机管理
│   └── main.py                     # 系统总入口
│
└── ... (其余现有文件不变)
```

### 4.2 检测目标

| 目标 | 方法 | 说明 |
| ------ | ------ | ------ |
| 人体位置 | YOLOv8 COCO person class (class_id=0) | bbox 中心深度 → 相机系 3D → T_cam_base → 基系 3D |

### 4.3 递送策略

**手势识别不适用于递杯场景**。人手接水杯时手指被杯身遮挡, MediaPipe Hands 21 landmarks 无法完整识别, open/close 分类不可靠.

替代方案: **固定 2s 定时松手**:

```text
机械臂到达递送位置 → 等待 2s (给用户足够的接杯时间) → RELEASE → 退回安全位置
```

优势:

- 无额外检测依赖, 不需要 MediaPipe
- 不受手部遮挡影响
- 2s 足够用户握住水杯

### 4.4 递送位置计算

**策略**: 基座 YZ 固定 (预示教的安全递送位置) + 基座 X 由人体检测动态调整.

```text
基系递送位置:
  X_person = person_pos_base.x   ← 外置相机检测人体 → bbox中心深度 → 相机系 → T_cam_base → 基系, 仅取 X 分量
  Y_deliver = Y_TEACH            ← 预示教固定值 (桌面边缘上方, 避免碰撞)
  Z_deliver = Z_TEACH            ← 预示教固定值 (递送高度)

姿态 (ABC): 预示教固定, 水杯保持水平
```

实际场景:

```text
桌边缘 ∥ 基座 X 轴

  人坐左边 (X小)    人坐中间            人坐右边 (X大)
       │                 │                  │
  ←X→  │            ←X→  │             ←X→  │
   ☕   │             ☕   │              ☕   │
  ┌──┐ │            ┌──┐ │             ┌──┐ │
  │机 │ │            │机 │ │             │机 │ │
  │器 │ │            │器 │ │             │器 │ │
  │人 │ │            │人 │ │             │人 │ │
  └──┘ │            └──┘ │             └──┘ │
───────┴───────────────┴────────────────┴──── 桌面
```

### 4.5 技术选型理由

| 决策点 | 选择 | 理由 |
| ------ | ------ | ------ |
| 人体检测 | YOLOv8 COCO person class | 与现有检测器技术栈一致, 无需额外模型 |
| 递送方式 | 固定 2s 定时松手 | 水杯遮挡手指, 手势识别不可靠; 定时松手简单可靠 |
| 递送位置 | YZ 固定 + X 动态 | 降低对检测精度的要求, 仅需人体 X 坐标即可 |
| 检测器基类 | 复用 `RealSense_R/BaseRealSenseDetector` | 外置相机内参损坏, 需手动像素映射; 复用现有基类 |
| 双相机通信 | 单进程多线程 | Handle+Cup 共享 pipeline 模式已有先例 (`inject_frames`) |

### 4.6 坐标系链路

```text
外置相机(RGB-D) → bbox中心深度 → 人体相机系3D → T_cam_base → 人体基系3D
                                                                  ↓
                                                          取 X 分量
                                                                  ↓
                                          递送目标 = (X_person, Y_TEACH, Z_TEACH)
```

### 4.7 外置相机标定方法

**不需要标定板**。灵巧手背面的矩形胶带作为参考标记, 复用现有的 HSV 检测 + 深度映射函数。

采集流程:

```text
┌─ 第 i 次采集 ───────────────────────────────────────────┐
│                                                          │
│  1. 手动控制机械臂移动到位置 i (覆盖工作区域 X 范围)        │
│  2. 记录 P_ee_base(i) = [x,y,z,A,B,C] (机器人反馈位姿)    │
│  3. 外置相机拍照 → HSV检测胶带中心 → 深度 → p_cam(i)       │
│                                                          │
│  p_base(i) = R_ee_base(i) @ p_ref_ee + t_ee_base(i)      │
│  (胶带中心在基系下的位置, 从机器人侧推算)                   │
│                                                          │
│  约束: p_base(i) = R_cam_base @ p_cam(i) + t_cam_base     │
│  未知: R_cam_base (3x3), t_cam_base (3x1)                │
└──────────────────────────────────────────────────────────┘

采集 4-5 组 (p_cam, p_base) → SVD 求解 T_cam_base
每对提供 3 个方程, 4 组即 12 方程 vs 12 未知 (R:9 + t:3), 充分.
```

与眼在手上的区别:

| | 眼在手上 (已有) | 眼在手外 (本模块) |
| ------ | ------ | ------ |
| 谁动 | 机械臂带着相机动 | **相机固定, 机械臂动** |
| 标记物 | 棋盘格 (固定桌面) | **灵巧手背胶带 (跟着机械臂走)** |
| 求解 | AX=XB (5 种算法 + 评优) | **SVD 直接求解** |
| 数据量 | 10-20 组 | **4-5 组** |
| 精度要求 | 高 (±1mm, 影响抓取) | **中等 (±2cm 可接受, 递送容错高)** |
| 新增代码 | 无 (已有 calib_data/) | **极小 (HSV检测 + SVD, ~50行)** |

---

## 5. 状态机设计

```text
                    ┌──────────┐
                    │  IDLE    │ 等待启动信号
                    └────┬─────┘
                         │ eye-to-hand 检测到人进入工作区域
                         ▼
            ┌────────────────────────────┐
            │  HANDLE_GRASP_AND_POUR     │ eye-in-hand: 抓把手+倒水 (现有流程)
            └───────────┬────────────────┘
                        │ 倒水完成, 已抓取水杯
                        ▼
            ┌────────────────────────────┐
            │  LOCATE_PERSON             │ eye-to-hand: 检测人体 X 坐标
            └───────────┬────────────────┘
                        │ 计算递送位置: (X_person, Y_TEACH, Z_TEACH)
                        ▼
            ┌────────────────────────────┐
            │  APPROACH_PERSON           │ 机械臂移动到递送位置
            └───────────┬────────────────┘
                        │ 到达递送位置
                        ▼
            ┌────────────────────────────┐
            │  WAIT_AND_RELEASE          │ 等待 2s → RELEASE (松手)
            └───────────┬────────────────┘
                        │ 释放完成
                        ▼
            ┌────────────────────────────┐
            │  RETREAT                   │ 机械臂退回安全位置
            └───────────┬────────────────┘
                        │
                        ▼
                    ┌──────────┐
                    │  IDLE    │
                    └──────────┘
```

### 5.1 异常处理

| 异常 | 检测方式 | 恢复策略 |
| ------ | ------ | ------ |
| 人体检测失败 | YOLO 连续 N 帧未检测到 person | 保持上次有效位置, 超时后回 IDLE |
| 物体滑落 | 灵巧手力矩反馈 (待硬件支持) | 紧急停止, 回 IDLE |
| 递送位置不可达 | 机械臂运动规划失败 | 回 IDLE, 通知用户 |

---

## 6. 开发计划

### Phase 1: 基础设施 (当前阶段)

- [x] **Step 1: Serial 绑定** — 两个 `_init_realsense()` 加入 `config.enable_device(serial)`, 确保双相机同时插入时各自连接正确的设备. 新增 `load_camera_serial()` 工具函数. ✅ 已测试通过 (2026-06-30)
- [ ] **Step 2: hand_to_eye/ 模块搭建** — `detector.py` 继承 `RealSense_R/BaseRealSenseDetector`, `person_detector.py` 实现人体位置检测 (YOLOv8 COCO person class).
- [ ] **Step 3: 眼在手外标定工具** — 灵巧手背胶带作参考标记, 采集 4-5 组 (P_ee_base, 胶带中心相机系 3D), SVD 求解 T_cam_base, 输出到 `hand_to_eye/calibration/result.yaml`; 预示教递送位置 YZ 写入 `hand_to_eye/config.yaml`.
- [ ] **Step 4: person_detector 3D 定位** — bbox 中心深度 → 相机系 3D → T_cam_base → 基系 X 坐标.

### Phase 2: 流程编排

- [ ] **Step 5: orchestrator/state_machine.py** — 实现 5 状态状态机 (IDLE → HANDLE_GRASP_AND_POUR → LOCATE_PERSON → APPROACH_PERSON → WAIT_AND_RELEASE → RETREAT).
- [ ] **Step 6: orchestrator/main.py** — 系统总入口, 单进程管理双检测器.
- [ ] **Step 7: 递送位置计算** — YZ 预示教固定 + X 从人体检测动态获取.

### Phase 3: 集成与验证

- [ ] **Step 8: 双相机联调** — 眼在手上 + 眼在手外全流程测试.
- [ ] **Step 9: 异常处理完善** — 人体检测失败/递送位置不可达等场景下的恢复策略.
- [ ] **Step 10: 全面测试** — 多场景、多人位置、不同光照条件下的可靠性验证.

---

## 7. 依赖项

无新增 pip 依赖. 所有功能使用已有库:

### 已有依赖

| 库 | 用途 |
| ------ | ------ |
| `opencv-python` | 图像处理 |
| `numpy` | 数值计算 |
| `pyrealsense2` | 外置相机驱动 (RealSense D435i) |
| `ultralytics` | YOLOv8 人体检测 |

---

## 8. 关键设计决策 (ADR)

| 决策 | 选择 | 理由 |
| ------ | ------ | ------ |
| 外置相机模块 | 复用 `RealSense_R/BaseRealSenseDetector` | 内参损坏, 需手动像素映射; 复用避免重复开发 |
| 人体检测 | YOLOv8 COCO person class | 与现有检测器技术栈一致, 无需额外模型 |
| 递送方式 | 固定 2s 定时松手 | 水杯遮挡手指, 手势识别不可靠; 定时松手简单可靠 |
| 递送位置 | YZ 固定 + X 动态 | 降低检测精度要求, 仅需人体 X 坐标 |
| 双相机通信 | 单进程多线程 | Handle+Cup 共享 pipeline 已有先例 |
| Serial 区分 | `config.enable_device(serial)` | librealsense2 原生支持, 无侵入 |
| 状态机实现 | 纯 Python 类 (非外部框架) | 5 状态足够简单, 不引入额外依赖 |

---

## 9. 文件结构总览 (完成态)

```text
Project_hand/
├── RealSense/                        # 眼在手上 (rs.align)
│   ├── D435i_info.yaml               # serial: 819612071851
│   ├── realsense_shared.py           # [已改] +serial 绑定
│   └── ...
├── RealSense_R/                      # 眼在手外 (手动像素映射)
│   ├── D435i_info.yaml               # serial: 342222071364
│   ├── realsense_shared.py           # [已改] +serial 绑定
│   └── ...
├── Estun/                            # 眼在手上入口
│   ├── estun_handle_cup_run.py       # 现有 Handle+Cup 融合
│   └── ...
├── hand_to_eye/                      # [新建] 眼在手外检测
│   ├── __init__.py
│   ├── config.yaml                   # 外置相机配置 + Y_TEACH/Z_TEACH 预示教值
│   ├── detector.py                   # 继承 RealSense_R 基类
│   ├── person_detector.py            # 人体检测 (YOLOv8 COCO person)
│   └── calibration/
│       ├── calibrate.py
│       └── result.yaml               # T_cam_base 标定结果
├── orchestrator/                     # [新建] 流程编排
│   ├── __init__.py
│   ├── state_machine.py
│   └── main.py
├── calib_data/                       # 手眼标定 (眼在手上, 已有)
├── memory-bank/                      # 上下文持久化
└── doc/
    └── Future_work.md                # 本文件
```
