# 未来工作规划 — 双相机体系

## 上次更新: 2026-07-03 (v4: Orchestrator 重构完成 + HandInCup 实现; 待实操测试)

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
│  │ 手外  │   ←────────── │ (防碰撞) │ ←──────── │ HandInCup│   │
│  │(已实  │  HandInCup    └──────────┘  CONTINUE/ └──────────┘   │
│  │ 现)   │  手-杯判定                  ABORT                     │
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
- 状态: ✅ 把手抓取 + 倒水 + 水杯抓取流程已完成 (back/left 双模式均已验证通过, 2026-06-30)

### 2.2 眼在手外 (Eye-to-Hand) — 大部分已完成, 待实测

- 相机: RealSense D435i (固定安装), Serial `342222071364`
- 模块: `RealSense_R/` (手动像素映射版本, 内参损坏)
- 功能:
  - ✅ 检测人的位置 (PersonDetector + PersonStabilizer)
  - ✅ 人手-杯把手重合检测 (HandInCupDetector, YOLOv8m+YOLOv8n-Pose)
  - ⚡ 控制机械臂将水杯递送到人手中 (需联调)
  - ✅ 作为整个系统的启动触发器 (检测到人 → 开启全流程)
- 状态: ⚡ 代码完成, 待实测验证 (标定阻塞)

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
├── hand_to_eye/ → EyeToHand/          # [新建] 眼在手外模块
│   ├── __init__.py
│   ├── config.yaml                    # 外置相机配置 + 预示教递送位置 + 稳定器参数
│   ├── D435i_info.yaml                # 相机内参 (serial: 342222071364)
│   ├── detector.py                    # 工具函数: load_calibration / camera_x_to_base_x
│   ├── person_detector.py             # 人体检测 (YOLO COCO person + PersonStabilizer)
│   ├── hand_in_cup.py                 # [新增] HandInCup (YOLOv8m+YOLOv8n-Pose)
│   └── calibration/
│       ├── __init__.py
│       ├── calibrate.py               # 1D最小二乘线性回归 (交互式, N≥2点)
│       └── result.yaml                # 标定结果 (scale, bias)
│
├── orchestrator/ → Orchestrator/       # [后续] 双相机流程编排
│   ├── __init__.py
│   ├── state_machine.py               # 状态机管理
│   └── main.py                        # 系统总入口
│
└── ... (其余现有文件不变)
```

### 4.2 检测目标

| 目标 | 方法 | 说明 |
| ------ | ------ | ------ |
| 人体位置 | YOLOv8 COCO person class (class_id=0) | bbox 中心深度 → 相机系 3D → T_cam_base → 基系 3D |

### 4.3 递送策略 — v4 升级为 HandInCup 检测

**原方案** (v3): 固定 2s 定时松手 — 简单但无法判断是否真有人接杯.

**新方案** (v4): **YOLOv8m + YOLOv8n-Pose HandInCup 检测**:

```text
机械臂到达 P3 → READY → wait 2s (同时 HandInCup 检测运行)
→ RELEASE 时 PC 判定:
    hand_on_cup=TRUE  → CONTINUE → release_object() → P4 退避 → Home
    hand_on_cup=FALSE → ABORT   → (不松手) → 返回 P1 放回 → P2 退避 → Home
```

HandInCup 判定逻辑:
- YOLOv8m 检测杯 bbox (COCO class 41)
- YOLOv8n-Pose 检测手腕关键点 (kpt 9=left_wrist, 10=right_wrist)
- 连续 5 帧手腕在杯 bbox 扩展区内 → True
- 2D 判定, 无需 3D 深度反投影, 避免深度噪声
- 可选深度辅助: `depth_tolerance=0.05m` 校验手腕-杯心深度一致性

实测修正 (2026-07-06):
- 无遮挡实测发现手腕在 ROI 外, 引入 `dist_factor` (判定半径 = bbox 对角线 × 系数) 和 `depth_tolerance` (深度容差, 防误报)
- 两参数当前为初始猜测值, **未实地验证**
- 有灵巧手遮挡场景 **未测试**

独立模块: `EyeToHand/hand_in_cup.py`, 可脱离状态机单独测试.

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

### 4.7 外置相机标定方法 (1D 最小二乘线性回归)

递送仅需基系 X 坐标, ±2cm 容错. 不需要 SVD, 不需要解胶带位置.

**流程:**

```text
采集:
  1. 机械臂移到 X_base = x1 处 (如 0.3m) → 外置相机检测胶带中心 X_cam_1
  2. 机械臂移到 X_base = x2 处 (如 0.6m) → 外置相机检测胶带中心 X_cam_2

拟合:
  scale = (x2 - x1) / (X_cam_2 - X_cam_1)      # 相机深度单位 → 基系米
  bias  = x1 - scale * X_cam_1

运行时:
  X_person = scale * X_cam_person + bias
```

| 项目 | SVD 方案 | 2 点方案 |
| ------ | ------ | ------ |
| 采集点数 | 5 组 | **2 组** |
| 求解 | SVD 解 15 元方程组 | **两行表达式** |
| 前提 | 需解 p_tape_ee 未知 | **scale+bias 隐式消去** |
| 耗时 | ~5 分钟 | **~2 分钟** |
| 精度 | 高 | 够用 (±2cm, 递送仅 X 方向) |

---

## 5. 状态机设计 (v4 已实现: 2 状态简化版)

实际实现简化为 2 状态 (单 TCP Session 消除多状态同步复杂性):

```text
       ┌─────────────────────────────────────────────┐
       │                                             │
       ▼                                             │
     IDLE ──(Camera B 检测到人)──→ HANDLE_GRASP_AND_POUR ──(SocketClose)──→ IDLE
       ▲                                                                   │
       └───────────────────────────────────────────────────────────────────┘

HANDLE_GRASP_AND_POUR 内部单 TCP Session:
  Phase A: 收初始位姿 → B: 发 Handle <P1>[P2]{P3}
  → C: PREGRIP/GRIPDONE/RESET/CUP → 发 Cup <P1>[P2]{P3} (P3=delivery)
  → D: READY/RELEASE → HandInCup → CONTINUE or ABORT
  → Robot 断开 → 回 IDLE
```

| 对比 | v3 设计 (6 状态) | v4 实现 (2 状态) |
|------|:---:|:---:|
| 状态数 | 6 | 2 |
| TCP Session | 2 | 1 |
| 递送位置 | 独立 Session 2 | Cup P3 = delivery |
| 松手判断 | 固定 2s | HandInCup 判定 |
| 复杂度 | 状态同步困难 | 协议内自包含 |

### 5.1 异常处理

| 异常 | 检测方式 | 恢复策略 |
| ------ | ------ | ------ |
| 人体检测失败 | YOLO 连续 N 帧未检测到 person | 保持上次有效位置, 超时后回 IDLE |
| 物体滑落 | 灵巧手力矩反馈 (待硬件支持) | 紧急停止, 回 IDLE |
| 递送位置不可达 | 机械臂运动规划失败 | 回 IDLE, 通知用户 |

---

## 6. 开发计划

### Phase 1: 基础设施

- [x] **Step 1: Serial 绑定** — 两个 `_init_realsense()` 加入 `config.enable_device(serial)`, 确保双相机同时插入时各自连接正确的设备. 新增 `load_camera_serial()` 工具函数. ✅ 已测试通过 (2026-06-30)
- [x] **Step 2: EyeToHand/ 模块搭建** — `person_detector.py` 直接继承 `RealSense_R/BaseRealSenseDetector` (对齐 test_3D.py 模式), 实现人体位置检测 (YOLOv8 COCO person class_id=0). `detector.py` 提供 `load_calibration()` / `camera_x_to_base_x()` 工具函数. `PersonStabilizer` 窗口稳定器 (参考 PoseStabilizer, window=10/pos_thresh=0.03m). ✅ person_detector 已测试通过 (2026-06-30)
- [x] **Step 3: 眼在手外标定工具** — `EyeToHand/calibration/calibrate.py`: 1D 最小二乘线性回归 (N≥2 点). 机械臂末端贴红色胶带, YZ/ABC 保持示教值不变, 仅沿 X 移动, 终端输入示教器 X_base 值采集. scale+bias 输出到 `EyeToHand/calibration/result.yaml`. 代码完成, 未实操验证 (机械臂被占用) (2026-06-30)
- [x] **Step 4: person_detector 3D 定位** — `detector.py` `camera_x_to_base_x()`: bbox 中心深度 → 相机系 3D → 取 X_cam → scale*X_cam + bias → 基系 X 坐标. 代码完成, 未端到端验证 (依赖 Step 3 标定) (2026-06-30)

### Phase 2: 流程编排 ✅ (2026-07-03)

- [x] **Step 5: Orchestrator/state_machine.py** — 2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR), 单 TCP Session 全流程 Handler ✅
- [x] **Step 6: Orchestrator/main.py** — 双相机主循环 + Person/HandInCup 检测切换 ✅
- [x] **Step 7: 递送位置计算** — X 动态 + YZ/ABC 预示教, Cup P3 = delivery ✅
- [x] **Step 7.5: HandInCup 检测器** — `EyeToHand/hand_in_cup.py` (YOLOv8m+YOLOv8n-Pose) ✅

### Phase 3: 集成与验证 ⚡

- [ ] **Step 8: 实操标定** — `EyeToHand/calibration/calibrate.py` (需机械臂) ⚡ P0 阻塞
- [ ] **Step 9: 填入递送参数** — `delivery.Y_TEACH/Z_TEACH/ABC` (需机械臂) ⚡
- [~] **Step 10: HandInCup 实测** — 无遮挡已测 (`dist_factor`/`depth_tolerance` 待调优); **有遮挡未测** ⚡
- [ ] **Step 11: 机器人 .crp 适配** — 新指令 READY/CONTINUE/ABORT ⚡
- [ ] **Step 12: 全流程联调** — 双相机 + 机器人 + 灵巧手 ⚡

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
| 状态机实现 | 纯 Python 类, 2 状态简化版 | 单 TCP Session, 消除多状态同步 |

---

## 9. ### 已补充的验证

- [x] P4 预倒水点: 已在示教器测试通过 ✅ (2026-07-02)

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
├── EyeToHand/                        # 眼在手外检测 (已实现)
│   ├── __init__.py
│   ├── config.yaml                   # Person + delivery + HandInCup 配置
│   ├── D435i_info.yaml               # serial: 342222071364
│   ├── detector.py                   # 工具函数 (load_calibration, camera_x_to_base_x)
│   ├── person_detector.py            # 人体检测 (PersonDetector + PersonStabilizer)
│   ├── hand_in_cup.py               # [新增] HandInCup (YOLOv8m+YOLOv8n-Pose, 可独立测试)
│   └── calibration/
│       ├── calibrate.py              # 1D最小二乘线性回归 (交互式, N≥2点)
│       └── result.yaml               # scale/bias (占位值, 待标定)
├── Orchestrator/                     # 流程编排 (已实现)
│   ├── __init__.py
│   ├── state_machine.py              # 2 状态状态机 + 单 Session TCP handler
│   └── main.py                       # 双相机主循环 + Person/HandInCup 切换
├── calib_data/                       # 手眼标定 (眼在手上, 已有)
├── memory-bank/                      # 上下文持久化
└── doc/
    └── Future_work.md                # 本文件
```
