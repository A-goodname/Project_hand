# 机械臂+灵巧手抓取系统

通过相机检测抓取目标，计算6D位姿，经手眼标定和工具偏移补偿后，通过TCP发送给ESTUN机器人，控制灵巧手完成抓取。

## 环境配置

### Python 版本

- Python >=3.11, <3.12

### 前置步骤 (手动安装)

依赖分三部分安装，请按顺序执行：

#### 步骤 1: 安装 PyTorch (CUDA 12.6)

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

#### 步骤 2: 安装 EpicEye SDK

参考 `epiceyesdk_samples_project/python/README.md`，打包 whl 并安装。

#### 步骤 3: 安装其余依赖

```bash
pip install -r requirements.txt
```

`requirements.txt` 包含所有固定版本的 pip 依赖（numpy, opencv-python, ultralytics, pyrealsense2, pyorbbecsdk2 等）。

## 目录结构

| 路径 | 说明 |
| ------ | ------ |
| `Estun/estun_run.py` | **主入口 (RealSense实时版-瓶身)**，整合检测→位姿转换→TCP服务器→灵巧手控制 |
| `Estun/estun_epiceye_run.py` | **主入口 (EpicEye单次触发版-瓶身)**，EpicEye相机单次拍照→6D抓取 |
| `Estun/estun_handle_run.py` | **主入口 (RealSense Handle 6D版)**，微调YOLOv8n+SAM2+扫描线分离把手→3点位姿 |
| `Estun/estun_config.yaml` | 机器人IP、灵巧手串口、灵巧手偏移、抓取模式、把手偏移等配置 |
| `Estun/utils_shared.py` | 共享工具库：坐标变换工具 + 灵巧手控制，瓶身入口共用 |
| `Estun/estun_demo.py` | 演示脚本：RealSense版位姿转换链路验证 (瓶身, 无硬件依赖) |
| `Estun/estun_epiceye_demo.py` | 演示脚本：EpicEye版位姿转换链路验证 (瓶身, 无硬件依赖) |
| `Estun/estun_handle_demo.py` | 演示脚本：RealSense Handle版位姿转换链路验证 (把手, 无硬件依赖) |
| `Estun/estun_cup_demo.py` | **演示脚本：水杯单检测位姿转换验证**，独立 pipeline，支持 back/left handle_side 多模式角度约束 |
| `Estun/estun_handle_cup_demo.py` | **演示脚本：Handle+Cup 融合版**，双检测器共享 pipeline 位姿转换验证 (无硬件依赖) |
| `Estun/estun_handle_cup_run.py` | **主入口 (Handle+Cup 融合版)**，Handle 抓取+倒水 + Cup 抓取+递杯，含 TCP+CUP 指令，支持 back/left 双模式 |
| `Estun/estun_teach_p3.py` | **P3 倒水位置示教脚本**，支持 back/left 分模式 + 多角度 (10°/15°/20°/30°) |
| `Estun/estun_test.py` | TCP 服务器模拟脚本，持续发送位姿并处理 GRIP/RELEASE 指令 |
| `Estun/__init__.py` | Estun 包标记文件 |
| `EyeToHand/` | **眼在手外模块**：人体检测 (PersonDetector + PersonStabilizer)、1D 标定工具、相机→基系 X 坐标转换、**人手-杯把手重合检测** (HandInCupDetector, YOLOv8m+YOLOv8n-Pose) |
| `Orchestrator/` | **双相机流程编排**：2 状态状态机 (IDLE ↔ HANDLE_GRASP_AND_POUR)、单 TCP Session 全流程 (抓壶倒水→抓杯递送→HandInCup判定→松手退避)、双相机主循环 |
| `calib_data/` | 手眼标定工具（采集+标定+可视化）及标定结果 |
| `RealSense_R/` | RealSense D435i + YOLO 6D检测 + SAM分割 + 把手检测 (旧版, 无 rs.align, 手动像素映射) |
| `RealSense/` | RealSense D435i + YOLO 6D检测 + SAM分割 + 把手检测 (新版, 使用 rs.align 简化深度对齐, 适用于内参完好的相机) |
| `Orbbec/` | Orbbec Gemini 215微距相机模块 |
| `EpicEye/` | Epic Eye Pixel Pro相机模块（灰度YOLO检测） |
| `Omnihand/` | 灵巧手USB串口控制（手势、速度、使能） |
| `KettleTraining/` | 水壶检测模型微调 (数据采集/标注/训练/测试) |
| `doc/` | 机器人通信协议、代码模板等文档 |
| `doc/Module_Rules.md` | 各硬件模块开发规则汇总 |
| `doc/Handle_Process.md` | 把手检测流程及关键设计决策记录 |
| `doc/Handle_Cup_Process.md` | **把手+水杯融合流程文档**，6 点位姿计算、共享 pipeline 架构、配置参数说明 |
| `doc/Future_work.md` | 未来工作规划: 双相机体系 (眼在手上+眼在手外) |
| `doc/Yolo_World.md` | YOLO-World 零样本检测配置与使用说明 |
| `doc/Grip_Process.md` | 位姿转换流程文档（末端接触+工具偏移两阶段） |
| `doc/Kettle_Training_Task.md` | 水壶检测模型微调详细流程 |
| `doc/python_script_template.*` | Python 代码架构模板 (含中文+英文标点规范) |

## 快速开始

### RealSense 实时版 (瓶身)

1. 连接 RealSense 相机和灵巧手（USB串口）
2. 修改 `Estun/estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 运行 `python Estun/estun_run.py`
4. 机器人作为TCP客户端连接本服务器，接收6D目标位姿

### EpicEye 单次触发版 (瓶身)

1. 连接 EpicEye 相机和灵巧手（USB串口）
2. 确认 `EpicEye/epiceye_config.yaml` 中 `handeye_matrix` 已正确填入
3. 修改 `Estun/estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
4. 运行 `python Estun/estun_epiceye_run.py`
5. 机器人作为TCP客户端连接本服务器，接收6D目标位姿

### RealSense Handle 6D 版 (把手)

1. 连接 RealSense 相机和灵巧手（USB串口）
2. 修改 `Estun/estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 确认 `RealSense_R/config.yaml` 中 `handle.skip_frames` 跳帧参数和 `handle.handle_side` 把手侧
4. 可选: 在把手上贴红色胶带并启用 `red_tape.enabled` 辅助检测
5. 运行 `python Estun/estun_handle_run.py`
6. 机器人作为TCP客户端连接本服务器，接收 P1(抓取点)/P2(预抓取点)/P3(目标点) 三组位姿
7. 机器人依次发送 PREGRIP/GRIPDONE/RESET 指令控制灵巧手

### RealSense Handle + Cup 融合版 (把手抓取+倒水+水杯抓取)

1. 连接 RealSense 相机和灵巧手（USB串口）
2. 修改 `Estun/estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 确认 `RealSense_R/config.yaml` 中 `cup.handle_side` 为 `back` 或 `left`
4. 配置 `Estun/estun_config.yaml` 中的 `handle_grasp.p3_teach_offset.{side}` 和 `pour_angle`
5. 运行 `python Estun/estun_handle_cup_run.py`
6. 机器人作为TCP客户端连接本服务器，接收 Handle P1/P2/P3 三组位姿
7. 倒水完成后机械臂发 `CUP` 指令获取 Cup P1/P2/P3 位姿
8. 支持 back/left 双模式，P3 支撑多倒水角度 (10°/15°/20°/30°)

### P3 倒水位置示教

1. 运行 `python Estun/estun_teach_p3.py`，按提示沿 X/Y/Z 方向微调末端位置
2. 设置 `Estun/estun_config.yaml` 中的 `handle_grasp.p3_teach_offset.pour_angle` 切换倒水角度
3. 示教数据按角度索引 (10/15/20/30) + back/left 分模式存储

## 配置说明

所有配置通过YAML文件管理，无需改代码：

- **RealSense 检测模型/目标类别 (瓶身)** → 修改对应模块 `config.yaml` 中的 `model.name` 和 `model.target_class_id` (`RealSense_R/` 或 `RealSense/`)
- **RealSense 水壶检测模型 (把手)** → 修改对应模块 `config.yaml` 中的 `kettle_model.model` (微调YOLOv8n) 或 `yolo_world.model` (YOLO-World)
- **RealSense 把手检测参数** → 修改对应模块 `config.yaml` 中的 `handle` 段 (side/skip_frames/center_deviation_ratio 等)
- **RealSense 水杯检测参数** → 修改对应模块 `config.yaml` 中的 `cup` 段 (height/rim_radius/rim_offset_ratio/handle_side)
- **EpicEye 检测模型/目标类别** → 修改 `EpicEye/epiceye_config.yaml` 中的 `model.gray_path` 和 `model.class_id`
- **切换运行设备** → 修改对应相机模块配置文件中的 `model.device`（gpu/cpu）
- **修改机器人参数** → 修改 `Estun/estun_config.yaml`
- **调整瓶身抓取位置** → 微调 `Estun/estun_config.yaml` 中的 `gripper_offset.t_offset`
- **调整把手抓取位置** → 微调 `Estun/estun_config.yaml` 中的 `handle_grasp.gripper_offset.t_handle_offset`
- **切换抓取模式** → 修改 `Estun/estun_config.yaml` 中的 `orientation_mode`（free/vertical/horizontal/fixed）
- **切换抓取方向** → 修改 `Estun/estun_config.yaml` 中的 `grasp_direction`（right/left）
- **固定姿态配置** → 修改 `Estun/estun_config.yaml` 中的 `fixed_abc.right` / `fixed_abc.left` (fixed 模式)
- **把手跳帧控制** → 修改对应模块 `config.yaml` 中的 `handle.skip_frames`（每N帧检测一次）
- **把手抓取点偏移** → 微调 `Estun/estun_config.yaml` 中的 `handle_grasp.pre_offset` / `handle_grasp.target_offset`
- **红色胶带辅助检测** → 修改对应模块 `config.yaml` 中的 `red_tape.enabled` / `red_tape.min_area`
- **SAM 模型选择** → 修改对应模块 `config.yaml` 中的 `sam.model` (当前为 sam2_b.pt)
- **调整水杯偏移 (back/left)** → 修改 `Estun/estun_config.yaml` 中的 `cup_grasp.gripper_offset.t_cup_offset` (back 模式) 或 `t_cup_offset_left` (left 模式)
- **P3 倒水角度切换** → 修改 `Estun/estun_config.yaml` 中的 `handle_grasp.p3_teach_offset.pour_angle` (10/15/20/30)
- **P3 示教偏移 (back/left)** → 修改 `Estun/estun_config.yaml` 中的 `handle_grasp.p3_teach_offset.back` / `.left` 各角度 `t_offset` 和 `r_abc`
- **眼在手外人体检测** → 修改 `EyeToHand/config.yaml` 中的 `model`、`stabilizer`、`delivery.*` 参数
- **眼在手外标定** → 运行 `EyeToHand/calibration/calibrate.py` 采集 N 点拟合 `scale`/`bias`

## 核心流程

### 瓶身抓取

```text
相机检测 → 手眼标定转末端系 → 工具偏移补偿 → 转基坐标系 → TCP发送 → 灵巧手抓取
```

详细位姿转换算法见 `doc/Grip_Process.md`。

### 把手抓取

```text
检测阶段:
  微调YOLOv8n → SAM2 (GPU/CPU) → 水平扫描线分离把手 → (可选)红色胶带替换位置

稳定阶段:
  PoseStabilizer (仅检测帧执行) → cam→ee → 补偿+对齐 → 基坐标系
                                                                        ↓
发送3点位姿: P1(抓取点)/P2(预抓取点)/P3(目标点)
            (P1 固定 C 值到 [-100, -80] 区间, 使手指垂直把手)
```

详细把手检测流程见 `doc/Handle_Process.md`。

### 把手+水杯融合 (Handle + Cup)

```text
Phase A: 机器人发送末端初始位姿 (5s 超时)
Phase B: 双检测器 (共享 pipeline) 同步检测 →
          Handle 稳定 → cam2ee → P1(抓取)/P2(预抓取)
          Cup 稳定 → cam2ee → P1/P2/P3
          P3_pour = Cup P1 基系 + P3_TEACH_OFFSET (按角度/模式索引)
          → 发送 Handle 三个位姿 <P1>[P2]{P3}
Phase C: 指令循环: PREGRIP → GRIPDONE → RESET → CUP → RELEASE → GRIP
```

流程文档见 `doc/Handle_Cup_Process.md`。

### 双相机全流程 (Orchestrator)

```text
       ┌─────────────────────────────────────────────┐
       │                                             │
       ▼                                             │
     IDLE ──(Camera B 检测到人)──→ HANDLE_GRASP_AND_POUR ──(SocketClose)──→ IDLE
       ▲                                                                   │
       └───────────────────────────────────────────────────────────────────┘

单 TCP Session 全流程:
  Phase A: Robot → PC  [x,y,z,A,B,C]       初始末端位姿
  Phase B: PC → Robot  <Handle_P1>[P2]{P3}  发送 Handle 三组位姿
  Phase C: 指令循环 PREGRIP → GRIPDONE → RESET → CUP
           PC → Robot  <Cup_P1>[Cup_P2]{P3}  回复 Cup 位姿 (P3=递送)
  Phase D: READY → RELEASE → HandInCup 判定 → CONTINUE/ABORT
           TRUE:  P4退避 → RESET → Home
           FALSE: 放回P1 → RELEASE → P2退避 → RESET → Home
```

- **Camera A** (眼在手上, serial `819612071851`, `RealSense/` rs.align): Handle+Cup 6D 检测
- **Camera B** (眼在手外, serial `342222071364`, `RealSense_R/`): 人体检测 + HandInCup 检测
- **Cup P3**: X = `scale × X_cam + bias` (Camera B 动态检测), YZ/ABC = config 预示教固定值
- **HandInCup**: YOLOv8m (杯 bbox) + YOLOv8n-Pose (手腕关键点), 连续 N 帧判定
- **松手**: Robot wait 2s → RELEASE → HandInCup 判定后才松
- 代码: `Orchestrator/state_machine.py` + `Orchestrator/main.py` + `EyeToHand/hand_in_cup.py`

## 关键概念

- **物体坐标系约定**：PCA 输出的 Z 轴指向相机 (test_6D 内部处理)，Y 轴统一向下 (utils_shared.compute_grasp_rotation 处理，由 `grasp_direction` 控制方向)，两次修正消除 PCA 符号歧义
- **手眼标定**：标定相机到末端法兰的变换矩阵，结果存储在 `calib_data/calibration_result.yaml`，支持Tsai/Park/Horaud等多种算法
- **工具偏移**：补偿灵巧手物理安装角度（C=-90°）带来的位置和旋转偏差
- **代际计数器**：防止客户端断开重连后旧帧数据污染新连接的目标位姿
- **抓取模式**：free（跟随物体朝向）、vertical（末端Z轴水平约束）、fixed（固定姿态）、horizontal（待实现）
- **红色胶带辅助**：HSV 双阈值红色检测，替换把手 3D 位置，旋转保持 PCA 不变
- **偏移参考系**：P2/P3 的 pre_offset / target_offset 定义在抓取点末端系 (grasp ee frame) 下，非当前末端系
- **扫描线分离**：按 `handle_side` 取最外段（right=最右段/left=最左段），底部+顶部双向剔除 x 中心偏离行
- **Cup handle_side 模式**：`RealSense_R/config.yaml` 中 `cup.handle_side` 控制水杯检测侧（back/left），影响 bbox 裁剪 + 角度约束与偏移选择: back 模式 B 角 [70,90]+C 角 [-120,-100]; left 模式仅 C 角 [-100,-80] 并选用独立偏移 t_cup_offset_left
- **P3 多角度示教**：`estun_config.yaml` 中 `p3_teach_offset.{side}` 按 B 角度数 key (10/15/20/30)，`pour_angle` 切换当前使用角度，加载自动转字符串键防整数键 bug
- **双相机体系**：两台 RealSense D435i，Camera A (眼在手上, serial 819612071851, `RealSense/` rs.align) + Camera B (眼在手外, serial 342222071364, `RealSense_R/` 手动像素映射)
- **Orchestrator 状态机**：2 状态 (IDLE ↔ HANDLE_GRASP_AND_POUR)，单 TCP Session 全流程，Handle+Cup 流程封装在 `HandleGraspAndPourHandler` 中
- **Cup P3 递送位姿**：X 来自 Camera B 人体检测 + EyeToHand 标定 (`scale × X_cam + bias`)，YZ/ABC 来自 `EyeToHand/config.yaml` delivery 段预示教
- **HandInCup 检测**：YOLOv8m (杯 bbox) + YOLOv8n-Pose (手腕关键点)，2D 判定人手是否在杯把手附近，连续 N 帧确认；独立模块 `EyeToHand/hand_in_cup.py`
- **新协议指令**：`READY` (Robot→PC, 到达 P3 等待)、`CONTINUE` (PC→Robot, HandInCup=True 继续退避)、`ABORT` (PC→Robot, HandInCup=False 放回杯子)
