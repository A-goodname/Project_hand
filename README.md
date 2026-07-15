# 机械臂+灵巧手抓取系统

双相机 (眼在手上 + 眼在手外) 全流程自动抓取-倒水-递杯. Camera A 检测把手+水杯 3D 位姿, Camera B 检测人就位+HandInCup 判定, TCP 发送给 ESTUN 机器人控制 OmniHand 灵巧手.

## 硬件需求

| 设备 | 型号 | 连接 | 用途 |
|------|------|------|------|
| Camera A | RealSense D435i (serial `342222071364`) | USB 3.0 | 眼在手上: Handle+Cup 3D 检测 |
| Camera B | UGREEN 1080P | USB | 眼在手外: 人就位 + HandInCup |
| 灵巧手 | OmniHand | USB 串口 (默认 COM5) | 抓取/手势/压力检测 |
| 机械臂 | ESTUN | TCP (LAN) | 执行抓取动作 |
| GPU | NVIDIA (CUDA 12.6) | — | YOLO 推理加速 |

## 运行前准备

首次运行前按顺序完成:

### 1. 环境安装

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

EpicEye SDK 需要手动打包 whl 安装 (参考 `epiceyesdk_samples_project/python/README.md`).

### 2. 确认硬件连接

- [ ] Camera A USB 连接, serial 与 `RealSense_R/D435i_info.yaml` 一致
- [ ] Camera B USB 连接, 记下 OpenCV 设备索引 (运行 `python -c "import cv2; [print(i) for i in range(5) if cv2.VideoCapture(i).isOpened()]"` 测试)
- [ ] OmniHand 串口连接, 确认 `Estun/estun_config.yaml` → `hand.port`
- [ ] 机械臂与控制 PC 在同一局域网

### 3. 配置 IP 与端口

编辑 `Estun/estun_config.yaml`:

| 参数 | 说明 |
|------|------|
| `estun.tcp_server.host` | **本机 IP** (机械臂作为客户端连接) |
| `estun.tcp_server.port` | 8080 (Phase A-D 抓取) |
| `estun.greeting.port` | 8600 (Phase 0 打招呼) |
| `estun.hand.port` | OmniHand 串口号 (如 COM5) |

编辑 `EyeToHand/config.yaml`:

| 参数 | 说明 |
|------|------|
| `ugreen.camera_id` | Camera B 的 OpenCV 设备索引 (0/1/2/3...) |

### 4. 标定 Camera B 杯 ROI

UGREEN Camera 无深度, YOLO 检测不到杯时回退到固定 ROI:

```bash
python EyeToHand/hand_in_cup.py --calibrate-roi
```

1. 确保 Camera B 对准杯放置区域
2. 鼠标拖拽框选杯区域
3. 按 `s` 保存 → 写入 `EyeToHand/config.yaml` → `cup_roi_fallback`

### 5. 手眼标定确认

- [ ] `calib_data/calibration_result.yaml` 存在
- [ ] `T_cam_to_gripper` 矩阵非单位阵 (查看 `handeye.use_method`)
- [ ] 如更换相机/重新安装 → 运行 `calib_data/collect_data.py` + `calib_data/calibrate_handeye.py`

### 6. 抓取偏移微调

首次部署或机械结构变更时必须重调:

| 偏移 | 配置文件 | 关键参数 |
|------|----------|----------|
| Handle 抓取 | `estun_config.yaml` | `handle_grasp.gripper_offset.t_handle_offset` |
| Cup 抓取 (back) | `estun_config.yaml` | `cup_grasp.gripper_offset.t_cup_offset` |
| Cup 抓取 (left) | `estun_config.yaml` | `cup_grasp.gripper_offset.t_cup_offset_left` |

微调方法: 运行对应 `_demo.py` (如 `estun_handle_cup_demo.py`), 启用 `fine_tune_mode`, 手动微调后自动回算新偏移.

### 7. 示教位姿 + 模型确认

- [ ] P3 倒水偏移: `estun_config.yaml` → `handle_grasp.p3_teach_offset` (按角度+模式索引)
- [ ] Cup P3 递送: `EyeToHand/config.yaml` → `delivery.*_TEACH`
- [ ] 打招呼位姿: `estun_config.yaml` → `greeting.zhaoshou*_pose / woshou_pose`
- [ ] 模型文件: `model/kettle_yolov8n_finetune.pt`, `model/hand_landmarker.task`
- [ ] YOLO 基础模型首次运行自动下载

## 快速开始

### 主入口: 双相机全流程

```bash
python Orchestrator/main.py
```

4 阶段顺序执行: `greeting → handle_detect → seated_check → hand_in_cup`.

1. 先启动 Python 端, 再运行机器人 CRP
2. CRP 文件: `doc/handlepluscup-orchestrator.crp` (80 widgets, 下载到示教器)
3. 按 `q` 退出

### 单独测试

```bash
# Camera A Handle+Cup 检测
python Orchestrator/handle_cup_detector.py

# Camera B 人就位检测
python EyeToHand/person_detector.py

# Camera B HandInCup 检测
python EyeToHand/hand_in_cup.py
```

## 配置速查

| 你想改什么 | 去哪改 |
|-----------|--------|
| 本机 IP / 串口号 | `Estun/estun_config.yaml` |
| Handle/Cup 抓取偏移 | `Estun/estun_config.yaml` → `handle_grasp` / `cup_grasp` |
| P3 倒水角度 | `Estun/estun_config.yaml` → `p3_teach_offset.pour_angle` |
| Camera B 设备索引 | `EyeToHand/config.yaml` → `ugreen.camera_id` |
| Cup P3 递送位姿 | `EyeToHand/config.yaml` → `delivery.*_TEACH` |
| 杯 ROI 回退 | `python EyeToHand/hand_in_cup.py --calibrate-roi` |
| 检测模型/固定ABC/跳帧 | `Orchestrator/config.yaml` |
| 白色把手/红色胶带 HSV | `Orchestrator/config.yaml` → `white_handle` / `red_handle` |
| 人就位/HandInCup 阈值 | `EyeToHand/config.yaml` → `chair` / `hand_in_cup` |

## 关键概念

- **4 阶段调度**: greeting → handle_detect → seated_check → hand_in_cup, 任何时刻只有一个检测器运行
- **pack_three**: `<P1>[P2]{P3}` 格式发送三组抓取位姿, CRP `TranStrToCpos` 按 bracket 标记解析
- **固定 ABC**: Handle+Cup 姿态角从 `Orchestrator/config.yaml` 直接读取基系固定值, 不通过 PCA 计算
- **PoseStabilizer**: 缓存 N 帧中值稳定, 代际计数器 `_gf` 防断连后旧帧污染
- **HandInCup**: YOLOv8n(杯) + MediaPipe Hands(21 landmarks), landmarks-in-bbox 计分 + curled gate 判定
- **CONTINUE/ABORT**: Python 发 `<x,y,z,A,B,C>` (CONTINUE) 或 `[x,y,z,A,B,C]` (ABORT), CRP 两个 `TranStrToCpos` (cmd=`<` / cmd=`[`) 解析, If `strflag_cont/abort == 0` 路由分支
- **RELEASE 双语义**: HandInCup 未启动 → 松把手; 已启动 → 判定触发

## 目录结构

| 路径 | 说明 |
|------|------|
| `Orchestrator/` | **主入口**: 4 阶段调度 + 双相机检测 + TCP 服务 |
| `Estun/` | 单相机入口 (RealSense/EpicEye), 坐标变换, 灵巧手控制 |
| `RealSense_R/` | Camera A 检测模块 (D435i 旧版, 手动像素映射) |
| `RealSense/` | Camera A 检测模块 (D435i 新版, rs.align) |
| `EyeToHand/` | Camera B: PersonDetector + HandInCup + UGREEN Camera |
| `Omnihand/` | 灵巧手 USB 串口控制 |
| `calib_data/` | 手眼标定数据 + 结果 |
| `KettleTraining/` | 水壶 YOLO 微调 |
| `EpicEye/` `Orbbec/` | 灰度/微距相机模块 (非主用) |
| `doc/` | 协议文档 + CRP 文件 + 流程文档 |
| `memory-bank/` | 跨会话项目上下文 |
| `model/` | YOLO + MediaPipe 模型文件 |

## 文档索引

| 文档 | 内容 |
|------|------|
| `doc/handlepluscup-orchestrator.crp` | **最新 CRP** (80 widgets, Phase 0 + A-D, 下载到示教器) |
| `doc/estun_handle_cup_code_orchestrator.txt` | CRP 伪代码 + 版本变更记录 |
| `doc/Greeting_Flow.md` | Phase 0 + Phase A-D 协议时间线 |
| `doc/Handle_Cup_Process.md` | Handle+Cup 位姿计算 + 偏移参数 |
| `doc/Handle_Process.md` | 把手检测设计决策记录 |
| `doc/Grip_Process.md` | 瓶身位姿转换算法 |
| `doc/Module_Rules.md` | 各硬件模块开发规则 |
| `doc/Project_Workflow.md` | 完整架构 + 双相机 + 标定流程 |
| `doc/Kettle_Training_Task.md` | 水壶模型微调 SOP |
