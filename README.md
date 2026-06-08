# 机械臂+灵巧手抓取系统

通过相机检测抓取目标，计算6D位姿，经手眼标定和工具偏移补偿后，通过TCP发送给ESTUN机器人，控制灵巧手完成抓取。

## 目录结构

| 路径 | 说明 |
|------|------|
| `estun_run.py` | **主入口 (RealSense实时版-瓶身)**，整合检测→位姿转换→TCP服务器→灵巧手控制 |
| `estun_epiceye_run.py` | **主入口 (EpicEye单次触发版-瓶身)**，EpicEye相机单次拍照→6D抓取 |
| `estun_handle_run.py` | **主入口 (RealSense Handle 6D版)**，YOLO-World+SAM+扫描线分离把手→3点位姿 |
| `estun_config.yaml` | 机器人IP、灵巧手串口、夹爪偏移、抓取模式、把手偏移等配置 |
| `utils_shared.py` | 共享工具库：坐标变换工具 + 灵巧手控制，瓶身入口共用 |
| `estun_demo.py` | 演示脚本：RealSense版位姿转换链路验证 (瓶身, 无硬件依赖) |
| `estun_epiceye_demo.py` | 演示脚本：EpicEye版位姿转换链路验证 (瓶身, 无硬件依赖) |
| `estun_handle_demo.py` | 演示脚本：RealSense Handle版位姿转换链路验证 (把手, 无硬件依赖) |
| `calib_data/` | 手眼标定工具（采集+标定+可视化）及标定结果 |
| `RealSense_R/` | RealSense D435i + YOLO 6D检测 + SAM分割 + 把手检测 |
| `Orbbec/` | Orbbec Gemini 215微距相机模块 |
| `EpicEye/` | Epic Eye Pixel Pro相机模块（灰度YOLO检测） |
| `Omnihand/` | 灵巧手USB串口控制（手势、速度、使能） |
| `kettle_training/` | 水壶检测模型微调 (数据采集/标注/训练/测试) |
| `doc/` | 机器人通信协议、代码模板等文档 |
| `doc/Module_Rules.md` | 各硬件模块开发规则汇总 |
| `doc/Handle_Process.md` | 把手检测流程：YOLO-World + SAM + 水平扫描线分离 |
| `doc/Yolo_World.md` | YOLO-World 零样本检测配置与使用说明 |
| `doc/Grip_Process.md` | 位姿转换流程文档（末端接触+工具偏移两阶段） |
| `doc/Kettle_Training_Task.md` | 水壶检测模型微调详细流程 |
| `doc/python_script_template.*` | Python 代码架构模板 (含中文+英文标点规范) |

## 快速开始

### RealSense 实时版 (瓶身)

1. 连接 RealSense 相机和灵巧手（USB串口）
2. 修改 `estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 运行 `python estun_run.py`
4. 机器人作为TCP客户端连接本服务器，接收6D目标位姿

### EpicEye 单次触发版 (瓶身)

1. 连接 EpicEye 相机和灵巧手（USB串口）
2. 确认 `EpicEye/epiceye_config.yaml` 中 `handeye_matrix` 已正确填入
3. 修改 `estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
4. 运行 `python estun_epiceye_run.py`
5. 机器人作为TCP客户端连接本服务器，接收6D目标位姿

### RealSense Handle 6D 版 (把手)

1. 连接 RealSense 相机和灵巧手（USB串口）
2. 修改 `estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 确认 `RealSense_R/config.yaml` 中 `handle.skip_frames` 跳帧参数
4. 运行 `python estun_handle_run.py`
5. 机器人作为TCP客户端连接本服务器，接收 P1(抓取点)/P2(预抓取点)/P3(目标点) 三组位姿
6. 机器人依次发送 PREGRIP/GRIPDONE/RESET 指令控制灵巧手

## 配置说明

所有配置通过YAML文件管理，无需改代码：

- **RealSense 检测模型/目标类别 (瓶身)** → 修改 `RealSense_R/config.yaml` 中的 `model.name` 和 `model.target_class_id`
- **RealSense 检测模型 (把手)** → 修改 `RealSense_R/config.yaml` 中的 `handle.model_name` 和 `handle.class_name` (YOLO-World)
- **EpicEye 检测模型/目标类别** → 修改 `EpicEye/epiceye_config.yaml` 中的 `model.gray_path` 和 `model.class_id`
- **切换运行设备** → 修改对应相机模块配置文件中的 `model.device`（gpu/cpu）
- **修改机器人参数** → 修改 `estun_config.yaml`
- **调整瓶身抓取位置** → 微调 `estun_config.yaml` 中的 `gripper_offset.t_offset`
- **调整把手抓取位置** → 微调 `estun_config.yaml` 中的 `handle_gripper_offset.t_handle_offset`
- **切换抓取模式** → 修改 `estun_config.yaml` 中的 `orientation_mode`（free/vertical/horizontal）
- **把手跳帧控制** → 修改 `RealSense_R/config.yaml` 中的 `handle.skip_frames`（每N帧检测一次）
- **把手抓取点偏移** → 微调 `estun_config.yaml` 中的 `handle_grasp.pre_offset` / `handle_grasp.target_offset`

## 核心流程

### 瓶身抓取

```
相机检测 → 手眼标定转末端系 → 工具偏移补偿 → 转基坐标系 → TCP发送 → 灵巧手抓取
```

详细位姿转换算法见 `doc/Grip_Process.md`。

### 把手抓取

```
YOLO-World (GPU) → SAM (CPU) → 水平扫描线分离把手 → 6D位姿 → 跳帧控制
                                                      ↓
稳定化 (PoseStabilizer) → cam→ee → 补偿+对齐 → 基坐标系
                                                      ↓
发送3点位姿: P1(抓取点)/P2(预抓取点)/P3(目标点)
```

详细把手检测流程见 `doc/Handle_Process.md`。

## 关键概念

- **物体坐标系约定**：PCA 输出的 Z 轴指向相机 (test_6D 内部处理)，Y 轴统一向下 (utils_shared.compute_grasp_rotation 处理)，两次修正消除 PCA 符号歧义
- **手眼标定**：标定相机到末端法兰的变换矩阵，结果存储在 `calib_data/calibration_result.yaml`，支持Tsai/Park/Horaud等多种算法
- **工具偏移**：补偿灵巧手物理安装角度（C=-90°）带来的位置和旋转偏差
- **代际计数器**：防止客户端断开重连后旧帧数据污染新连接的目标位姿
- **抓取模式**：free（跟随物体朝向）、vertical（末端Z轴水平约束）、horizontal（待实现）