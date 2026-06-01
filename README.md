# 机械臂+灵巧手抓取系统

通过相机检测抓取目标，计算6D位姿，经手眼标定和工具偏移补偿后，通过TCP发送给ESTUN机器人，控制灵巧手完成抓取。

## 目录结构

| 路径 | 说明 |
|------|------|
| `estun_run.py` | **主入口**，整合检测→位姿转换→TCP服务器→灵巧手控制 |
| `estun_config.yaml` | 机器人IP、灵巧手串口、夹爪偏移、抓取模式等配置 |
| `GRIP_Process.md` | 位姿转换流程文档（末端接触+工具偏移两阶段） |
| `calib_data/` | 手眼标定工具（采集+标定+可视化）及标定结果 |
| `RealSense_R/` | RealSense D435i + YOLO 6D检测 + SAM分割 |
| `Orbbec/` | Orbbec Gemini 215微距相机模块 |
| `EpicEye/` | Epic Eye Pixel Pro相机模块（60-150cm） |
| `Omnihand/` | 灵巧手USB串口控制（手势、速度、使能） |
| `doc/` | 机器人通信协议、代码模板等文档 |

## 快速开始

1. 连接相机和灵巧手（USB串口）
2. 修改 `estun_config.yaml` 中的 `host`（本机IP）和 `hand.port`（串口号）
3. 运行 `python estun_run.py`
4. 机器人作为TCP客户端连接本服务器，接收6D目标位姿

## 配置说明

所有配置通过YAML文件管理，无需改代码：

- **更换检测模型/目标类别** → 修改 `RealSense_R/config.yaml` 中的 `model.name` 和 `model.target_class_id`（COCO类别ID）
- **切换运行设备** → 修改 `RealSense_R/config.yaml` 中的 `model.device`（gpu/cpu）
- **更换相机** → 替换对应相机模块目录即可（各模块接口模式一致）
- **修改机器人参数** → 修改 `estun_config.yaml`
- **调整抓取位置** → 微调 `estun_config.yaml` 中的 `gripper_offset.t_offset`
- **切换抓取模式** → 修改 `estun_config.yaml` 中的 `orientation_mode`（free/vertical/horizontal）

## 核心流程

```
相机检测 → 手眼标定转末端系 → 工具偏移补偿 → 转基坐标系 → TCP发送 → 灵巧手抓取
```

详细位姿转换算法见 `GRIP_Process.md`。

## 关键概念

- **手眼标定**：标定相机到末端法兰的变换矩阵，结果存储在 `calib_data/calibration_result.yaml`，支持Tsai/Park/Horaud等多种算法
- **工具偏移**：补偿灵巧手物理安装角度（C=-90°）带来的位置和旋转偏差
- **代际计数器**：防止客户端断开重连后旧帧数据污染新连接的目标位姿
- **抓取模式**：free（跟随物体朝向）、vertical（末端Z轴水平约束）、horizontal（待实现）