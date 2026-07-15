# Tech Context — Project_hand

## 技术栈

| 类别 | 技术 |
| ------ | ------ |
| 语言 | Python 3.11+ |
| 深度学习 | PyTorch 2.x (CUDA 12.6), Ultralytics YOLOv8 / YOLOv8-Pose |
| 计算机视觉 | OpenCV, NumPy, MediaPipe Hands |
| 3D 处理 | pyrealsense2 (仅 Camera A), 深度反投影 |
| 通信 | TCP socket, USB 串口 (PySerial), OpenCV VideoCapture (Camera B UGREEN) |
| 配置 | PyYAML |

## 开发环境

- **OS**: Windows 10
- **IDE**: VS Code
- **Conda 环境**: `D:\Environment\miniconda\envs\Proj\python.exe`
- **GPU**: MX450 (可选)
- **Python 版本**: 3.11+

## 项目目录结构

- `Orchestrator/`: **主入口** — 4 阶段调度 + 双相机检测 + TCP 服务
- `Estun/`: 单相机入口 (运行/演示/工具)
- `RealSense_R/`: Camera A D435i 旧版 (手动像素映射)
- `RealSense/`: Camera A D435i 新版 (rs.align)
- `EyeToHand/`: Camera B UGREEN 1080P — PersonDetector + HandInCup
- `EpicEye/`: Epic Eye 灰度相机 (非主用)
- `Orbbec/`: Orbbec Gemini 215 微距相机 (非主用)
- `Omnihand/`: 灵巧手控制
- `calib_data/`: 手眼标定工具 + 标定结果
- `KettleTraining/`: 水壶检测模型微调
- `doc/`: 文档 + CRP 文件
- `logs/`: 运行日志

## 依赖安装

1. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
2. EpicEye SDK (手动打包 whl)
3. `pip install -r requirements.txt`

## 关键约束

- RealSense_R 不依赖 `rs.align` (旧相机内参损坏)
- Camera B 为 UGREEN 1080P, 纯 RGB 无深度 — 3D→1D 人体标定已废弃
- 水壶微调模型路径: `model/kettle_yolov8n_finetune.pt`
- 手眼标定结果存储在 YAML 文件, 各模块独立读取
- HandInCup: YOLOv8n (杯检测, class=41) + MediaPipe Hands (手部 21 landmarks, hand_landmarker.task)
- HandInCup 模型路径可配置: `EyeToHand/config.yaml` → `hand_in_cup` 段
