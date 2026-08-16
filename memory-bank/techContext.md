# Tech Context — Project_hand

## 技术栈

| 类别 | 技术 |
| ------ | ------ |
| 语言 | Python 3.11+ |
| 深度学习 | PyTorch 2.x (CUDA 12.6), Ultralytics YOLOv8 / YOLOv8-Pose |
| 计算机视觉 | OpenCV, NumPy, MediaPipe Hands |
| 3D 处理 | pyrealsense2 (仅 Camera A), 深度反投影 |
| 通信 | TCP socket, USB 串口 (PySerial), OpenCV VideoCapture (Camera B UGREEN: Linux V4L2 / Windows DSHOW) |
| 设备探测 | Linux: `/sys/class/video4linux` 名称匹配 + ID_PATH 物理口 (同名相机区分); Windows: pygrabber (DSHOW) |
| 配置 | PyYAML |

## 开发环境

- **OS**: Windows 10 (原开发机)
- **IDE**: VS Code
- **Conda 环境**: `D:\Environment\miniconda\envs\Proj\python.exe`
- **GPU**: MX450 (可选)
- **Python 版本**: 3.11+

### Linux 工作机 (2026-08-14 起, 主用)

- **OS**: Ubuntu (Linux 7.0.0-28-generic)
- **Conda 环境**: `/home/user/miniconda3/envs/Proj` (Python 3.11.15)
- **GPU**: 可用 (torch 2.13.0+cu126, CUDA 可用)
- **已装依赖**: numpy 2.4.4, opencv-python / opencv-contrib-python 4.13.0.92, PyYAML 6.0.3, pyserial 3.5, pyrealsense2 2.56.5.9235, ultralytics 8.4.48, mediapipe 0.10.35
- **未装 (按需)**: EpicEye SDK, pyorbbecsdk2, pygrabber/comtypes (Windows 专属), 语音控制包 (sherpa-onnx / sounddevice / pypinyin / openai / websocket-client)
- **git 配置**: `core.autocrlf input` + `.gitattributes` (`* text=auto`), 与 Windows 端 `autocrlf=true` 兼容
- **灵巧手串口**: `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00` (STM32 VCP, 枚举为 /dev/ttyACM0; **禁 ttyACM 编号**); 460800; 节点 0x01; 当前用户已在 dialout 组直连 (2026-08-16 起全项目默认端口已统一为此路径)

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
4. 注意: pyrealsense2 需带构建后缀版本 (`==2.56.5.9235`),`==2.56.5` 匹配不到; mediapipe 会拉 opencv-contrib-python,需与 opencv-python 版本对齐 (当前均 4.13.0.92)

## 关键约束

- RealSense_R 不依赖 `rs.align` (旧相机内参损坏)
- Camera B 为 UGREEN 1080P, 纯 RGB 无深度 — 3D→1D 人体标定已废弃
- 水壶微调模型路径: `model/kettle_yolov8n_finetune.pt`
- 手眼标定结果存储在 YAML 文件, 各模块独立读取
- HandInCup: YOLOv8n (杯检测, class=41) + MediaPipe Hands (手部 21 landmarks, hand_landmarker.task)
- HandInCup 模型路径可配置: `EyeToHand/config.yaml` → `hand_in_cup` 段
- **/dev/videoN 编号不稳定** (插拔/驱动重绑会重排): 严禁硬编码索引, Camera B 必须按名称枚举 (EyeToHand/camera.py 的 sysfs 探测)
