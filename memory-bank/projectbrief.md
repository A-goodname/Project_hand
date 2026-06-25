# Project Brief — Project_hand

## 项目概述

机械臂 + 灵巧手抓取系统. 通过相机检测抓取目标, 计算 6D 位姿, 经手眼标定和工具偏移补偿后, 通过 TCP 发送给 ESTUN 机器人, 控制灵巧手完成抓取动作.

## 核心需求

1. **视觉检测**: 支持多相机 (RealSense D435i / Epic Eye / Orbbec), 检测目标并计算 6D 位姿
2. **抓取控制**: 通过 TCP 协议向 ESTUN 机器人发送目标位姿, 控制灵巧手 (Omnihand) 完成抓取
3. **多抓取模式**: 瓶身抓取 / 把手抓取 / 水杯抓取 / 把手+水杯融合抓取

## 项目范围

- 机械臂: ESTUN 机器人 (TCP 通信)
- 灵巧手: Omnihand (USB 串口通信)
- 相机: RealSense D435i (主用), Epic Eye Pixel Pro (灰度备用), Orbbec Gemini 215 (微距备用)
- 检测算法: YOLOv8 / YOLO-World / 微调 YOLOv8n / SAM2
- 位姿计算: PCA + PointNet++ / 深度投影
- 环境: Python 3.11+, Conda (Proj), PyTorch CUDA 12.6
