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
- 相机: Camera A = RealSense D435i (眼在手上), Camera B = UGREEN 1080P (眼在手外, 纯RGB)
- 检测算法: YOLOv8n (微调kettle) + YOLOv8m/YOLOv8n (cup) + MediaPipe Hands + HSV 青色贴纸 (把手, 含旋转判档)
- 位姿计算: 检测动态 6D + 旋转 band 补偿 (C=C0-θ) + 深度反投影; 固定 ABC 为回退模式
- 环境: Python 3.11+, Conda (Proj), PyTorch CUDA 12.6
