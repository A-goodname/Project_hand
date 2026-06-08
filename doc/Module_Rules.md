# 模块规则文档

本文档汇总本项目中各硬件模块的开发规则与约定.

## EpicEye 模块

存放路径: `EpicEye/`

- 配置文件 `EpicEye/epiceye_config.yaml` 是唯一配置入口
- YOLO 检测仅使用灰度模型 `yolo11n-grayscale.pt`, 不保留彩色模型路径
- 3D/6D 算法统一: online/offline 均使用 PLY 点云投影筛选 bbox 内点的方案 (不再区分点云格式)
- 点云/位置单位统一为**米** (PLY 文件存储单位为毫米, 代码内部需转换)
- `_filter_ply_by_bbox`: PLY 点云投影回图像 UV, 筛选 bbox 内点 + 深度离群过滤 (使用 bbox 中心 20% 区域 z_median 作为锚点)
- `test_3D.py` 和 `test_6D.py` 共享 `_filter_ply_by_bbox` 和 `process_and_display()` 接口模式
- 物体坐标系约定: PCA 输出的 Z 轴指向相机 (target_z[2] > 0 时翻转), Y 轴为主方向 (瓶身轴线)
- 相机内参: online 从相机自动获取, offline 从配置文件读取
- 保存目录相对 EpicEye/, 日志目录相对项目根目录
- `epiceye_config.yaml` 内含 `handeye_matrix` (相机->末端法兰, 4x4 齐次矩阵), 供 `estun_epiceye_run.py` 使用

## RealSense_R 模块

存放路径: `RealSense_R/`

- 相机参数从 `config.yaml` 和 `D435i_info.yaml` 加载 (不依赖 SDK 读取的 NaN 参数)
- 彩色->深度像素映射通过 `rs2_project_color_pixel_to_depth_pixel` 手动计算
- 不依赖 `rs.align` (相机硬件参数损坏, 结果不可靠)
- YOLO 使用标准彩色模型 (非灰度模型)
- **瓶身检测脚本**:
  - `test_6D.py`: 标准 6D 检测 (PCA+PointNet++ 计算物体位姿)
  - `test_3D.py`: 纯 3D 检测 (无 6D 姿态输出)
  - `yolo_detect.py`: 纯 YOLO bbox 检测
  - `yolo_world_detect.py`: YOLO-World 零样本检测 (支持文本 prompt)
- **把手检测脚本**:
  - `test_SAM_handle.py`: RealSenseHandleDetector 类, YOLO-World + SAM + 扫描线分离把手, 输出 handle_mask + bbox
  - `test_handle_6D.py`: RealSenseHandle6DDetector 类, 继承 HandleDetector 并扩展 6D 位姿计算 (PCA+PointNet++), 供 estun_handle_run.py / estun_handle_demo.py 使用
  - `test_handle_3D.py`: 把手纯 3D 检测

## Orbbec 模块

存放路径: `Orbbec/`

- 配置文件 `Orbbec/config.yaml` 管理相机参数
- 设备信息通过 `get_device_info_orb.py` 获取并保存到 `device_info.yaml`
- 3D 测试脚本 `test_3D.py` 使用 SDK 直接获取 depth 对齐数据

## Omnihand 灵巧手模块

存放路径: `Omnihand/`

- USB 串口通信: `omnihand_usb_serial.py` 封装串口收发
- 手势控制: `omnihand_gesture.py` 通过 `actlist.json` 解析动作列表
- 速度控制: `omnihand_speed_control.py` 控制手指运动速度
- 启用/关闭: `omnihand_enable.py` 控制手部使能

## Estun 机器人模块

存放路径: 项目根目录, 主要文件见目录表

- 配置入口 `estun_config.yaml`, 支持多机器人配置
- 通信协议参数在 `estun_config.yaml` 中定义 (host/port/remote_ip 等)
- 标定结果从 `calib_data/calibration_result.yaml` 读取 (RealSense 版使用), 或从 `EpicEye/epiceye_config.yaml` 读取手眼矩阵 (EpicEye 版使用)
- **三个主入口**:
  - `estun_run.py`: RealSense 实时版 (瓶身), 整合 RealSense 6D 检测 + 位姿转换 + TCP + 灵巧手
  - `estun_epiceye_run.py`: EpicEye 单次触发版, 单帧拍照 + 灰度 YOLO 6D 检测 + 位姿转换 + TCP + 灵巧手
  - `estun_handle_run.py`: RealSense Handle 6D 抓取系统, 整合 YOLO-World + SAM + 扫描线分离把手, 独立于瓶身检测; 发送 3 点位姿 (P1 抓取点 / P2 预抓取点 / P3 目标点), 直接使用 OmniHandGesture 手势 (不依赖 utils_shared)
- **共享工具库 `utils_shared.py`**: 两个瓶身入口共用以下函数:
  - 坐标变换: `euler_to_rotation_matrix`, `transform_pose_to_base`, `compute_gripper_offset_transform`, `compute_grasp_rotation`, `post_correct_angles`
  - 灵巧手控制: `init_hand`, `grip_object`, `release_object`, `disable_hand`
  - 物体 Y 轴统一约定: `compute_grasp_rotation` 中将物体 Y 轴通过基系 Z 分量判断符号, 统一指向竖直向下, 消除 PCA 符号歧义
- **Handle 入口特有机制**:
  - `HandController` 类: 封装 OmniHandGesture 的初始化/使能/手势执行/清理流程, 使用 PREGRIP/GRIPDONE/RESET 手势
  - `PoseStabilizer` 类: 缓存最近 N 帧位姿, 当位置+角度变化均低于阈值时输出稳定均值; 内置代际计数器防止跨连接数据污染
  - 把手偏移配置: `estun_config.yaml` 中 `handle_gripper_offset` (t_handle_offset / r_handle_offset) 独立于瓶身偏移; `handle_grasp` 段定义 P2/P3 的 pre_offset / target_offset
  - 跳帧控制: 每 `handle.skip_frames` (来自 RealSense_R/config.yaml) 帧调用一次 detect_handle_6d, 大幅降低 GPU 负载
- **演示脚本** (无硬件依赖):
  - `estun_demo.py`: RealSense 版位姿转换链路验证 (瓶身)
  - `estun_epiceye_demo.py`: EpicEye 版位姿转换链路验证 (瓶身)
  - `estun_handle_demo.py`: RealSense Handle 版位姿转换链路验证 (把手), 剥离 TCP/灵巧手/CRP 指令循环; 支持 fine_tune_mode 反算新偏移

## 手眼标定模块

存放路径: `calib_data/`

- 标定数据目录: `calib_data/images/` (图片), `poses.csv` (位姿)
- 棋盘格参数在 `calib_data/calib_config.yaml` 中配置
- 求解 5 种手眼方法 (Tsai/Park/Horaud/Andreff/Daniilidis), 用 Board-in-base 一致性评分选最优
- 输出文件: `calibration_result.yaml` (含全部方法), `board_in_base.yaml` (仅最佳方法的标定板位姿)
- 数据采集: `collect_data.py`, 角点可视化: `visualize_corners.py`

## Kettle_training 模块

存放路径: `kettle_training/`

- 独立的水壶检测模型微调流程, 不依赖项目其他模块的配置文件
- 使用 YOLOv8n (轻量级) 替代 YOLO-World 进行水壶零样本检测
- 目录结构: `images/` (原始图片), `labels/` (YOLO 格式标注), `dataset.yaml` (数据集配置)
- 脚本清单:
  - `collect_kettle_data.py`: 使用 RealSense 采集多角度水壶照片
  - `label_tool.py`: OpenCV 手动 bbox 标注工具 (替代 LabelImg, 规避 PyQt5 类型错误)
  - `train_kettle.py`: 基于 COCO 预训练权重的微调训练脚本
  - `test_finetuned.py`: 独立测试脚本, 直接加载微调模型进行实时检测 (不依赖 config.yaml)
- 训练配置: epochs=50, imgsz=640, batch=8, amp=False, lr0=0.001
- 预期指标: mAP50 ≥ 0.99, Recall=1.0, Precision ≥ 0.99
- 训练时间: YOLOv8n 在 MX450 上约 9 分钟完成 50 epochs
- 模型输出路径: `runs/train/kettle_finetune/weights/best.pt`, 复制至 `model/kettle_yolov8n_finetune.pt`
- `labels.cache` 为 Ultralytics 自动生成的标签验证缓存, 已在 `.gitignore` 中忽略
