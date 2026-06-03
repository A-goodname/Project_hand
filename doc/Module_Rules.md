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
- SAM 分割脚本 `test_SAM.py` 用于把手检测

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
- **两个主入口**:
  - `estun_run.py`: RealSense 实时版, 整合 RealSense 6D 检测 + 位姿转换 + TCP + 灵巧手
  - `estun_epiceye_run.py`: EpicEye 单次触发版, 单帧拍照 + 灰度 YOLO 6D 检测 + 位姿转换 + TCP + 灵巧手
- **共享工具库 `utils_shared.py`**: 两个主入口共用以下函数:
  - 坐标变换: `euler_to_rotation_matrix`, `transform_pose_to_base`, `compute_gripper_offset_transform`, `compute_grasp_rotation`, `post_correct_angles`
  - 灵巧手控制: `init_hand`, `grip_object`, `release_object`, `disable_hand`
  - 物体 Y 轴统一约定: `compute_grasp_rotation` 中将物体 Y 轴通过基系 Z 分量判断符号, 统一指向竖直向下, 消除 PCA 符号歧义
- **演示脚本** (无硬件依赖):
  - `estun_demo.py`: RealSense 版位姿转换链路验证
  - `estun_epiceye_demo.py`: EpicEye 版位姿转换链路验证

## 手眼标定模块

存放路径: `calib_data/`

- 标定数据目录: `calib_data/images/` (图片), `poses.csv` (位姿)
- 棋盘格参数在 `calib_data/calib_config.yaml` 中配置
- 求解 5 种手眼方法 (Tsai/Park/Horaud/Andreff/Daniilidis), 用 Board-in-base 一致性评分选最优
- 输出文件: `calibration_result.yaml` (含全部方法), `board_in_base.yaml` (仅最佳方法的标定板位姿)
- 数据采集: `collect_data.py`, 角点可视化: `visualize_corners.py`