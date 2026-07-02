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
- `epiceye_config.yaml` 内含 `handeye_matrix` (相机->末端法兰, 4x4 齐次矩阵), 供 `Estun/estun_epiceye_run.py` 使用

## RealSense_R 模块

存放路径: `RealSense_R/`

- 相机参数从 `config.yaml` 和 `D435i_info.yaml` 加载 (不依赖 SDK 读取的 NaN 参数)
- 彩色->深度像素映射通过 `rs2_project_color_pixel_to_depth_pixel` 手动计算
- **不依赖 `rs.align`** (相机硬件参数损坏, 结果不可靠)
- 与 `RealSense/` (新版) 的基类 `BaseRealSenseDetector` 及共享工具函数完全独立, 互不依赖
- `BaseRealSenseDetector.__init__` 支持 `serial_number=None` 参数: `None` 时从 `RealSense_R/D435i_info.yaml` 读取 serial, 非 `None` 时使用传入值 (用于 Orchestrator 双相机场景)

## RealSense 模块

存放路径: `RealSense/`

- **rs.align 简化版**: 适用于内参完好的 D435i 相机, 使用 `rs.align(rs.stream.color)` 代替 `rs2_project_color_pixel_to_depth_pixel` 手动映射
- 相机参数从 `D435i_info.yaml` 加载彩色内参 (与 `RealSense_R/` 共用 YAML 格式)
- 基类 `BaseRealSenseDetector` 及共享工具函数 (`realsense_shared.py`) 与 `RealSense_R/` 版本独立, 但结构和接口高度相似
- 三种互斥检测模式: 标准 YOLOv8 + YOLO-World + 微调 YOLOv8n (优先级: 微调 > YOLO-World > 标准)
- 把手检测依赖 SAM, 使用 `scanline_find_handle` 水平扫描线分离壶身与把手
- 红色胶带辅助检测通过 `detect_red_tape` 实现
- **检测脚本清单**:
  - `test_6D.py`: 标准 6D 检测 (瓶身)
  - `test_3D.py`: 纯 3D 检测
  - `test_handle_6D.py`: 把手 6D 位姿检测 (SAM + 扫描线分离)
  - `test_handle_3D.py`: 把手纯 3D 检测
  - `test_cup_6D.py`: 水杯 6D 检测 + 杯口位置 + 倒水目标点 (新增)
  - `test_align.py`: rs.align 对齐链路验证
- 配置入口: `RealSense/config.yaml`, 包含 `cup` 段 (水杯几何参数)
- 日志输出至 `logs/RealSense/`
- 两种模块差异小结:

  | 特性 | RealSense_R (旧) | RealSense (新) |
  | ------ | ---------------- | ---------------- |
  | 深度对齐 | `rs2_project_color_pixel_to_depth_pixel` 手动映射 | `rs.align` 自动对齐 |
  | 基类文件 | `RealSense_R/realsense_shared.py` | `RealSense/realsense_shared.py` |
  | 配置文件 | `RealSense_R/config.yaml` | `RealSense/config.yaml` |
  | 适用相机 | 内参受损的旧相机 | 内参完好的新相机 |
  | 检测脚本 | `test_6D.py` / `test_handle_6D.py` / `test_cup_6D.py` | 同名脚本, 使用 `rs.align` 简化版本 |
  | 融合入口 | `Estun/estun_handle_cup_run.py` 引用 `RealSense_R/` | 待迁移 |

- `BaseRealSenseDetector.__init__` 支持 `serial_number=None` 参数: `None` 时从 `RealSense/D435i_info.yaml` 读取 serial, 非 `None` 时使用传入值 (与 `RealSense_R/` 同步, 用于双相机场景)
- YOLO 使用标准彩色模型 (非灰度模型)
- **支持三种水壶检测模型**: 标准 YOLOv8 (瓶身), YOLO-World (零样本), 微调 YOLOv8n (高置信度水壶, 需预先训练)
- **红色胶带辅助检测**: 通过 `red_tape` 配置段控制 (enabled/min_area), `detect_red_tape()` 使用 HSV 双阈值 + 中位数中心 + 最大连通分量; 替换把手 3D 位置, 旋转保持 PCA 不变
- **共享工具库 `realsense_shared.py`**: 包含检测器基类 `BaseRealSenseDetector` 及通用工具函数
- **基类类属性配置**: 子类通过类属性开关控制: `USE_YOLO_WORLD`, `USE_FINETUNED_KETTLE`, `USE_SAM`, `USE_RED_TAPE` 等
- **瓶身检测脚本**:
  - `test_6D.py`: 标准 6D 检测 (PCA+PointNet++ 计算物体位姿); 使用 `SMOOTH_POSE_ALPHA` 进行位姿平滑
  - `test_3D.py`: 纯 3D 检测 (无 6D 姿态输出)
  - `yolo_detect.py`: 纯 YOLO bbox 检测
  - `yolo_world_detect.py`: YOLO-World 零样本检测 (支持文本 prompt)
- **把手检测脚本**:
  - `test_SAM_handle.py`: RealSenseHandleDetector 类, **微调 YOLOv8n + SAM2** + 扫描线分离把手, 输出 handle_mask + bbox; 已从 YOLO-World+跳帧模式 重构为 微调YOLO+手动触发模式
  - `test_handle_6D.py`: RealSenseHandle6DDetector 类, 继承 HandleDetector 并扩展 6D 位姿计算 (PCA+PointNet++), 供 Estun/estun_handle_run.py / Estun/estun_handle_demo.py 使用; 含 PoseSmoother 位姿平滑 (仅检测帧执行)
  - `test_handle_3D.py`: 把手纯 3D 检测
- **把手参数配置**: `RealSense_R/config.yaml` 中 `handle` 段配置跳帧 `skip_frames` (默认 90)、ROI 扩展 `roi_expand_ratio`、把手侧 `handle_side`、中心偏差 `handle_center_deviation_ratio` (双向剔除) 等
- **水杯检测脚本**:
  - `test_cup_6D.py`: RealSenseCup6DDetector 类, 继承 BaseRealSenseDetector, 使用标准 YOLOv8m COCO cup (class_id=41), 输出 6D 位姿 + 杯口位置 (rim_pos) + 倒水目标 (pour_target)
  - 杯口方向确定依赖像素 Y 坐标 (Y 小=画面顶部=杯口), 替代旧版 Z 深度启发式
  - 主轴方向指数平滑 (AXIS_SMOOTH_ALPHA=0.7) 抑制帧间 B 角跳动
  - 配置见 `config.yaml` 中 `cup` 段: height/rim_radius/rim_offset_ratio, 需实际测量填写
- **把手+水杯融合脚本**:
  - `Estun/estun_handle_cup_demo.py`: 双检测器共享 pipeline 的位姿转换验证 Demo
  - `Estun/estun_handle_cup_run.py`: 全流程运行版, 含 TCP 通信 + CUP 指令扩展
  - Cup 检测器以 `start_pipeline=False` 初始化, 通过 `inject_frames()` 复用 Handle 检测器的帧
  - 完整流程文档见 `doc/Handle_Cup_Process.md`

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
- **使能可靠性**: `init_hand()` 在 `enable()` 后通过 `is_enabled()` (→ `query_enable()`) 验证硬件实际状态, 未确认则自动重试 (默认 10 次 / 0.5s 间隔). 全部失败自动 fallback 到模拟模式
- **状态检查**: `is_hand_available()` 检查 `_hand is not None`, 供 run 脚本判断灵巧手是否真正可用. run 脚本应使用此函数而非乐观假设 `_hand_available = True`

## Estun 机器人模块

存放路径: `Estun/` 包目录

- 配置入口 `Estun/estun_config.yaml`, 支持多机器人配置
- 通信协议参数在 `Estun/estun_config.yaml` 中定义 (host/port/remote_ip 等)
- 标定结果从 `calib_data/calibration_result.yaml` 读取 (RealSense 版使用), 或从 `EpicEye/epiceye_config.yaml` 读取手眼矩阵 (EpicEye 版使用)
- **顶层配置项**:
  - `orientation_mode`: 放置方向修正模式: vertical | horizontal | free | fixed
  - `grasp_direction`: 抓取方向: right (右手/掌心向左, Y朝下) | left (左手/掌心向右, Y朝上)
  - `fixed_abc`: PCA 失效时固定姿态, 左右手独立配置 `right`/`left`
- **三个主入口**:
  - `Estun/estun_run.py`: RealSense 实时版 (瓶身), 整合 RealSense 6D 检测 + 位姿转换 + TCP + 灵巧手
  - `Estun/estun_epiceye_run.py`: EpicEye 单次触发版, 单帧拍照 + 灰度 YOLO 6D 检测 + 位姿转换 + TCP + 灵巧手
  - `Estun/estun_handle_run.py`: RealSense Handle 6D 抓取系统, 整合微调 YOLOv8n + SAM2 + 扫描线分离把手, 独立于瓶身检测; 发送 3 点位姿 (P1 抓取点 / P2 预抓取点 / P3 目标点)
- **共享工具库 `Estun/utils_shared.py`**: 所有入口共用以下函数:
  - 坐标变换: `euler_to_rotation_matrix`, `transform_pose_to_base`, `compute_gripper_offset_transform`, `compute_grasp_rotation`, `post_correct_angles`
  - 灵巧手控制: `init_hand`, `grip_object`, `release_object`, `disable_hand`, `hand_pre_grip`, `hand_grip_done`, `hand_reset`
  - 物体 Y 轴统一约定: `compute_grasp_rotation` 中将物体 Y 轴通过基系 Z 分量判断符号, 根据 `grasp_direction` 统一指向 (right=向下, left=向上), 消除 PCA 符号歧义
- **Handle 入口特有机制**:
  - 2026-06-11 已重构: `HandController` 类拆为 `utils_shared.py` 中函数式接口 (`hand_pre_grip`/`hand_grip_done`/`hand_reset`), 移除各入口对 `OmniHandGesture` 的直接依赖
  - `PoseStabilizer` 类: 缓存最近 N 帧位姿, 当位置+角度变化均低于阈值时输出稳定均值; 内置代际计数器防止跨连接数据污染; **注意**: `add()` 仅检测帧调用, 在相机系稳定后做一次 `cam→ee`, 避免跨帧零值/旧值污染 (commit e347717)
  - 把手偏移配置: `Estun/estun_config.yaml` 中 `handle_grasp.gripper_offset` (t_handle_offset / r_handle_offset) 独立于瓶身偏移; `handle_grasp` 段定义 P2/P3 的 pre_offset / target_offset, **偏移参考系为 grasp ee frame** (commit 24e26ba)
  - 跳帧控制: 每 `handle.skip_frames` (来自 RealSense_R/config.yaml) 帧调用一次 detect_handle_6d, 大幅降低 GPU 负载
  - C 值约束: vertical/free 模式下 P1 的 C 角固定到 `[-100, -80]`, 使手指垂直于把手
- **演示脚本** (无硬件依赖):
  - `Estun/estun_demo.py`: RealSense 版位姿转换链路验证 (瓶身)
  - `Estun/estun_epiceye_demo.py`: EpicEye 版位姿转换链路验证 (瓶身)
  - `Estun/estun_handle_demo.py`: RealSense Handle 版位姿转换链路验证 (把手), 剥离 TCP/灵巧手/CRP 指令循环; 支持 fine_tune_mode 反算新偏移
  - `Estun/estun_cup_demo.py`: 水杯单检测位姿转换验证 (独立 pipeline); 从 RealSense config 读取 `cup.handle_side`, 按 handle_side 选择偏移 (t_cup_offset / t_cup_offset_left) 和角度约束 (back: B[70,90]+C[-120,-100]; left: 仅 C[-100,-80])
  - `Estun/estun_handle_cup_demo.py`: Handle+Cup 融合版位姿转换验证 (共享 pipeline)
- **水杯偏移配置**: `estun_config.yaml` 中 `cup_grasp.gripper_offset.t_cup_offset` (back 模式) 和 `t_cup_offset_left` (left 模式)
- **TCP 模拟脚本**:
  - `Estun/estun_test.py`: TCP 服务器模拟, 持续发送位姿并处理 GRIP/RELEASE 指令
- **导入约定**:
  - 所有文件统一使用 `from Estun.utils_shared import ...` 绝对导入
  - 文件顶部固定 `_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` + `sys.path.insert(0, _project_root)`

## 手眼标定模块

存放路径: `calib_data/`

- 标定数据目录: `calib_data/images/` (图片), `poses.csv` (位姿)
- 棋盘格参数在 `calib_data/calib_config.yaml` 中配置
- 求解 5 种手眼方法 (Tsai/Park/Horaud/Andreff/Daniilidis), 用 Board-in-base 一致性评分选最优
- 输出文件: `calibration_result.yaml` (含全部方法), `board_in_base.yaml` (仅最佳方法的标定板位姿)
- 数据采集: `collect_data.py`, 角点可视化: `visualize_corners.py`

## KettleTraining 模块

存放路径: `KettleTraining/`

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

## EyeToHand 眼在手外模块

存放路径: `EyeToHand/`

- 外置固定相机 (RealSense D435i, serial 342222071364), 使用 `RealSense_R/` 基类 (手动像素映射, 内参损坏)
- 人体检测使用标准 YOLOv8 COCO person (class_id=0), 无需额外模型
- 位姿稳定: `PersonStabilizer` 窗口稳定器 (window=10, pos_thresh=0.03m), 参考 `PoseStabilizer`
- 坐标转换: `detector.py` 提供 `load_calibration()` / `camera_x_to_base_x()` 工具函数
- 标定工具: `calibration/calibrate.py` — 1D 最小二乘线性回归 (N≥2 点, 建议 3-5 点, 范围尽量大), 红色胶带辅助检测
  - 前置: 机械臂末端 YZ/ABC = 预示教递送值, 仅沿 X 移动
  - 操作: 'c' 采集 → 'f' 拟合 → 's' 保存
- 递送策略: YZ/ABC 预示教固定 + X 动态从人体检测获取 + 固定 2s 定时松手 (不依赖手势识别)
- 配置入口: `EyeToHand/config.yaml`, `stabilizer` 段控制稳定参数, `delivery` 段控制递送位置
- `CalibrationDetector` 直接继承 `BaseRealSenseDetector`, 与 `person_detector.py` 共享同一相机绑定模式
- 导入约定: 与 `RealSense_R/` 模块约定一致 (BaseRealSenseDetector + 手动像素映射)

## Orchestrator 双相机流程编排

存放路径: `Orchestrator/`

- **模块角色**: 串联眼在手上 (Handle+Cup 抓取倒水) + 眼在手外 (人体检测递送) 的全流程状态机
- **状态流转**: IDLE → HANDLE_GRASP_AND_POUR → LOCATE_PERSON → APPROACH_PERSON → WAIT_AND_RELEASE → RETREAT → IDLE
- 核心文件:
  - `state_machine.py`: `StateMachine` 引擎 + 6 个 `StateHandler` 实现 + `HandleCupRunner` (封装 Handle+Cup 全流程) + 共享工具 (PoseStabilizer/fmt_pose/pack_three)
  - `main.py`: 双相机主循环, 每 tick 获取 Camera B 帧 → 驱动状态机 → 可视化
- **关键设计**:
  - Handle+Cup 逻辑从 `estun_handle_cup_run.py` 复制适配 (不改原文件), 仅增加 completion event 通知
  - Camera B (eye-to-hand) 全程运行; Camera A (eye-in-hand) 由状态机管理生命周期
  - 递送阶段使用独立 TCP 单 pose 协议: `<x,y,z,A,B,C>` + READY 确认 + RELEASE 指令
  - 异常回退: 任何状态异常 → on_error → IDLE
- **serial 绑定**: `BaseRealSenseDetector.__init__` 新增 `serial_number=None` 参数, Orchestrator 传入眼在手外 serial
- 配置入口: `Estun/estun_config.yaml` (Handle+Cup + TCP), `EyeToHand/config.yaml` (人体检测 + 递送), `RealSense_R/config.yaml` (Handle+Cup 检测)
