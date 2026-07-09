# System Patterns — Project_hand

## 系统架构

```text
Camera A (眼在手上, 手动像素映射)    Camera B (眼在手外, rs.align)
        ↓                                    ↓
  Handle+Cup 6D 检测                   PersonDetector (人体X)
        ↓                                    ↓
  手眼标定 (cam→ee)                    全示教 delivery.X_TEACH
        ↓                                    ↓
  工具偏移补偿                              ──→ Cup P3 X
        ↓                               YZ/ABC ← config 预示教
  转基系 (TCP pack_three)                    ↓
        ↓                               HandInCup 判定
        └──────────┬──────────────────────┘
                   ↓
          [ESTUN 机器人 + Omnihand 灵巧手]
```

## 关键技术决策

| 决策 | 选择 | 理由 |
| ------ | ------ | ------ |
| 检测框架 | YOLOv8 / YOLO-World / 微调 YOLOv8n / YOLOv8-Pose | 场景多样化, 各有利弊; Pose 用于 HandInCup |
| 位姿计算 | PCA + PointNet++ (主), SAM2 (把手) | PCA 稳定高效, SAM 适合不规则物体 |
| 深度对齐 | rs.align (新相机) / 手动像素映射 (旧相机) | 旧相机内参损坏无法使用 rs.align |
| 通信协议 | TCP 单 Session 全流程 | 简单可靠, 与 ESTUN 兼容; 避免多连接状态同步 |
| 配置管理 | YAML 文件 (模块级 config.yaml) | 不改代码可调参数 |
| HandInCup | YOLOv8m(cup bbox) + YOLOv8n-Pose(wrist) | 2D 判定, 零额外依赖, 独立可测试 |
| Cup P3 递送 | X 动态 (Camera B) + YZ/ABC 固定 (示教) | 安全: YZ/ABC 预教避免碰撞, X 自适应人体位置 |

## 设计模式

### 检测器基类模式

- `BaseRealSenseDetector` (RealSense/realsense_shared.py) 为基类
- 子类通过类属性开关控制: `USE_YOLO_WORLD`, `USE_FINETUNED_KETTLE`, `USE_SAM`, `USE_RED_TAPE`
- 检测与位姿计算分离: `Detector` 产出 mask/bbox, `6DDetector` 扩展位姿

### 共享工具库

- `Estun/utils_shared.py`: 所有主入口共用坐标变换和灵巧手控制
- 函数式接口替代类封装: `init_hand`, `grip_object`, `release_object` 等

### 位姿稳定

- `PoseStabilizer` 类: 缓存最近 N 帧, 位置+角度变化均低于阈值时输出均值
- 代际计数器 (`_gf`): 客户端断开重连时自增, 防旧帧污染新连接
- 仅检测帧执行, 避免跨帧零值/旧值污染

### 双检测器共享 Pipeline

- Cup 检测器以 `start_pipeline=False` 初始化, 通过 `inject_frames()` 复用 Handle 检测器的帧
- 避免重复打开相机硬件

### 通信协议: `pack_three` + 新指令

- `pack_three`: `<P1>[P2]{P3}`, 与 ESTUN `TranStrToCpos` 括号标记约定匹配
- 新增 Phase D 指令:
  - `READY` (Robot→PC): 到达 P3, 等待 HandInCup 判定
  - `CONTINUE` (PC→Robot): HandInCup=TRUE, 松手退避
  - `ABORT` (PC→Robot): HandInCup=FALSE, 放回杯子

### P3 多倒水角度示教

- `estun_config.yaml` 的 `p3_teach_offset.{side}` 按 B 角度数 key (10/15/20/30) 索引
- `pour_angle` 顶层 key 控制当前使用哪个角度
- 加载时按 `str(pour_angle)` 查找, 不存在则回退到第一个可用角度

### Cup P3 递送位姿模式

- X: 全示教固定值 `delivery.X_TEACH` (3D→1D 人体标定工具就绪, 暂未接入)
- Y, Z, A, B, C: 固定 — `EyeToHand/config.yaml` delivery 段预示教
- 人体位置仅在 CUP 指令时取一次最新稳定值, PersonDetector 随后暂停

### HandInCup 检测模式

- 双模型 2D 判定: YOLOv8m (杯 bbox class=41) + YOLOv8n-Pose (手腕 kpt 9/10)
- 连续 N 帧手腕在杯 bbox 扩展区内 → `is_hand_on_cup = True`
- 独立文件 `EyeToHand/hand_in_cup.py`, 可脱离状态机单独测试

### PersonDetector / HandInCup 共用 Camera B Pipeline

- `PersonDetector` 打开 Camera B pipeline
- main loop 通过标志位 (`stop_person_detector` / `hand_in_cup_active`) 切换检测逻辑
- Pipeline 始终保持运行, 帧来源不变

### 状态机 — 2 状态线性往返

```text
IDLE  ←──────────────────────→  HANDLE_GRASP_AND_POUR
    (PersonDetector 检测到人)       (SocketClose)
```

- 全流程在单 TCP Session 内完成: Phase A (初始位姿) → B (Handle) → C (指令循环 + Cup) → D (HandInCup 判定)
- 异常回 IDLE, 清理所有资源后重新等待

## 模块关系

```text
Orchestrator/                        ← 系统入口
  ├── main.py                        ← 双相机主循环 + Person/HandInCup 切换
  └── state_machine.py               ← 2 状态 + 单 Session TCP handler

Estun/                               ← 顶层入口 (单相机)
  ├── estun_handle_cup_run.py        ← RealSense Handle+Cup 融合 (参考实现)
  ├── estun_handle_run.py            ← RealSense 把手抓取
  ├── estun_run.py                   ← RealSense 实时瓶身
  ├── estun_epiceye_run.py           ← EpicEye 单次瓶身
  └── utils_shared.py                ← 共享坐标变换+灵巧手

RealSense_R/  (旧相机, 手动像素映射) ← Camera A 检测模块
  ├── test_6D.py / test_handle_6D.py / test_cup_6D.py
  └── realsense_shared.py            ← BaseRealSenseDetector

RealSense/  (新相机, rs.align)       ← Camera B 基类
  └── realsense_shared.py            ← BaseRealSenseDetector

EyeToHand/                           ← 眼在手外模块
  ├── person_detector.py             ← PersonDetector (人体 X)
  ├── hand_in_cup.py                 ← HandInCupDetector (YOLOv8m+YOLOv8n-Pose)
  ├── detector.py                    ← load_calibration / person_to_base_x
  ├── config.yaml                    ← Person + 递送(全示教) + HandInCup 配置
  └── calibration/
      ├── calibrate.py               ← 3D→1D 人体标定工具
      └── result.yaml                ← a,b,c,d 标定参数

EpicEye/   Omnihand/   calib_data/   ← 灰度相机 / 灵巧手 / 手眼标定
