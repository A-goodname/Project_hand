# System Patterns — Project_hand

## 系统架构

```text
Camera A (眼在手上, 手动像素映射)    Camera B (眼在手外, UGREEN 1080P)
        ↓                                    ↓
  Handle+Cup 6D 检测                   PersonDetector (椅子+人就位)
        ↓                                    ↓
  手眼标定 (cam→ee)                    HandInCup (纯RGB判定)
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
| HandInCup | YOLOv8m(cup) + MediaPipe Hands(21 landmarks) | landmarks-in-bbox 计分 + 握拳 gate; 遮挡时杯ROI回退 |
| Cup P3 递送 | XYZ/ABC 全示教固定 (delivery.X_TEACH) | 3D→1D 人体标定工具就绪但暂未接入, X 暂用示教值 |

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

- **杯检测**: YOLOv8m (class=41) → 失败回退固定 ROI (`cup_roi_fallback`)
- **手检测**: MediaPipe Hands (21 landmarks, 帧间跟踪, RunningMode.VIDEO)
- **判定**: landmarks-in-bbox 计分制 — 最近手的 wrist(×2) + MCP(×1) + TIP(×1)
  - wrist 在框内 → score ≥ 2 即 hit
  - wrist 在框外 → score ≥ 5 即 hit (多手指在框内); 否则需 curled ≥ 2
- **curled_threshold**: 35px (720p), bbox_margin: 40px (720p) — 分辨率相关
- 纯 RGB 判定 (深度辅助已移除)
- 独立文件 `EyeToHand/hand_in_cup.py`, 可脱离状态机单独测试 (`--calibrate-roi` 标定工具)

### PersonDetector / HandInCup 共用 Camera B Pipeline

- `PersonDetector` 打开 UGREEN Camera B (OpenCV VideoCapture)
- Camera B 生命周期: IDLE 关 → GREETING enter 开 → HANDLE 继续
- main loop 通过标志位 (`run_seated_detection` / `hand_in_cup_active`) 切换检测逻辑
- 椅子检测: 每帧实时 YOLO, 取画面中心最近 (不锁定)
- 人就位: 非对称阈值 (进10出5) + `_chair_baseline` 尺寸锚定 (面积差≤30%更新)
- 纯 RGB 无深度 — HandInCup 判定仅依赖 landmarks-in-bbox 计分 + curled gate

### 4 阶段顺序调度 (v12 重构, 替代状态机)

```text
greeting ──(Phase 0 done)──→ handle_detect ──(targets_sent)──→ seated_check ──(ready_received)──→ hand_in_cup ──(hand_judged)──→ exit
```

| 阶段 | Camera A | Camera B | 检测器 | TCP |
|------|----------|----------|--------|-----|
| greeting | Phase 0 人检测→4步协议 | 椅子标定 (竞争不大) | YOLOv8n person | 8600 |
| handle_detect | Handle+Cup 全速 | **PAUSED** (不拉帧) | HandleCupDetector | 8080 Phase A-B |
| seated_check | 已关闭 | 人就位全速 | PersonDetector | 8080 Phase C |
| hand_in_cup | 已关闭 | HandInCup 全速 | HandInCupDetector | 8080 Phase D |

- **任何时刻只有一个检测器运行** — 消除 YOLO 竞争, 移除跳帧补丁
- Phase 过渡: TCP daemon 线程写入 `SharedContext` Event, main loop 轮询
- `hand_on_cup is None` → Phase C 松把手; `is not None` → Phase D 判定
- `_camera_a_to_close`: Phase B 后 daemon 线程保存引用, main 线程安全关闭 pipeline

## 模块关系

```text
Orchestrator/                        ← 系统入口
  ├── main.py                        ← 4-phase 主循环 + 可视化
  ├── state_machine.py               ← SharedContext + GreetingHandler + HandleHandler
  ├── handle_cup_detector.py         ← Camera A 轻量检测器
  ├── config.yaml                    ← 检测参数
  └── logger.py                      ← 终端+文件双输出

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
  ├── hand_in_cup.py                 ← HandInCupDetector (YOLOv8m+MediaPipe Hands)
  ├── detector.py                    ← load_calibration / person_to_base_x
  ├── config.yaml                    ← Person + 递送(全示教) + HandInCup 配置
  └── calibration/
      ├── calibrate.py               ← 3D→1D 人体标定工具
      └── result.yaml                ← a,b,c,d 标定参数

EpicEye/   Omnihand/   calib_data/   ← 灰度相机 / 灵巧手 / 手眼标定
