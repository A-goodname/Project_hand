# System Patterns — Project_hand

## 系统架构

```text
[相机] → 6D检测 → 手眼标定(相机→法兰)
                          ↓
                   工具偏移补偿(法兰→夹爪)
                          ↓
                   转基坐标系(TCP发送)
                          ↓
                   [ESTUN机器人 + Omnihand灵巧手]
```

## 关键技术决策

| 决策 | 选择 | 理由 |
| ------ | ------ | ------ |
| 检测框架 | YOLOv8 (标准) / YOLO-World (零样本) / 微调 YOLOv8n | 场景多样化, 各有利弊 |
| 位姿计算 | PCA + PointNet++ (主), SAM2 (把手) | PCA 稳定高效, SAM 适合不规则物体 |
| 深度对齐 | rs.align (新相机) / 手动像素映射 (旧相机) | 旧相机内参损坏无法使用 rs.align |
| 通信协议 | TCP JSON 协议 | 简单可靠, 与 ESTUN 兼容 |
| 配置管理 | YAML 文件 (模块级 config.yaml) | 不改代码可调参数 |

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
- 仅检测帧执行, 避免跨帧零值/旧值污染

### 双检测器共享 Pipeline

- `Estun/estun_handle_cup_demo.py` / `estun_handle_cup_run.py`
- Cup 检测器以 `start_pipeline=False` 初始化, 通过 `inject_frames()` 复用 Handle 检测器的帧
- 避免重复打开相机硬件

### 通信协议: `pack_three` 三括号标记

- 格式: `<P1>[P2]{P3}`, P1 用 `<>` (原为 `()` 但机器人 `TranStrToCpos` 不支持), P2 用 `[]`, P3 用 `{}`
- 机械臂端 `TranStrToCpos` 按括号类型分别解析, 与 PC 端 `pack_three` 一一对应

### P3 多倒水角度示教

- `estun_config.yaml` 的 `p3_teach_offset.{side}` 按 B 角度数 key (10/15/20/30) 索引
- `pour_angle` 顶层 key 控制当前使用哪个角度
- 加载时按 `str(pour_angle)` 查找, 不存在则回退到第一个可用角度

## 模块关系

```text
Estun/                              ← 顶层入口
  ├── estun_run.py                  ← RealSense 实时瓶身
  ├── estun_handle_run.py           ← RealSense 把手抓取
  ├── estun_handle_cup_run.py       ← RealSense 把手+水杯融合
  ├── estun_epiceye_run.py          ← EpicEye 单次瓶身
  └── utils_shared.py               ← 共享坐标变换+灵巧手

RealSense_R/ 或 RealSense/          ← 检测模块
  ├── test_6D.py                    ← 瓶身 6D
  ├── test_handle_6D.py             ← 把手 6D
  ├── test_cup_6D.py                ← 水杯 6D
  └── realsense_shared.py           ← 基类+工具

EpicEye/                            ← 灰度相机检测
Omnihand/                           ← 灵巧手控制
calib_data/                         ← 手眼标定
