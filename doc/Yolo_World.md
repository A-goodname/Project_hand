# YOLO-World 使用说明

## 概述

本项目使用 YOLO-World v2 实现零样本**水壶本体检测**（kettle/teapot/jug），为后续的把手分割提供 bbox.

YOLO-World 的核心优势在于：无需训练即可通过文本描述（prompt）检测任意类别物体.

**重要**: YOLO-World 不适合零样本把手（handle）检测——实测会将纸杯误检为 "cup handle"，且检测框无法分离把手与瓶身。因此本项目分两阶段处理:

1. **YOLO-World 水壶本体检测** (本文件, `yolo_world_detect.py`)
2. **SAM + 几何筛选把手分割** (后续 `test_SAM_handle.py`, 详见 `doc/Handle_Process.md`)

**当前状态**: 把手检测脚本 `test_SAM_handle.py` 已从 YOLO-World+跳帧模式 重构为 **微调 YOLOv8n + 手动触发模式** (commit `c3a7913`). YOLO-World 当前仅作为瓶身检测/备用方案保留.

## 模型文件

| 文件                 | 路径                       | 大小   | 说明                                  |
| -------------------- | -------------------------- | ------ | ------------------------------------- |
| `yolov8m-worldv2.pt` | `model/yolov8m-worldv2.pt` | ~100MB | YOLO-World 主模型权重（已在 repo 中） |
| `yolov8s-worldv2.pt` | `model/yolov8s-worldv2.pt` | ~60MB  | YOLO-World 轻量版（已在 repo 中）     |

## CLIP 依赖说明

### 作用

YOLO-World 的工作机制是将用户传入的文本描述（如 `"cup handle"`）通过 **OpenAI CLIP 文本编码器** 编码为文本 embedding，再与图像特征做匹配。

首次调用 `model.set_classes()` 时，Ultralytics 会**自动下载** CLIP 的 ViT-B/32 权重文件。

### 文件位置

```text
C:\Users\荒\.cache\clip\ViT-B-32.pt   (~338MB)
```

该路径由 Ultralytics 内部硬编码，**不可移动或重命名**。若移动，程序会重新下载。

### 弃用清理

若后续项目不再使用 YOLO-World（例如回退到普通 YOLOv8），建议手动删除以下文件以释放磁盘空间：

```text
# CLIP 权重文件（~338MB）
C:\Users\荒\.cache\clip\ViT-B-32.pt

# Ultralytics 模型缓存（可选）
C:\Users\荒\.cache\ultralytics\
```

> **注意**：`model/yolov8m-worldv2.pt` 和 `model/yolov8s-worldv2.pt` 虽然体积较小（~100MB），不在弃用时建议一并清理。

## 检测配置

### 水壶本体检测 — 当前使用的 prompt

```python
from ultralytics import YOLO

model = YOLO("model/yolov8m-worldv2.pt")
model.set_classes([
    "kettle",           # id=0, 水壶 (主目标)
    "electric kettle",  # id=1, 电热水壶
    "water jug",        # id=2, 水罐
    "teapot",           # id=3, 茶壶
])
```

- 所有检测结果均为水壶本体（无背景类），无需额外过滤
- prompt 中**不含任何 'handle' 或 'cup' 词汇**，避免语义干扰
- 只取最高置信度检测框作为后续 SAM 分割的输入
- 置信度阈值推荐 0.25（水壶置信度通常 ≤0.45，阈值不宜过高）
- **注意**: 本阶段仅检测水壶本体 bbox，把手分离在后续 `test_SAM_handle.py` 中通过 SAM 多 mask + 几何筛选完成

### 注意事项

1. **首次运行耗时**：首次加载 YOLO-World 时会自动安装 `clip` 依赖包 + 下载 `ViT-B-32.pt`（~338MB）。后续加载为秒级。
2. **返回类型**：加载 `yolov8m-worldv2.pt` 返回的对象类型为 `YOLOWorld`（非普通 `YOLO`），包含 `set_classes()` 方法。
3. **`set_classes()` 调用顺序**：必须在 `model.predict()` / `model()` 之前调用，否则仍使用默认 COCO 80 类。
4. **`model.names` 变化**：调用后 `model.names` 会替换为自定义类别映射。
