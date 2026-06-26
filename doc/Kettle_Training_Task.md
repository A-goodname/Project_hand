# 水壶检测模型微调流程: 自采数据集 + YOLOv8n 微调

## 流程概览

```text
[1] 数据采集 → [2] 数据标注 → [3] 组织数据集 → [4] 模型训练 → [5] 部署到项目
```

## 背景

### 为什么需要自采数据集微调

1. **YOLO-World 零样本检测效果不理想**: 当前使用 `yolov8m-worldv2.pt` + 文本 prompt 检测水壶本体, 实测置信度偏低 (≤0.45), 且纸杯等非目标物存在误检
2. **COCO 数据集不含 kettle 类**: 官方 YOLOv8 权重仅在 COCO 80 类上训练, 不含 kettle / teapot / water jug 等水壶类别
3. **LVIS 虽有但官方模型未使用**: LVIS 数据集 (1203 类) 包含 kettle 等类别, 但 YOLO 官方预训练权重仅基于 COCO
4. **微调是最佳方案**: 在自采的实际水壶照片上, 基于 `yolov8n.pt` COCO 预训练权重继续微调, 只需少量数据即可获得高精度的水壶检测模型

### 预期成果

- 训练好的模型文件: `model/kettle_yolov8n_finetune.pt`
- 检测置信度显著提升 (mAP50 ≥ 0.99), 误检率大幅降低
- 独立测试脚本: `KettleTraining/test_finetuned.py`

---

## Step 1: 数据采集

使用 RealSense D435i 拍摄实际水壶照片, 多角度覆盖.

### 采集脚本

`KettleTraining/collect_kettle_data.py`: 使用 RealSense 彩色流拍摄, 按空格键保存, 按 q 退出.

### 采集要求

| 要求 | 说明 |
| ------ | ------ |
| 图片数量 | 50 张 (由转台+多角度拍摄, 50 张已足够) |
| 旋转角度 | 水壶在转台上 0-360 度旋转, 每 30-45 度拍一张 |
| 俯仰变化 | 相机高度变化: 水平视角、俯视 45 度、俯视 60 度 |
| 距离变化 | 0.3m ~ 1.0m, 覆盖不同深度 |
| 是否包含把手 | **必须包含把手**, 因为 bbox 框的是水壶整体 (壶身+把手) |
| 格式要求 | JPG, 分辨率 640x480 |

> **注意**: 采集时水壶周围不要有太多同类物品, 避免后期标注混淆.

---

## Step 2: 数据标注

使用 OpenCV 自研标注工具 `KettleTraining/label_tool.py` 标注.

### 标注工具 (替代 LabelImg)

由于 LabelImg 在 PyQt5 环境下存在 float→int 类型错误, 改用 OpenCV 手动标注工具:

- 鼠标拖拽绘制 bbox
- 自动保存 YOLO 格式标签
- 按 `n` 切换下一张, `q` 退出

### 标注质量要求

- bbox 紧密贴合水壶边缘, **必须包含把手部分**
- 确保所有图片都标注完毕 (检查 `images/` 和 `labels/` 文件数一致)
- 若水壶在图片中被严重遮挡 (<30% 可见), 跳过该图片

---

## Step 3: 组织数据集目录结构

```text
Project_hand/
└── KettleTraining/
    ├── images/              # 原始图片 (JPG)
    │   ├── kettle_000.jpg
    │   ├── kettle_001.jpg
    │   └── ...
    ├── labels/              # YOLO 格式标注 (TXT)
    │   ├── kettle_000.txt
    │   ├── kettle_001.txt
    │   └── ...
    └── dataset.yaml         # 数据集配置文件
```

### 数据集配置文件 `KettleTraining/dataset.yaml`

```yaml
path: ./KettleTraining   # 数据集根目录 (相对项目根目录)
train: images              # 训练集图片路径
val: images                # 验证集图片路径 (数据量少, 复用训练集)
nc: 1                      # 类别数 (仅 kettle)
names: ["kettle"]           # 类别名称
```

> **注意**: 由于自采数据量少 (50 张), 不划分 train/val, 全部用于训练.

---

## Step 4: 模型训练

### 训练脚本

`KettleTraining/train_kettle.py`: 基于 COCO 预训练的 YOLOv8n 微调.

### 训练参数

| 参数 | 值 | 说明 |
| ------ | ----- | ------ |
| `model` | `yolov8n.pt` | 轻量级模型, MX450 上 9 分钟跑完 |
| `epochs` | 50 | 小数据集 50 epoch 足够, 太多易过拟合 |
| `imgsz` | 640 | 标准 YOLOv8 输入尺寸 |
| `batch` | 8 | 小数据集用小 batch |
| `device` | `cuda:0` (自动检测) | MX450 2GB 可用 |
| `amp` | `False` | 关闭 AMP 避免 Ultralytics 自动下载 yolo26n.pt |
| `lr0` | 0.001 | 微调使用较小学习率 |
| `patience` | 20 | 若验证集 20 epoch 无提升则提前停止 |

### 运行训练

```bash
D:\Environment\miniconda\envs\Proj\python.exe KettleTraining/train_kettle.py
```

### 训练输出

```text
runs/train/kettle_finetune/weights/best.pt   ← 最佳模型 (推荐使用)
runs/train/kettle_finetune/weights/last.pt   ← 最后 epoch 模型
```

### 预期指标

| 指标 | 值 | 说明 |
| ------ | ----- | ------ |
| Precision | ≥ 0.99 | 几乎无假阳性 |
| Recall | 1.0 | 所有水壶都检到 |
| mAP50 | ≥ 0.99 | 近乎完美 |

### 复制模型

```bash
copy runs\train\kettle_finetune\weights\best.pt model\kettle_yolov8n_finetune.pt
```

---

## Step 5: 部署与测试

### 独立测试脚本

`KettleTraining/test_finetuned.py`: 直接加载微调模型进行实时检测, 不依赖对 `config.yaml` 的修改.

```bash
D:\Environment\miniconda\envs\Proj\python.exe KettleTraining/test_finetuned.py
```

### 检测结果可视化

- 蓝色检测框 + 置信度文字
- 控制台实时输出检测信息和 bbox 坐标
- 按 `q` 键退出

### 集成到主链路 (TODO)

后续将微调模型接入 `RealSense_R/realsense_shared.py`, 替换 YOLO-World 作为水壶本体检测入口. 需修改:

1. `RealSense_R/config.yaml`: 新增 `kettle_detector` 配置段
2. `RealSense_R/realsense_shared.py`: 新增 `_detect_kettle_bbox_finetuned()` 方法
3. `RealSense_R/test_SAM_handle.py`: 使用 USE_FINETUNED 标志切换检测器

---

## 关键文件

| 文件 | 说明 |
| ------ | ------ |
| `KettleTraining/collect_kettle_data.py` | 数据采集脚本 |
| `KettleTraining/label_tool.py` | OpenCV 标注工具 |
| `KettleTraining/train_kettle.py` | 训练脚本 (YOLOv8n) |
| `KettleTraining/test_finetuned.py` | 独立测试脚本 |
| `KettleTraining/dataset.yaml` | 数据集配置 |
| `model/kettle_yolov8n_finetune.pt` | 训练好的模型 |

## 注意事项

1. 训练时确保 `KettleTraining/images/` 与 `KettleTraining/labels/` 文件数一致, 否则 Ultralytics 会跳过无标签图片
2. 每次训练前建议删除旧的 `runs/train/kettle_finetune/` 目录, 避免混淆
3. `labels.cache` 是 Ultralytics 自动生成的标签缓存, 已在 `.gitignore` 中忽略, 可安全删除
4. YOLOv8n 在 MX450 上约 9 分钟跑完 50 epochs, 比 v8m 快约 3 倍
