# -*- coding: utf-8 -*-
"""
水壶检测模型微调脚本 (CUDA - YOLOv8n)
======================================
基于 COCO 预训练的 YOLOv8n, 用自采 50 张水壶照片微调.
使用 CUDA 加速训练.

使用方法:
    python kettle_training/train_kettle.py

训练完成后:
    - 最佳模型: runs/train/kettle_finetune/weights/best.pt
    - 复制: copy runs/train/kettle_finetune/weights/best.pt model/kettle_yolov8n.pt
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from ultralytics import YOLO


def main():
    # 检测 CUDA
    import torch
    cuda_available = torch.cuda.is_available()
    device = 'cuda:0' if cuda_available else 'cpu'
    print("[INFO] CUDA available: %s, using device: %s" % (cuda_available, device))

    # 加载 COCO 预训练权重 (轻量级 yolov8n)
    model_path = os.path.join(_project_root, "model", "yolov8n.pt")
    if not os.path.exists(model_path):
        print("[ERROR] 预训练权重未找到: %s" % model_path)
        sys.exit(1)
    print("[INFO] Loading model: %s" % model_path)
    model = YOLO(model_path)

    # 微调
    # 确保输出目录存在 (防止 plotting 线程在目录创建前写入)
    save_dir = os.path.join(_project_root, "runs", "train", "kettle_finetune")
    os.makedirs(save_dir, exist_ok=True)
    print("[INFO] Starting training, save to: %s" % save_dir)
    model.train(
        data=os.path.join(_project_root, "kettle_training", "dataset.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        device=device,
        amp=False,
        lr0=0.001,
        patience=20,
        save=True,
        project=os.path.join(_project_root, "runs", "train"),
        name='kettle_finetune',
        exist_ok=True,
        pretrained=True,
        workers=0,
    )
    print("[INFO] Training completed!")


if __name__ == "__main__":
    main()