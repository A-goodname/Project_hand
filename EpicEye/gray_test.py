# -*- coding: utf-8 -*-
"""
灰度 YOLO 2D 检测测试 — 直接检测 image* 图片，OpenCV 显示结果
==============================================================
用法:
    python EpicEye/gray_test.py

操作:
    n       下一张
    p       上一张
    q       退出
"""

import os
import glob
import cv2
import numpy as np
from ultralytics import YOLO

# ---- COCO 类别名称（id -> name） ----
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite", 34: "baseball bat",
    35: "baseball glove", 36: "skateboard", 37: "surfboard", 38: "tennis racket",
    39: "bottle", 40: "wine glass", 41: "cup", 42: "fork", 43: "knife",
    44: "spoon", 45: "bowl", 46: "banana", 47: "apple", 48: "sandwich",
    49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog", 53: "pizza",
    54: "donut", 55: "cake", 56: "chair", 57: "couch", 58: "potted plant",
    59: "bed", 60: "dining table", 61: "toilet", 62: "TV", 63: "laptop",
    64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone", 68: "microwave",
    69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator", 73: "book",
    74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear", 78: "hair drier",
    79: "toothbrush",
}

# ---- 路径 ----
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAPTURES_DIR = os.path.join(SCRIPT_DIR, ".", "captures")
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "model", "yolo11n-grayscale.pt")

# ---- 加载模型 ----
print(f"[INFO] 加载模型: {MODEL_PATH}")
model = YOLO(MODEL_PATH)
print("[INFO] 模型就绪")

# ---- 扫描图片 ----
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_files = []
for ext in image_exts:
    pattern = os.path.join(CAPTURES_DIR, f"image_{ext}")
    image_files.extend(glob.glob(pattern))
# 也匹配 image 开头但可能不带日期前缀
for ext in image_exts:
    pattern = os.path.join(CAPTURES_DIR, f"image*.{ext.split('.')[-1]}")
    image_files.extend(glob.glob(pattern))
image_files = sorted(set(image_files))

if not image_files:
    print(f"[ERROR] captures 目录中未找到 image* 图片: {CAPTURES_DIR}")
    exit(1)

print(f"[INFO] 找到 {len(image_files)} 张图片")

# ---- 主循环 ----
WIN_NAME = "Gray YOLO Detection"
cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_NAME, 960, 720)

idx = 0
while True:
    img_path = image_files[idx]
    basename = os.path.basename(img_path)
    print(f"\n[{idx + 1}/{len(image_files)}] {basename}")

    # 读取灰度图 (H,W)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"  [ERROR] 无法读取: {img_path}")
        idx = (idx + 1) % len(image_files)
        continue

    # YOLO 推理 (灰度图直接送入)
    results = model(gray)[0]

    # 转 BGR 用于绘制彩色框
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if results.boxes is not None and len(results.boxes) > 0:
        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)

        for box, conf, cls_id in zip(boxes, confs, cls_ids):
            x1, y1, x2, y2 = map(int, box[:4])
            label = COCO_NAMES.get(cls_id, f"class_{cls_id}")
            text = f"{label} {conf:.2f}"

            # 绘制框
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 背景 + 标签文字
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(display, text, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            print(f"  {text} | box: ({x1},{y1})-({x2},{y2})")
    else:
        print("  未检测到目标")
        cv2.putText(display, "No detections", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    # 信息栏
    info_text = f"{basename}  [{idx + 1}/{len(image_files)}]  [n] next  [p] prev  [q] quit"
    cv2.putText(display, info_text, (10, display.shape[0] - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    # 缩放到合适显示大小
    h, w = display.shape[:2]
    max_w = 960
    if w > max_w:
        scale = max_w / w
        display = cv2.resize(display, (max_w, int(h * scale)))

    cv2.imshow(WIN_NAME, display)

    # 按键处理
    key = cv2.waitKey(0) & 0xFF
    if key == ord("q") or key == 27:
        break
    elif key == ord("n"):
        idx = (idx + 1) % len(image_files)
    elif key == ord("p"):
        idx = (idx - 1) % len(image_files)

cv2.destroyAllWindows()
print("[INFO] 退出")