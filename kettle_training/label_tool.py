# -*- coding: utf-8 -*-
"""
简易 YOLO 标注工具 (基于 OpenCV)
=================================
鼠标在图片上拖动画框, 自动保存 YOLO 格式标注文件.
类别: kettle (class_id=0)

使用方法:
1. 运行: python kettle_training/label_tool.py
2. 鼠标拖动画框 (左上→右下)
3. 按 [Space] 保存并跳转到下一张
4. 按 [d] 跳过当前图片
5. 按 [q] 退出
"""

import sys
import os
import glob

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

IMAGES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
LABELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labels")
os.makedirs(LABELS_DIR, exist_ok=True)

# 获取所有图片
image_exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
image_paths = []
for ext in image_exts:
    image_paths.extend(glob.glob(os.path.join(IMAGES_DIR, ext)))
image_paths.sort()

if len(image_paths) == 0:
    print("[ERROR] 在 %s 中未找到图片" % IMAGES_DIR)
    exit(1)

print("=" * 60)
print("  简易标注工具 - kettle")
print("  共 %d 张图片" % len(image_paths))
print("  [鼠标拖动] 画框")
print("  [Space] 保存并下一张")
print("  [d] 跳过 (不保存)")
print("  [q] 退出")
print("=" * 60)

current_idx = 0
drawing = False
ix, iy = -1, -1
bbox = None
saved_count = 0

def mouse_callback(event, x, y, flags, param):
    global ix, iy, drawing, bbox
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        bbox = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        bbox = (ix, iy, x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        bbox = (ix, iy, x, y)

cv2.namedWindow("Label Tool")
cv2.setMouseCallback("Label Tool", mouse_callback)

while current_idx < len(image_paths):
    img_path = image_paths[current_idx]
    img = cv2.imread(img_path)
    if img is None:
        print("[WARN] 无法读取: %s, 跳过" % img_path)
        current_idx += 1
        continue

    h, w = img.shape[:2]

    while True:
        display = img.copy()
        draw_bbox = bbox
        info = "[%d/%d] %s" % (current_idx + 1, len(image_paths),
                                os.path.basename(img_path))
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "[Space] Save & Next  [d] Skip  [q] Quit",
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if draw_bbox:
            x1, y1, x2, y2 = draw_bbox
            # 标准化
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "kettle", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("Label Tool", display)
        key = cv2.waitKey(30) & 0xFF

        if key == ord(' '):  # Space: 保存
            if draw_bbox is None:
                print("[SKIP] 未画框, 跳过: %s" % os.path.basename(img_path))
                break
            x1, y1, x2, y2 = draw_bbox
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            # YOLO 格式: class_id x_center y_center width height (归一化)
            x_center = ((x1 + x2) / 2.0) / w
            y_center = ((y1 + y2) / 2.0) / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            # 保存
            basename = os.path.splitext(os.path.basename(img_path))[0]
            label_path = os.path.join(LABELS_DIR, basename + ".txt")
            with open(label_path, "w") as f:
                f.write("0 %.6f %.6f %.6f %.6f\n" % (x_center, y_center, bw, bh))
            saved_count += 1
            print("[SAVE] %s -> labels/%s.txt" % (os.path.basename(img_path), basename))
            draw_bbox = None
            bbox = None
            break

        elif key == ord('d'):  # 跳过
            print("[SKIP] %s" % os.path.basename(img_path))
            draw_bbox = None
            bbox = None
            break

        elif key == ord('q'):  # 退出
            print("[INFO] 用户退出")
            current_idx = len(image_paths)
            break

    current_idx += 1

cv2.destroyAllWindows()
print("=" * 60)
print("  标注完成!")
print("  总图片: %d, 已标注: %d" % (len(image_paths), saved_count))
print("  标注文件保存至: %s" % LABELS_DIR)
print("=" * 60)