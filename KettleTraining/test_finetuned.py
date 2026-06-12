# -*- coding: utf-8 -*-
"""
微调 YOLOv8n 水壶检测测试脚本
==============================
独立测试脚本, 不依赖 config.yaml / yolo_world_detect.py 的修改.
直接加载 kettle_yolov8n_finetune.pt 进行实时检测.

    使用方法:
        python KettleTraining/test_finetuned.py

按 'q' 键退出.
"""

import sys
import os

# 将项目根目录加入 sys.path
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO


# ---- 模型配置 (硬编码, 不依赖外部配置文件) ----
MODEL_PATH = os.path.join(_project_root, "model", "kettle_yolov8n_finetune.pt")
CONF_THRESHOLD = 0.45

# ---- 分辨率 ----
FRAME_W = 640
FRAME_H = 480
FRAME_FPS = 30


def main():
    # ---- 检查模型是否存在 ----
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] 模型文件不存在: %s" % MODEL_PATH)
        sys.exit(1)

    # ---- 设备选择 ----
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print("[INFO] 使用设备: %s" % device)

    # ---- 加载模型 ----
    print("[INFO] 加载微调模型: %s" % MODEL_PATH)
    model = YOLO(MODEL_PATH)
    model.to(device)
    print("[INFO] 模型类别: %s" % str(model.names))

    # ---- 初始化 RealSense ----
    print("[INFO] 初始化 RealSense 彩色流...")
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
    pipeline.start(config)
    print("[INFO] RealSense 已启动 (%dx%d @ %dfps)" % (FRAME_W, FRAME_H, FRAME_FPS))

    print("=" * 60)
    print("  微调 YOLOv8n 水壶检测")
    print("  置信度阈值: %.2f" % CONF_THRESHOLD)
    print("  按 'q' 键退出")
    print("=" * 60)

    try:
        while True:
            # ---- 获取帧 ----
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            color_image = np.asanyarray(color_frame.get_data())

            # ---- 检测 ----
            results = model(color_image, conf=CONF_THRESHOLD, verbose=False)

            # ---- 解析结果 ----
            best = None
            for result in results:
                if result.boxes is None:
                    continue
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    if best is None or conf > best["conf"]:
                        best = {
                            "bbox": [int(box[0]), int(box[1]),
                                     int(box[2]), int(box[3])],
                            "conf": float(conf),
                            "class_id": int(cls_id),
                            "label": model.names[cls_id]
                        }

            # ---- 绘制 ----
            display_img = color_image.copy()
            if best is not None:
                x1, y1, x2, y2 = best["bbox"]
                conf = best["conf"]
                label = best["label"]

                cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                text = "%s %.2f" % (label, conf)
                cv2.putText(display_img, text, (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                cv2.putText(display_img, "Kettle detected: %.2f" % conf,
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                print("\r[KETTLE] %s | conf=%.3f | bbox=(%d,%d,%d,%d)     " % (
                    label, conf, x1, y1, x2, y2), end="", flush=True)
            else:
                cv2.putText(display_img, "No kettle detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("\r[KETTLE] No kettle detected     ", end="", flush=True)

            cv2.imshow("YOLOv8n Finetuned Kettle Detection", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n[INFO] 用户按 q 退出")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print("\n[ERROR] %s" % str(e))
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("[INFO] 程序已退出")


if __name__ == "__main__":
    main()