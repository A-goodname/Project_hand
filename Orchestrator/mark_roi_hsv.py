# -*- coding: utf-8 -*-
"""
静态帧 HSV 圈定分析
===================
打开 D435i 抓一帧原图 (无检测/mask 遮挡), 静态显示;
鼠标左键拖拽圈定把手区域 → HSV 统计 (青色像素/直方图/H-S-V 范围)
+ 保存原图裁剪与标注图; 按 r 重抓一帧, q 退出.

用途 (2026-08-17): 青色贴纸 HSV 参数标定 — 用户圈定把手, 离线分析
贴纸真实 H/S/V 范围, 修正 test_tag_points.py 的 HUE/SAT/VAL 参数.

用法:
  conda run -n Proj python Orchestrator/mark_roi_hsv.py
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 复用青色参数与 analyze_roi (模块级只定义, 安全 import)
from Orchestrator.test_tag_points import analyze_roi


def grab_frame(serial):
    """抓一帧彩色原图 (D435i)."""
    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    config = rs.config()
    if serial:
        config.enable_device(serial)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    # 丢弃前几帧 (自动曝光收敛)
    for _ in range(10):
        pipeline.wait_for_frames()
    frame = pipeline.wait_for_frames().get_color_frame()
    color = np.asanyarray(frame.get_data()).copy()
    pipeline.stop()
    return color


def main():
    # ---- 读取 serial ----
    serial = ""
    info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            serial = str(yaml.safe_load(f)["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")

    save_dir = os.path.join(_project_root, "logs")
    os.makedirs(save_dir, exist_ok=True)

    # ---- 鼠标圈定 (直接拖拽, 不依赖按键) ----
    mark_state = {"p0": None, "p1": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mark_state["p0"] = (x, y)
            mark_state["p1"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            mark_state["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mark_state["p1"] = (x, y)
            mark_state["done"] = True

    cv2.namedWindow("ROI Mark (static)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("ROI Mark (static)", on_mouse)

    color = grab_frame(serial)
    print("[INFO] 静态帧已抓取 (640x480); 左键拖拽圈定把手区域, r 重抓, q 退出")

    try:
        while True:
            display = color.copy()
            if mark_state["p0"] and mark_state["p1"]:
                cv2.rectangle(display, mark_state["p0"], mark_state["p1"],
                              (0, 255, 0), 1)
            cv2.imshow("ROI Mark (static)", display)

            # ---- 圈定完成处理 ----
            if mark_state["done"]:
                mark_state["done"] = False
                p0, p1 = mark_state["p0"], mark_state["p1"]
                mark_state["p0"] = mark_state["p1"] = None
                if p0 and p1:
                    rx0, ry0 = min(p0[0], p1[0]), min(p0[1], p1[1])
                    rx1, ry1 = max(p0[0], p1[0]), max(p0[1], p1[1])
                    if rx1 - rx0 >= 10 and ry1 - ry0 >= 10:
                        analyze_roi(color, (rx0, ry0, rx1, ry1), save_dir)
                    else:
                        print("[MARK] 区域太小, 忽略")

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                print("[INFO] 重抓一帧...")
                color = grab_frame(serial)
                print("[INFO] 已重抓; 可重新圈定")

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
