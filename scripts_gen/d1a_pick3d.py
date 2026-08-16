# -*- coding: utf-8 -*-
"""D1-A Task 3 辅助: D435i 取一帧, 鼠标点选参考物 -> 相机系 3D 坐标 (只读).

用法: 运行后窗口显示 D435i color 帧; 鼠标左键点选参考物像素,
终端输出该点的相机系 3D 坐标 (m). 按 q 退出.
内参/深度尺度来自 RealSense_R/D435i_info.yaml 与硬件.
"""
import os
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402
import pyrealsense2 as rs  # noqa: E402

from RealSense_R.realsense_shared import (  # noqa: E402
    load_camera_params, color_to_depth_pixel, depth_median_around,
)

SERIAL = "342222071364"
YAML = os.path.join(_PROJECT_ROOT, "RealSense_R", "D435i_info.yaml")

_click = None


def on_mouse(event, x, y, flags, param):
    global _click
    if event == 1:  # EVENT_LBUTTONDOWN
        _click = (x, y)


def main():
    global _click
    params = load_camera_params(YAML)
    color_intr = params["color_intr"]
    depth_intr = params["depth_intr"]
    c2d_extr = params["c2d_extr"]
    d2c_extr = params["d2c_extr"]
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    depth_sensor = pipe.get_active_profile().get_device().first_depth_sensor()
    scale = depth_sensor.get_depth_scale()

    # 丢弃前 15 帧 (自动曝光/深度稳定)
    for _ in range(15):
        pipe.wait_for_frames()
    frames = pipe.wait_for_frames()
    color = np.asanyarray(frames.get_color_frame().get_data())
    depth_frame = frames.get_depth_frame()

    cv2 = __import__("cv2")
    win = "Pick3D - D435i (click reference, q quit)"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    display = color.copy()
    print("相机系 3D 输出 (m): [x, y, z]  (z=深度, x右+, y下+... 相机系)")
    print(f"color 内参 fx={color_intr.fx:.2f} fy={color_intr.fy:.2f} "
          f"cx={color_intr.ppx:.2f} cy={color_intr.ppy:.2f}")
    print(f"depth scale={scale:.6f}")

    while True:
        if _click is not None:
            cx, cy = _click
            _click = None
            # 深度反投影: color 像素 -> depth 像素 -> 深度 -> 相机系 3D
            res = color_to_depth_pixel(cx, cy, depth_frame, scale,
                                       depth_intr, color_intr, c2d_extr, d2c_extr)
            if res is None:
                print(f"  点 ({cx},{cy}) -> 深度映射失败 (无有效深度)")
            else:
                dx, dy = res
                depth = depth_median_around(dx, dy, depth_frame)
                if depth is None:
                    print(f"  点 ({cx},{cy}) -> 深度无效 (无有效点)")
                    continue
                z = float(depth)          # get_distance 已返回米, 不再乘 scale
                x = (cx - color_intr.ppx) * z / color_intr.fx
                y = (cy - color_intr.ppy) * z / color_intr.fy
                print(f"  点 ({cx},{cy}) -> 相机系 [{x:.4f}, {y:.4f}, {z:.4f}] m")
                cv2.circle(display, (cx, cy), 4, (0, 0, 255), -1)
        cv2.imshow(win, display)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break

    pipe.stop()
    cv2.destroyAllWindows()
    print("已退出 (只读采集, 未启动任何服务)")


if __name__ == "__main__":
    main()
