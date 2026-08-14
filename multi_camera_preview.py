# -*- coding: utf-8 -*-
"""
三相机实时预览 (D435i + UGREEN x2)
===================================
同时打开:
  - Camera A: RealSense D435i (彩色 + 深度伪彩)
  - Camera B1/B2: UGREEN Camera 1080P x2 (V4L2 + MJPG 720P)

用法:
  python multi_camera_preview.py

按键: q / ESC 退出

说明:
  - UGREEN 节点由 sysfs 自动枚举 (video6/video8, 跳过 metadata 节点)
  - D435i 不设置 laser_power (mid-stream 设置会卡死流, 默认值已是 150)
  - 三相机同开时 USB 带宽共享, 帧率可能下降属正常
"""

import sys
import os
import time

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np

from EyeToHand.camera import UGreenCamera, _list_v4l2_devices


def find_ugreen_nodes():
    """从 sysfs 找出所有可用的 UGREEN capture 节点 (跳过 metadata)."""
    nodes = []
    for idx, name in _list_v4l2_devices():
        if "ugreen" in name.lower():
            nodes.append(idx)
    return nodes


def open_ugreen_cameras():
    """打开全部 UGREEN 相机, 返回 (UGreenCamera 列表, 失败信息)."""
    cams, errors = [], []
    for idx in find_ugreen_nodes():
        try:
            cam = UGreenCamera(camera_id=idx, width=1280, height=720, fps=30)
            cams.append(cam)
        except RuntimeError as e:
            errors.append(f"video{idx}: {e}")
    return cams, errors


class FpsCounter:
    """简易 FPS 计数器 (1 秒窗口)."""

    def __init__(self, name):
        self.name = name
        self._t0 = time.time()
        self._n = 0
        self.fps = 0.0

    def tick(self):
        self._n += 1
        now = time.time()
        if now - self._t0 >= 1.0:
            self.fps = self._n / (now - self._t0)
            self._n = 0
            self._t0 = now


def main():
    # ---- D435i (pyrealsense2, 不设 laser_power) ----
    import pyrealsense2 as rs
    rs_pipe = rs.pipeline()
    rs_cfg = rs.config()
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_profile = rs_pipe.start(rs_cfg)
    rs_colorizer = rs.colorizer()
    print(f"[D435i] serial={rs_profile.get_device().get_info(rs.camera_info.serial_number)}")

    # ---- UGREEN x2 ----
    ugreen_cams, ugreen_errors = open_ugreen_cameras()
    for e in ugreen_errors:
        print(f"[WARN] {e}")
    if not ugreen_cams:
        print("[FATAL] 没有可用的 UGREEN 相机")
        rs_pipe.stop()
        return 1

    fps_ugreen = [FpsCounter(f"UGREEN video{u.camera_id}") for u in ugreen_cams]
    fps_rs = FpsCounter("D435i")

    print(f"已打开 {len(ugreen_cams)} 台 UGREEN + D435i, 按 q / ESC 退出")

    try:
        while True:
            # ---- UGREEN 帧 ----
            ugreen_frames = []
            for cam, fc in zip(ugreen_cams, fps_ugreen):
                img = cam.read()[0]
                ugreen_frames.append(img)
                if img is not None:
                    fc.tick()

            # ---- D435i 帧 ----
            rs_frames = rs_pipe.wait_for_frames()
            rs_color = np.asanyarray(rs_frames.get_color_frame().get_data())
            rs_depth = np.asanyarray(rs_colorizer.colorize(rs_frames.get_depth_frame()).get_data())
            fps_rs.tick()

            # ---- 显示 ----
            for cam, img, fc in zip(ugreen_cams, ugreen_frames, fps_ugreen):
                if img is None:
                    continue
                cv2.putText(img, f"UGREEN video{cam.camera_id}  {fc.fps:.1f} fps",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow(f"UGREEN video{cam.camera_id}", img)

            cv2.putText(rs_color, f"D435i color  {fps_rs.fps:.1f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("D435i color", rs_color)
            cv2.imshow("D435i depth", rs_depth)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                print("用户退出")
                break
    except KeyboardInterrupt:
        print("Ctrl+C 退出")
    finally:
        rs_pipe.stop()
        for cam in ugreen_cams:
            cam.release()
        cv2.destroyAllWindows()
        print("所有相机已释放")
    return 0


if __name__ == "__main__":
    sys.exit(main())
