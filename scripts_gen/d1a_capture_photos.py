# -*- coding: utf-8 -*-
"""D1-A Task 1: 三相机视角照片采集 (只读, 各取 1 帧).

- UGREEN 迎宾 (usb-0:4.1)  -> photo_guest.jpg
- UGREEN 桌面 Camera B (usb-0:4.2) -> photo_scene.jpg
- D435i 手腕 (serial 342222071364) -> photo_wrist.jpg
不启动任何服务; 仅相机读取.
"""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

OUT_DIR = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "VLA-BD", "docs",
                                       "d1-measurements-2026-08-15"))
os.makedirs(OUT_DIR, exist_ok=True)

import cv2  # noqa: E402


def find_video_by_id_path(prefix):
    """按 ID_PATH 前缀找 /dev/videoN (如 pci-0000:2a:00.1-usb-0:4.1)."""
    for i in range(16):
        path = f"/dev/video{i}"
        if not os.path.exists(path):
            continue
        try:
            info = os.popen(f"udevadm info --query=property --name={path} 2>/dev/null").read()
        except Exception:
            continue
        for line in info.splitlines():
            if line.startswith("ID_PATH=") and prefix in line:
                return path
    return None


def capture_ugreen(id_path_prefix, out_name, expected_w=1280, expected_h=720):
    dev = find_video_by_id_path(id_path_prefix)
    if dev is None:
        print(f"[SKIP] {out_name}: 未找到 ID_PATH={id_path_prefix}")
        return False
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[FAIL] {out_name}: 无法打开 {dev}")
        return False
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, expected_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, expected_h)
    # 丢弃前 5 帧 (自动曝光稳定)
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[FAIL] {out_name}: 取帧失败 {dev}")
        return False
    out = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out, frame)
    h, w = frame.shape[:2]
    print(f"[OK] {out_name}: {dev} {w}x{h} -> {out}")
    return True


def capture_d435i(serial, out_name):
    import pyrealsense2 as rs
    ctx = rs.context()
    dev = None
    for d in ctx.query_devices():
        if d.get_info(rs.camera_info.serial_number) == serial:
            dev = d
            break
    if dev is None:
        print(f"[SKIP] {out_name}: 未找到 D435i serial={serial}")
        return False
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipe.start(cfg)
    # 丢弃前 10 帧 (自动曝光稳定)
    for _ in range(10):
        pipe.wait_for_frames()
    frames = pipe.wait_for_frames()
    color = frames.get_color_frame()
    pipe.stop()
    if not color:
        print(f"[FAIL] {out_name}: 无 color 帧")
        return False
    img = np_asanyarray(color)
    out = os.path.join(OUT_DIR, out_name)
    cv2.imwrite(out, img)
    h, w = img.shape[:2]
    print(f"[OK] {out_name}: D435i {w}x{h} -> {out}")
    return True


def np_asanyarray(frame):
    import numpy as np
    return np.asanyarray(frame.get_data())


if __name__ == "__main__":
    print(f"照片输出目录: {OUT_DIR}")
    capture_ugreen("pci-0000:2a:00.1-usb-0:4.1", "photo_guest.jpg")
    capture_ugreen("pci-0000:2a:00.1-usb-0:4.2", "photo_scene.jpg")
    capture_d435i("342222071364", "photo_wrist.jpg")
    print("完成 (每张照片 = 对应相机实际视角 1 帧)")
