# -*- coding: utf-8 -*-
"""
UGREEN Camera 1080P — OpenCV VideoCapture 封装 (跨平台)
========================================================
替代 RealSense D435i 的纯 RGB 相机.

后端选择:
  - Linux:   V4L2 (CAP_V4L2), 720P/1080P 需 MJPG 格式 (参考 Project/camera_view.py)
  - Windows: DirectShow (CAP_DSHOW), 按名称探测走 pygrabber

按名称自动探测:
  - Linux:   遍历 /sys/class/video4linux/videoN/name 匹配关键词
  - Windows: pygrabber DSHOW 设备名匹配

使用:
  cam = UGreenCamera(camera_id=1)
  # 或自动探测:
  cam_id = find_camera_id(["UGREEN"])
  cam = UGreenCamera(camera_id=cam_id)
  color = cam.read()
  cam.release()
"""

import os
import sys
from typing import Optional, List

import cv2
import numpy as np

_IS_LINUX = sys.platform.startswith("linux")


# ------------------------------------------------------------
#  设备名探测
# ------------------------------------------------------------

def _list_v4l2_devices() -> List[tuple]:
    """列出 Linux 下 /dev/video* 设备名 (sysfs).

    Returns:
        [(index, name), ...] — index 即 OpenCV VideoCapture 用的节点号.
    """
    devices = []
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return devices
    for entry in sorted(os.listdir(base)):
        if not entry.startswith("video"):
            continue
        name_path = os.path.join(base, entry, "name")
        if not os.path.exists(name_path):
            continue
        try:
            with open(name_path, "r", encoding="utf-8", errors="ignore") as f:
                name = f.read().strip()
        except OSError:
            continue
        index = int(entry[len("video"):])
        devices.append((index, name))
    return devices


def find_camera_id(keywords: Optional[List[str]] = None) -> Optional[int]:
    """查找摄像头 ID，按设备名关键词匹配。找不到返回 None，调用方决定如何处理。

    Linux:   遍历 /sys/class/video4linux/videoN/name (无需 pygrabber)
    Windows: pygrabber DSHOW 设备名匹配
    """
    if keywords is None:
        keywords = ["UGREEN"]

    if _IS_LINUX:
        devices = _list_v4l2_devices()
        print(f"[CameraFinder] V4L2 设备 ({len(devices)} 个):")
        for idx, name in devices:
            print(f"  [{idx}] {name}")

        if not devices:
            return None

        for keyword in keywords:
            for idx, name in devices:
                if keyword.lower() in name.lower():
                    print(f"[CameraFinder] 匹配 \"{keyword}\" → [{idx}] {name}")
                    return idx

        return None

    # Windows: DSHOW 探测
    try:
        from pygrabber.dshow_graph import FilterGraph
    except ImportError:
        raise RuntimeError(
            "[CameraFinder] pygrabber 未安装, 无法按名称查找摄像头. "
            "请执行: pip install pygrabber"
        )

    graph = FilterGraph()
    devices = graph.get_input_devices()

    print(f"[CameraFinder] DSHOW 设备 ({len(devices)} 个):")
    for i, name in enumerate(devices):
        print(f"  [{i}] {name}")

    if not devices:
        return None

    for keyword in keywords:
        for i, name in enumerate(devices):
            if keyword.lower() in name.lower():
                print(f"[CameraFinder] 匹配 \"{keyword}\" → [{i}] {name}")
                return i

    return None


class UGreenCamera:
    """UGREEN 1080P USB Camera (OpenCV backend)."""

    def __init__(self, camera_id: int = 1, camera_name: str = "",
                 width: int = 1920, height: int = 1080, fps: int = 30):
        """
        Args:
            camera_id:   OpenCV 设备索引. -1 = 按 camera_name 自动搜索.
            camera_name: 摄像头名称关键词 (camera_id=-1 时启用).
            width:       目标宽度 (默认 1920).
            height:      目标高度 (默认 1080).
            fps:         目标帧率 (默认 30).
        """
        # camera_id == -1: 按名称自动搜索 (Linux: sysfs / Windows: DSHOW)
        if camera_id == -1:
            if not camera_name:
                raise RuntimeError("camera_id=-1 但 camera_name 未指定")
            resolved = find_camera_id([camera_name])
            if resolved is None:
                raise RuntimeError(f"未找到摄像头: \"{camera_name}\"")
            camera_id = resolved

        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        backend = cv2.CAP_V4L2 if _IS_LINUX else cv2.CAP_DSHOW
        self._cap = cv2.VideoCapture(camera_id, backend)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 camera_id={camera_id}")

        if _IS_LINUX:
            # V4L2: 先设 MJPG 格式再设分辨率, 否则部分摄像头只支持 640x480
            self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[UGreenCamera] camera_id={camera_id}  {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

    # ------------------------------------------------------------
    def read(self) -> tuple:
        """读取一帧.

        Returns:
            (color_image,) 单元素元组 — 与旧 get_frames() 接口兼容.
            color_image: np.ndarray (H, W, 3) BGR, 或 None (读取失败时).
        """
        if self._cap is None:
            return (None,)
        ret, frame = self._cap.read()
        if not ret:
            return (None,)
        return (frame,)

    # ------------------------------------------------------------
    def release(self):
        """释放相机."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("[UGreenCamera] 已释放")

    # ------------------------------------------------------------
    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
