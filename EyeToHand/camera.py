# -*- coding: utf-8 -*-
"""
UGREEN Camera 1080P — OpenCV VideoCapture 封装
================================================
替代 RealSense D435i 的纯 RGB 相机, 通过 DirectShow 后端连接.

使用:
  cam = UGreenCamera(camera_id=1)
  color = cam.read()
  cam.release()
"""

import cv2
import numpy as np


class UGreenCamera:
    """UGREEN 1080P USB Camera (OpenCV backend)."""

    def __init__(self, camera_id: int = 1,
                 width: int = 1920, height: int = 1080, fps: int = 30):
        """
        Args:
            camera_id: OpenCV 设备索引 (0=第一个摄像头).
            width:     目标宽度 (默认 1920).
            height:    目标高度 (默认 1080).
            fps:       目标帧率 (默认 30).
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps

        self._cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开相机 camera_id={camera_id}")

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
