# -*- coding: utf-8 -*-
"""
RealSense YOLO-World + SAM + 水平扫描线 水壶把手定位
=====================================================
重构版本: 继承 BaseRealSenseDetector 基类.
使用 YOLO-World(GPU) 检测水壶本体 + SAM(CPU) 分割 + 水平扫描线分离把手.

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/test_SAM_handle.py
  3. 按 'q' 键退出, 按 's' 键保存当前帧
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import time
import yaml
import cv2
import numpy as np

from RealSense_R.realsense_shared import (
    BaseRealSenseDetector,
    scanline_find_handle,
)


# ============================================================
#  1. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  2. 主检测器
# ============================================================

class RealSenseHandleDetector(BaseRealSenseDetector):
    """
    YOLO-World + SAM + 水平扫描线分离 水壶把手检测器.

    流程:
      1. YOLO-World 检测水壶本体 bbox
      2. SAM 在 bbox 内生成完整 mask (壶身+把手)
      3. 水平扫描线 mask|gap|mask 分离壶身和把手
    """
    USE_YOLO_WORLD = True
    USE_SAM = True
    LOG_NAME = "test_SAM_handle.log"

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

        # --- 耗时统计 ---
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # ---- 跳帧缓存配置 ----
        self._frame_counter = 0
        self._skip_frames = 10
        self._cached_result = None  # (handle_mask, handle_bbox, body_mask, stage, yolo_bbox)

    # ------------------------------------------------------------
    def _calc_handle_bbox(self, handle_mask):
        """从 handle_mask 计算 axis-aligned 包围盒."""
        mask_uint8 = handle_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            rect = cv2.minAreaRect(largest_contour)
            box_pts = cv2.boxPoints(rect)
            xs_pts = [int(p[0]) for p in box_pts]
            ys_pts = [int(p[1]) for p in box_pts]
            handle_bbox = [
                float(min(xs_pts)), float(min(ys_pts)),
                float(max(xs_pts)), float(max(ys_pts)),
            ]
        else:
            ys, xs = np.where(handle_mask)
            handle_bbox = [
                float(np.min(xs)), float(np.min(ys)),
                float(np.max(xs)), float(np.max(ys)),
            ]
        return handle_bbox

    # ------------------------------------------------------------
    def detect_handle(self, color_image, depth_frame=None):
        """
        主检测函数: YOLO-World → SAM → 水平扫描线分离 → 输出.

        返回:
            handle_mask    : np.ndarray (H, W), bool, 把手 mask 或 None
            handle_bbox    : [x1, y1, x2, y2], 把手 bbox 或 None
            body_mask      : np.ndarray (H, W), bool, 壶身 mask 或 None
            full_mask      : np.ndarray (H, W), bool, 完整水壶 mask 或 None
            stage          : str, "success" / "none"
            yolo_bbox      : [x1, y1, x2, y2], YOLO-World 原始检测框, 或 None
        """
        # === Step 1: YOLO-World 检测水壶本体 ===
        _t_yolo_start = time.perf_counter()
        best = self._detect_kettle_bbox(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t_yolo_start) * 1000

        if best is None:
            return None, None, None, None, "none", None

        bbox = best["bbox"]

        # === Step 2: SAM 生成完整水壶 mask ===
        _t_sam_start = time.perf_counter()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        full_mask = self._segment_full_mask(rgb_image, bbox)
        self._sam_time_ms = (time.perf_counter() - _t_sam_start) * 1000

        if full_mask is None:
            return None, None, None, None, "none", bbox

        # === Step 3: 水平扫描线分离把手 ===
        body_mask, handle_mask = scanline_find_handle(
            full_mask, bbox,
            handle_min_ratio=self.HANDLE_MIN_RATIO,
            handle_max_ratio=self.HANDLE_MAX_RATIO,
            handle_side=self.HANDLE_SIDE,
            handle_center_deviation_ratio=self.HANDLE_CENTER_DEVIATION_RATIO
        )

        if handle_mask is not None and body_mask is not None:
            handle_bbox = self._calc_handle_bbox(handle_mask)

            handle_area = float(np.sum(handle_mask))
            body_area = float(np.sum(body_mask))
            print("[INFO] Handle detected: area_ratio=%.1f%%" % (
                handle_area / body_area * 100))
            return handle_mask, handle_bbox, body_mask, full_mask, "success", bbox

        print("[WARN] No handle found (scanline detection failed)")
        return None, None, full_mask, full_mask, "none", bbox

    # ------------------------------------------------------------
    def visualize(self, display_img, body_mask, handle_mask,
                  handle_bbox, stage, yolo_bbox=None):
        """可视化检测结果."""
        # ---- 先绘制 YOLO-World 原始检测框 (紫色) ----
        if yolo_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display_img, "YOLO bbox", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        overlay = display_img.copy()

        if body_mask is not None:
            body_only = body_mask.copy()
            if handle_mask is not None:
                body_only = body_only & (~handle_mask)
            overlay[body_only] = (255, 0, 0)
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)

        if handle_mask is not None:
            overlay2 = display_img.copy()
            overlay2[handle_mask] = (0, 255, 0)
            cv2.addWeighted(overlay2, 0.4, display_img, 0.6, 0, display_img)

            mask_uint8 = handle_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)

        if body_mask is not None:
            mask_uint8 = body_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255, 0, 0), 1)

        if handle_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in handle_bbox]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 额外绘制旋转矩形轮廓
            if handle_mask is not None:
                mask_uint8 = handle_mask.astype(np.uint8) * 255
                ctrs, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if ctrs:
                    big_ctr = max(ctrs, key=cv2.contourArea)
                    rect = cv2.minAreaRect(big_ctr)
                    box_pts = cv2.boxPoints(rect)
                    box_pts = box_pts.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(display_img, [box_pts], 0, (0, 255, 255), 2)

        if stage == "success":
            color = (0, 255, 0)
            text = "Handle OK"
        else:
            color = (0, 0, 255)
            text = "No handle"

        cv2.putText(display_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_img, "YOLO: %dms  SAM: %dms" % (
            int(self._yolo_time_ms), int(self._sam_time_ms)),
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ------------------------------------------------------------
    def run(self):
        """
        主循环: 每 N 帧执行一次完整检测, 中间帧复用缓存.
        按 'q' 退出, 按 's' 保存当前帧 (原图+mask) 至 images/ 目录.
        """
        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        print("按 'q' 键退出程序...")
        print("按 's' 键保存当前帧 (原图+mask) 至 %s" % _save_dir)
        print("水壶检测: YOLO-World (%s)" % str(self.YW_CLASSES))
        print("把手侧: %s, 把手面积比: [%.0f%%, %.0f%%]" % (
            self.HANDLE_SIDE, self.HANDLE_MIN_RATIO * 100, self.HANDLE_MAX_RATIO * 100))
        print("跳帧策略: 每 %d 帧执行一次完整检测" % self._skip_frames)
        print("ROI 左右拓展: %.0f%%" % (self.ROI_EXPAND_RATIO * 100))

        _last_color = None
        _last_full_mask = None

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                self._frame_counter += 1
                display_img = color_image.copy()

                # --- 每 N 帧执行一次完整检测 ---
                if self._frame_counter % self._skip_frames == 1:
                    handle_mask, handle_bbox, body_mask, full_mask, stage, yolo_bbox = \
                        self.detect_handle(color_image, depth_frame)
                    self._cached_result = (
                        handle_mask, handle_bbox, body_mask, stage, yolo_bbox,
                    )
                    _last_color = color_image.copy()
                    _last_full_mask = full_mask.copy() if full_mask is not None else None
                else:
                    if self._cached_result is not None:
                        handle_mask, handle_bbox, body_mask, stage, yolo_bbox = self._cached_result
                    else:
                        handle_mask = None
                        handle_bbox = None
                        body_mask = None
                        stage = "none"
                        yolo_bbox = None

                self.visualize(display_img, body_mask, handle_mask,
                               handle_bbox, stage, yolo_bbox)

                cv2.imshow('YOLO-World+SAM Handle Detection', display_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if _last_color is not None:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        color_path = os.path.join(_save_dir, "frame_%s_color.png" % ts)
                        cv2.imwrite(color_path, _last_color)
                        print("[SAVE] %s" % color_path)
                        if _last_full_mask is not None:
                            mask_path = os.path.join(_save_dir, "frame_%s_mask.png" % ts)
                            cv2.imwrite(mask_path, (_last_full_mask.astype(np.uint8) * 255))
                            print("[SAVE] %s" % mask_path)
                    else:
                        print("[WARN] No frame saved yet (wait for detection)")

        finally:
            self.stop()
            print("[INFO] 程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseHandleDetector(config_dict=_MODULE_CFG)
    detector.run()