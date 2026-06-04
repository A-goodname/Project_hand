# -*- coding: utf-8 -*-
"""
RealSense YOLO-World + SAM + 水平扫描线 + 3D 水壶把手定位
=========================================================
重构版本: 继承 BaseRealSenseDetector 基类.
在 test_SAM_handle.py 基础上增加 3D 定位.

流程:
  YOLO-World → SAM → 水平扫描线分离 → 3D 定位

深度处理仅在 3D 定位阶段进行 (Z 离群过滤), 不预处理 mask,
避免透明壶身深度异常干扰扫描线分离.

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/test_handle_3D.py
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
    color_to_depth_pixel,
    scanline_find_handle,
    PositionSmoother,
)


# ============================================================
#  1. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  2. 3D 定位工具函数
# ============================================================

def _calc_mask_3d(mask, depth_frame, anchor_depth,
                  depth_scale, depth_intr, color_intr, c2d_extr, d2c_extr,
                  step=4, z_threshold=0.05,
                  depth_min=0.15, depth_max=1.5):
    """
    在 mask 区域内采样, 计算 3D 位置 (彩色坐标系, 点云均值).

    流程:
      1. np.where(mask) 获取有效像素
      2. 隔步长采样
      3. 每个彩色像素映射到深度图 → 读深度
      4. Z 离群过滤 |d - anchor_depth| < z_threshold
      5. 彩色内参反投影 → 彩色坐标系 3D → 点云均值

    参数:
        mask             : np.ndarray (H, W), bool
        depth_frame      : pyrealsense2 depth frame
        anchor_depth     : float, 锚点深度
        depth_scale      : float
        depth_intr       : rs.intrinsics
        color_intr       : rs.intrinsics
        c2d_extr         : rs.extrinsics
        d2c_extr         : rs.extrinsics
        step             : int, 步长
        z_threshold      : float, 离群过滤阈值 (米), None 时不过滤
        depth_min        : float, 有效深度最小值
        depth_max        : float, 有效深度最大值

    返回:
        (x, y, z) 或 None (点不足时)
    """
    ys, xs = np.where(mask)
    if len(ys) < 5:
        return None

    points = []
    indices = np.arange(0, len(ys), step)
    for idx in indices:
        cy = ys[idx]
        cx = xs[idx]
        dp = color_to_depth_pixel(cx, cy, depth_frame, depth_scale,
                                  depth_intr, color_intr, c2d_extr, d2c_extr)
        if dp is None:
            continue
        dx, dy = dp
        d = depth_frame.get_distance(dx, dy)
        if not (depth_min < d < depth_max):
            continue
        if z_threshold is not None and anchor_depth is not None:
            if abs(d - anchor_depth) > z_threshold:
                continue

        # 彩色内参反投影 → 彩色坐标系
        x = (cx - color_intr.ppx) * d / color_intr.fx
        y = (cy - color_intr.ppy) * d / color_intr.fy
        points.append([x, y, d])

    if len(points) < 10:
        return None

    point_cloud = np.array(points)
    center = np.mean(point_cloud, axis=0)
    return (center[0], center[1], center[2])


# ============================================================
#  3. 主检测器
# ============================================================

class RealSenseHandle3DDetector(BaseRealSenseDetector):
    """
    YOLO-World + SAM + 水平扫描线 + 3D 定位.
    """
    USE_YOLO_WORLD = True
    USE_SAM = True
    LOG_NAME = "test_handle_3D.log"

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

        # --- 耗时统计 ---
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # ---- 跳帧缓存 ----
        self._frame_counter = 0
        self._skip_frames = 10
        self._cached_result = None  # (body_mask, handle_mask, stage, yolo_bbox, handle_pos)

        # ---- EMA 平滑 (仅 handle) ----
        self._handle_smoother = PositionSmoother(alpha=0.4)

    # ------------------------------------------------------------
    def _get_anchor_depth(self, bbox, depth_frame):
        """
        覆盖基类: bbox 内均匀采样 50 个点, 取有效深度中位数 (原 test_handle_3D 策略).
        """
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_roi = y2 - y1
        w_roi = x2 - x1

        depths = []
        np.random.seed(0)
        for _ in range(50):
            cy = y1 + np.random.randint(0, max(h_roi, 1))
            cx = x1 + np.random.randint(0, max(w_roi, 1))
            dp = self._color_to_depth_pixel(cx, cy, depth_frame)
            if dp is None:
                continue
            dx, dy = dp
            d = depth_frame.get_distance(dx, dy)
            if self.DEPTH_MIN < d < self.DEPTH_MAX:
                depths.append(d)

        if len(depths) < 5:
            return None
        depths.sort()
        return depths[len(depths) // 2]

    # ------------------------------------------------------------
    def detect_handle(self, color_image, depth_frame):
        """
        主检测函数: YOLO-World → SAM → 扫描线分离 → 3D.

        返回:
            body_mask      : np.ndarray (H, W), bool, 壶身 mask
            handle_mask    : np.ndarray (H, W), bool, 把手 mask
            stage          : str, "success" / "none"
            yolo_bbox      : [x1, y1, x2, y2], YOLO 原始检测框
            handle_pos     : (x, y, z) 或 None, 把手 3D 位置
        """
        # === Step 1: YOLO-World 检测水壶本体 ===
        _t_yolo_start = time.perf_counter()
        best = self._detect_kettle_bbox(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t_yolo_start) * 1000

        if best is None:
            return None, None, "none", None, None

        bbox = best["bbox"]

        # === Step 2: 获取锚点深度 (用于 3D 定位的 Z 过滤) ===
        anchor_depth = self._get_anchor_depth(bbox, depth_frame)
        if anchor_depth is None:
            print("[WARN] No valid anchor depth, skip 3D positioning")

        # === Step 3: SAM 生成完整水壶 mask ===
        _t_sam_start = time.perf_counter()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        full_mask = self._segment_full_mask(rgb_image, bbox)
        self._sam_time_ms = (time.perf_counter() - _t_sam_start) * 1000

        if full_mask is None:
            return None, None, "none", bbox, None

        # === Step 4: 水平扫描线分离把手 ===
        body_mask, handle_mask = scanline_find_handle(
            full_mask, bbox,
            handle_min_ratio=self.HANDLE_MIN_RATIO,
            handle_max_ratio=self.HANDLE_MAX_RATIO,
            handle_side=self.HANDLE_SIDE
        )

        # === Step 5: 3D 定位 ===
        handle_pos = None
        if handle_mask is not None and anchor_depth is not None:
            handle_pos = _calc_mask_3d(
                handle_mask, depth_frame, anchor_depth,
                self.depth_scale, self.depth_intrinsics, self.color_intrinsics,
                self.color_to_depth_extrinsics, self.depth_to_color_extrinsics,
                step=2, z_threshold=self.DEPTH_Z_THRESHOLD,
                depth_min=self.DEPTH_MIN, depth_max=self.DEPTH_MAX
            )

        if handle_mask is not None and body_mask is not None:
            handle_area = float(np.sum(handle_mask))
            body_area_val = float(np.sum(body_mask))
            pos_str = "(%.3f, %.3f, %.3f)" % handle_pos if handle_pos else "(none)"
            print("[INFO] Handle detected: area_ratio=%.1f%%, handle_3D=%s" % (
                handle_area / body_area_val * 100, pos_str))
            return body_mask, handle_mask, "success", bbox, handle_pos

        print("[WARN] No handle found (scanline detection failed)")
        return body_mask, None, "none", bbox, None

    # ------------------------------------------------------------
    def visualize(self, display_img, body_mask, handle_mask,
                  stage, yolo_bbox, handle_pos):
        """可视化检测结果 + 3D 坐标."""
        # ---- YOLO 检测框 (紫色) ----
        if yolo_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display_img, "YOLO bbox", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        # ---- mask 叠加 ----
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

        # ---- 3D 文字标注 ----
        if handle_pos is not None:
            text_h = "Handle: (%.3f, %.3f, %.3f) m" % handle_pos
            cv2.putText(display_img, text_h, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 把手中心点
            if handle_mask is not None:
                ys_h, xs_h = np.where(handle_mask)
                if len(ys_h) > 0:
                    cy_h = int(np.mean(ys_h))
                    cx_h = int(np.mean(xs_h))
                    cv2.circle(display_img, (cx_h, cy_h), 4, (0, 255, 0), -1)

        # ---- 状态文字 ----
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
        主循环: 每 N 帧执行一次完整检测 + 3D 定位.
        按 'q' 退出, 按 's' 保存当前帧.
        """
        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        print("按 'q' 键退出程序...")
        print("按 's' 键保存当前帧至 %s" % _save_dir)
        print("水壶检测: YOLO-World (%s)" % str(self.YW_CLASSES))
        print("把手侧: %s, 把手面积比: [%.0f%%, %.0f%%]" % (
            self.HANDLE_SIDE, self.HANDLE_MIN_RATIO * 100, self.HANDLE_MAX_RATIO * 100))
        print("深度 Z 过滤阈值: %.2fm" % self.DEPTH_Z_THRESHOLD)
        print("跳帧策略: 每 %d 帧执行一次完整检测" % self._skip_frames)
        print("ROI 左右拓展: %.0f%%" % (self.ROI_EXPAND_RATIO * 100))

        _last_color = None

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                self._frame_counter += 1
                display_img = color_image.copy()

                # --- 每 N 帧执行一次完整检测 ---
                if self._frame_counter % self._skip_frames == 1:
                    body_mask, handle_mask, stage, yolo_bbox, handle_pos = \
                        self.detect_handle(color_image, depth_frame)

                    # EMA 平滑 handle_pos
                    if handle_pos is not None:
                        handle_pos = self._handle_smoother.update(handle_pos)
                    else:
                        self._handle_smoother.reset()

                    self._cached_result = (
                        body_mask, handle_mask, stage, yolo_bbox, handle_pos,
                    )
                    _last_color = color_image.copy()
                else:
                    if self._cached_result is not None:
                        body_mask, handle_mask, stage, yolo_bbox, handle_pos = \
                            self._cached_result
                    else:
                        body_mask = None
                        handle_mask = None
                        stage = "none"
                        yolo_bbox = None
                        handle_pos = None

                self.visualize(display_img, body_mask, handle_mask,
                               stage, yolo_bbox, handle_pos)

                # 叠加深度热力图
                if depth_colormap is not None:
                    h_vis = display_img.shape[0]
                    depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                    vis_panel = np.hstack((display_img, depth_small))
                else:
                    vis_panel = display_img

                cv2.imshow('Handle 3D Detection (YOLO-World+SAM)', vis_panel)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if _last_color is not None:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        color_path = os.path.join(_save_dir, "handle3d_%s.png" % ts)
                        cv2.imwrite(color_path, _last_color)
                        print("[SAVE] %s" % color_path)

        finally:
            self.stop()
            print("[INFO] 程序已退出")


# ============================================================
#  4. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseHandle3DDetector(config_dict=_MODULE_CFG)
    detector.run()