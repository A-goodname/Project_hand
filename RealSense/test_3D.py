# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶3D定位监控 (新)
=====================================
基于内参完好的新相机, 使用 rs.align 简化深度对齐链路.
继承 BaseRealSenseDetector 基类.
使用 COCO 数据集预训练的 YOLOv8m 模型, 水瓶类别 ID 为 39.

使用方法:
1. 连接 Intel RealSense 相机
2. 运行: python RealSense/test_3D.py
3. 按 'q' 键退出
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np

from RealSense.realsense_shared import BaseRealSenseDetector, PositionSmoother


# ============================================================
#  1. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  2. 主检测器
# ============================================================

class RealSenseYOLO3DDetector(BaseRealSenseDetector):
    """YOLOv8m 水瓶 3D 定位检测器."""
    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "test_3D.log"

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

        # 平滑器 (使用基类的 SMOOTH_ALPHA)
        self._smoother = PositionSmoother(alpha=self.SMOOTH_ALPHA)

    # ------------------------------------------------------------
    def get_3d_position(self, bbox, aligned_depth_frame):
        """
        计算目标 3D 位置 (彩色坐标系, 点云均值方案).

        由于深度已通过 rs.align 对齐至彩色坐标系:
        1. 直接在彩色 bbox 中心读取深度作为锚点 (用于离群过滤)
        2. 对 bbox 区域内每个彩色像素直接读对齐深度
        3. Z 值离群过滤 (|d - center_depth| < DEPTH_Z_THRESHOLD)
        4. 彩色内参反投影 -> 彩色坐标系 3D 点
        5. 全部有效点云取均值作为最终位置
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, aligned_depth_frame)
        if anchor is None:
            return None
        _, _, center_depth = anchor

        # ---- Step 2: bbox 区域逐像素直接读对齐深度 ----
        x1_c, y1_c = int(bbox[0]), int(bbox[1])
        x2_c, y2_c = int(bbox[2]), int(bbox[3])

        points = []
        for cy in range(y1_c, y2_c + 1, 2):
            for cx in range(x1_c, x2_c + 1, 2):
                d = aligned_depth_frame.get_distance(cx, cy)
                if not (self.DEPTH_MIN < d < self.DEPTH_MAX):
                    continue
                if abs(d - center_depth) > self.DEPTH_Z_THRESHOLD:
                    continue
                # 彩色内参反投影 (depth 已在彩色坐标系)
                x = (cx - self.color_intrinsics.ppx) * d / self.color_intrinsics.fx
                y = (cy - self.color_intrinsics.ppy) * d / self.color_intrinsics.fy
                points.append([x, y, d])

        if len(points) < 10:
            # 点太少, 不限制 Z 值再试一次
            points_wide = []
            for cy in range(y1_c, y2_c + 1, 2):
                for cx in range(x1_c, x2_c + 1, 2):
                    d = aligned_depth_frame.get_distance(cx, cy)
                    if self.DEPTH_MIN < d < self.DEPTH_MAX:
                        x = (cx - self.color_intrinsics.ppx) * d / self.color_intrinsics.fx
                        y = (cy - self.color_intrinsics.ppy) * d / self.color_intrinsics.fy
                        points_wide.append([x, y, d])
            if len(points_wide) < 10:
                return None
            point_cloud = np.array(points_wide)
            print("[INFO] anchor depth: %.3fm, wide points: %d" % (center_depth, len(points_wide)))
        else:
            point_cloud = np.array(points)
            print("[INFO] anchor depth: %.3fm, filtered points: %d (bbox %dx%d px)" % (
                center_depth, len(points), x2_c - x1_c, y2_c - y1_c))

        # ---- Step 3: 点云均值作为最终位置 (XYZ 均为均值) ----
        center = np.mean(point_cloud, axis=0)
        return (center[0], center[1], center[2])

    # ------------------------------------------------------------
    def run(self):
        """主循环."""
        print("按 'q' 键退出程序...")

        try:
            while True:
                color_image, depth_image, depth_colormap, aligned_depth_frame = self.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue

                # ---- YOLO 检测 ----
                detections = self.detect_targets(color_image)

                # 只保留置信度最高的一个检测
                if detections:
                    detections = [max(detections, key=lambda d: d[4])]

                # ---- 3D 位置计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    bbox = det[:4]
                    position = self.get_3d_position(bbox, aligned_depth_frame)

                    if position is not None:
                        # 时域平滑
                        smoothed = self._smoother.update(position)
                        if smoothed is None:
                            continue
                        print(("\n[3D] pos=(%.3f, %.3f, %.3f) m") % smoothed)
                        self._log_write("[3D]", "pos=(%.3f, %.3f, %.3f) m" % smoothed)

                        # 绘制检测框和中心点
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

                        text_pos = '(%.2f, %.2f, %.2f)' % (smoothed[0], smoothed[1], smoothed[2])
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 并排显示彩色图 + 对齐深度热力图
                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8m 3D Position (RealSense/test_3D)', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()
            print("程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseYOLO3DDetector(config_dict=_MODULE_CFG)
    detector.run()