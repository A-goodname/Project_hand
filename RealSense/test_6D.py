# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶6D位姿估计 (新)
====================================
基于内参完好的新相机, 使用 rs.align 简化深度对齐链路.
继承 BaseRealSenseDetector 基类.
依赖 RealSense.realsense_shared 中的共享工具函数/类.

使用方法:
1. 连接 Intel RealSense 相机
2. 运行: python RealSense/test_6D.py
3. 按 'q' 键退出

输出格式: (X, Y, Z, A, B, C)
  X, Y, Z = 位置 (m)
  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np
from typing import Optional

from RealSense.realsense_shared import (
    BaseRealSenseDetector,
    rotation_matrix_to_euler_angles as _shared_r2euler,
    PoseSmoother,
    visualize_pose,
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

class RealSenseYOLO6DPoseDetector(BaseRealSenseDetector):
    """
    YOLOv8m 水瓶 6D 位姿估计检测器.

    外部引用 (供 estun_run.py 使用):
      - last_known_pose
      - rotation_matrix_to_euler_angles (static method)
    """
    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "test_6D.log"

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

        # 平滑器
        self._pose_smoother = PoseSmoother(alpha=self.SMOOTH_POSE_ALPHA)

        # --- 外部使用的属性 (供 estun_run.py 使用) ---
        self.last_known_pose: Optional[tuple] = None

    # ------------------------------------------------------------
    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """委托到 realsense_shared 中的实现 (保持外部引用兼容)."""
        return _shared_r2euler(R)

    # ------------------------------------------------------------
    def compute_6d_pose(self, bbox, aligned_depth_frame):
        """
        计算水瓶的 6D 位姿 (彩色坐标系).

        两阶段策略:
        1. 锚点深度: bbox 彩色中心 -> 直接读对齐深度
        2. 多点采样: 对 bbox 区域内每个彩色像素直接读对齐深度
           - 彩色内参反投影 -> 彩色坐标系 3D 点
           - |d - center_depth| 过滤离群点
        3. PCA 主成分分析 -> 旋转矩阵

        参数:
            bbox                : [x1, y1, x2, y2], 彩色坐标系检测框
            aligned_depth_frame : rs.depth_frame, 对齐后的深度帧

        返回:
            position: (x, y, z) 重心坐标 (m) — 彩色坐标系
            rotation: 3x3 旋转矩阵
            point_cloud: Nx3 点云
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, aligned_depth_frame)
        if anchor is None:
            return None, None, None
        _, _, center_depth = anchor

        # ---- Step 2: bbox 区域逐像素直接读对齐深度 + 彩色内参反投影 ----
        x1_c, y1_c = int(bbox[0]), int(bbox[1])
        x2_c, y2_c = int(bbox[2]), int(bbox[3])

        points = []
        for cy in range(y1_c, y2_c + 1, 2):
            for cx in range(x1_c, x2_c + 1, 2):
                pt = self._color_pixel_to_3d(cx, cy, aligned_depth_frame)
                if pt is None:
                    continue
                x, y, d = pt
                if abs(d - center_depth) > self.DEPTH_Z_THRESHOLD:
                    continue
                points.append([x, y, d])

        if len(points) < 10:
            # 点太少, 不限制 Z 值再试一次
            points_wide = []
            for cy in range(y1_c, y2_c + 1, 2):
                for cx in range(x1_c, x2_c + 1, 2):
                    pt = self._color_pixel_to_3d(cx, cy, aligned_depth_frame)
                    if pt is not None:
                        points_wide.append(list(pt))
            if len(points_wide) < 10:
                return None, None, None
            point_cloud = np.array(points_wide)
            print("[INFO] anchor depth: %.3fm, no-filter points: %d" % (center_depth, len(points_wide)))
        else:
            point_cloud = np.array(points)
            print("[INFO] anchor depth: %.3fm, filtered points: %d (bbox %dx%d px)" % (
                center_depth, len(points), x2_c - x1_c, y2_c - y1_c))

        # ---- Step 3: PCA 位姿 ----
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        target_y_main = np.real(eig_vec[:, 0])
        target_z = np.real(eig_vec[:, 2])

        if target_z[2] > 0:
            target_z = -target_z

        target_x = np.cross(target_y_main, target_z)
        target_x = target_x / np.linalg.norm(target_x)
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)

        return center, rot, point_cloud

    # ------------------------------------------------------------
    def run(self):
        """主循环."""
        print("按 'q' 键退出程序...")
        print("\n6D位姿格式: (X, Y, Z, A, B, C)")
        print("  X, Y, Z = 位置 (m) — 彩色相机坐标系")
        print("  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw")
        print("坐标轴颜色: 红=X 绿=Y(主方向, 水瓶纵向) 蓝=Z\n")

        try:
            while True:
                color_image, depth_image, depth_colormap, aligned_depth_frame = self.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue

                # ---- YOLO 检测 ----
                det = self._detect_targets(color_image)

                # ---- 6D 位姿计算与可视化 ----
                display_img = color_image.copy()

                if det is not None:
                    position, rotation, point_cloud = self.compute_6d_pose(det["bbox"], aligned_depth_frame)

                    if position is not None and rotation is not None:
                        smooth_pos, smooth_rot = self._pose_smoother.update(position, rotation)
                        if smooth_pos is None:
                            continue

                        A, B, C = self.rotation_matrix_to_euler_angles(smooth_rot)

                        print(("\n[6D] pos=(%.3f, %.3f, %.3f) "
                               "euler=(A=%.1f, B=%.1f, C=%.1f)") % (
                                   smooth_pos[0], smooth_pos[1], smooth_pos[2],
                                   A, B, C))
                        self._log_write("[6D]", "pos=(%.3f, %.3f, %.3f) euler=(A=%.1f, B=%.1f, C=%.1f)" % (
                                        smooth_pos[0], smooth_pos[1], smooth_pos[2], A, B, C))
                        if point_cloud is not None:
                            print("\n[PointCloud] points=%d" % (point_cloud.shape[0]))
                            self._log_write("[PointCloud]", "points=%d" % (point_cloud.shape[0]))

                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 使用彩色内参投影 (彩色坐标系 -> 彩色图)
                        K = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                                      [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                                      [0, 0, 1]], dtype=float)
                        visualize_pose(display_img, smooth_pos, smooth_rot, K, scale=0.08)

                        text_pos = 'Pos: (%.2f, %.2f, %.2f)' % (smooth_pos[0], smooth_pos[1], smooth_pos[2])
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        text_euler = 'Euler: (A=%.1f, B=%.1f, C=%.1f)' % (A, B, C)
                        cv2.putText(display_img, text_euler, (x1, y1 - 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    else:
                        print("Warning: 点云数量不足, 无法计算位姿")

                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8 6D Pose (RealSense/test_6D)', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.stop()
            print("程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseYOLO6DPoseDetector(config_dict=_MODULE_CFG)
    detector.run()