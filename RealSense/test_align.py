# -*- coding: utf-8 -*-
"""
RealSense D435i 对齐验证脚本
============================
验证 rs.align 能否在新 D435i (内参完好) 上正常工作.

验证方法:
取一帧数据, 对每个彩色像素 (cx,cy) 分别走两条链路计算 3D 坐标:

  链路A (rs.align: 深度→彩色):
    原始frameset → rs.align(rs.stream.color) → 对齐深度图 (彩色坐标系下)
    → aligned_depth[cy, cx] → 彩色内参反投影 → P_align (彩色相机系)

  链路B (手动映射: 彩色→深度, 同 RealSense_R 做法):
    彩色像素 (cx,cy) → rs2_project_color_pixel_to_depth_pixel → 原始深度图坐标 (dx,dy)
    → 读原始深度 d_raw → 深度内参反投影 → 深度相机系 3D点
    → depth→color 外参变换 → P_manual (彩色相机系)

比较 ||P_align - P_manual|| 的 3D 距离, 统计平均/最大差异.

使用方法:
  python RealSense/test_align.py
  按 'q' 退出
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pyrealsense2 as rs


class AlignValidator:
    """验证 rs.align 与手动映射两种链路的 3D 坐标一致性."""

    def __init__(self):
        # ---- 启动 pipeline ----
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.profile = self.pipeline.start(config)

        # ---- 深度缩放系数 ----
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("[INFO] depth_scale = %f" % self.depth_scale)

        # ---- 获取 SDK 内参 ----
        depth_profile = self.profile.get_stream(rs.stream.depth)
        color_profile = self.profile.get_stream(rs.stream.color)
        self.depth_intr = depth_profile.as_video_stream_profile().get_intrinsics()
        self.color_intr = color_profile.as_video_stream_profile().get_intrinsics()
        self.c2d_extr = color_profile.get_extrinsics_to(depth_profile)
        self.d2c_extr = depth_profile.get_extrinsics_to(color_profile)

        print("[INFO] 深度内参: fx=%.3f, fy=%.3f, ppx=%.3f, ppy=%.3f" % (
            self.depth_intr.fx, self.depth_intr.fy, self.depth_intr.ppx, self.depth_intr.ppy))
        print("[INFO] 彩色内参: fx=%.3f, fy=%.3f, ppx=%.3f, ppy=%.3f" % (
            self.color_intr.fx, self.color_intr.fy, self.color_intr.ppx, self.color_intr.ppy))
        print("[INFO] d2c 外参 t: (%.4f, %.4f, %.4f)" % (
            self.d2c_extr.translation[0], self.d2c_extr.translation[1], self.d2c_extr.translation[2]))

        # ---- 对齐器 ----
        self.align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

    # ------------------------------------------------------------
    def _depth_pixel_to_3d(self, dx, dy, d, intr):
        """
        深度图坐标 (dx,dy) + 深度值 d(米) → 该相机坐标系下 3D 点.
        intr: rs.intrinsics
        """
        point = rs.rs2_deproject_pixel_to_point(intr, [dx, dy], d)
        return np.array([point[0], point[1], point[2]])

    # ------------------------------------------------------------
    def _color_pixel_to_3d_align(self, cx, cy, aligned_depth, color_intr):
        """
        链路A: 从对齐深度图取深度, 用彩色内参反投影.
        返回 P_align (彩色相机系)
        """
        d = float(aligned_depth[cy, cx])
        if d <= 0:
            return None
        pt = rs.rs2_deproject_pixel_to_point(color_intr, [cx, cy], d)
        return np.array([pt[0], pt[1], pt[2]])

    # ------------------------------------------------------------
    def _color_pixel_to_3d_manual(self, cx, cy, raw_depth_frame):
        """
        链路B: 手动映射, 同 RealSense_R 做法.

        步骤:
          1. 彩色像素 → rs2_project_color_pixel_to_depth_pixel → 深度图坐标 (dx,dy)
          2. 读原始深度 d_raw
          3. 深度内参反投影 → 深度相机系 3D 点
          4. depth→color 外参变换 → 彩色相机系 3D 点

        返回 P_manual (彩色相机系) 或 None
        """
        # Step 1: 彩色→深度像素映射
        try:
            dp = rs.rs2_project_color_pixel_to_depth_pixel(
                raw_depth_frame.get_data(), self.depth_scale,
                0.1, 6.0,
                self.depth_intr, self.color_intr,
                self.c2d_extr, self.d2c_extr,
                [cx, cy]
            )
        except Exception:
            return None
        if dp is None or not np.isfinite(dp).all():
            return None

        dx, dy = int(dp[0]), int(dp[1])
        w = raw_depth_frame.get_width()
        h = raw_depth_frame.get_height()
        if not (0 <= dx < w and 0 <= dy < h):
            return None

        # Step 2: 读原始深度
        d_raw = raw_depth_frame.get_distance(dx, dy)
        if not (0.1 < d_raw < 6.0):
            return None

        # Step 3: 深度内参反投影 → 深度相机系
        pt_depth = self._depth_pixel_to_3d(dx, dy, d_raw, self.depth_intr)

        # Step 4: depth→color 外参变换 → 彩色相机系
        # 旋转: R_d2c @ pt_depth + t_d2c
        R = np.array(self.d2c_extr.rotation).reshape(3, 3)
        t = np.array(self.d2c_extr.translation)
        pt_color = R @ pt_depth + t

        return pt_color

    # ------------------------------------------------------------
    def run(self):
        """取一帧, 对比 ROI 内所有像素的两条链路 3D 坐标差异."""
        print("按 'q' 键退出...")
        print("=" * 60)

        WIN_NAME = 'rs.align verification | color vs aligned depth'
        cv2.namedWindow(WIN_NAME)

        # ROI: 中心区域, 避开边缘畸变严重的区域
        roi_x1, roi_y1 = 120, 90
        roi_x2, roi_y2 = 520, 390

        # 等待获取一帧稳定数据
        print("[INFO] 正在获取帧数据...")
        for _ in range(10):
            self.pipeline.wait_for_frames()

        # ---- 取一帧 ----
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        raw_depth_frame = frames.get_depth_frame()

        if not aligned_depth_frame or not color_frame or not raw_depth_frame:
            print("[ERROR] 获取帧失败")
            self.pipeline.stop()
            return

        # ---- 图像数据 ----
        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth = np.asanyarray(aligned_depth_frame.get_data()).astype(np.float32)
        aligned_depth *= self.depth_scale
        colormap = np.asanyarray(self.colorizer.colorize(aligned_depth_frame).get_data())

        # ---- 预计算深度梯度 mask (排除跳变边缘) ----
        sobel_x = cv2.Sobel(aligned_depth, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(aligned_depth, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        # 梯度阈值: 深度变化 > 5cm/像素视为跳变边缘
        edge_mask = grad_mag > 0.05

        # ---- 逐像素对比 ----
        results_all = []    # 全部像素
        results_flat = []   # 排除边缘后的平坦区域

        for cy in range(roi_y1, roi_y2 + 1, 2):  # 步长2加快速度
            for cx in range(roi_x1, roi_x2 + 1, 2):
                # 链路A: rs.align
                d_align = float(aligned_depth[cy, cx])
                if not (0.1 < d_align < 6.0):
                    continue

                # 链路B: 手动映射
                P_manual = self._color_pixel_to_3d_manual(cx, cy, raw_depth_frame)
                if P_manual is None:
                    continue

                # 链路A 的 3D 坐标
                P_align = self._color_pixel_to_3d_align(cx, cy, aligned_depth, self.color_intr)
                if P_align is None:
                    continue

                # 3D 距离
                diff_3d = np.linalg.norm(P_align - P_manual)
                is_edge = bool(edge_mask[cy, cx])

                # Z 差异过大说明手动映射到了错误深度, 标记为异常
                z_diff = abs(P_align[2] - P_manual[2])
                is_z_abnormal = z_diff > 0.5  # 50cm Z差异

                item = (cx, cy, d_align, P_align, P_manual, diff_3d, is_edge, is_z_abnormal)
                results_all.append(item)
                if not is_edge and not is_z_abnormal:
                    results_flat.append(item)

        # ---- 统计 ----
        if len(results_all) == 0:
            print("[ERROR] 无有效像素用于对比")
            self.pipeline.stop()
            return

        diffs_all = np.array([r[5] for r in results_all])
        avg_all = np.mean(diffs_all)
        max_all = np.max(diffs_all)

        print("=" * 60)
        print("验证结果统计 (全部 %d px):" % len(results_all))
        print("  3D 距离 平均差异: %.2f mm" % (avg_all * 1000))
        print("  3D 距离 最大差异: %.2f mm" % (max_all * 1000))

        # ---- 平坦区域统计 ----
        if len(results_flat) > 0:
            diffs_flat = np.array([r[5] for r in results_flat])
            avg_flat = np.mean(diffs_flat)
            max_flat = np.max(diffs_flat)
            median_flat = np.median(diffs_flat)
            std_flat = np.std(diffs_flat)
            print("")
            print("平坦区域 (排除边缘+Z异常) %d px:" % len(results_flat))
            print("  3D 距离 平均差异: %.2f mm" % (avg_flat * 1000))
            print("  3D 距离 中值差异: %.2f mm" % (median_flat * 1000))
            print("  3D 距离 标准差:   %.2f mm" % (std_flat * 1000))
            print("  3D 距离 最大差异: %.2f mm" % (max_flat * 1000))
        else:
            avg_flat = avg_all
            max_flat = max_all
            median_flat = 0
            std_flat = 0

        # ---- 查看异常点详情 ----
        abnormal = [r for r in results_all if r[7]]  # is_z_abnormal
        if abnormal:
            print("")
            print("Z异常像素 (|Z_diff| > 50cm): %d 个" % len(abnormal))
            if len(abnormal) > 0:
                # 取前3个
                for cx, cy, d_align, P_align, P_manual, diff, _, _ in abnormal[:3]:
                    print("  (%d,%d) d_align=%.3f  P_align_z=%.3f  P_manual_z=%.3f  diff=%.0f mm" % (
                        cx, cy, d_align, P_align[2], P_manual[2], diff * 1000))

        # ---- 判断 ----
        threshold_avg = 0.005  # 5mm
        threshold_max = 0.030  # 30mm
        print("")
        print("=" * 60)
        print("判断 (基于平坦区域):")
        print("  平均差异 < %d mm + 最大差异 < %d mm => rs.align 可用" % (
            threshold_avg * 1000, threshold_max * 1000))
        if len(results_flat) > 0:
            if avg_flat < threshold_avg and max_flat < threshold_max:
                print("  => rs.align 在 D435i 上验证通过!")
            elif avg_flat < threshold_avg:
                print("  => 平均差异合格 (%.1f mm), 但最大差异超标 (%.1f mm > %d mm), 边缘区域需注意" % (
                    avg_flat * 1000, max_flat * 1000, threshold_max * 1000))
                print("  => rs.align 可用, 但建议使用时避开深度跳变边缘")
            else:
                print("  => rs.align 差异较大 (平均 %.1f mm > %d mm), 需要进一步排查" % (
                    avg_flat * 1000, threshold_avg * 1000))
        print("  全部像素平均差异: %.1f mm (含 %d 个Z异常像素)" % (
            avg_all * 1000, len(abnormal)))

        # ---- 显示 ----
        display_img = color_image.copy()
        cv2.rectangle(display_img, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)

        info_lines = [
            "Total=%d  Flat=%d px" % (len(results_all), len(results_flat)),
            "Flat: avg=%.1f  max=%.1f mm" % (
                (avg_flat if len(results_flat) > 0 else 0) * 1000,
                (max_flat if len(results_flat) > 0 else 0) * 1000),
            "All:  avg=%.1f  max=%.1f mm" % (avg_all * 1000, max_all * 1000),
            "Z-abnormal=%d px (excluded)" % len(abnormal),
        ]
        for i, line in enumerate(info_lines):
            cv2.putText(display_img, line, (10, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        images = np.hstack((display_img, colormap))
        cv2.imshow(WIN_NAME, images)
        print("")
        print("窗口已显示, 按 'q' 键退出...")

        while True:
            key = cv2.waitKey(25) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

        cv2.destroyAllWindows()
        for _ in range(3):
            cv2.waitKey(1)
        self.pipeline.stop()
        print("程序已退出")


if __name__ == "__main__":
    validator = AlignValidator()
    validator.run()