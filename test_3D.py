# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶3D定位监控

使用方法：
1. 连接Intel RealSense相机
2. 运行：python test_3D.py
3. 按 'q' 键退出

注意：
- 使用COCO数据集预训练的YOLOv8m模型，水瓶类别ID为39
- 3D位置计算：锚点深度 + bbox采样 + 深度离群过滤 + 点云均值

- 原方案：单点映射 → 单点反投影，受单个像素噪声影响大
- 新方案：锚点深度 + bbox内多点采样 + 深度离群过滤 + 全部点云均值
"""

model_name = './model/yolov8n'
class_id = 39
depth_min = 0.15   # 最小有效深度（米）
depth_max = 1.5    # 最大有效深度（米）
DEPTH_Z_THRESHOLD = 0.05  # 深度离群过滤阈值（米）

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO

# ---- GPU/CPU 设备选择 ----
if torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print(f"[INFO] Running on device: {DEVICE}")

class PositionSmoother:
    """EMA 时域平滑器"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered = None

    def update(self, pos):
        if pos is None:
            return None
        if self.filtered is None:
            self.filtered = list(pos)
        else:
            for i in range(3):
                self.filtered[i] = (1 - self.alpha) * self.filtered[i] + self.alpha * pos[i]
        return tuple(self.filtered)

    def reset(self):
        self.filtered = None


class RealSenseYOLO3DDetector:
    def __init__(self, model_name='yolov8m', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.target_class_id = class_id  # COCO 'bottle'

        # --- RealSense ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.profile = self.pipeline.start(config)

        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] depth_scale = {self.depth_scale}")

        # 激光功率设适中，减少近距光斑过曝
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
            print("[INFO] laser_power set to 150")
        except Exception as e:
            print(f"[WARN] laser_power set failed: {e}")

        # --- 对齐对象 ---
        self.align = rs.align(rs.stream.color)

        # --- 深度相机内参 ---
        depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
        self.depth_intrinsics = depth_profile.get_intrinsics()
        print(f"[INFO] depth intrinsics: {self.depth_intrinsics}")

        # --- 彩色相机内参（延迟初始化） ---
        self.color_intrinsics = None

        # --- 彩色→深度 外参（用于精确像素映射） ---
        self.color_to_depth_extrinsics = None

        # --- YOLO ---
        self.model = YOLO(f'{model_name}.pt').to(DEVICE)
        self.colorizer = rs.colorizer()

        # --- 平滑 ---
        self._smoother = PositionSmoother(alpha=0.4)

        # --- 统计 ---
        self._aligned_zero_count = 0
        self._total_count = 0

    # ------------------------------------------------------------
    def _depth_median_around(self, cx, cy, depth_frame, half_size=3):
        """
        在 (cx, cy) 周围 half_size 窗口取深度中值
        此处的 (cx,cy) 已经是深度图坐标系下的坐标
        """
        w = depth_frame.get_width()
        h = depth_frame.get_height()

        x_s = max(0, cx - half_size)
        y_s = max(0, cy - half_size)
        x_e = min(w, cx + half_size + 1)
        y_e = min(h, cy + half_size + 1)

        depths = []
        for vy in range(y_s, y_e):
            for vx in range(x_s, x_e):
                d = depth_frame.get_distance(vx, vy)
                if depth_min < d < depth_max:
                    depths.append(d)

        if len(depths) < 3:
            return None
        depths.sort()
        return depths[len(depths) // 2]

    # ------------------------------------------------------------
    def get_frames(self):
        """获取彩色帧、对齐深度帧、原始深度帧"""
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        raw_depth_frame = frames.get_depth_frame()

        if not color_frame or (not aligned_depth_frame and not raw_depth_frame):
            return None, None, None, None, False

        # ---- 延迟初始化彩色内参 和 外参 ----
        if self.color_intrinsics is None:
            try:
                cp = color_frame.profile.as_video_stream_profile()
                self.color_intrinsics = cp.get_intrinsics()
                print(f"[INFO] color intrinsics (from frame): {self.color_intrinsics}")
                if np.isnan([self.color_intrinsics.ppx, self.color_intrinsics.ppy,
                             self.color_intrinsics.fx, self.color_intrinsics.fy]).any():
                    print("[WARN] color intrinsics contains NaN, falling back to depth intrinsics")
                    self.color_intrinsics = self.depth_intrinsics
            except Exception as e:
                print(f"[WARN] Failed to get color intrinsics from frame: {e}, using depth intrinsics")
                self.color_intrinsics = self.depth_intrinsics

        # ---- 延迟初始化外参 ----
        if self.color_to_depth_extrinsics is None:
            try:
                color_profile = self.profile.get_stream(rs.stream.color).as_video_stream_profile()
                depth_profile = self.profile.get_stream(rs.stream.depth).as_video_stream_profile()
                self.color_to_depth_extrinsics = color_profile.get_extrinsics_to(depth_profile)
                print(f"[INFO] color→depth extrinsics: translation={np.round(self.color_to_depth_extrinsics.translation, 4)} m")
            except Exception as e:
                print(f"[WARN] Failed to get extrinsics: {e}")
                self.color_to_depth_extrinsics = None

        # ---- 判断对齐深度帧是否有效 ----
        use_aligned = True
        aligned_valid_ratio = 0.0
        if aligned_depth_frame:
            aligned_data = np.asanyarray(aligned_depth_frame.get_data())
            aligned_valid_ratio = np.count_nonzero(aligned_data) / aligned_data.size
            if aligned_valid_ratio < 0.01:
                use_aligned = False
                self._aligned_zero_count += 1
        else:
            use_aligned = False

        self._total_count += 1
        if self._total_count % 30 == 1:
            if raw_depth_frame:
                raw_data = np.asanyarray(raw_depth_frame.get_data())
                raw_valid = np.count_nonzero(raw_data)
                print(f"[DIAG] aligned valid: {aligned_valid_ratio*100:.1f}% ({self._aligned_zero_count}/{self._total_count} zero)")
                print(f"[DIAG] raw valid: {100*raw_valid/raw_data.size:.1f}%")

        # 选择深度帧：优先用对齐帧，无效则回退到原始帧
        if use_aligned:
            depth_frame = aligned_depth_frame
        else:
            depth_frame = raw_depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image, depth_colormap, depth_frame, use_aligned

    # ------------------------------------------------------------
    def detect_targets(self, color_image):
        """YOLO 检测"""
        results = self.model(color_image, conf=self.conf_threshold, classes=[self.target_class_id])

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(boxes, confs, class_ids):
                if class_id == self.target_class_id:
                    detections.append([*box, conf, class_id])

        return detections

    # ------------------------------------------------------------
    def _get_anchor_depth(self, bbox, depth_frame):
        """
        获取 bbox 中心的锚点深度（单点）
        优先用 rs2_project_color_pixel_to_depth_pixel 精确映射，
        失败则用 _fallback_search_single 回退

        返回 (center_x, center_y, center_depth) 或 None
        """
        cx_color = (int(bbox[0]) + int(bbox[2])) // 2
        cy_color = (int(bbox[1]) + int(bbox[3])) // 2

        # ---- 尝试精确映射（彩色→深度） ----
        if self.color_to_depth_extrinsics is not None and self.color_intrinsics is not None:
            try:
                depth_data = depth_frame.as_frame().get_data()
                depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                    depth_data,
                    self.depth_scale,
                    0.1, 6.0,
                    self.depth_intrinsics,
                    self.color_intrinsics,
                    self.color_to_depth_extrinsics,
                    [cx_color, cy_color]
                )
                if depth_pixel and np.isfinite(depth_pixel).all():
                    dx, dy = int(depth_pixel[0]), int(depth_pixel[1])
                    w = depth_frame.get_width()
                    h = depth_frame.get_height()
                    if 0 <= dx < w and 0 <= dy < h:
                        depth_val = self._depth_median_around(dx, dy, depth_frame, half_size=3)
                        if depth_val is not None:
                            return cx_color, cy_color, depth_val
            except Exception:
                pass

        # ---- 单点回退：在 bbox 周围搜索最近有效深度 ----
        return self._fallback_search_single(bbox, depth_frame)

    def _fallback_search_single(self, bbox, depth_frame):
        """
        单点回退：在 bbox 周围 ±80px 搜索最近的有效深度点
        返回 (cx_color, cy_color, depth_val) 或 None
        """
        w = depth_frame.get_width()
        h = depth_frame.get_height()

        cx_bbox = (int(bbox[0]) + int(bbox[2])) // 2
        cy_bbox = (int(bbox[1]) + int(bbox[3])) // 2

        x1 = max(0, min(int(bbox[0]) - 80, w - 1))
        y1 = max(0, min(int(bbox[1]) - 80, h - 1))
        x2 = max(0, min(int(bbox[2]) + 80, w - 1))
        y2 = max(0, min(int(bbox[3]) + 80, h - 1))

        best_dist = float('inf')
        best_d = None
        best_vx = None
        best_vy = None

        for vy in range(y1, y2 + 1, 2):
            for vx in range(x1, x2 + 1, 2):
                d = depth_frame.get_distance(vx, vy)
                if depth_min < d < depth_max:
                    dist = (vx - cx_bbox) ** 2 + (vy - cy_bbox) ** 2
                    if dist < best_dist:
                        best_dist = dist
                        best_d = d
                        best_vx = vx
                        best_vy = vy

        if best_d is None:
            return None

        depth_val = self._depth_median_around(best_vx, best_vy, depth_frame, half_size=3)
        if depth_val is None:
            depth_val = best_d

        return cx_bbox, cy_bbox, depth_val

    # ------------------------------------------------------------
    def get_3d_position(self, bbox, depth_frame, use_aligned):
        """
        计算目标3D位置（点云均值方案）

        改进点（vs 原单点映射方案）：
        1. 锚点深度：映射 bbox 中心获取 center_depth（同原方案）
        2. 多点采样：在 bbox 区域内密集采样，反投影为 3D 点云
        3. 深度离群过滤：|d - center_depth| < 0.05，剔除背景/前景噪声
        4. 统计平均：全部点云取均值作为最终位置，大幅降低单像素噪声
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, depth_frame)
        if anchor is None:
            return None
        _, _, center_depth = anchor

        # ---- Step 2: bbox 区域采样 + 深度过滤 + 反投影 ----
        x1 = max(0, min(int(bbox[0]), depth_frame.get_width() - 1))
        y1 = max(0, min(int(bbox[1]), depth_frame.get_height() - 1))
        x2 = max(0, min(int(bbox[2]), depth_frame.get_width() - 1))
        y2 = max(0, min(int(bbox[3]), depth_frame.get_height() - 1))

        fx = self.depth_intrinsics.fx
        fy = self.depth_intrinsics.fy
        ppx = self.depth_intrinsics.ppx
        ppy = self.depth_intrinsics.ppy

        points = []
        for vy in range(y1, y2 + 1, 2):
            for vx in range(x1, x2 + 1, 2):
                d = depth_frame.get_distance(vx, vy)
                if not (depth_min < d < depth_max):
                    continue
                # 深度离群过滤：只保留与锚点深度相近的点
                if abs(d - center_depth) > DEPTH_Z_THRESHOLD:
                    continue
                depth_x = (vx - ppx) * d / fx
                depth_y = (vy - ppy) * d / fy
                points.append([depth_x, depth_y, d])

        if len(points) < 10:
            # 点太少，不做过滤再试一次
            points_wide = []
            for vy in range(y1, y2 + 1, 2):
                for vx in range(x1, x2 + 1, 2):
                    d = depth_frame.get_distance(vx, vy)
                    if depth_min < d < depth_max:
                        depth_x = (vx - ppx) * d / fx
                        depth_y = (vy - ppy) * d / fy
                        points_wide.append([depth_x, depth_y, d])
            if len(points_wide) < 10:
                return None
            point_cloud = np.array(points_wide)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, wide points: {len(points_wide)}")
        else:
            point_cloud = np.array(points)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)} (bbox {x2-x1}x{y2-y1} px)")

        # ---- Step 3: 点云均值作为最终位置 ----
        center = np.mean(point_cloud, axis=0)
        return (center[0], center[1], center[2])

    # ------------------------------------------------------------
    def run(self):
        print("按 'q' 键退出程序...")

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame, use_aligned = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- YOLO检测 ----
                detections = self.detect_targets(color_image)

                # ---- 3D位置计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    bbox = det[:4]
                    position = self.get_3d_position(bbox, depth_frame, use_aligned)

                    if position is not None:
                        # 时域平滑
                        smoothed = self._smoother.update(position)
                        print(f"\nTarget 3D position (x, y, z): ({smoothed[0]:.3f}, {smoothed[1]:.3f}, {smoothed[2]:.3f}) m")

                        # 绘制检测框和中心点
                        x1, y1, x2, y2 = map(int, bbox)
                        conf = det[4]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

                        text_pos = f'({smoothed[0]:.2f}, {smoothed[1]:.2f}, {smoothed[2]:.2f})'
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 并排显示
                if display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8m 3D Position (test_3D)', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")


if __name__ == "__main__":
    detector = RealSenseYOLO3DDetector(model_name=model_name, conf_threshold=0.5)
    detector.run()