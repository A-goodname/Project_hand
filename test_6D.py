# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶6D位姿估计 (v9)

锚点 + 对齐帧深度采样 + PCA 6D 位姿估计:
- YOLOv8m 检测水瓶 (COCO class 39)
- 锚点: 用 rs2_project_color_pixel_to_depth_pixel 映射 bbox 中心 (失败则单点回退)
- 点云: 在 aligned depth frame 的 bbox 区域内采样, 过滤 |d - center_depth| < 0.05 m
- PCA 计算点云主方向, 构建 6D 位姿 (位置 + 旋转矩阵)

==== v9 修正 (2026-05-11) ====
问题: rs2_project_color_pixel_to_depth_pixel 在逐像素遍历时始终返回 0 有效点,
导致全走回退, 回退收集 5000+ 背景点, 质心严重偏离.

方案: 放弃全像素映射, 改用两阶段策略:
1. 仅映射/回退 bbox 中心 -> 得到锚点深度 center_depth (与 test_3D 一致)
2. 在对齐深度帧上采样 bbox 像素, 用 |d - center_depth| < 0.05 过滤 -> 点云
"""

model_name = './model/yolov8m'
class_id = 39
depth_min = 0.15   # 最小有效深度 (米)
depth_max = 1.5    # 最大有效深度 (米)
DEPTH_Z_THRESHOLD = 0.05  # 深度离群过滤阈值 (米)

import os
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
    """EMA 时域平滑器 (用于3D位置)"""
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


class PoseSmoother:
    """6D位姿平滑器: 位置EMA + 旋转矩阵指数移动平均 + SVD正交化"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_pos = None
        self.filtered_rot = None

    def update(self, pos, rot):
        if pos is None or rot is None:
            return None, None
        if self.filtered_pos is None:
            self.filtered_pos = list(pos)
            self.filtered_rot = rot.copy()
        else:
            for i in range(3):
                self.filtered_pos[i] = (1 - self.alpha) * self.filtered_pos[i] + self.alpha * pos[i]
            self.filtered_rot = (1 - self.alpha) * self.filtered_rot + self.alpha * rot
            # SVD正交化, 确保是有效的旋转矩阵
            U, _, Vt = np.linalg.svd(self.filtered_rot)
            self.filtered_rot = U @ Vt
        return tuple(self.filtered_pos), self.filtered_rot

    def reset(self):
        self.filtered_pos = None
        self.filtered_rot = None


class RealSenseYOLO6DPoseDetector:
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

        # 激光功率设适中, 减少近距光斑过曝
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

        # --- 彩色相机内参 (延迟初始化) ---
        self.color_intrinsics = None

        # --- 彩色->深度 外参 (用于精确像素映射) ---
        self.color_to_depth_extrinsics = None

        # --- YOLO ---
        self.model = YOLO(f'{model_name}.pt').to(DEVICE)
        self.colorizer = rs.colorizer()

        # --- 平滑 ---
        self._pos_smoother = PositionSmoother(alpha=0.4)
        self._pose_smoother = PoseSmoother(alpha=0.3)

        # --- 统计: 对齐帧全零计数 ---
        self._aligned_zero_count = 0
        self._total_count = 0

    # ------------------------------------------------------------
    def _depth_median_around(self, cx, cy, depth_frame, half_size=3):
        """在 (cx, cy) 周围 half_size 窗口取深度中值."""
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
        """
        获取彩色帧 + 对齐深度帧 (warp 到彩色视口).
        返回 (color_image, depth_image, depth_colormap, depth_frame).
        """
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        aligned_depth_frame = aligned_frames.get_depth_frame()
        raw_depth_frame = frames.get_depth_frame()

        if not color_frame or (not aligned_depth_frame and not raw_depth_frame):
            return None, None, None, None

        # ---- 延迟初始化彩色内参 ----
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
                print(f"[INFO] color->depth extrinsics: translation={np.round(self.color_to_depth_extrinsics.translation, 4)} m")
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

        # 选择深度帧
        if use_aligned:
            depth_frame = aligned_depth_frame
        else:
            depth_frame = raw_depth_frame

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image, depth_colormap, depth_frame

    # ------------------------------------------------------------
    def detect_targets(self, color_image):
        """YOLO 检测, 返回 [(x1,y1,x2,y2,conf), ...]"""
        results = self.model(color_image, conf=self.conf_threshold, classes=[self.target_class_id], verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confs):
                detections.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf)])

        # ---- 将 YOLO verbose 日志写文件 (覆盖式), 控制台不再打印 ----
        if results and len(results[0].boxes) > 0:
            speed = results[0].speed
            h, w = results[0].orig_shape
            cls_name = results[0].names.get(int(results[0].boxes.cls[0]), '?')
            log_lines = [
                f"0: {w}x{h} {len(results[0].boxes)} {cls_name}, {speed.get('inference', 0):.1f}ms",
                f"Speed: {speed.get('preprocess', 0):.1f}ms preprocess, {speed.get('inference', 0):.1f}ms inference, {speed.get('postprocess', 0):.1f}ms postprocess per image at shape (1, 3, {h}, {w})"
            ]
            log_path = 'doc/yolo_detect.log'
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines) + '\n')

        return detections

    # ------------------------------------------------------------
    def _get_anchor_depth(self, bbox, depth_frame):
        """
        获取 bbox 中心的锚点深度 (单点, 与 test_3D 一致).

        优先用 rs2_project_color_pixel_to_depth_pixel 精确映射,
        失败则用 _fallback_search 单点回退.

        返回 (center_x, center_y, center_depth) 或 None
        """
        cx_color = (int(bbox[0]) + int(bbox[2])) // 2
        cy_color = (int(bbox[1]) + int(bbox[3])) // 2

        # ---- 尝试精确映射 ----
        if self.color_to_depth_extrinsics is not None and self.color_intrinsics is not None:
            try:
                raw_depth_frame = self._cached_raw_depth
                if raw_depth_frame is not None:
                    depth_data = raw_depth_frame.as_frame().get_data()
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

        # ---- 单点回退 (与 test_3D 一致) ----
        return self._fallback_search_single(bbox, depth_frame)

    def _fallback_search_single(self, bbox, depth_frame):
        """
        与 test_3D 一致的单点回退:
        在 bbox 周围 +/-80px 逐行搜索最近的有效深度点.
        返回 (cx_color, cy_color, depth_val) 或 None.
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

        # 在找到的最优点周围再取中值
        depth_val = self._depth_median_around(best_vx, best_vy, depth_frame, half_size=3)
        if depth_val is None:
            depth_val = best_d

        return cx_bbox, cy_bbox, depth_val

    # ------------------------------------------------------------
    def compute_6d_pose(self, bbox, depth_frame):
        """
        计算水瓶的6D位姿.

        两阶段策略:
        1. 锚点: 获取 bbox 中心深度 center_depth (映射 -> 单点回退)
        2. 点云: 在对齐深度帧的 bbox 区域采样, 过滤 |d - center_depth| < 0.05 m

        返回:
            position: (x, y, z) 重心坐标 (米)
            rotation: 3x3 旋转矩阵
            point_cloud: Nx3 点云
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, depth_frame)
        if anchor is None:
            return None, None, None
        cx_color, cy_color, center_depth = anchor

        # ---- Step 2: 在对齐深度帧 bbox 区域采样 ----
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
                # 深度离群过滤: 只保留与锚点深度相差 < threshold 的点
                if abs(d - center_depth) > DEPTH_Z_THRESHOLD:
                    continue
                depth_x = (vx - ppx) * d / fx
                depth_y = (vy - ppy) * d / fy
                points.append([depth_x, depth_y, d])

        if len(points) < 10:
            # 点太少, 不做过滤再试一次
            points_wide = []
            for vy in range(y1, y2 + 1, 2):
                for vx in range(x1, x2 + 1, 2):
                    d = depth_frame.get_distance(vx, vy)
                    if depth_min < d < depth_max:
                        depth_x = (vx - ppx) * d / fx
                        depth_y = (vy - ppy) * d / fy
                        points_wide.append([depth_x, depth_y, d])
            if len(points_wide) < 10:
                return None, None, None
            point_cloud = np.array(points_wide)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, no-filter points: {len(points_wide)}")
        else:
            point_cloud = np.array(points)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)} (bbox {x2-x1}x{y2-y1} px)")

        # ---- Step 3: PCA 位姿 ----
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主方向 (最大特征值) -> Y轴 (绿色), 水瓶竖直时为主轴
        target_y_main = eig_vec[:, 0].real
        # 最小方差方向 -> Z轴 (蓝色), 指向相机
        target_z = eig_vec[:, 2].real

        if target_z[2] > 0:
            target_z = -target_z

        # X轴 = Y x Z, 确保右手系
        target_x = np.cross(target_y_main, target_z)
        target_x = target_x / np.linalg.norm(target_x)
        # 重新正交化 Y = Z x X
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)

        return center, rot, point_cloud

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """
        从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw).
        返回角度值 (度).
        """
        # A (roll)  = atan2(R[2,1], R[2,2])
        # B (pitch) = -asin(R[2,0])
        # C (yaw)   = atan2(R[1,0], R[0,0])
        if abs(abs(R[2, 0]) - 1.0) < 1e-6:
            # 奇异情况 (万向锁)
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = 0.0
                yaw = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = 0.0
                yaw = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = -np.arcsin(R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        A = np.degrees(roll)
        B = np.degrees(pitch)
        C = np.degrees(yaw)
        return A, B, C

    # ------------------------------------------------------------
    def visualize_pose(self, image, center_3d, rot, K, scale=0.08):
        """
        在图像上绘制6D位姿坐标轴.
        红=X轴, 绿=Y轴, 蓝=Z轴
        """
        center_2d = K @ np.array(center_3d)
        center_2d = center_2d / center_2d[2]
        cx_2d = int(center_2d[0])
        cy_2d = int(center_2d[1])

        axes = [
            (rot[:, 0], (0, 0, 255)),   # X -> 红色
            (rot[:, 1], (0, 255, 0)),   # Y -> 绿色
            (rot[:, 2], (255, 0, 0)),   # Z -> 蓝色
        ]

        cv2.circle(image, (cx_2d, cy_2d), 5, (124, 0, 255), 3)

        for axis, color in axes:
            end_3d = np.array(center_3d) + scale * axis
            end_2d = K @ end_3d
            end_2d = end_2d / end_2d[2]
            ex_2d = int(end_2d[0])
            ey_2d = int(end_2d[1])

            cv2.arrowedLine(image, (cx_2d, cy_2d), (ex_2d, ey_2d), color, 3, tipLength=0.2)

    # ------------------------------------------------------------
    def run(self):
        print("按 'q' 键退出程序...")
        print("\n6D位姿格式: (X, Y, Z, A, B, C)")
        print("  X, Y, Z = 位置 (米)")
        print("  A, B, C = ZYX欧拉角 (度), 即 Roll, Pitch, Yaw")
        print("坐标轴颜色: 红=X 绿=Y(主方向, 水瓶纵向) 蓝=Z\n")

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # 缓存原始深度帧供锚点映射使用
                try:
                    frames = self.pipeline.wait_for_frames()
                    self._cached_raw_depth = frames.get_depth_frame()
                except Exception:
                    self._cached_raw_depth = None

                # ---- YOLO检测 ----
                detections = self.detect_targets(color_image)

                # 只保留置信度最高的一个检测 (避免重复检测/误检干扰输出)
                if detections:
                    detections = [max(detections, key=lambda d: d[4])]

                # ---- 6D位姿计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    position, rotation, point_cloud = self.compute_6d_pose(det, depth_frame)

                    if position is not None and rotation is not None:
                        # 平滑
                        smooth_pos, smooth_rot = self._pose_smoother.update(position, rotation)
                        if smooth_pos is None:
                            continue

                        # 旋转矩阵 -> ZYX欧拉角 (A=roll, B=pitch, C=yaw)
                        A, B, C = self.rotation_matrix_to_euler_angles(smooth_rot)

                        # 输出
                        print(f"\nTarget 6D Pose: (X, Y, Z, A, B, C) = ({smooth_pos[0]:.3f}, {smooth_pos[1]:.3f}, {smooth_pos[2]:.3f}, {A:.1f}, {B:.1f}, {C:.1f})")
                        print(f"Point cloud points: {point_cloud.shape[0]}")

                        # 绘制检测框
                        x1, y1, x2, y2, conf = det
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 可视化坐标轴 (用深度内参投影到图像)
                        K = np.array([[self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
                                      [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                                      [0, 0, 1]], dtype=float)

                        self.visualize_pose(display_img, smooth_pos, smooth_rot, K, scale=0.08)

                        # 显示位姿文本 (位置 + 欧拉角)
                        text_pos = f'Pos: ({smooth_pos[0]:.2f}, {smooth_pos[1]:.2f}, {smooth_pos[2]:.2f})'
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        text_euler = f'Euler: (A={A:.1f}, B={B:.1f}, C={C:.1f})'
                        cv2.putText(display_img, text_euler, (x1, y1 - 28),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    else:
                        print("Warning: 点云数量不足, 无法计算位姿")

                # 并排显示
                if display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8 6D Pose', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")


if __name__ == "__main__":
    detector = RealSenseYOLO6DPoseDetector(model_name=model_name, conf_threshold=0.5)
    detector.run()