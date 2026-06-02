# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶6D位姿估计 (RealSense_R 版)
=================================================
重构版本: 从 D435i_info.yaml 加载整组相机参数(深度+彩色+外参),
替代 SDK 读取的 NaN 参数, 实现稳定的 depth->color 像素映射.

核心策略:
  - 不依赖 rs.align (该函数使用相机硬件内部已损坏的参数, 结果不可靠)
  - 获取原始深度帧 + 原始彩色帧
  - 所有彩色->深度坐标映射均通过 rs2_project_color_pixel_to_depth_pixel
    手动计算 (使用 YAML 中保存的正确参数)
  - PCA 计算点云主方向, 构建 6D 位姿 (位置 + 旋转矩阵)

使用方法:
1. 连接 Intel RealSense 相机
2. 运行: python RealSense_R/test_6D.py
3. 按 'q' 键退出

输出格式: (X, Y, Z, A, B, C)
  X, Y, Z = 位置 (m)
  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw
"""

import sys
import os
import yaml

# ---- 将项目根目录加入 sys.path, 以导入 pyrealsense2 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from typing import Optional
from ultralytics import YOLO

# ============================================================
#  0. 参数加载工具函数
# ============================================================

def _load_camera_params(yaml_path):
    """
    从标准 YAML 文件加载相机参数(深度+彩色+外参).

    返回 dict:
      {
          'depth_intr': rs.intrinsics,
          'color_intr': rs.intrinsics,
          'd2c_extr': rs.extrinsics,
          'c2d_extr': rs.extrinsics,
      }
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    def _intr(src):
        i = rs.intrinsics()
        i.width, i.height = data['camera']['resolution']
        i.ppx, i.ppy = src['ppx'], src['ppy']
        i.fx, i.fy = src['fx'], src['fy']
        i.model = getattr(rs.distortion, src['model'])
        i.coeffs = src['distortion']
        return i

    def _extr(src):
        e = rs.extrinsics()
        e.rotation = src['rotation']
        e.translation = src['translation']
        return e

    return {
        'depth_intr': _intr(data['depth']['intrinsics']),
        'color_intr': _intr(data['color']['intrinsics']),
        'd2c_extr': _extr(data['extrinsics']['depth_to_color']),
        'c2d_extr': _extr(data['extrinsics']['color_to_depth']),
    }


# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_name = os.path.join(_project_root, cfg["model"]["name"])
class_id = cfg["model"]["target_class_id"]
conf_threshold = cfg["model"]["conf_threshold"]
depth_min = cfg["depth"]["min"]
depth_max = cfg["depth"]["max"]
DEPTH_Z_THRESHOLD = cfg["depth"]["z_threshold"]
SMOOTH_ALPHA = cfg["smoothing"]["alpha"]
FRAME_W = cfg["resolution"]["width"]
FRAME_H = cfg["resolution"]["height"]
FRAME_FPS = cfg["resolution"]["fps"]

_cam_params = _load_camera_params(os.path.join(os.path.dirname(__file__), 'D435i_info.yaml'))

# ---- 设备选择: config device="gpu" 时优先 GPU, 否则回退 CPU ----
_device_pref = cfg["model"].get("device", "cpu")
if _device_pref == "gpu" and torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print("[INFO] 运行设备: %s" % DEVICE)


# ============================================================
#  1. 平滑工具类
# ============================================================

class PositionSmoother:
    """EMA 时域平滑器 (用于3D位置)"""
    def __init__(self, alpha=SMOOTH_ALPHA):
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
    """6D 位姿平滑器: 位置 EMA + 旋转矩阵指数移动平均 + SVD 正交化"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_pos = None
        self.filtered_rot = None

    def update(self, pos, rot):
        if pos is None or rot is None:
            return None, None
        if self.filtered_pos is None or self.filtered_rot is None:
            self.filtered_pos = list(pos)
            self.filtered_rot = rot.copy()
            return tuple(self.filtered_pos), self.filtered_rot
        for i in range(3):
            self.filtered_pos[i] = (1 - self.alpha) * self.filtered_pos[i] + self.alpha * pos[i]
        self.filtered_rot = (1 - self.alpha) * self.filtered_rot + self.alpha * rot
        # SVD 正交化, 确保是有效的旋转矩阵
        U, _, Vt = np.linalg.svd(self.filtered_rot)
        self.filtered_rot = U @ Vt
        return tuple(self.filtered_pos), self.filtered_rot

    def reset(self):
        self.filtered_pos = None
        self.filtered_rot = None


# ============================================================
#  2. 主检测器
# ============================================================

class RealSenseYOLO6DPoseDetector:
    def __init__(self, model_name=model_name, conf_threshold=conf_threshold):
        self.conf_threshold = conf_threshold
        self.target_class_id = class_id

        # --- RealSense 初始化 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FRAME_FPS)

        self.profile = self.pipeline.start(config)

        # --- 深度缩放系数 ---
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] depth_scale = {self.depth_scale}")

        # 激光功率设适中, 减少近距光斑过曝
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
            print("[INFO] laser_power set to 150")
        except Exception as e:
            print(f"[WARN] laser_power set failed: {e}")

        # ---- 不创建 rs.align 对象 ----
        # rs.align 依赖相机硬件内部参数 (已损坏), 对齐结果不可靠.
        # 改为手动映射: 彩色像素 -> rs2_project_color_pixel_to_depth_pixel -> 原始深度图

        # --- 相机参数: 全部从 D435i_info.yaml 加载 ---
        self.depth_intrinsics = _cam_params['depth_intr']
        self.color_intrinsics = _cam_params['color_intr']
        self.color_to_depth_extrinsics = _cam_params['c2d_extr']
        self.depth_to_color_extrinsics = _cam_params['d2c_extr']

        print(f"[INFO] 深度内参 (YAML): fx={self.depth_intrinsics.fx:.2f}, fy={self.depth_intrinsics.fy:.2f}, "
              f"ppx={self.depth_intrinsics.ppx:.2f}, ppy={self.depth_intrinsics.ppy:.2f}")
        print(f"[INFO] 彩色内参 (YAML): fx={self.color_intrinsics.fx:.2f}, fy={self.color_intrinsics.fy:.2f}, "
              f"ppx={self.color_intrinsics.ppx:.2f}, ppy={self.color_intrinsics.ppy:.2f}")
        print(f"[INFO] Color->Depth 外参 (YAML): T={np.round(self.color_to_depth_extrinsics.translation, 4)} m")

        # --- YOLO ---
        model_path = f"{model_name}.pt"
        if not os.path.exists(model_path):
            model_path = model_name
        self.model = YOLO(model_path).to(DEVICE)
        self.colorizer = rs.colorizer()

        # --- 平滑 ---
        self._pos_smoother = PositionSmoother(alpha=SMOOTH_ALPHA)
        self._pose_smoother = PoseSmoother(alpha=0.3)

        # --- estun_run.py 外部使用的属性 (定义于此以消除 Pylance 报错) ---
        self.last_known_pose: Optional[tuple] = None  # 存储上次稳定的 (pos, euler)
        self._cached_raw_depth = None                  # 缓存原始深度帧

    # ------------------------------------------------------------
    #  工具: 将单个彩色像素映射到深度图坐标
    # ------------------------------------------------------------
    def _color_to_depth_pixel(self, cx, cy, depth_frame):
        """
        将彩色图上的像素 (cx, cy) 映射到深度图坐标系.
        返回 (dx, dy) 或 None (映射失败时)
        """
        try:
            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                depth_frame.get_data(),  # BufData
                self.depth_scale,
                0.1, 6.0,
                self.depth_intrinsics,
                self.color_intrinsics,
                self.color_to_depth_extrinsics,
                self.depth_to_color_extrinsics,
                [cx, cy]
            )
            if depth_pixel and np.isfinite(depth_pixel).all():
                dx, dy = int(depth_pixel[0]), int(depth_pixel[1])
                w = depth_frame.get_width()
                h = depth_frame.get_height()
                if 0 <= dx < w and 0 <= dy < h:
                    return dx, dy
        except Exception:
            pass
        return None

    # ------------------------------------------------------------
    def _depth_median_around(self, cx, cy, depth_frame, half_size=3):
        """在深度图坐标 (cx, cy) 周围 half_size 窗口取深度中值."""
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
    def _get_anchor_depth(self, bbox, depth_frame):
        """
        获取 bbox 中心的锚点深度 (单点).

        步骤:
        1. 获取 bbox 中心彩色坐标 (cx_color, cy_color)
        2. 映射到深度图坐标 (dx, dy)
        3. 在深度图 (dx, dy) 周围取 3x3 中值作为锚点深度

        返回 (cx_color, cy_color, center_depth) 或 None
        """
        cx_color = (int(bbox[0]) + int(bbox[2])) // 2
        cy_color = (int(bbox[1]) + int(bbox[3])) // 2

        # ---- 映射到深度坐标 ----
        depth_pixel = self._color_to_depth_pixel(cx_color, cy_color, depth_frame)
        if depth_pixel is None:
            return None

        dx, dy = depth_pixel

        # ---- 在深度图局部窗口取中值 ----
        depth_val = self._depth_median_around(dx, dy, depth_frame, half_size=3)
        if depth_val is None:
            return None

        return cx_color, cy_color, depth_val

    # ------------------------------------------------------------
    def get_frames(self):
        """
        获取原始彩色帧和原始深度帧.
        不经过 rs.align
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

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
            log_path = os.path.join(_project_root, 'doc', 'yolo_detect.log')
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(log_lines) + '\n')

        return detections

    # ------------------------------------------------------------
    def compute_6d_pose(self, bbox, depth_frame):
        """
        计算水瓶的 6D 位姿.

        两阶段策略:
        1. 锚点深度: bbox 彩色中心 -> 映射到深度图 -> 读深度
        2. 多点采样: 对 bbox 区域内每个彩色像素逐像素映射到深度坐标,
           反投影到 3D, 用 |d - center_depth| 过滤离群点
        3. PCA 主成分分析 -> 旋转矩阵

        返回:
            position: (x, y, z) 重心坐标 (m)
            rotation: 3x3 旋转矩阵
            point_cloud: Nx3 点云
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, depth_frame)
        if anchor is None:
            return None, None, None
        _, _, center_depth = anchor

        # ---- Step 2: bbox 区域逐像素映射 + 采样 ----
        x1_c, y1_c = int(bbox[0]), int(bbox[1])
        x2_c, y2_c = int(bbox[2]), int(bbox[3])

        fx = self.depth_intrinsics.fx
        fy = self.depth_intrinsics.fy
        ppx = self.depth_intrinsics.ppx
        ppy = self.depth_intrinsics.ppy

        points = []
        for cy in range(y1_c, y2_c + 1, 2):
            for cx in range(x1_c, x2_c + 1, 2):
                # 彩色像素 -> 映射到深度图坐标
                depth_pixel = self._color_to_depth_pixel(cx, cy, depth_frame)
                if depth_pixel is None:
                    continue
                dx, dy = depth_pixel

                # 读深度
                d = depth_frame.get_distance(dx, dy)
                if not (depth_min < d < depth_max):
                    continue
                # Z 值离群过滤
                if abs(d - center_depth) > DEPTH_Z_THRESHOLD:
                    continue

                # 反投影 (使用深度内参)
                depth_x = (dx - ppx) * d / fx
                depth_y = (dy - ppy) * d / fy
                points.append([depth_x, depth_y, d])

        if len(points) < 10:
            # 点太少, 不限制 Z 值再试一次
            points_wide = []
            for cy in range(y1_c, y2_c + 1, 2):
                for cx in range(x1_c, x2_c + 1, 2):
                    depth_pixel = self._color_to_depth_pixel(cx, cy, depth_frame)
                    if depth_pixel is None:
                        continue
                    dx, dy = depth_pixel
                    d = depth_frame.get_distance(dx, dy)
                    if depth_min < d < depth_max:
                        depth_x = (dx - ppx) * d / fx
                        depth_y = (dy - ppy) * d / fy
                        points_wide.append([depth_x, depth_y, d])
            if len(points_wide) < 10:
                return None, None, None
            point_cloud = np.array(points_wide)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, no-filter points: {len(points_wide)}")
        else:
            point_cloud = np.array(points)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)} "
                  f"(bbox {x2_c - x1_c}x{y2_c - y1_c} px)")

        # ---- Step 3: PCA 位姿 ----
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主方向 (最大特征值) -> Y轴 (绿色), 水瓶竖直时为主轴
        target_y_main = np.real(eig_vec[:, 0])
        # 最小方差方向 -> Z轴 (蓝色), 指向相机
        target_z = np.real(eig_vec[:, 2])

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

    # ------------------------------------------------------------
    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """
        从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw).
        返回角度值 (deg).
        """
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

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    # ------------------------------------------------------------
    def visualize_pose(self, image, center_3d, rot, K, scale=0.08):
        """
        在图像上绘制 6D 位姿坐标轴.
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
        print("  X, Y, Z = 位置 (m)")
        print("  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw")
        print("坐标轴颜色: 红=X 绿=Y(主方向, 水瓶纵向) 蓝=Z\n")

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- YOLO 检测 ----
                detections = self.detect_targets(color_image)

                # 只保留置信度最高的一个检测 (避免重复检测/误检干扰输出)
                if detections:
                    detections = [max(detections, key=lambda d: d[4])]

                # ---- 6D 位姿计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    position, rotation, point_cloud = self.compute_6d_pose(det, depth_frame)

                    if position is not None and rotation is not None:
                        # 平滑
                        smooth_pos, smooth_rot = self._pose_smoother.update(position, rotation)
                        if smooth_pos is None:
                            continue

                        # 旋转矩阵 -> ZYX 欧拉角 (A=roll, B=pitch, C=yaw)
                        A, B, C = self.rotation_matrix_to_euler_angles(smooth_rot)

                        # 输出
                        print(f"\nTarget 6D Pose: (X, Y, Z, A, B, C) = "
                              f"({smooth_pos[0]:.3f}, {smooth_pos[1]:.3f}, {smooth_pos[2]:.3f}, "
                              f"{A:.1f}, {B:.1f}, {C:.1f})")
                        assert point_cloud is not None
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
                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8 6D Pose (RealSense_R/test_6D)', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseYOLO6DPoseDetector(model_name=model_name, conf_threshold=conf_threshold)
    detector.run()