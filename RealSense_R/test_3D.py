# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水瓶3D定位监控
===============================
重构版本: 从 D435i_info.yaml 加载整组相机参数(深度+彩色+外参),
替代 SDK 读取的 NaN 参数, 实现稳定的 depth->color 像素映射.

核心策略:
  - 不依赖 rs.align (该函数使用相机硬件内部已损坏的参数, 结果不可靠)
  - 获取原始深度帧 + 原始彩色帧
  - 所有彩色->深度坐标映射均通过 rs2_project_color_pixel_to_depth_pixel
    手动计算 (使用 YAML 中保存的正确参数)

使用方法:
1. 连接 Intel RealSense 相机
2. 运行: python RealSense_R/test_3D.py
3. 按 'q' 键退出

注意:
- 使用 COCO 数据集预训练的 YOLOv8m 模型, 水瓶类别 ID 为 39
- 3D 位置计算: 锚点深度 + bbox 采样 + 深度离群过滤 + 点云均值
- 相机参数来源: RealSense_R/D435i_info.yaml (另一台 D435i 标定值)
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
#  2. 工具类
# ============================================================

class PositionSmoother:
    """EMA 时域平滑器"""
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


# ============================================================
#  3. 主检测器
# ============================================================

class RealSenseYOLO3DDetector:
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
        self._smoother = PositionSmoother(alpha=SMOOTH_ALPHA)

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
        """
        在深度图坐标 (cx, cy) 周围 half_size 窗口取深度中值.
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
    def get_3d_position(self, bbox, depth_frame):
        """
        计算目标 3D 位置 (点云均值方案).

        步骤:
        1. 锚点深度: bbox 彩色中心 -> 映射到深度图 -> 读深度
        2. 多点采样: 对 bbox 区域内每个彩色像素:
           a) 映射到深度图坐标
           b) 读深度值
           c) Z 值离群过滤 (|d - center_depth| < 0.05)
           d) 反投影到 3D
        3. 统计平均: 全部有效点云取均值作为最终位置
        """
        # ---- Step 1: 获取锚点深度 ----
        anchor = self._get_anchor_depth(bbox, depth_frame)
        if anchor is None:
            return None
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
                return None
            point_cloud = np.array(points_wide)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, wide points: {len(points_wide)}")
        else:
            point_cloud = np.array(points)
            print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)} "
                  f"(bbox {x2_c-x1_c}x{y2_c-y1_c} px)")

        # ---- Step 3: 点云均值作为最终位置 ----
        center = np.mean(point_cloud, axis=0)
        return (center[0], center[1], center[2])

    # ------------------------------------------------------------
    def run(self):
        print("按 'q' 键退出程序...")

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- YOLO 检测 ----
                detections = self.detect_targets(color_image)

                # ---- 3D 位置计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    bbox = det[:4]
                    position = self.get_3d_position(bbox, depth_frame)

                    if position is not None:
                        # 时域平滑
                        smoothed = self._smoother.update(position)
                        print(f"\nTarget 3D position (x, y, z): "
                              f"({smoothed[0]:.3f}, {smoothed[1]:.3f}, {smoothed[2]:.3f}) m")

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

                # 并排显示彩色图 + 原始深度热力图
                if display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow('RealSense YOLOv8m 3D Position (RealSense_R/test_3D)', images)

                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")


# ============================================================
#  4. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseYOLO3DDetector(model_name=model_name, conf_threshold=conf_threshold)
    detector.run()