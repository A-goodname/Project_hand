# -*- coding: utf-8 -*-
"""
Orbbec YOLOv8m 水瓶3D定位监控
==============================
使用 Orbbec Gemini 215 相机 + pyorbbecsdk2

使用方法：
1. 连接 Orbbec Gemini 215 相机
2. 运行：python Orbbec/test_3D.py
3. 按 'q' 键退出

注意：
- 使用COCO数据集预训练的YOLOv8m模型，水瓶类别ID为39
- 3D位置计算：锚点深度 + bbox采样 + 深度离群过滤 + 点云均值
- 使用 AlignFilter（软对齐），将深度图对齐到彩色坐标系
"""

import yaml
import os

# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_name = cfg["model"]["path"]
class_id = cfg["model"]["class_id"]
conf_threshold = cfg["model"]["conf_threshold"]
depth_min = cfg["depth"]["min"]
depth_max = cfg["depth"]["max"]
DEPTH_Z_THRESHOLD = cfg["depth"]["z_threshold"]
COLOR_W = cfg["stream"]["color_width"]
COLOR_H = cfg["stream"]["color_height"]
DEPTH_W = cfg["stream"]["depth_width"]
DEPTH_H = cfg["stream"]["depth_height"]
COLOR_FPS = cfg["stream"]["color_fps"]
DEPTH_FPS = cfg["stream"]["depth_fps"]
SMOOTH_ALPHA = cfg["smoothing"]["alpha"]

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---- GPU/CPU 设备选择（优先读取 config.yaml 的 device 字段） ----
device_cfg = cfg["model"].get("device", "gpu")
if device_cfg == "gpu" and torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"[INFO] Running on device: {DEVICE} (config: {device_cfg})")

from pyorbbecsdk import (
    Pipeline, Config, OBSensorType, OBFormat, OBAlignMode,
    AlignFilter, OBStreamType, OBFrameAggregateOutputMode,
)
from utils import frame_to_bgr_image


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


class OrbbecYOLO3DDetector:
    def __init__(self, model_name='../model/yolov8m', conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.target_class_id = class_id

        # --- YOLO 模型加载 ---
        model_path = f"{model_name}.pt"
        if not os.path.exists(model_path):
            model_path = model_name
        self.model = YOLO(model_path).to(DEVICE)

        # --- 内参占位（在 get_frames 中首次填充） ---
        self.color_intrinsics = None
        self.depth_intrinsics = None

        # --- Orbbec Pipeline ---
        self.pipeline = Pipeline()
        try:
            self.pipeline.enable_frame_sync()
        except Exception as e:
            print(f"[WARN] frame sync not available: {e}")

        config = Config()

        # 使用 get_stream_profile_list 自动获取可用的 RGB profile（hw_d2c_align.py 样式）
        # 但 Gemini 215 上 get_stream_profile_list 在 start 前可能返回空，
        # 所以我们手动使用已知支持的配置
        config.enable_video_stream(OBSensorType.COLOR_SENSOR, COLOR_W, COLOR_H, COLOR_FPS, OBFormat.RGB)
        config.enable_video_stream(OBSensorType.DEPTH_SENSOR, DEPTH_W, DEPTH_H, DEPTH_FPS, OBFormat.Y16)

        # 保证 frameset 必须同时包含 Color + Depth
        config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)

        self.pipeline.start(config)

        # --- 使用自动曝光（默认开启） ---
        print("[INFO] exposure: auto")

        # --- 对齐滤波器：将深度图 warp 到彩色坐标系 ---
        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)

        # --- 平滑 ---
        self._smoother = PositionSmoother(alpha=SMOOTH_ALPHA)

        print(f"[INFO] model: {model_name}.pt | conf: {conf_threshold} | class_id: {class_id}")
        print(f"[INFO] stream: color {COLOR_W}x{COLOR_H}@{COLOR_FPS}, depth {DEPTH_W}x{DEPTH_H}@{DEPTH_FPS}")
        print(f"[INFO] align: AlignFilter(COLOR_STREAM)")

    # ------------------------------------------------------------
    def _init_intrinsics(self):
        """从 pipeline 获取相机内参"""
        if self.color_intrinsics is not None:
            return
        cp = self.pipeline.get_camera_param()
        self.depth_intrinsics = cp.depth_intrinsic
        self.color_intrinsics = cp.rgb_intrinsic
        print(f"[INFO] color intrinsics: fx={self.color_intrinsics.fx:.4f}, "
              f"fy={self.color_intrinsics.fy:.4f}, "
              f"cx={self.color_intrinsics.cx:.4f}, cy={self.color_intrinsics.cy:.4f}")
        print(f"[INFO] depth intrinsics: fx={self.depth_intrinsics.fx:.4f}, "
              f"fy={self.depth_intrinsics.fy:.4f}, "
              f"cx={self.depth_intrinsics.cx:.4f}, cy={self.depth_intrinsics.cy:.4f}")

    # ------------------------------------------------------------
    def get_frames(self):
        """获取彩色帧与对齐后深度帧"""
        frames = self.pipeline.wait_for_frames(2000)
        if frames is None:
            return None, None, None

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None

        # ---- 初始化内参 ----
        self._init_intrinsics()

        # ---- 软对齐（AlignFilter） ----
        frames = self.align_filter.process(frames)
        if frames is None:
            return None, None, None
        frames = frames.as_frame_set()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None

        # ---- 彩色帧转 BGR 图像 ----
        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            return None, None, None

        # ---- 对齐后的深度数据 ----
        # 对齐后深度图分辨率 = 彩色图分辨率（1280x720）
        depth_data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            (depth_frame.get_height(), depth_frame.get_width()))
        # 原始单位为毫米，转为米
        depth_data = depth_data.astype(np.float32) * depth_frame.get_depth_scale()
        # 实际 Orbbec depth_scale 通常为 1.0（毫米），所以 * 0.001 转米
        depth_data = depth_data * 0.001

        # ---- 深度伪彩图 ----
        depth_valid = (depth_data > depth_min).astype(np.uint8) * 255
        depth_norm = cv2.normalize(depth_data, np.zeros_like(depth_data), 0, 255, cv2.NORM_MINMAX)
        depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
        depth_colormap[depth_valid == 0] = 0

        return color_image, depth_data, depth_colormap

    # ------------------------------------------------------------
    def detect_targets(self, color_image):
        """YOLO 检测"""
        results = self.model(color_image, conf=self.conf_threshold, classes=[self.target_class_id])

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if cls_id == self.target_class_id:
                    detections.append([*box, conf, cls_id])

        return detections

    # ------------------------------------------------------------
    def get_3d_position(self, bbox, depth_data):
        """
        计算目标3D位置（点云均值方案）
        使用对齐深度图 + rgb_intrinsics（对齐后深度在彩色坐标系下）
        """
        # ---- Step 1: 获取锚点深度 ----
        cx = int((bbox[0] + bbox[2]) / 2)
        cy = int((bbox[1] + bbox[3]) / 2)
        cx = np.clip(cx, 0, depth_data.shape[1] - 1)
        cy = np.clip(cy, 0, depth_data.shape[0] - 1)

        half = 3
        x_s = max(0, cx - half)
        y_s = max(0, cy - half)
        x_e = min(depth_data.shape[1], cx + half + 1)
        y_e = min(depth_data.shape[0], cy + half + 1)

        patch = depth_data[y_s:y_e, x_s:x_e]
        valid = patch[(patch > depth_min) & (patch < depth_max)]
        if len(valid) < 3:
            # debug: 输出锚点和 patch 信息
            print(f"[DEBUG] anchor=({cx},{cy}) patch_min={patch.min():.4f} patch_max={patch.max():.4f} "
                  f"valid_cnt={len(valid)} depth_min={depth_min} depth_max={depth_max}")
            return None
        center_depth = float(np.median(valid))

        # ---- Step 2: bbox 区域采样 + 深度过滤 + 反投影 ----
        x1 = max(0, min(int(bbox[0]), depth_data.shape[1] - 1))
        y1 = max(0, min(int(bbox[1]), depth_data.shape[0] - 1))
        x2 = max(0, min(int(bbox[2]), depth_data.shape[1] - 1))
        y2 = max(0, min(int(bbox[3]), depth_data.shape[0] - 1))

        assert self.color_intrinsics is not None
        fx = self.color_intrinsics.fx
        fy = self.color_intrinsics.fy
        ppx = self.color_intrinsics.cx
        ppy = self.color_intrinsics.cy

        points = []
        for vy in range(y1, y2 + 1, 2):
            for vx in range(x1, x2 + 1, 2):
                d = depth_data[vy, vx]
                if not (depth_min < d < depth_max):
                    continue
                if abs(d - center_depth) > DEPTH_Z_THRESHOLD:
                    continue
                depth_x = (vx - ppx) * d / fx
                depth_y = (vy - ppy) * d / fy
                points.append([depth_x, depth_y, d])

        if len(points) < 10:
            points_wide = []
            for vy in range(y1, y2 + 1, 2):
                for vx in range(x1, x2 + 1, 2):
                    d = depth_data[vy, vx]
                    if depth_min < d < depth_max:
                        depth_x = (vx - ppx) * d / fx
                        depth_y = (vy - ppy) * d / fy
                        points_wide.append([depth_x, depth_y, d])
            if len(points_wide) < 10:
                return None
            point_cloud = np.array(points_wide)
        else:
            point_cloud = np.array(points)

        center = np.mean(point_cloud, axis=0)
        return (center[0], center[1], center[2])

    # ------------------------------------------------------------
    def run(self):
        print("按 'q' 键退出程序...")

        # ---- 创建窗口（只执行一次） ----
        WIN_NAME = 'Orbbec YOLOv8m 3D Position'
        display_w, display_h = 640, 360  # 显示缩放尺寸（原始为 1280x720 / 1280x800）
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, display_w * 2, display_h)

        try:
            while True:
                # 检查窗口是否被手动关闭（点 X）
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                color_image, depth_data, depth_colormap = self.get_frames()
                if color_image is None or depth_data is None:
                    continue

                # ---- YOLO检测 ----
                detections = self.detect_targets(color_image)

                # ---- 3D位置计算与可视化 ----
                display_img = color_image.copy()

                for det in detections:
                    bbox = det[:4]

                    # ---- 始终画检测框和中心点（不论深度是否有效） ----
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

                    # ---- 计算 3D 位置（深度有效时才标注文字） ----
                    position = self.get_3d_position(bbox, depth_data)

                    if position is not None:
                        smoothed = self._smoother.update(position)
                        assert smoothed is not None
                        print(f"\nTarget 3D position (x, y, z): ({smoothed[0]:.3f}, {smoothed[1]:.3f}, {smoothed[2]:.3f}) m")

                        text_pos = f'({smoothed[0]:.2f}, {smoothed[1]:.2f}, {smoothed[2]:.2f})'
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # ---- 调整显示尺寸 ----
                img_rgb = cv2.resize(display_img, (display_w, display_h))
                if depth_colormap is not None:
                    img_depth = cv2.resize(depth_colormap, (display_w, display_h))
                    images = np.hstack((img_rgb, img_depth))
                else:
                    images = img_rgb

                cv2.imshow(WIN_NAME, images)

                key = cv2.waitKey(25) & 0xFF
                if key == ord('q') or key == ord('Q') or key == 27:  # q / Q / ESC
                    break

        finally:
            # 先销毁 OpenCV 窗口并刷新事件队列，再停止 pipeline，避免死锁
            cv2.destroyAllWindows()
            for _ in range(3):
                cv2.waitKey(1)
            self.pipeline.stop()
            print("程序已退出")


if __name__ == "__main__":
    detector = OrbbecYOLO3DDetector(model_name=model_name, conf_threshold=conf_threshold)
    detector.run()
