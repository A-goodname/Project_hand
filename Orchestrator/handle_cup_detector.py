# -*- coding: utf-8 -*-
"""
Orchestrator Camera A Handle+Cup 轻量检测器
=============================================
独立于 RealSense_R 体系, 不继承 BaseRealSenseDetector.

Handle 检测: YOLO kettle bbox → 白色把手 HSV+RGB 分割 → 红色胶带(R通道优势) → 3D 位置
Cup 检测:   YOLO cup bbox → bbox 采样 → 深度反投影 → 3D 位置

ABC 固定:
  - Handle:    [-90, 89, -90]
  - Cup back:  [-90, 89, -110]
  - Cup left:  [-90, 89, -90]

参考: reference/机械臂_同事运行包_20260713/visionpro.py + detect_red_tag_to_robot.py
"""

import os
import sys
import time
from typing import Optional

# 项目根目录必须在 import pyrealsense2 之前加入 sys.path,
# 因为 pyrealsense2 是项目根目录下的 .pyd 文件 (非 pip 安装).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

from RealSense_R.realsense_shared import (
    load_camera_params,
    color_to_depth_pixel,
    depth_median_around,
)


# ============================================================
#  固定 ABC
# ============================================================
HANDLE_EULER = (-90.0, 89.0, -90.0)
CUP_EULER_BACK = (-90.0, 89.0, -110.0)
CUP_EULER_LEFT = (-90.0, 89.0, -90.0)


# ============================================================
#  1. 白色把手 ROI 定位 (移植自 visionpro.py)
# ============================================================

def _clamp_roi(roi, width, height):
    x0, y0, x1, y1 = roi
    return (
        max(0, min(width - 1, int(x0))),
        max(0, min(height - 1, int(y0))),
        max(0, min(width - 1, int(x1))),
        max(0, min(height - 1, int(y1))),
    )


def _mask_to_roi(white_mask, candidate_mask_out, cfg_white):
    """从白色 mask 提取最佳 ROI 候选. 返回 (x0,y0,x1,y1) 或 None."""
    # 形态学
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)

    contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(cfg_white.get("min_component_area", 80.0))
    max_area = float(cfg_white.get("max_component_area", 18000.0))
    min_w = int(cfg_white.get("min_width", 18))
    min_h = int(cfg_white.get("min_height", 35))
    max_w = int(cfg_white.get("max_width", 180))
    max_h = int(cfg_white.get("max_height", 210))

    h, w = white_mask.shape[:2]
    candidate_mask = np.zeros_like(white_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(cnt)
        if bw < min_w or bh < min_h:
            continue
        if bw > max_w or bh > max_h:
            continue
        if bw > w * 0.95 and bh > h * 0.85:
            continue
        cv2.drawContours(candidate_mask, [cnt], -1, 255, thickness=-1)

    if cv2.countNonZero(candidate_mask) == 0:
        return None

    # dilate 合并近邻碎片
    join_px = max(3, int(cfg_white.get("join_px", 25)))
    joined = cv2.dilate(candidate_mask, np.ones((join_px, join_px), np.uint8), iterations=1)
    groups, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_px = int(cfg_white.get("min_pixels", 220))
    candidates = []
    for grp in groups:
        gx, gy, gw, gh = cv2.boundingRect(grp)
        if gw < min_w or gh < min_h:
            continue
        if gw > max_w or gh > max_h:
            continue
        white_count = cv2.countNonZero(candidate_mask[gy:gy + gh, gx:gx + gw])
        if white_count < min_px:
            continue
        # right_score 和 height_score 需要外部传入的 kettle bbox 信息,
        # 这里简化: 按 white_count 排序, 调用方再微调
        candidates.append((white_count, gx, gy, gw, gh))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    x, y, rw, rh = candidates[0][1:]
    if candidate_mask_out is not None:
        candidate_mask_out[:] = candidate_mask
    return (x, y, x + rw, y + rh)


def find_white_handle_roi(color_image, kettle_bbox, cfg_white):
    """在 kettle bbox 右侧区域检测白色把手, 返回 ROI 或 None.

    两遍检测:
      Pass 1: 严格阈值 (正常光照)
      Pass 2: 宽松阈值 (暗光/关灯), 仅在第一遍无结果时触发.

    移植自 visionpro.py:VisionServer.find_white_handle_shape_roi().
    """
    image_h, image_w = color_image.shape[:2]
    kx0, ky0, kx1, ky1 = _clamp_roi(kettle_bbox, image_w, image_h)
    kw = kx1 - kx0
    kh = ky1 - ky0
    if kw <= 0 or kh <= 0:
        return None

    # 搜索区域: kettle bbox 右侧
    search_left_frac = float(cfg_white.get("search_left_frac", 0.68))
    sx0 = kx0 + int(kw * search_left_frac)
    sx0, sy0, sx1, sy1 = _clamp_roi((sx0, ky0, kx1, ky1), image_w, image_h)
    if sx1 <= sx0 or sy1 <= sy0:
        return None

    crop = color_image[sy0:sy1 + 1, sx0:sx1 + 1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    b_ch, g_ch, r_ch = cv2.split(crop)
    rgb_min = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    rgb_max = np.maximum(np.maximum(r_ch, g_ch), b_ch)

    max_sat = int(cfg_white.get("max_saturation", 115))
    max_spread = int(cfg_white.get("max_rgb_spread", 150))

    # 候选 mask (被 _mask_to_roi 填充)
    candidate_mask = None

    # ---- Two-pass: strict → relaxed ----
    for pass_idx, (min_val, min_rgb_val) in enumerate([
        (int(cfg_white.get("min_value", 110)), int(cfg_white.get("min_rgb", 120))),   # Pass 1: strict
        (45, 50),                                                                       # Pass 2: relaxed (暗光)
    ]):
        white_mask = (
            (s_ch < max_sat)
            & (v_ch > min_val)
            & (rgb_min > min_rgb_val)
            & ((rgb_max - rgb_min) < max_spread)
        ).astype(np.uint8) * 255

        roi_raw = _mask_to_roi(white_mask, candidate_mask, cfg_white)
        if roi_raw is not None:
            if pass_idx == 1:
                print(f"[WhiteHandle] Pass 2 (暗光) 命中: min_val={min_val} min_rgb={min_rgb_val}")
            x0, y0, x1, y1 = roi_raw
            full_x0, full_y0 = sx0 + x0, sy0 + y0
            full_x1, full_y1 = sx0 + x1, sy0 + y1
            margin_x = int(cfg_white.get("roi_margin_x", 14))
            margin_y = int(cfg_white.get("roi_margin_y", 10))
            return _clamp_roi(
                (full_x0 - margin_x, full_y0 - margin_y,
                 full_x1 + margin_x, full_y1 + margin_y),
                image_w, image_h,
            )

    return None


# ============================================================
#  2. 红色胶带检测 (移植自 visionpro.py, 含 R 通道优势)
# ============================================================

def find_handle_red_tag(color_image, cfg_red, roi=None):
    """在 ROI 内检测红色胶带, 返回 (contour, mask) 或 (None, mask).

    基于 RealSense_R/detect_red_tape 简化: HSV 双阈值 → 开运算去噪 → 取最大连通域 → min_area 过滤.
    移除了 visionpro.py 的 R 通道优势检查、评分系统等复杂逻辑.
    """
    image_h, image_w = color_image.shape[:2]
    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    low_h_max = int(cfg_red.get("low_hue_max", 10))
    high_h_min = int(cfg_red.get("high_hue_min", 170))
    min_s = int(cfg_red.get("min_saturation", 50))
    min_v = int(cfg_red.get("min_value", 50))
    min_area = float(cfg_red.get("min_area", 20.0))

    mask1 = cv2.inRange(hsv, (0, min_s, min_v), (low_h_max, 255, 255))
    mask2 = cv2.inRange(hsv, (high_h_min, min_s, min_v), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)

    # 开运算去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # ROI 约束
    if roi is not None:
        x0, y0, x1, y1 = roi
        roi_mask = np.zeros_like(mask)
        roi_mask[y0:y1 + 1, x0:x1 + 1] = 255
        mask = cv2.bitwise_and(mask, roi_mask)

    # 取最大连通域
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None, mask
    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    largest_idx, largest_area = max(areas, key=lambda x: x[1])
    if largest_area < min_area:
        return None, mask
    mask = (labels == largest_idx).astype(np.uint8) * 255

    # 从 mask 提取轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, mask
    return contours[0], mask


# ============================================================
#  3. 红色胶带轮廓 → 3D (移植自 detect_red_tag_to_robot.py)
# ============================================================

def deproject_red_tag(depth_frame, color_intrinsics, depth_intrinsics,
                      color_to_depth_extrinsics, depth_to_color_extrinsics,
                      depth_scale, contour, depth_min=0.15, depth_max=1.5):
    """红色胶带轮廓内逐像素采样 → 中值 3D 位置.

    移植自 detect_red_tag_to_robot.py:deproject_roi(), 适配本项目的
    color_to_depth_pixel API (不使用 rs2_deproject_pixel_to_point).
    """
    x, y, w, h = cv2.boundingRect(contour)
    frame_w = depth_frame.get_width()
    frame_h = depth_frame.get_height()
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(frame_w, x + w), min(frame_h, y + h)
    if x1 <= x0 or y1 <= y0:
        return None

    # 轮廓 mask
    mask = np.zeros((frame_h, frame_w), dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, thickness=-1)

    points = []
    for yy in range(y0, y1 + 1, 2):  # stride=2
        for xx in range(x0, x1 + 1, 2):
            if mask[yy, xx] == 0:
                continue
            dp = color_to_depth_pixel(
                xx, yy, depth_frame, depth_scale,
                depth_intrinsics, color_intrinsics,
                color_to_depth_extrinsics, depth_to_color_extrinsics,
            )
            if dp is None:
                continue
            dx, dy = dp
            d = depth_frame.get_distance(dx, dy)
            if not (depth_min < d < depth_max):
                continue
            # 彩色内参反投影
            px = (xx - color_intrinsics.ppx) * d / color_intrinsics.fx
            py = (yy - color_intrinsics.ppy) * d / color_intrinsics.fy
            points.append([px, py, d])

    if len(points) < 10:
        return None

    pts = np.array(points)
    return {
        "pos_cam": tuple(np.median(pts, axis=0)),
        "valid_count": len(points),
    }


# ============================================================
#  4. HandleCupDetector
# ============================================================

class HandleCupDetector:
    """Camera A Handle+Cup 轻量检测器."""

    def __init__(self, orch_config: dict, serial: str = "", device: str = "cuda"):
        """
        Args:
            orch_config: Orchestrator/config.yaml 加载后的字典 (全部配置).
            serial:      Camera A 序列号 (D435i_info.yaml 中的 serial).
            device:      "cuda" | "cpu".
        """
        _c = orch_config
        self._serial = serial
        self._device = device

        # ---- 检测参数 ----
        self._cfg_white = _c.get("white_handle", {})
        self._cfg_red = _c.get("red_handle", {})
        self._depth_min = float(_c.get("depth_min", 0.15))
        self._depth_max = float(_c.get("depth_max", 1.5))
        self._depth_z_thr = float(_c.get("depth_z_threshold", 0.05))
        self._cup_handle_side = _c.get("cup_handle_side", "back")
        self._cup_rim_radius = float(_c.get("cup_rim_radius", 0.035))
        self._kettle_conf = float(_c.get("kettle_conf", 0.7))
        self._cup_conf = float(_c.get("cup_conf", 0.5))

        # ---- 模型路径 ----
        self._kettle_model_path = os.path.join(
            _project_root, _c.get("kettle_model", "model/kettle_yolov8n_finetune.pt"))
        _cup_m = _c.get("cup_model", "model/yolov8m")
        self._cup_model_path = os.path.join(_project_root, _cup_m)
        if not self._cup_model_path.endswith(".pt"):
            self._cup_model_path += ".pt"

        # ---- 固定 ABC ----
        _h = _c.get("handle_euler")
        self._handle_euler = tuple(_h) if _h else HANDLE_EULER
        _cb = _c.get("cup_euler_back")
        _cl = _c.get("cup_euler_left")
        self._cup_euler_back = tuple(_cb) if _cb else CUP_EULER_BACK
        self._cup_euler_left = tuple(_cl) if _cl else CUP_EULER_LEFT
        self._cup_euler = self._cup_euler_left if self._cup_handle_side == "left" else self._cup_euler_back

        # ---- RealSense pipeline ----
        self._pipeline = None
        self._colorizer = None
        self._depth_scale = 0.0
        self._color_intr = None
        self._depth_intr = None
        self._c2d_extr = None
        self._d2c_extr = None
        self._init_pipeline()

        # ---- YOLO 模型 ----
        self._kettle_model: Optional[YOLO] = None
        self._cup_model: Optional[YOLO] = None
        self._init_models()

        # ---- 状态缓存 ----
        self._last_kettle_bbox = None
        self._last_kettle_time = 0.0
        self._kettle_hold_sec = 1.0  # YOLO 漏检缓存时间
        self._last_white_roi = None
        self._last_red_contour = None
        self._last_red_bbox = None
        self._last_cup_bbox = None
        self._last_handle_pos = None   # (x, y, z) or None
        self._last_cup_pos = None      # (x, y, z) or None

        # ---- 供 state_machine 写入 (类型声明防 Pylance 报错) ----
        self.last_known_pose: Optional[tuple] = None      # (pos_ee, euler_ee) or None
        self.last_known_cup_pose: Optional[tuple] = None  # (pos_cam, euler_cam) or None

    # ------------------------------------------------------------

    def _init_pipeline(self):
        """打开 Camera A pipeline (color + depth)."""
        yaml_path = os.path.join(
            _project_root, "RealSense_R", "D435i_info.yaml"
        )
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
            print(f"[HandleCup] 绑定 serial: {self._serial}")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self._pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
        except Exception:
            pass

        params = load_camera_params(yaml_path)
        self._color_intr = params["color_intr"]
        self._depth_intr = params["depth_intr"]
        self._c2d_extr = params["c2d_extr"]
        self._d2c_extr = params["d2c_extr"]
        self._colorizer = rs.colorizer()

        print(f"[HandleCup] depth_scale={self._depth_scale:.6f}")
        print(f"[HandleCup] color_intr: fx={self._color_intr.fx:.1f} fy={self._color_intr.fy:.1f}")

    # ------------------------------------------------------------

    def _init_models(self):
        """加载两个独立 YOLO 模型."""
        print(f"[HandleCup] Kettle: {self._kettle_model_path}")
        self._kettle_model = YOLO(self._kettle_model_path)
        self._kettle_model.to(self._device)

        print(f"[HandleCup] Cup: {self._cup_model_path}")
        self._cup_model = YOLO(self._cup_model_path)
        self._cup_model.to(self._device)

    # ------------------------------------------------------------
    #  帧获取
    # ------------------------------------------------------------

    def get_frames(self):
        """返回 (color, depth_image, depth_colormap, depth_frame)."""
        if self._pipeline is None:
            return None, None, None, None
        assert self._colorizer is not None  # type guard for Pylance
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None, None
        color = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_cmap = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
        return color, depth_img, depth_cmap, depth_frame

    # ------------------------------------------------------------
    #  Kettle YOLO
    # ------------------------------------------------------------

    def _detect_kettle(self, color_image):
        """YOLO 检测水壶, 带缓存."""
        assert self._kettle_model is not None  # type guard for Pylance
        results = self._kettle_model(
            color_image, conf=self._kettle_conf, verbose=False
        )
        best = None
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if best is None or conf > best["conf"]:
                    best = {
                        "bbox": [int(b) for b in box],
                        "conf": float(conf),
                    }
        if best is not None:
            self._last_kettle_bbox = best["bbox"]
            self._last_kettle_time = time.time()
            return best["bbox"]
        # 缓存回退
        if self._last_kettle_bbox is not None:
            if time.time() - self._last_kettle_time < self._kettle_hold_sec:
                return self._last_kettle_bbox
        return None

    # ------------------------------------------------------------
    #  Cup YOLO
    # ------------------------------------------------------------

    def _detect_cup_bbox(self, color_image):
        """YOLO 检测水杯 (class=41)."""
        assert self._cup_model is not None  # type guard for Pylance
        results = self._cup_model(
            color_image, classes=[41], conf=self._cup_conf, verbose=False
        )
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            if float(confs[best_idx]) >= self._cup_conf:
                xyxy = boxes.xyxy.cpu().numpy()[best_idx]
                return [int(x) for x in xyxy]
        return None

    # ------------------------------------------------------------
    #  Cup 3D 位置
    # ------------------------------------------------------------

    def _compute_cup_3d(self, color_image, depth_frame, cup_bbox):
        """bbox 内采样 → 3D 点云 → 中值位置.

        移植自 test_cup_6D.py compute_cup_6d_pose, 含 left/right 模式 bbox 裁剪.
        """
        assert self._color_intr is not None  # type guard for Pylance
        x1, y1, x2, y2 = cup_bbox

        # 锚点深度
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dp = color_to_depth_pixel(
            cx, cy, depth_frame, self._depth_scale,
            self._depth_intr, self._color_intr,
            self._c2d_extr, self._d2c_extr,
        )
        if dp is None:
            return None
        anchor_d = depth_median_around(
            dp[0], dp[1], depth_frame,
            depth_min=self._depth_min, depth_max=self._depth_max,
        )
        if anchor_d is None:
            return None

        # ---- 根据 handle_side 裁剪 bbox (排除把手侧的背景点) ----
        # 移植自 test_cup_6D.py:159-171
        _x1_orig, _x2_orig = x1, x2
        if self._cup_handle_side in ("left", "right"):
            rim_px = max(1, int(self._cup_rim_radius * self._color_intr.fx / anchor_d))
            if self._cup_handle_side == "left":
                # 把手在左侧, 从右边界向左取 2 倍杯口半径
                x1 = max(int(x2) - 2 * rim_px, int(x1))
            elif self._cup_handle_side == "right":
                # 把手在右侧, 从左边界向右取 2 倍杯口半径
                x2 = min(int(x1) + 2 * rim_px, int(x2))
            if x2 - x1 < 4:
                x1, x2 = _x1_orig, _x2_orig  # 回退到原始 bbox
            # 更新锚点深度为裁剪后 bbox 中心
            _cx2 = (x1 + x2) // 2
            _dp2 = color_to_depth_pixel(
                _cx2, cy, depth_frame, self._depth_scale,
                self._depth_intr, self._color_intr,
                self._c2d_extr, self._d2c_extr,
            )
            if _dp2 is not None:
                _ad2 = depth_median_around(
                    _dp2[0], _dp2[1], depth_frame,
                    depth_min=self._depth_min, depth_max=self._depth_max,
                )
                if _ad2 is not None:
                    anchor_d = _ad2
            # 存储裁剪后 bbox 供可视化
            self._last_cup_bbox = [x1, y1, x2, y2]

        # 步长采样
        z_thr = self._depth_z_thr
        points = []
        for cy_i in range(y1, y2 + 1, 2):
            for cx_i in range(x1, x2 + 1, 2):
                dp2 = color_to_depth_pixel(
                    cx_i, cy_i, depth_frame, self._depth_scale,
                    self._depth_intr, self._color_intr,
                    self._c2d_extr, self._d2c_extr,
                )
                if dp2 is None:
                    continue
                dx2, dy2 = dp2
                d = depth_frame.get_distance(dx2, dy2)
                if not (self._depth_min < d < self._depth_max):
                    continue
                if abs(d - anchor_d) > z_thr:
                    continue
                px = (cx_i - self._color_intr.ppx) * d / self._color_intr.fx
                py = (cy_i - self._color_intr.ppy) * d / self._color_intr.fy
                points.append([px, py, d])

        if len(points) < 10:
            # 放宽 Z 过滤
            points2 = []
            for cy_i in range(y1, y2 + 1, 2):
                for cx_i in range(x1, x2 + 1, 2):
                    dp3 = color_to_depth_pixel(
                        cx_i, cy_i, depth_frame, self._depth_scale,
                        self._depth_intr, self._color_intr,
                        self._c2d_extr, self._d2c_extr,
                    )
                    if dp3 is None:
                        continue
                    dx3, dy3 = dp3
                    d = depth_frame.get_distance(dx3, dy3)
                    if not (self._depth_min < d < self._depth_max):
                        continue
                    px = (cx_i - self._color_intr.ppx) * d / self._color_intr.fx
                    py = (cy_i - self._color_intr.ppy) * d / self._color_intr.fy
                    points2.append([px, py, d])
            if len(points2) < 10:
                return None
            points = points2

        pts = np.array(points)
        return tuple(np.median(pts, axis=0))

    # ------------------------------------------------------------
    #  主检测接口
    # ------------------------------------------------------------

    def detect(self, color_image, depth_frame):
        """同步检测 Handle + Cup.

        Returns:
            dict with keys "handle", "cup". 每个子 dict:
              {"pos_cam": (x,y,z) or None,
               "euler_cam": (A,B,C),
               "bbox": [...] or None,
               "red_center": (cx,cy) or None,   # handle only
               }
        """
        h, w = color_image.shape[:2]
        handle_result = {
            "pos_cam": None, "euler_cam": self._handle_euler,
            "kettle_bbox": None, "red_center": None,
            "white_roi": None, "red_contour": None,
        }
        cup_result = {
            "pos_cam": None, "euler_cam": self._cup_euler,
            "cup_bbox": None,
        }

        # === Handle 检测 ===
        kettle_bbox = self._detect_kettle(color_image)
        handle_result["kettle_bbox"] = kettle_bbox

        if kettle_bbox is not None:
            # ---- 新策略: 先在 kettle 右侧直接搜红色胶带 ----
            # 红色在 HSV 空间比白色稳定 (光照只影响 V, 不影响 Hue),
            # 无需白色把手前置步骤.
            kx0, ky0, kx1, ky1 = _clamp_roi(kettle_bbox, w, h)
            search_left_frac = float(self._cfg_white.get("search_left_frac", 0.68))
            red_sx0 = kx0 + int((kx1 - kx0) * search_left_frac)
            red_roi = _clamp_roi((red_sx0, ky0, kx1, ky1), w, h)
            if red_roi[2] > red_roi[0] and red_roi[3] > red_roi[1]:
                red_contour, _red_mask = find_handle_red_tag(
                    color_image, self._cfg_red, roi=red_roi,
                )
            else:
                red_contour = None

            if red_contour is not None:
                # 红色找到 → 直接用它做 3D, 不再依赖白色检测
                x, y, rw, rh = cv2.boundingRect(red_contour)
                self._last_red_bbox = (x, y, rw, rh)
                handle_result["red_center"] = (x + rw // 2, y + rh // 2)
                handle_result["red_contour"] = red_contour
                self._last_red_contour = red_contour
                # 用红胶带 + margin 构造 white_roi (给可视化, 也兼容后续引用)
                white_roi = _clamp_roi(
                    (x - 20, y - 15, x + rw + 20, y + rh + 15), w, h,
                )
                handle_result["white_roi"] = white_roi
                self._last_white_roi = white_roi
            else:
                # 红色未找到 → 回退: 白色把手检测 (原两遍流程)
                white_roi = find_white_handle_roi(color_image, kettle_bbox, self._cfg_white)
                handle_result["white_roi"] = white_roi
                self._last_white_roi = white_roi

                if white_roi is not None:
                    red_contour, _red_mask = find_handle_red_tag(
                        color_image, self._cfg_red, roi=white_roi,
                    )
                else:
                    red_contour = None

            handle_result["red_contour"] = red_contour
            self._last_red_contour = red_contour

            if red_contour is not None:
                x, y, rw, rh = cv2.boundingRect(red_contour)
                self._last_red_bbox = (x, y, rw, rh)
                handle_result["red_center"] = (x + rw // 2, y + rh // 2)

                # 3D 位置
                dep_result = deproject_red_tag(
                    depth_frame,
                    self._color_intr, self._depth_intr,
                    self._c2d_extr, self._d2c_extr,
                    self._depth_scale, red_contour,
                    depth_min=self._depth_min, depth_max=self._depth_max,
                )
                if dep_result is not None:
                    handle_result["pos_cam"] = dep_result["pos_cam"]
                    self._last_handle_pos = dep_result["pos_cam"]
            else:
                self._last_red_bbox = None
                self._last_handle_pos = None

        # === Cup 检测 ===
        cup_bbox = self._detect_cup_bbox(color_image)
        cup_result["cup_bbox"] = cup_bbox
        self._last_cup_bbox = cup_bbox

        if cup_bbox is not None:
            cup_pos = self._compute_cup_3d(color_image, depth_frame, cup_bbox)
            cup_result["pos_cam"] = cup_pos
            self._last_cup_pos = cup_pos

        return {"handle": handle_result, "cup": cup_result}

    # ------------------------------------------------------------
    #  可视化
    # ------------------------------------------------------------

    def visualize(self, color_image):
        """绘制检测结果, 返回带标注的 BGR 图像."""
        display = color_image.copy()
        h, w = display.shape[:2]

        # ---- 状态行 ----
        h_status = "H:OK" if self._last_handle_pos is not None else "H:..."
        c_status = "C:OK" if self._last_cup_pos is not None else "C:..."
        cv2.putText(display, f"{h_status} {c_status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---- 3D 位姿文字 ----
        _y_off = 55
        if self._last_handle_pos is not None:
            p = self._last_handle_pos
            cv2.putText(display, f"H: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) A={self._handle_euler[0]:.0f} B={self._handle_euler[1]:.0f} C={self._handle_euler[2]:.0f}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            _y_off += 20
        if self._last_cup_pos is not None:
            p = self._last_cup_pos
            cv2.putText(display, f"C: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) A={self._cup_euler[0]:.0f} B={self._cup_euler[1]:.0f} C={self._cup_euler[2]:.0f}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # ---- Kettle bbox (橙色) ----
        if self._last_kettle_bbox is not None:
            x1, y1, x2, y2 = self._last_kettle_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(display, "kettle", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # ---- 白色把手 ROI (黄色) ----
        if self._last_white_roi is not None:
            x1, y1, x2, y2 = self._last_white_roi
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 0), 2)
            cv2.putText(display, "white handle", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # ---- 红色胶带 (绿色) ----
        if self._last_red_contour is not None:
            cv2.drawContours(display, [self._last_red_contour], -1, (0, 255, 255), 2)
            if self._last_red_bbox is not None:
                x, y, rw, rh = self._last_red_bbox
                cv2.rectangle(display, (x, y), (x + rw, y + rh), (0, 255, 0), 2)
                cv2.putText(display, "RED TAG", (x, max(20, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cx, cy = x + rw // 2, y + rh // 2
                cv2.circle(display, (cx, cy), 5, (0, 255, 255), -1)

        # ---- Cup bbox (品红) ----
        if self._last_cup_bbox is not None:
            x1, y1, x2, y2 = self._last_cup_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display, "cup", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return display

    # ------------------------------------------------------------
    #  清理
    # ------------------------------------------------------------

    def close(self):
        """停止 pipeline. (不调 destroyAllWindows — 由 main 线程统一清理)"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("[HandleCup] pipeline 已停止")
            except Exception:
                pass
            self._pipeline = None

    # ------------------------------------------------------------
    #  独立测试
    # ------------------------------------------------------------

    def run(self):
        """交互式测试主循环."""
        print("=" * 55)
        print("  HandleCupDetector 独立测试")
        print("  Handle: YOLO kettle + 白色把手 + 红色胶带")
        print("  Cup:   YOLO cup + 深度反投影")
        print("  按 'q' 退出, 's' 保存当前帧")
        print("=" * 55)

        frame_counter = 0
        test_skip = 5  # 独立测试固定间隔
        print(f"[INFO] 检测间隔: 每 {test_skip} 帧")

        _save_dir = os.path.join(_project_root, "logs")
        os.makedirs(_save_dir, exist_ok=True)

        try:
            while True:
                color, depth_img, depth_cmap, depth_frame = self.get_frames()
                if color is None or depth_frame is None:
                    time.sleep(0.03)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                    continue

                frame_counter += 1
                display = color.copy()

                if frame_counter % test_skip == 1:
                    result = self.detect(color, depth_frame)
                    h_info = result["handle"]
                    c_info = result["cup"]

                    # 终端输出 (相机系 3D 位置; 末端 ABC 由 state_machine 计算)
                    if h_info["pos_cam"] is not None:
                        p = h_info["pos_cam"]
                        print(f"[Handle] cam=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
                    else:
                        kettle = "Y" if h_info["kettle_bbox"] is not None else "N"
                        white = "Y" if h_info["white_roi"] is not None else "N"
                        red = "Y" if h_info["red_contour"] is not None else "N"
                        dots = f"kettle={kettle} white={white} red={red}"
                        print(f"[Handle] no detection ({dots})")

                    if c_info["pos_cam"] is not None:
                        p = c_info["pos_cam"]
                        print(f"[Cup]    cam=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
                    else:
                        print("[Cup]    no detection")

                display = self.visualize(color)
                if depth_cmap is not None:
                    h_vis = display.shape[0]
                    d_small = cv2.resize(depth_cmap, (h_vis, h_vis))
                    vis_panel = np.hstack((display, d_small))
                else:
                    vis_panel = display
                cv2.imshow("HandleCupDetector Test", vis_panel)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(_save_dir, f"handlecup_{ts}.png")
                    cv2.imwrite(path, vis_panel)
                    print(f"[SAVE] {path}")

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            self.close()
            print("[INFO] 已退出")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import yaml

    _orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(_orch_path, "r", encoding="utf-8") as f:
        _orch_cfg = yaml.safe_load(f)

    _serial = ""
    _info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(_info_path, "r", encoding="utf-8") as f:
            _info = yaml.safe_load(f)
        _serial = str(_info["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")

    import torch
    _dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    detector = HandleCupDetector(orch_config=_orch_cfg, serial=_serial, device=_dev)
    detector.run()
