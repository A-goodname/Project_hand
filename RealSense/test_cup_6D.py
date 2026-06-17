# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水杯6D位姿 + 杯口位置估计 (新)
=============================================
基于内参完好的新相机, 使用 rs.align 简化深度对齐链路.
继承 BaseRealSenseDetector 基类.
依赖 RealSense.realsense_shared 中的共享工具函数/类.

使用标准 YOLO COCO 模型, class_id=41 (cup).
检测流程:
  1. YOLO 检测水杯 bbox
  2. bbox 内点云提取 + PCA → 6D 位姿
  3. 沿主轴方向偏移杯高/2 → 杯口 3D 位置
  4. 径向偏移 → 倒水目标点

输入:
  - RealSense/config.yaml 中的 cup 配置段
  - 实际水杯的杯高 (cup.height) 和杯口半径 (cup.rim_radius) [用户必填]

输出:
  6D 位姿: (x, y, z, A, B, C) — 水杯在彩色相机系下的位姿
  杯口位置: (x, y, z) — 杯口中心在彩色相机系下的 3D 坐标
  倒水目标: (x, y, z) — 壶嘴应对准的位置 (杯口中心 + 径向偏移)

使用方法:
  python RealSense/test_cup_6D.py
  按 'q' 退出, 按 's' 保存当前帧

配置注意事项:
  请确保 config.yaml 中 cup.height 和 cup.rim_radius 已填写实际测量值!
  单位为米 (m), 例如 12cm = 0.12m, 4cm = 0.04m
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
from typing import Optional

import yaml
import cv2
import numpy as np
import torch

from RealSense.realsense_shared import (
    BaseRealSenseDetector,
    rotation_matrix_to_euler_angles as _r2euler,
    PoseSmoother,
)


# ============================================================
#  1. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  2. 杯口位置计算工具函数
# ============================================================

def estimate_rim_position(point_cloud, cup_height, rim_radius,
                          rim_offset_ratio=0.6, axis_in=None):
    """
    从水杯点云估计杯口 3D 位置和倒水目标点.

    流程:
      1. 确定主轴方向 (外部传入或内部 PCA)
      2. 重心沿主轴偏移 cup_height/2 → 杯口位置
      3. 径向偏移 → 倒水目标点

    参数:
        point_cloud      : np.ndarray (N, 3), bbox 内点云 (彩色坐标系)
        cup_height       : float, 杯高先验 (m). 当前版本改用投影边界, 此参数保留兼容.
        rim_radius       : float, 杯口半径先验 (m)
        rim_offset_ratio : float, 倒水偏移比例 (0~1). 补偿单面检测的偏心,
                           0=杯壁正上方, 1=杯口中心
        axis_in          : (3,) 或 None. 外部传入已修正方向的主轴 (杯底→杯口),
                           不为 None 时跳过内部 PCA 方向计算.

    返回:
        (axis, rim_pos, pour_target, r_est)
          axis        : (3,) 主轴方向单位向量 (杯底→杯口)
          rim_pos     : (3,) 杯口 3D 位置 (彩色坐标系)
          pour_target : (3,) 倒水目标点 (彩色坐标系)
          r_est       : float, 从点云估算的杯口半径 (m), 失败时返回 rim_radius
    """
    if point_cloud is None or len(point_cloud) < 10:
        return None, None, None, rim_radius

    # ---- 确定主轴方向 ----
    center = np.mean(point_cloud, axis=0)

    if axis_in is None:
        # 内部 PCA: 仅用于从未修正的场景 (例如独立运行 test_cup_6D 但未传 axis_in)
        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)
        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]
        axis = np.real(eig_vec[:, 0])
    else:
        axis = axis_in

    # ---- 杯口位置: 重心沿主轴偏移 cup_height/2 ----
    # 注: 俯视时点云重心 bias 已偏杯口端, 因此用 cup_height/2 偏移可能导致略超杯口.
    #     可通过减小偏移比例 (如 0.4) 或增大 rim_offset_ratio 补偿.
    rim_pos = center + axis * (cup_height / 2.0)

    # ---- 从点云估算实际杯口半径 ----
    # 在垂直于主轴的平面上投影点云, 取最大距离
    # 构建垂直于 axis 的平面基向量
    if abs(axis[2]) < 0.9:
        ref = np.array([0.0, 0.0, 1.0])
    else:
        ref = np.array([1.0, 0.0, 0.0])
    u = np.cross(axis, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(axis, u)
    v = v / np.linalg.norm(v)

    # 投影到 uv 平面
    offsets = point_cloud - center
    u_proj = offsets @ u
    v_proj = offsets @ v
    radii = np.sqrt(u_proj ** 2 + v_proj ** 2)
    r_est = float(np.max(radii))
    if r_est < 0.001 or r_est > 1.0:
        r_est = rim_radius  # 估算异常时回退先验值

    # ---- 径向方向: 从相机指向 rim_pos 的方向在垂直轴平面上的投影 ----
    # 相机在彩色坐标系原点 (0,0,0)
    cam_to_rim = rim_pos - np.array([0.0, 0.0, 0.0])
    # 减去沿轴分量 → 垂直轴的分量
    radial = cam_to_rim - np.dot(cam_to_rim, axis) * axis
    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-6:
        # 相机正好在水杯正上方, 用 u 作为径向
        radial_unit = u
    else:
        radial_unit = radial / radial_norm

    # ---- 倒水目标: 从杯壁上方往杯口中心偏移 ----
    pour_target = rim_pos + radial_unit * rim_offset_ratio * r_est

    return axis, rim_pos, pour_target, r_est


# ============================================================
#  3. 主检测器 (独立运行, 继承基类)
# ============================================================

class RealSenseCup6DDetector(BaseRealSenseDetector):
    """
    水杯 6D 位姿 + 杯口位置检测器.

    与 test_6D.py 的差异:
      - TARGET_CLASS_ID = 41 (cup)
      - 增加 cup_height / rim_radius / rim_offset_ratio 配置
      - compute_6d_pose 后附加杯口位置和倒水目标点计算
      - 可视化杯口圆圈 + 倒水目标点
    """

    # ---- 覆盖基类类属性 ----
    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "test_cup_6D.log"

    # ---- 水杯参数 (从 config.yaml 加载) ----
    CUP_HEIGHT = 0.12             # 杯高先验 (m), 需用户填写
    RIM_RADIUS = 0.04             # 杯口半径先验 (m), 需用户填写
    RIM_OFFSET_RATIO = 0.6        # 倒水偏移比例 (0~1)
    HANDLE_SIDE = "back"          # 把手方向: back | left | right (config fallback)
    CUP_MODEL_NAME = ""           # cup 专用模型, 为空时使用基类 MODEL_NAME

    def __init__(self, config_dict=None, start_pipeline=True):
        super().__init__(config_dict=config_dict, start_pipeline=start_pipeline)

        # 6D 位姿平滑器
        self._pose_smoother = PoseSmoother(alpha=self.SMOOTH_POSE_ALPHA)

        # --- 耗时统计 ----
        self._yolo_time_ms = 0.0

        # --- 外部访问的动态属性 (用于 demo/run 共享 pipeline) ---
        self.last_known_pose = None   # (pos_cam, euler_cam) 或 None
        self._cached_pour_target = None  # pour_target 3D 点或 None
        self._last_axis_vec = None   # 最近一帧的主轴方向 (杯底→杯口), 供 smooth 路径使用
        # HANDLE_SIDE 直接从 config 读取 (self.HANDLE_SIDE)

    # ------------------------------------------------------------
    def _apply_config(self, cfg):
        """覆盖基类, 增加 cup 配置加载."""
        super()._apply_config(cfg)

        _cup = cfg.get("cup", {})
        if not _cup.get("enabled", True):
            return  # cup 未启用, 保持基类配置

        # --- 模型名 (不为空时独立加载, 与基类 MODEL_NAME 完全解耦) ---
        if "model_name" in _cup and _cup["model_name"]:
            self.CUP_MODEL_NAME = os.path.join(_project_root, _cup["model_name"])
            self.MODEL_NAME = self.CUP_MODEL_NAME
        # --- 目标类 ID / 置信度 / 设备 (覆盖基类配置) ---
        if "target_class_id" in _cup:
            self.TARGET_CLASS_ID = _cup["target_class_id"]
        if "conf_threshold" in _cup:
            self.CONF_THRESHOLD = _cup["conf_threshold"]
        if "device" in _cup:
            pref = _cup["device"]
            self.DEVICE = 'cuda:0' if (pref == "gpu" and torch.cuda.is_available()) else 'cpu'
        # --- 水杯专属几何参数 ---
        if "height" in _cup:
            self.CUP_HEIGHT = _cup["height"]
        if "rim_radius" in _cup:
            self.RIM_RADIUS = _cup["rim_radius"]
        if "rim_offset_ratio" in _cup:
            self.RIM_OFFSET_RATIO = _cup["rim_offset_ratio"]
        if "handle_side" in _cup:
            self.HANDLE_SIDE = _cup["handle_side"]

    # ------------------------------------------------------------
    def compute_cup_6d_pose(self, color_image, aligned_depth_frame):
        """
        主检测函数: YOLO → bbox 点云 → PCA → 6D 位姿 + 杯口位置.

        流程:
          Phase 1: YOLO 检测水杯 bbox (class_id=41)
          Phase 2: bbox 内逐像素采点 → 彩色内参反投影 → 3D 点云
          Phase 3: PCA → 6D 位姿 (同 test_6D)
          Phase 4: estimate_rim_position → 杯口位置 + 倒水目标

        参数:
            color_image        : np.ndarray (H, W, 3), BGR 彩色图
            aligned_depth_frame: rs.depth_frame, 对齐后的深度帧

        返回:
            (bbox, position, rotation, euler, point_cloud,
             rim_pos, pour_target, rim_r)
            全部为 None 时表示检测失败
        """
        # ---- Phase 1: YOLO 检测 ----
        _t0 = time.perf_counter()
        best = self._detect_targets(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t0) * 1000

        if best is None:
            return (None, None, None, None, None, None, None, None)

        bbox = best["bbox"]

        # ---- Phase 2: 点云提取 ----
        anchor = self._get_anchor_depth(bbox, aligned_depth_frame)
        if anchor is None:
            return (bbox, None, None, None, None, None, None, None)
        _, _, center_depth = anchor

        # ---- 根据 handle_side 裁剪 bbox (排除把手侧的背景点) ----
        if self.HANDLE_SIDE in ("left", "right"):
            fx = self.color_intrinsics.fx
            rim_px = max(1, int(self.RIM_RADIUS * fx / center_depth))
            x1_orig, x2_orig = bbox[0], bbox[2]
            if self.HANDLE_SIDE == "left":
                # 把手在左侧, 从右边界向左取 2 倍杯口半径
                bbox[0] = max(int(x2_orig) - 2 * rim_px, int(x1_orig))
            elif self.HANDLE_SIDE == "right":
                # 把手在右侧, 从左边界向右取 2 倍杯口半径
                bbox[2] = min(int(x1_orig) + 2 * rim_px, int(x2_orig))
            # 确保裁剪后 bbox 宽度合理, 否则回退到原始 bbox
            if bbox[2] - bbox[0] < 4:
                bbox[0], bbox[2] = x1_orig, x2_orig

        x1_c, y1_c = int(bbox[0]), int(bbox[1])
        x2_c, y2_c = int(bbox[2]), int(bbox[3])

        points = []
        pixels = []  # 同步记录像素坐标, 用于杯口方向判断
        for cy in range(y1_c, y2_c + 1, 2):
            for cx in range(x1_c, x2_c + 1, 2):
                pt = self._color_pixel_to_3d(cx, cy, aligned_depth_frame)
                if pt is None:
                    continue
                x, y, d = pt
                if abs(d - center_depth) > self.DEPTH_Z_THRESHOLD:
                    continue
                points.append([x, y, d])
                pixels.append([cx, cy])

        if len(points) < 10:
            # 放宽 Z 过滤再试一次
            points_wide = []
            pixels_wide = []
            for cy in range(y1_c, y2_c + 1, 2):
                for cx in range(x1_c, x2_c + 1, 2):
                    pt = self._color_pixel_to_3d(cx, cy, aligned_depth_frame)
                    if pt is not None:
                        points_wide.append(list(pt))
                        pixels_wide.append([cx, cy])
            if len(points_wide) < 10:
                return (bbox, None, None, None, None, None, None, None)
            point_cloud = np.array(points_wide)
            pixels_arr = np.array(pixels_wide)
        else:
            point_cloud = np.array(points)
            pixels_arr = np.array(pixels)

        # ---- Phase 3: PCA 6D 位姿 ----
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        target_y_main = np.real(eig_vec[:, 0])  # 主轴 = 水杯轴向

        # 用像素 Y 坐标判断杯口方向:
        #   target_y_main 正方向端 (max_idx) 在图像中应位于上方 (Y 像素小 = 杯口)
        #   若正方向端在下方 (Y 大), 则翻转 target_y_main
        proj = point_cloud @ target_y_main
        min_idx = np.argmin(proj)
        max_idx = np.argmax(proj)
        if pixels_arr[max_idx, 1] > pixels_arr[min_idx, 1]:
            target_y_main = -target_y_main

        # 对主轴方向做指数平滑, 抑制帧间 B 角跳动
        AXIS_SMOOTH_ALPHA = 0.7  # 0~1, 越大越平滑
        if self._last_axis_vec is not None:
            # 检查方向一致性: 若夹角>90度则翻转, 避免正反交替拖慢
            if np.dot(target_y_main, self._last_axis_vec) < 0:
                target_y_main = -target_y_main
            target_y_main = (AXIS_SMOOTH_ALPHA * self._last_axis_vec +
                             (1 - AXIS_SMOOTH_ALPHA) * target_y_main)
            target_y_main = target_y_main / np.linalg.norm(target_y_main)

        target_z = np.real(eig_vec[:, 2])

        if target_z[2] > 0:
            target_z = -target_z

        target_x = np.cross(target_y_main, target_z)
        norm_x = np.linalg.norm(target_x)
        if norm_x < 1e-10:
            return (bbox, None, None, None, None, None, None, None)
        target_x = target_x / norm_x
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)
        euler = _r2euler(rot)

        position = center
        rotation = rot

        # ---- Phase 4: 杯口位置估计 ----
        # 传入已修正方向的 target_y_main, 避免 estimate_rim_position 内部重新 PCA
        axis_vec, rim_pos, pour_target, rim_r = estimate_rim_position(
            point_cloud,
            cup_height=self.CUP_HEIGHT,
            rim_radius=self.RIM_RADIUS,
            rim_offset_ratio=self.RIM_OFFSET_RATIO,
            axis_in=target_y_main
        )

        # 缓存主轴方向 (杯底→杯口), 供 run() 平滑路径使用
        self._last_axis_vec = axis_vec

        return (bbox, position, rotation, euler, point_cloud,
                rim_pos, pour_target, rim_r)

    # ------------------------------------------------------------
    def visualize_cup(self, image, bbox, position, rotation, euler,
                      rim_pos, pour_target, rim_r):
        """
        在图像上绘制检测结果.

        绘制内容:
          - YOLO bbox (绿色)
          - 6D 坐标轴 (红X 绿Y 蓝Z)
          - 倒水目标点 (红色十字 + 圆圈)
          - 杯口位置 (黄色圆点)
          - 文字标注
        """
        display = image.copy()

        # ---- 内参矩阵 (用于 3D→2D 投影) ----
        K = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                      [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                      [0, 0, 1]], dtype=float)

        # ---- YOLO bbox (绿色) ----
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "Cup bbox", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- 6D 坐标轴 ----
        if position is not None and rotation is not None:
            center_2d = K @ np.array(position)
            center_2d = center_2d / center_2d[2]
            cx_2d = int(center_2d[0])
            cy_2d = int(center_2d[1])

            cv2.circle(display, (cx_2d, cy_2d), 5, (124, 0, 255), 3)

            axes = [
                (rotation[:, 0], (0, 0, 255)),   # X -> 红色
                (rotation[:, 1], (0, 255, 0)),   # Y -> 绿色
                (rotation[:, 2], (255, 0, 0)),   # Z -> 蓝色
            ]
            for axis, color in axes:
                end_3d = np.array(position) + 0.08 * axis
                end_2d = K @ end_3d
                end_2d = end_2d / end_2d[2]
                ex_2d = int(end_2d[0])
                ey_2d = int(end_2d[1])
                cv2.arrowedLine(display, (cx_2d, cy_2d), (ex_2d, ey_2d),
                                color, 3, tipLength=0.2)

        # ---- 杯口位置 (黄色圆圈) ----
        if rim_pos is not None:
            rim_2d = K @ np.array(rim_pos)
            rim_2d = rim_2d / rim_2d[2]
            rim_cx = int(rim_2d[0])
            rim_cy = int(rim_2d[1])

            # 杯口圆圈 (黄色)
            cv2.circle(display, (rim_cx, rim_cy), 8, (0, 255, 255), 2)
            cv2.putText(display, "RIM", (rim_cx + 10, rim_cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # ---- 倒水目标点 (红色十字 + 圆圈) ----
        if pour_target is not None:
            pour_2d = K @ np.array(pour_target)
            pour_2d = pour_2d / pour_2d[2]
            px = int(pour_2d[0])
            py = int(pour_2d[1])

            # 红色十字
            cv2.line(display, (px - 8, py), (px + 8, py), (0, 0, 255), 2)
            cv2.line(display, (px, py - 8), (px, py + 8), (0, 0, 255), 2)
            cv2.circle(display, (px, py), 6, (0, 0, 255), 1)
            cv2.putText(display, "POUR TARGET", (px + 12, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # ---- 文字标注 ----
        cv2.putText(display, "YOLO: %dms" % int(self._yolo_time_ms),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, "Handle: %s" % self.HANDLE_SIDE,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if position is not None and euler is not None:
            text_p = "Pos: (%.2f, %.2f, %.2f) m" % tuple(position)
            text_e = "Euler: (A=%.1f, B=%.1f, C=%.1f)" % tuple(euler)
            cv2.putText(display, text_p, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display, text_e, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if rim_pos is not None:
            text_rim = "Rim: (%.2f, %.2f, %.2f) m" % tuple(rim_pos)
            cv2.putText(display, text_rim, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if pour_target is not None:
            text_pour = "Pour: (%.2f, %.2f, %.2f) m" % tuple(pour_target)
            cv2.putText(display, text_pour, (10, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        if rim_r is not None:
            cv2.putText(display, "Rim_r: %.2fm" % rim_r, (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

        return display

    # ------------------------------------------------------------
    def run(self):
        """主循环: 检测 → 6D → 杯口 → 显示 + 终端输出."""
        print("=" * 60)
        print("  RealSense 水杯 6D + 杯口位置检测")
        print("  模型: YOLOv8m COCO (class_id=%d=cup)" % self.TARGET_CLASS_ID)
        print("  分辨率: %dx%d @ %dfps" % (self.FRAME_W, self.FRAME_H, self.FRAME_FPS))
        print("  置信度阈值: %.2f" % self.CONF_THRESHOLD)
        print("-" * 60)
        print("  [杯参数] 杯高: %.2fm, 杯口半径: %.2fm, 偏移比例: %.1f" % (
            self.CUP_HEIGHT, self.RIM_RADIUS, self.RIM_OFFSET_RATIO))
        if self.CUP_HEIGHT < 0.01 or self.RIM_RADIUS < 0.01:
            print("  [WARN] 杯高或杯口半径未填写! 请在 config.yaml 中设置 cup.height 和 cup.rim_radius")
        print("=" * 60)
        print("  按 'q' 退出, 按 's' 保存当前帧")
        print("  输出格式: 6D位姿 (X, Y, Z, A, B, C) + 杯口位置 + 倒水目标")
        print("")

        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        try:
            while True:
                color_image, depth_image, depth_colormap, aligned_depth_frame = self.get_frames()
                if color_image is None or aligned_depth_frame is None:
                    continue

                # ---- 检测 ----
                (bbox, position, rotation, euler, point_cloud,
                 rim_pos, pour_target, rim_r) = self.compute_cup_6d_pose(
                    color_image, aligned_depth_frame
                )

                # ---- 6D 位姿平滑 ----
                if position is not None and rotation is not None:
                    smooth_pos, smooth_rot = self._pose_smoother.update(
                        position, rotation
                    )
                    if smooth_rot is not None:
                        position = smooth_pos
                        rotation = smooth_rot
                        euler = _r2euler(rotation)

                        # 用平滑后的位姿重新计算杯口位置
                        # 使用已确定的 _last_axis_vec (像素 Y 修正), 不再重复 PCA
                        if point_cloud is not None and len(point_cloud) >= 10 and self._last_axis_vec is not None:
                            axis = self._last_axis_vec.copy()

                            # 从点云重心沿轴偏移 cup_height/2 (与 estimate_rim_position 一致)
                            assert position is not None  # 外层 if 已保证
                            rim_pos = position + axis * (self.CUP_HEIGHT / 2.0)

                            # 重新计算倒水目标
                            cam_to_rim = rim_pos
                            radial = cam_to_rim - np.dot(cam_to_rim, axis) * axis
                            radial_norm = np.linalg.norm(radial)
                            if radial_norm > 1e-6:
                                radial_unit = radial / radial_norm
                            else:
                                if abs(axis[2]) < 0.9:
                                    ref = np.array([0.0, 0.0, 1.0])
                                else:
                                    ref = np.array([1.0, 0.0, 0.0])
                                u = np.cross(axis, ref)
                                radial_unit = u / np.linalg.norm(u)

                            if rim_pos is not None and rim_r is not None:
                                pour_target = rim_pos + radial_unit * self.RIM_OFFSET_RATIO * rim_r
                else:
                    self._pose_smoother.reset()

                # ---- 可视化 ----
                display = self.visualize_cup(
                    color_image, bbox, position, rotation, euler,
                    rim_pos, pour_target, rim_r
                )

                # 并排显示深度图
                if depth_colormap is not None:
                    h_vis = display.shape[0]
                    depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                    vis_panel = np.hstack((display, depth_small))
                else:
                    vis_panel = display

                cv2.imshow("RealSense Cup 6D + Rim Position", vis_panel)

                # ---- 终端输出 ----
                if position is not None and euler is not None:
                    rim_str = "(%.3f, %.3f, %.3f)" % tuple(rim_pos) if rim_pos is not None else "(None)"
                    pour_str = "(%.3f, %.3f, %.3f)" % tuple(pour_target) if pour_target is not None else "(None)"
                    print(("\n[Cup 6D] pos=(%.3f, %.3f, %.3f) "
                           "euler=(A=%.1f, B=%.1f, C=%.1f) "
                           "rim=%s pour=%s") % (
                               position[0], position[1], position[2],
                               euler[0], euler[1], euler[2],
                               rim_str, pour_str
                           ), end="", flush=True)
                    self._log_write("[Cup 6D]", "pos=(%.3f, %.3f, %.3f) euler=(A=%.1f, B=%.1f, C=%.1f) rim=%s pour=%s" % (
                        position[0], position[1], position[2],
                        euler[0], euler[1], euler[2],
                        rim_str, pour_str))
                else:
                    print(("\n[Cup 6D] no detection" + " " * 30), end="", flush=True)

                # ---- 按键处理 ----
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 用户按 q 退出")
                    break
                elif key == ord('s') and color_image is not None:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(_save_dir, "cup6d_%s.png" % ts)
                    cv2.imwrite(save_path, color_image)
                    print("\n[SAVE] %s" % save_path)

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断 (Ctrl+C)")
        except Exception as e:
            print("\n[ERROR] 运行异常: %s" % str(e))
        finally:
            self.stop()
            print("[INFO] 程序已退出")


# ============================================================
#  4. 主入口 (独立运行)
# ============================================================

def main():
    print("=" * 60)
    print("  RealSense 水杯 6D + 杯口位置检测")
    print("  基于 BaseRealSenseDetector 基类")
    print("=" * 60)

    # 检查配置
    cup_cfg = _MODULE_CFG.get("cup", {})
    if not cup_cfg.get("enabled", True):
        print("[INFO] 水杯检测未启用 (cup.enabled=false)")
        return

    height = cup_cfg.get("height", 0.0)
    radius = cup_cfg.get("rim_radius", 0.0)
    if height < 0.01 or radius < 0.01:
        print("[WARN] cup.height=%.2f 或 cup.rim_radius=%.2f 未正确设置!" % (height, radius))
        print("[WARN] 请在 RealSense/config.yaml 中填写实际测量值 (单位: 米)")
        print("[WARN] 例如: 杯高12cm -> height: 0.12, 杯口半径4cm -> rim_radius: 0.04")
        print("[WARN] 继续运行, 但杯口位置可能不准确")

    detector = RealSenseCup6DDetector(config_dict=_MODULE_CFG)
    detector.run()


if __name__ == "__main__":
    main()