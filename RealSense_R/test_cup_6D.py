# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m 水杯6D位姿检测
=============================================
继承 BaseRealSenseDetector 基类.
使用标准 YOLO COCO 模型, class_id=41 (cup).

检测流程:
  1. YOLO 检测水杯 bbox
  2. bbox 内点云提取 + PCA → 6D 位姿
  3. 输出 5 元组: (bbox, position, rotation, euler, point_cloud)

注意事项:
  - 杯口/倒水位置不再由本模块计算, 改由上层 estun_handle_cup_demo.py
    在基坐标系下基于 Cup TCP 位置几何推导.
  - 配置文件中 cup.height / cup.rim_radius 仍用于可视化参考和上层计算.

使用方法:
  python RealSense_R/test_cup_6D.py
  按 'q' 退出, 按 's' 保存当前帧
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

from RealSense_R.realsense_shared import (
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
#  2. 主检测器 (独立运行, 继承基类)
# ============================================================

class RealSenseCup6DDetector(BaseRealSenseDetector):
    """
    水杯 6D 位姿检测器.

    与 test_6D.py 的差异:
      - TARGET_CLASS_ID = 41 (cup)
      - 增加 cup_height / rim_radius / rim_offset_ratio / handle_side 配置 (供上层使用)
      - compute_cup_6d_pose 返回 5 元组 (去掉 rim/pour 检测)
      - 可视化仅显示 bbox + 6D 坐标轴 + PCA 中心
    """

    # ---- 覆盖基类类属性 ----
    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "test_cup_6D.log"

    # ---- 水杯参数 (从 config.yaml 加载, 供上层使用) ----
    CUP_HEIGHT = 0.12             # 杯高先验 (m), 需用户填写
    RIM_RADIUS = 0.04             # 杯口半径先验 (m), 需用户填写
    RIM_OFFSET_RATIO = 0.6        # 倒水偏移比例 (0~1), 上层使用
    CUP_HANDLE_SIDE = "back"      # 把手方向: back | left | right (config fallback)
    CUP_MODEL_NAME = ""           # cup 专用模型, 为空时使用基类 MODEL_NAME

    def __init__(self, config_dict=None, start_pipeline=True):
        super().__init__(config_dict=config_dict, start_pipeline=start_pipeline)

        # 6D 位姿平滑器
        self._pose_smoother = PoseSmoother(alpha=self.SMOOTH_POSE_ALPHA)

        # --- 耗时统计 ----
        self._yolo_time_ms = 0.0

        # --- 外部访问的动态属性 (用于 demo/run 共享 pipeline) ---
        self.last_known_pose = None   # (pos_cam, euler_cam) 或 None

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
            self.CUP_HANDLE_SIDE = _cup["handle_side"]

    # ------------------------------------------------------------
    def compute_cup_6d_pose(self, color_image, depth_frame):
        """
        主检测函数: YOLO → bbox 点云 → PCA → 6D 位姿.

        流程:
          Phase 1: YOLO 检测水杯 bbox (class_id=41)
          Phase 2: bbox 内逐像素采点 → 彩色内参反投影 → 3D 点云
          Phase 3: PCA → 6D 位姿

        参数:
            color_image: np.ndarray (H, W, 3), BGR 彩色图
            depth_frame: rs.depth_frame, 深度帧 (RealSense_R 版使用手动像素映射)

        返回:
            (bbox, position, rotation, euler, point_cloud)
            全部为 None 时表示检测失败
        """
        # ---- Phase 1: YOLO 检测 ----
        _t0 = time.perf_counter()
        best = self._detect_targets(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t0) * 1000

        if best is None:
            return (None, None, None, None, None)

        bbox = best["bbox"]

        # ---- Phase 2: 点云提取 ----
        anchor = self._get_anchor_depth(bbox, depth_frame)
        if anchor is None:
            return (bbox, None, None, None, None)
        _, _, center_depth = anchor

        # ---- 根据 handle_side 裁剪 bbox (排除把手侧的背景点) ----
        if self.CUP_HANDLE_SIDE in ("left", "right"):
            fx = self.color_intrinsics.fx
            rim_px = max(1, int(self.RIM_RADIUS * fx / center_depth))
            x1_orig, x2_orig = bbox[0], bbox[2]
            if self.CUP_HANDLE_SIDE == "left":
                # 把手在左侧, 从右边界向左取 2 倍杯口半径
                bbox[0] = max(int(x2_orig) - 2 * rim_px, int(x1_orig))
            elif self.CUP_HANDLE_SIDE == "right":
                # 把手在右侧, 从左边界向右取 2 倍杯口半径
                bbox[2] = min(int(x1_orig) + 2 * rim_px, int(x2_orig))
            # 确保裁剪后 bbox 宽度合理, 否则回退到原始 bbox
            if bbox[2] - bbox[0] < 4:
                bbox[0], bbox[2] = x1_orig, x2_orig

        x1_c, y1_c = int(bbox[0]), int(bbox[1])
        x2_c, y2_c = int(bbox[2]), int(bbox[3])

        points = []
        for cy in range(y1_c, y2_c + 1, 2):
            for cx in range(x1_c, x2_c + 1, 2):
                pt = self._color_pixel_to_3d(cx, cy, depth_frame)
                if pt is None:
                    continue
                x, y, d = pt
                if abs(d - center_depth) > self.DEPTH_Z_THRESHOLD:
                    continue
                points.append([x, y, d])

        if len(points) < 10:
            # 放宽 Z 过滤再试一次
            points_wide = []
            for cy in range(y1_c, y2_c + 1, 2):
                for cx in range(x1_c, x2_c + 1, 2):
                    pt = self._color_pixel_to_3d(cx, cy, depth_frame)
                    if pt is not None:
                        points_wide.append(list(pt))
            if len(points_wide) < 10:
                return (bbox, None, None, None, None)
            point_cloud = np.array(points_wide)
        else:
            point_cloud = np.array(points)

        # ---- Phase 3: PCA 6D 位姿 ----
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主轴 = 水杯轴向
        target_y_main = np.real(eig_vec[:, 0])

        # Z 轴指向相机
        target_z = np.real(eig_vec[:, 2])
        if target_z[2] > 0:
            target_z = -target_z

        target_x = np.cross(target_y_main, target_z)
        norm_x = np.linalg.norm(target_x)
        if norm_x < 1e-10:
            return (bbox, None, None, None, None)
        target_x = target_x / norm_x
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)
        euler = _r2euler(rot)

        return (bbox, center, rot, euler, point_cloud)

    # ------------------------------------------------------------
    def visualize_cup(self, image, bbox, position, rotation, euler):
        """
        在图像上绘制检测结果.

        绘制内容:
          - YOLO bbox (绿色)
          - 6D 坐标轴 (红X 绿Y 蓝Z)
          - PCA 中心 (紫色圆点)
          - 文字标注 (位姿信息)
        """
        display = image.copy()

        # ---- YOLO bbox (绿色) ----
        if bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, "Cup bbox", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ---- 6D 坐标轴 ----
        if position is not None and rotation is not None:
            K = np.array([[self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
                          [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
                          [0, 0, 1]], dtype=float)

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

        # ---- 文字标注 ----
        cv2.putText(display, "YOLO: %dms" % int(self._yolo_time_ms),
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display, "Handle: %s" % self.CUP_HANDLE_SIDE,
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        if position is not None and euler is not None:
            text_p = "Pos: (%.2f, %.2f, %.2f) m" % tuple(position)
            text_e = "Euler: (A=%.1f, B=%.1f, C=%.1f)" % tuple(euler)
            cv2.putText(display, text_p, (10, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display, text_e, (10, 95),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return display

    # ------------------------------------------------------------
    def run(self):
        """主循环: 检测 → 6D → 显示 + 终端输出."""
        from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles as _r2e

        print("=" * 60)
        print("  RealSense 水杯 6D 位姿检测")
        print("  模型: YOLOv8m COCO (class_id=%d=cup)" % self.TARGET_CLASS_ID)
        print("  分辨率: %dx%d @ %dfps" % (self.FRAME_W, self.FRAME_H, self.FRAME_FPS))
        print("  置信度阈值: %.2f" % self.CONF_THRESHOLD)
        print("-" * 60)
        print("  [杯参数] 杯高: %.2fm, 杯口半径: %.2fm" % (
            self.CUP_HEIGHT, self.RIM_RADIUS))
        if self.CUP_HEIGHT < 0.01 or self.RIM_RADIUS < 0.01:
            print("  [WARN] 杯高或杯口半径未填写! 请在 config.yaml 中设置 cup.height 和 cup.rim_radius")
        print("=" * 60)
        print("  按 'q' 退出, 按 's' 保存当前帧")
        print("  输出格式: 6D位姿 (X, Y, Z, A, B, C)")
        print("")

        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- 检测 ----
                (bbox, position, rotation, euler, point_cloud) = self.compute_cup_6d_pose(
                    color_image, depth_frame
                )

                # ---- 6D 位姿平滑 ----
                if position is not None and rotation is not None:
                    smooth_pos, smooth_rot = self._pose_smoother.update(
                        position, rotation
                    )
                    if smooth_rot is not None:
                        position = smooth_pos
                        rotation = smooth_rot
                        euler = _r2e(rotation)
                else:
                    self._pose_smoother.reset()

                # ---- 可视化 ----
                display = self.visualize_cup(
                    color_image, bbox, position, rotation, euler
                )

                # 并排显示深度图
                if depth_colormap is not None:
                    h_vis = display.shape[0]
                    depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                    vis_panel = np.hstack((display, depth_small))
                else:
                    vis_panel = display

                cv2.imshow("RealSense Cup 6D", vis_panel)

                # ---- 终端输出 ----
                if position is not None and euler is not None:
                    print(("\n[Cup 6D] pos=(%.3f, %.3f, %.3f) "
                           "euler=(A=%.1f, B=%.1f, C=%.1f)") % (
                               position[0], position[1], position[2],
                               euler[0], euler[1], euler[2],
                           ), end="", flush=True)
                    self._log_write("[Cup 6D]", "pos=(%.3f, %.3f, %.3f) euler=(A=%.1f, B=%.1f, C=%.1f)" % (
                        position[0], position[1], position[2],
                        euler[0], euler[1], euler[2]))
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
#  3. 主入口 (独立运行)
# ============================================================

def main():
    print("=" * 60)
    print("  RealSense 水杯 6D 位姿检测")
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
        print("[WARN] 请在 RealSense_R/config.yaml 中填写实际测量值 (单位: 米)")
        print("[WARN] 例如: 杯高12cm -> height: 0.12, 杯口半径4cm -> rim_radius: 0.04")
        print("[WARN] 继续运行, 但杯口位置可能不准确")

    detector = RealSenseCup6DDetector(config_dict=_MODULE_CFG)
    detector.run()


if __name__ == "__main__":
    main()