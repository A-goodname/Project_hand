# -*- coding: utf-8 -*-
"""
EyeToHand 人手-杯把手重合检测 (Hand-in-cup)
=============================================
双模型判定: YOLOv8m (cup bbox) + YOLOv8n-Pose (wrist keypoints).

判定策略:
  - 手腕到杯 bbox 中心的 2D 像素距离 < 阈值 (bbox 对角线 × dist_factor) → 邻近
  - 可选: 手腕深度与杯中心深度差值 < depth_tolerance → 确认在同一平面
  - 连续 N 帧满足 → hand_on_cup = True

实测状态 (2026-07-06):
  ✅ 无灵巧手遮挡: 人手侧握杯把时发现手腕在 ROI 外, 已引入 dist_factor 和
     depth_tolerance 作为可调修正, 但二者仍为初始猜测值, 未实地验证.
  ❌ 有灵巧手遮挡: 尚未测试.
  详细记录见 memory-bank/progress.md 的 "HandInCup 实测记录" 段.

模型:
  - Cup:     model/yolov8m.pt       (COCO class 41 "cup")
  - Pose:    model/yolov8n-pose.pt  (COCO keypoints, wrist=9,10)
             model/yolov8m-pose.pt  (备用, 已下载)

接口:
  - update(color_image, depth_frame=None) : 每帧调用
  - is_hand_on_cup : bool, 人手是否在杯把手上
  - stable_count   : int, 当前连续确认帧数
  - reset()        : 重置内部状态
  - run()          : 交互式测试主循环 (连接 Camera B)
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np
from ultralytics import YOLO

from RealSense_R.realsense_shared import BaseRealSenseDetector


# ============================================================
#  HandInCupDetector
# ============================================================

class HandInCupDetector:
    """人手-杯把手重合检测器.

    使用邻近判定而非包含判定: 手侧握把手时, 手腕在杯 bbox 外部,
    但到 bbox 中心的距离足够近.
    """

    # COCO keypoint indices
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    # ------------------------------------------------------------
    @staticmethod
    def _load_default_models():
        """从 EyeToHand/config.yaml 的 hand_in_cup 段读取默认模型路径."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(_cfg_path, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f)
        except Exception:
            _cfg = {}
        hic = _cfg.get("hand_in_cup", {})
        cup_path = hic.get("cup_model", os.path.join(_project_root, "model", "yolov8m.pt"))
        pose_path = hic.get("pose_model", os.path.join(_project_root, "model", "yolov8n-pose.pt"))
        return cup_path, pose_path

    # ------------------------------------------------------------

    def __init__(self,
                 cup_model_path: str = "",
                 pose_model_path: str = "",
                 device: str = "cuda",
                 conf_threshold: float = 0.5,
                 required_stable: int = 5,
                 dist_factor: float = 0.8,
                 depth_tolerance: float = 0.05):
        """
        Args:
            cup_model_path:  YOLO 杯检测模型路径.
            pose_model_path: YOLO-Pose 模型路径.
            device:          "cuda" | "cpu".
            conf_threshold:  检测置信度阈值.
            required_stable: 连续确认帧数 (默认 5, ~0.17s@30fps).
            dist_factor:     距离阈值 = bbox 对角线 × dist_factor
                             默认 0.8, 覆盖杯身半径 + 把手延伸.
            depth_tolerance: 手腕-杯心深度差阈值 (米), 0.05=5cm.
        """
        _default_cup, _default_pose = self._load_default_models()
        if not cup_model_path:
            cup_model_path = _default_cup
        if not pose_model_path:
            pose_model_path = _default_pose

        self._conf_threshold = conf_threshold
        self._required_stable = required_stable
        self._dist_factor = dist_factor
        self._depth_tolerance = depth_tolerance
        self._stable_count = 0
        self._on_cup = False

        # 模型
        print(f"[HandInCup] Cup: {cup_model_path}")
        self._cup_model = YOLO(cup_model_path)
        self._cup_model.to(device)

        print(f"[HandInCup] Pose: {pose_model_path}")
        self._pose_model = YOLO(pose_model_path)
        self._pose_model.to(device)

        # 可视化用
        self._last_cup_bbox = None
        self._last_cup_center = None
        self._last_wrists = []
        self._last_distances = []       # 各手腕到杯心的像素距离
        self._last_depth_ok = False     # 深度是否匹配

    # ------------------------------------------------------------
    #  核心逻辑
    # ------------------------------------------------------------

    def update(self, color_image: np.ndarray, depth_frame=None):
        """每帧调用.

        Args:
            color_image: BGR 图像 (H, W, 3).
            depth_frame: rs.depth_frame, 用于深度辅助判定 (可选).
        """
        # 1. 检测杯 bbox
        cup_bbox = self._detect_cup(color_image)
        self._last_cup_bbox = cup_bbox

        # 2. 检测手腕
        wrists = self._detect_wrists(color_image)
        self._last_wrists = wrists

        # 3. 邻近判定
        hit = False
        cup_center = None
        distances = []
        depth_ok = False

        if cup_bbox is not None and len(wrists) > 0:
            # 杯心
            cx = (cup_bbox[0] + cup_bbox[2]) // 2
            cy = (cup_bbox[1] + cup_bbox[3]) // 2
            cup_center = (cx, cy)

            # 阈值 = bbox 对角线 × dist_factor
            w = cup_bbox[2] - cup_bbox[0]
            h = cup_bbox[3] - cup_bbox[1]
            dist_thresh = ((w ** 2 + h ** 2) ** 0.5) * self._dist_factor

            for wx, wy in wrists:
                d = ((wx - cx) ** 2 + (wy - cy) ** 2) ** 0.5
                distances.append(d)
                if d < dist_thresh:
                    hit = True

            # 深度辅助 (可选): 手腕与杯心深度差 < tolerance
            if hit and depth_frame is not None:
                # 取杯心深度 (中值 5x5)
                cup_depths = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        py, px = cy + dy, cx + dx
                        if 0 <= py < color_image.shape[0] and 0 <= px < color_image.shape[1]:
                            d = depth_frame.get_distance(px, py)
                            if 0 < d < 2.0:
                                cup_depths.append(d)
                if cup_depths:
                    cup_z = np.median(cup_depths)
                    # 检查每个手腕的深度
                    for wx, wy in wrists:
                        wz = depth_frame.get_distance(wx, wy)
                        if 0 < wz < 2.0 and abs(wz - cup_z) < self._depth_tolerance:
                            depth_ok = True
                            break
                    if not depth_ok:  # 深度差异大 → 手在杯前/后, 非抓取
                        hit = False

        self._last_cup_center = cup_center
        self._last_distances = distances
        self._last_depth_ok = depth_ok

        # 4. 累积
        if hit:
            self._stable_count += 1
        else:
            self._stable_count = 0

        self._on_cup = self._stable_count >= self._required_stable

    def _detect_cup(self, color_image):
        """YOLOv8m 检测水杯, 返回最高置信度 bbox [x1,y1,x2,y2] 或 None."""
        results = self._cup_model(color_image, classes=[41], verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        best_idx = confs.argmax()
        if confs[best_idx] < self._conf_threshold:
            return None
        xyxy = boxes.xyxy.cpu().numpy()[best_idx]
        return [int(x) for x in xyxy]

    def _detect_wrists(self, color_image):
        """YOLOv8n-Pose 检测, 返回可见手腕像素坐标列表 [(x, y), ...]."""
        results = self._pose_model(color_image, verbose=False)
        if not results or results[0].keypoints is None:
            return []

        kpts = results[0].keypoints
        if kpts.xy is None or kpts.conf is None:
            return []

        n_people = kpts.conf.shape[0]
        n_kpts = kpts.conf.shape[1]
        if n_people == 0 or n_kpts == 0:
            return []

        wrist_indices = [self.LEFT_WRIST, self.RIGHT_WRIST]
        wrists = []
        for idx in wrist_indices:
            if idx >= n_kpts:
                continue
            confs = kpts.conf[:, idx].cpu().numpy()
            if len(confs) == 0:
                continue
            best_person = confs.argmax()
            if confs[best_person] >= self._conf_threshold:
                xy = kpts.xy[best_person, idx].cpu().numpy()
                wrists.append((int(xy[0]), int(xy[1])))
        return wrists

    # ------------------------------------------------------------
    #  公开属性
    # ------------------------------------------------------------

    @property
    def is_hand_on_cup(self) -> bool:
        return self._on_cup

    @property
    def stable_count(self) -> int:
        return self._stable_count

    def reset(self):
        self._stable_count = 0
        self._on_cup = False
        self._last_cup_bbox = None
        self._last_cup_center = None
        self._last_wrists = []
        self._last_distances = []
        self._last_depth_ok = False

    # ------------------------------------------------------------
    #  交互式测试
    # ------------------------------------------------------------

    def run(self):
        """交互式测试主循环. 需要连接 Camera B (RealSense D435i)."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(_cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)

        class _TestRunner(BaseRealSenseDetector):
            USE_YOLO_WORLD = False
            USE_SAM = False
            LOG_NAME = "hand_in_cup_test.log"
            TARGET_CLASS_ID = 0

        runner = _TestRunner(config_dict=_cfg)
        print("=" * 55)
        print("  Hand-in-cup 检测测试")
        print("  YOLOv8m (cup) + YOLOv8n-Pose (wrist)")
        print("  邻近判定 + 深度辅助")
        print("  按 'q' 退出")
        print("=" * 55)

        try:
            while True:
                color, depth_img, depth_cmap, depth_frame = runner.get_frames()
                if color is None:
                    continue

                self.update(color, depth_frame)
                display = color.copy()

                # ---- 状态行 ----
                status = "ON CUP" if self._on_cup else f"wait... ({self._stable_count}/{self._required_stable})"
                status_color = (0, 255, 0) if self._on_cup else (0, 165, 255)
                cv2.putText(display, f"HandInCup: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # ---- 杯 bbox (紫) ----
                bbox = self._last_cup_bbox
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    w, h = x2 - x1, y2 - y1
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.putText(display, "cup", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                    # 杯心 + 距离圆
                    cx, cy = self._last_cup_center or ((x1 + x2) // 2, (y1 + y2) // 2)
                    dist_thresh = ((w ** 2 + h ** 2) ** 0.5) * self._dist_factor
                    cv2.circle(display, (cx, cy), int(dist_thresh), (255, 128, 255), 1, cv2.LINE_AA)
                    cv2.drawMarker(display, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)

                # ---- 手腕 + 连线 (黄) ----
                for i, (wx, wy) in enumerate(self._last_wrists):
                    cv2.drawMarker(display, (wx, wy), (0, 255, 255), cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(display, "wrist", (wx + 5, wy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                    # 到杯心的连线 + 距离值
                    if self._last_cup_center and i < len(self._last_distances):
                        cx, cy = self._last_cup_center
                        d = self._last_distances[i]
                        cv2.line(display, (wx, wy), (cx, cy), (0, 255, 255), 1, cv2.LINE_AA)
                        mid_x, mid_y = (wx + cx) // 2, (wy + cy) // 2
                        cv2.putText(display, f"{d:.0f}px", (mid_x, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # ---- 深度状态 ----
                dep_str = f"depth:{'OK' if self._last_depth_ok else '--'}" if self._last_cup_bbox else ""
                if dep_str:
                    cv2.putText(display, dep_str, (10, 55),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 0) if self._last_depth_ok else (128, 128, 128), 1)

                # ---- 并排显示 ----
                if depth_cmap is not None and display.shape == depth_cmap.shape:
                    panel = np.hstack((display, depth_cmap))
                else:
                    panel = display
                cv2.imshow("HandInCup Test", panel)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
        finally:
            runner.stop()
            cv2.destroyAllWindows()
            print("[INFO] 已退出")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    detector = HandInCupDetector()
    detector.run()
