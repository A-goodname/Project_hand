# -*- coding: utf-8 -*-
"""
EyeToHand 人手-杯把手重合检测 (Hand-in-cup)
=============================================
双模型 2D 判定: YOLOv8m (cup bbox) + YOLOv8n-Pose (wrist keypoints).
手腕关键点落入杯 bbox 扩展区域内且连续 N 帧 → 判定手在杯上.

使用方法:
  1. 下载模型: 首次运行 ultralytics 自动下载 yolov8n-pose.pt
  2. 测试:   python EyeToHand/hand_in_cup.py
  3. 接入:   from EyeToHand.hand_in_cup import HandInCupDetector

模型:
  - Cup:     model/yolov8m.pt       (COCO class 41 "cup")
  - Pose:    model/yolov8n-pose.pt  (COCO keypoints, wrist=9,10)

接口:
  - update(color_image, depth_frame=None) : 每帧调用
  - is_hand_on_cup : bool, 人手是否在杯把手上
  - reset()        : 重置内部状态
  - run()          : 交互式测试主循环
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

# ---- PersonDetector 复用 Camera B pipeline ----
from RealSense_R.realsense_shared import BaseRealSenseDetector


# ============================================================
#  HandInCupDetector
# ============================================================

class HandInCupDetector:
    """人手-杯把手重合检测器.

    每帧调用 update(), 内部累积连续帧判定结果.
    连续 N 帧手腕在杯 bbox 内 → is_hand_on_cup = True.
    """

    # COCO keypoint indices
    LEFT_WRIST = 9
    RIGHT_WRIST = 10

    # Cup bbox 扩展系数 (容差)
    BBOX_EXPAND = 0.20

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
                 required_stable: int = 5):
        """
        Args:
            cup_model_path:  YOLO 杯检测模型路径, 默认从 config 读取或 model/yolov8m.pt
            pose_model_path: YOLO-Pose 模型路径, 默认从 config 读取或 model/yolov8n-pose.pt
            device:          "cuda" | "cpu"
            conf_threshold:  检测置信度阈值
            required_stable: 连续确认帧数 (默认 5, ~0.17s@30fps)
        """
        # 从模块配置读取默认模型路径
        _default_cup, _default_pose = self._load_default_models()

        if not cup_model_path:
            cup_model_path = _default_cup
        if not pose_model_path:
            pose_model_path = _default_pose

        self._conf_threshold = conf_threshold
        self._required_stable = required_stable
        self._stable_count = 0
        self._on_cup = False

        # 模型
        print(f"[HandInCup] Cup: {cup_model_path}")
        self._cup_model = YOLO(cup_model_path)
        self._cup_model.to(device)

        print(f"[HandInCup] Pose: {pose_model_path}")
        self._pose_model = YOLO(pose_model_path)
        self._pose_model.to(device)

        # 可视化用 (run() 中设置)
        self._last_cup_bbox = None
        self._last_wrists = []

    # ------------------------------------------------------------
    #  核心逻辑
    # ------------------------------------------------------------

    def update(self, color_image: np.ndarray, depth_frame=None):
        """每帧调用. depth_frame 保留为接口兼容, 当前不使用.

        Args:
            color_image: BGR 图像 (H, W, 3)
            depth_frame: 保留参数 (未使用)
        """
        # 1. 检测杯 bbox
        cup_bbox = self._detect_cup(color_image)
        self._last_cup_bbox = cup_bbox

        # 2. 检测手腕关键点
        wrists = self._detect_wrists(color_image)
        self._last_wrists = wrists

        # 3. 判定重合
        if cup_bbox is not None and len(wrists) > 0:
            exp_bbox = self._expand_bbox(cup_bbox, color_image.shape)
            hit = any(self._point_in_bbox(w, exp_bbox) for w in wrists)
        else:
            hit = False

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
        # 取最高置信度
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

        wrist_indices = [self.LEFT_WRIST, self.RIGHT_WRIST]
        wrists = []
        for idx in wrist_indices:
            if idx < kpts.conf.shape[1]:
                # 取所有检测到的人中该关键点置信度最高的
                confs = kpts.conf[:, idx].cpu().numpy()
                best_person = confs.argmax()
                if confs[best_person] >= self._conf_threshold:
                    xy = kpts.xy[best_person, idx].cpu().numpy()
                    wrists.append((int(xy[0]), int(xy[1])))
        return wrists

    # ------------------------------------------------------------
    #  几何工具
    # ------------------------------------------------------------

    @staticmethod
    def _expand_bbox(bbox, shape):
        """扩展 bbox 以增加容差."""
        x1, y1, x2, y2 = bbox
        h, w = shape[:2]
        dw = int((x2 - x1) * HandInCupDetector.BBOX_EXPAND)
        dh = int((y2 - y1) * HandInCupDetector.BBOX_EXPAND)
        return [
            max(0, x1 - dw),
            max(0, y1 - dh),
            min(w, x2 + dw),
            min(h, y2 + dh),
        ]

    @staticmethod
    def _point_in_bbox(point, bbox):
        """点是否在 bbox 内."""
        px, py = point
        x1, y1, x2, y2 = bbox
        return x1 <= px <= x2 and y1 <= py <= y2

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
        self._last_wrists = []

    # ------------------------------------------------------------
    #  交互式测试
    # ------------------------------------------------------------

    def run(self):
        """交互式测试主循环. 需要连接 Camera B (RealSense D435i)."""
        # 模块级配置
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
        print("  按 'q' 退出")
        print("=" * 55)

        try:
            while True:
                color, depth_img, depth_cmap, depth_frame = runner.get_frames()
                if color is None:
                    continue

                self.update(color)
                display = color.copy()

                # 状态
                status = "ON CUP" if self._on_cup else f"waiting... ({self._stable_count}/{self._required_stable})"
                status_color = (0, 255, 0) if self._on_cup else (0, 165, 255)
                cv2.putText(display, f"HandInCup: {status}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

                # 杯 bbox
                if self._last_cup_bbox is not None:
                    x1, y1, x2, y2 = self._last_cup_bbox
                    exp = self._expand_bbox(self._last_cup_bbox, color.shape)
                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
                    cv2.rectangle(display, (exp[0], exp[1]), (exp[2], exp[3]), (255, 128, 255), 1)
                    cv2.putText(display, "cup", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

                # 手腕
                for wx, wy in self._last_wrists:
                    cv2.drawMarker(display, (wx, wy), (0, 255, 255),
                                   cv2.MARKER_CROSS, 15, 2)
                    cv2.putText(display, "wrist", (wx + 5, wy - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # 并排
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
