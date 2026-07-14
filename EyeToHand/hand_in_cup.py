# -*- coding: utf-8 -*-
"""
EyeToHand 人手-杯把手重合检测 (Hand-in-cup)
=============================================
杯检测: YOLOv8m (class 41) → 失败时回退固定 ROI (cup_roi_fallback)
手检测: MediaPipe Hands (21 landmarks, 帧间跟踪)

判定策略 (landmarks-in-bbox 计分制):
  1. 取距离杯 bbox 中心最近的手
  2. 统计该手落在杯 bbox (外扩 bbox_margin px) 内的关键 landmarks:
       wrist (lm 0)  ×2 分
       MCP  (lm 5,9,13,17)  ×1 分
       TIP  (lm 4,8,12,16,20) ×1 分
  3. score ≥ 2 → hit
     - 仅手腕在 bbox 内 = 2 分 → hit
     - 手腕不在但 2 MCP 在内 = 2 分 → hit (抓握场景)
  4. 判定: score >= 2 → 累积 → hand_on_cup
  5. 连续 N 帧满足 → hand_on_cup = True

模型:
  - Cup:  model/yolov8m.pt  (COCO class 41)
  - Hand: model/hand_landmarker.task  (MediaPipe Hands)

接口:
  - update(color_image) : 每帧调用
  - is_hand_on_cup : bool
  - stable_count   : int
  - reset()        : 重置内部状态
  - run()          : 交互式测试主循环 (连接 Camera B)
"""

import os
import sys
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np
from ultralytics import YOLO

from EyeToHand.camera import UGreenCamera

# MediaPipe Hands (Task API)
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp

# MediaPipe landmark 索引
MP_WRIST = 0
MP_MCP = {5: "INDEX_MCP", 9: "MIDDLE_MCP", 13: "RING_MCP", 17: "PINKY_MCP"}
MP_TIP  = {4: "THUMB_TIP", 8: "INDEX_TIP", 12: "MIDDLE_TIP", 16: "RING_TIP", 20: "PINKY_TIP"}
# 手部骨架连线 (仅关键点)
HAND_BONES = [
    (0,5),(0,9),(0,13),(0,17),           # wrist → MCPs
    (5,8),(9,12),(13,16),(17,20),         # MCP → TIP (简化)
]


# ============================================================
#  HandInCupDetector
# ============================================================

class HandInCupDetector:
    """人手-杯把手重合检测器.

    使用 landmarks-in-bbox 计分制: 统计 MediaPipe 手部关键点中
    落在杯 bbox (外扩 margin) 内的个数, ≥2 分即判定为 hand_on_cup.
    """

    # ------------------------------------------------------------
    @staticmethod
    def _load_default_config():
        """从 EyeToHand/config.yaml 的 hand_in_cup 段读取所有默认参数."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(_cfg_path, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f)
        except Exception:
            _cfg = {}
        hic = _cfg.get("hand_in_cup", {})
        # 杯 ROI 回退 (顶层 key)
        _roi = _cfg.get("cup_roi_fallback", {})
        _roi_bbox = None
        if _roi.get("enabled", False):
            _roi_bbox = [
                int(_roi.get("x1", 0)), int(_roi.get("y1", 0)),
                int(_roi.get("x2", 0)), int(_roi.get("y2", 0)),
            ]
            if _roi_bbox[2] <= _roi_bbox[0] or _roi_bbox[3] <= _roi_bbox[1]:
                _roi_bbox = None
        return {
            "cup_model": hic.get("cup_model", os.path.join(_project_root, "model", "yolov8m.pt")),
            "hand_model": hic.get("hand_model", os.path.join(_project_root, "model", "hand_landmarker.task")),
            "conf_threshold": float(hic.get("conf_threshold", 0.5)),
            "hand_min_detection_conf": float(hic.get("hand_min_detection_conf", 0.5)),
            "hand_min_tracking_conf": float(hic.get("hand_min_tracking_conf", 0.5)),
            "required_stable": int(hic.get("required_stable", 5)),
            "dist_factor": float(hic.get("dist_factor", 1.2)),
            "bbox_margin": int(hic.get("bbox_margin", 15)),
            "curled_threshold": int(hic.get("curled_threshold", 25)),
            "curled_min_fingers": int(hic.get("curled_min_fingers", 2)),
            "cup_roi_fallback": _roi_bbox,
        }

    # ------------------------------------------------------------

    def __init__(self,
                 cup_model_path: str = "",
                 device: str = "cuda",
                 conf_threshold: float = 0.0,
                 required_stable: int = 0,
                 dist_factor: float = 0.0):
        """
        Args:
            cup_model_path:  YOLO 杯检测模型路径 (空=从 config 读).
            device:          "cuda" | "cpu".
            conf_threshold:  杯检测置信度阈值 (0=从 config 读).
            required_stable: 连续确认帧数 (0=从 config 读).
            dist_factor:     距离阈值因子 (保留, 未在计分制中使用).
        """
        cfg = self._load_default_config()
        if not cup_model_path:
            cup_model_path = cfg["cup_model"]

        self._conf_threshold = conf_threshold if conf_threshold > 0 else cfg["conf_threshold"]
        self._required_stable = required_stable if required_stable > 0 else cfg["required_stable"]
        self._dist_factor = dist_factor if dist_factor > 0 else cfg["dist_factor"]
        self._bbox_margin = cfg["bbox_margin"]
        self._curled_threshold = cfg["curled_threshold"]
        self._curled_min_fingers = cfg["curled_min_fingers"]
        self._cup_roi_fallback = cfg.get("cup_roi_fallback")
        self._stable_count = 0
        self._on_cup = False

        # Cup: YOLOv8m
        print(f"[HandInCup] Cup: {cup_model_path}")
        self._cup_model = YOLO(cup_model_path)
        self._cup_model.to(device)

        # Hand: MediaPipe Hands
        hand_model_path = cfg["hand_model"]
        print(f"[HandInCup] Hand: {hand_model_path}")
        _options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=cfg["hand_min_detection_conf"],
            min_tracking_confidence=cfg["hand_min_tracking_conf"],
        )
        self._hand_landmarker = HandLandmarker.create_from_options(_options)
        self._mp_timestamp = 0

        # 可视化用
        self._last_cup_bbox = None
        self._last_cup_center = None
        self._last_hands = []           # 每手 {handedness, conf, wrist, mcps, tips, finger_dists}
        self._last_best_hand = None     # 最近手 (同上)
        self._last_score = 0
        self._last_curled = 0           # 弯曲手指数

    def close(self):
        """释放 MediaPipe 资源."""
        if hasattr(self, "_hand_landmarker") and self._hand_landmarker is not None:
            try:
                self._hand_landmarker.close()
            except Exception:
                pass

    # ------------------------------------------------------------
    #  核心逻辑
    # ------------------------------------------------------------

    # ------------------------------------------------------------
    #  核心逻辑
    # ------------------------------------------------------------

    def update(self, color_image: np.ndarray):
        """每帧调用 — landmarks-in-bbox 计分制 (纯 RGB).

        Args:
            color_image: BGR 图像 (H, W, 3).
        """
        # 1. 检测杯 bbox
        cup_bbox, _ = self._detect_cup(color_image)
        self._last_cup_bbox = cup_bbox
        self._last_hands = []
        self._last_best_hand = None
        self._last_score = 0
        self._last_curled = 0
        cup_center = None
        hit = False

        if cup_bbox is not None:
            x1, y1, x2, y2 = cup_bbox
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            cup_center = (cx, cy)

            # 2. 检测手部 landmarks
            hands = self._detect_hands(color_image)
            self._last_hands = hands

            if hands:
                # 3. 找最近手 (按 wrist 到杯心 2D 距离)
                best_hand = min(hands, key=lambda h: (
                    (h["wrist"][0] - cx) ** 2 + (h["wrist"][1] - cy) ** 2
                ) ** 0.5)
                self._last_best_hand = best_hand

                # 4. 计分: 统计落在杯 bbox (外扩 margin) 内的 landmarks
                mx1 = max(0, x1 - self._bbox_margin)
                my1 = max(0, y1 - self._bbox_margin)
                mx2 = min(color_image.shape[1], x2 + self._bbox_margin)
                my2 = min(color_image.shape[0], y2 + self._bbox_margin)

                score = 0
                wx, wy = best_hand["wrist"]
                if mx1 <= wx <= mx2 and my1 <= wy <= my2:
                    score += 2  # wrist 权重 ×2
                for (px, py) in best_hand["mcps"]:
                    if mx1 <= px <= mx2 and my1 <= py <= my2:
                        score += 1
                for (px, py) in best_hand["tips"]:
                    if mx1 <= px <= mx2 and my1 <= py <= my2:
                        score += 1

                # 握拳检测: 几根手指弯曲 (MCP→TIP 距离 < 阈值)?
                curled_count = sum(1 for d in best_hand.get("finger_dists", []) if d < self._curled_threshold)
                wrist_in_bbox = (mx1 <= wx <= mx2 and my1 <= wy <= my2)

                self._last_score = score
                self._last_curled = curled_count

                # 判定: 手腕在 bbox 内 → 跳过握拳检测
                #       手腕不在 bbox 内: score≥5(多手指在框内)→信任; 低→curl兜底
                if wrist_in_bbox:
                    hit = (score >= 2)
                elif score >= 5:
                    hit = True
                else:
                    hit = (score >= 2) and (curled_count >= self._curled_min_fingers)

        self._last_cup_center = cup_center

        # 6. 累积
        if hit:
            self._stable_count += 1
        else:
            self._stable_count = 0
        self._on_cup = self._stable_count >= self._required_stable

    # ------------------------------------------------------------

    def _detect_cup(self, color_image):
        """YOLOv8m 检测水杯, 失败时回退到固定 ROI (cup_roi_fallback).

        Returns:
            (bbox, source) 或 (None, None).
        """
        results = self._cup_model(color_image, classes=[41], verbose=False)
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            best_conf = float(confs[best_idx])
            if best_conf >= self._conf_threshold:
                xyxy = boxes.xyxy.cpu().numpy()[best_idx]
                return [int(x) for x in xyxy], "yolo"

        # YOLO 未检测到 → 回退到固定 ROI
        if self._cup_roi_fallback is not None:
            return list(self._cup_roi_fallback), "fallback"
        return None, None

    # ------------------------------------------------------------

    def _detect_hands(self, color_image):
        """MediaPipe Hands 检测 → 返回每只手的 key landmarks.

        Returns:
            list of dict: [
              {"handedness": "Right", "conf": 1.0, "wrist": (x,y),
               "mcps": [(x,y),...], "tips": [(x,y),...], "landmark_count": 17},
              ...
            ], 按 conf 降序.
        """
        h, w = color_image.shape[:2]
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._mp_timestamp += 33  # ~30fps 帧间隔

        result = self._hand_landmarker.detect_for_video(mp_image, self._mp_timestamp)
        if not result.hand_landmarks:
            return []

        hands = []
        for i, hand_lms in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name if result.handedness else "?"
            conf = result.handedness[i][0].score if result.handedness else 0.0

            def _to_px(lm):
                return (int(lm.x * w), int(lm.y * h))

            wrist = _to_px(hand_lms[MP_WRIST])
            mcps = [_to_px(hand_lms[idx]) for idx in MP_MCP]
            tips = [_to_px(hand_lms[idx]) for idx in MP_TIP]
            # MCP→TIP 像素距离 (4 根手指: INDEX, MIDDLE, RING, PINKY)
            _mcp_ids = list(MP_MCP.keys())   # [5, 9, 13, 17]
            _tip_ids = list(MP_TIP.keys())   # [4, 8, 12, 16, 20]
            # finger pairs: INDEX(5→8), MIDDLE(9→12), RING(13→16), PINKY(17→20)
            finger_dists = []
            for mcp_id, tip_id in [(_mcp_ids[0], _tip_ids[1]), (_mcp_ids[1], _tip_ids[2]),
                                    (_mcp_ids[2], _tip_ids[3]), (_mcp_ids[3], _tip_ids[4])]:
                mx, my = _to_px(hand_lms[mcp_id])
                tx, ty = _to_px(hand_lms[tip_id])
                d = ((mx - tx) ** 2 + (my - ty) ** 2) ** 0.5
                finger_dists.append(d)

            hands.append({
                "handedness": handedness,
                "conf": conf,
                "wrist": wrist,
                "mcps": mcps,
                "tips": tips,
                "finger_dists": finger_dists,  # [d_index, d_middle, d_ring, d_pinky]
            })
        hands.sort(key=lambda h: -h["conf"])
        return hands

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
        self._last_hands = []
        self._last_best_hand = None
        self._last_score = 0
        self._last_curled = 0

    # ------------------------------------------------------------
    #  可视化绘制 (复用)
    # ------------------------------------------------------------

    def draw_visualization(self, color_image: np.ndarray) -> np.ndarray:
        """在图像上绘制检测结果可视化 (MediaPipe landmarks + cup bbox).

        Returns:
            带标注的 BGR 图像副本.
        """
        display = color_image.copy()

        # ---- 状态行 ----
        status = "ON CUP" if self._on_cup else f"wait... ({self._stable_count}/{self._required_stable})"
        status_color = (0, 255, 0) if self._on_cup else (0, 165, 255)
        cv2.putText(display, f"HandInCup: {status}  score={self._last_score} curled={self._last_curled}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

        # ---- 杯 bbox (紫) + expanded bbox (紫虚线) ----
        bbox = self._last_cup_bbox
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display, "cup", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            # expanded bbox
            m = self._bbox_margin
            ex1, ey1 = max(0, x1 - m), max(0, y1 - m)
            ex2 = min(color_image.shape[1], x2 + m)
            ey2 = min(color_image.shape[0], y2 + m)
            cv2.rectangle(display, (ex1, ey1), (ex2, ey2), (255, 128, 255), 1, cv2.LINE_AA)

            # 杯心
            cx, cy = self._last_cup_center or ((x1 + x2) // 2, (y1 + y2) // 2)
            cv2.drawMarker(display, (cx, cy), (255, 0, 255), cv2.MARKER_CROSS, 10, 1)

        # ---- 最近手 landmarks (MediaPipe) ----
        best = self._last_best_hand
        if best is not None:
            # 骨架连线 (绿)
            for a, b in HAND_BONES:
                pts = {
                    MP_WRIST: best["wrist"],
                    **{k: best["mcps"][i] for i, k in enumerate(MP_MCP)},
                    **{k: best["tips"][i] for i, k in enumerate(MP_TIP)},
                }
                if a in pts and b in pts:
                    cv2.line(display, pts[a], pts[b], (0, 255, 0), 1, cv2.LINE_AA)

            # 手腕 (大红点)
            wx, wy = best["wrist"]
            cv2.circle(display, (wx, wy), 7, (0, 0, 255), -1)
            cv2.putText(display, "wrist", (wx + 5, wy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # MCPs (黄/灰 — 在 bbox 内=绿)
            def _in_expanded(px, py):
                if bbox is None:
                    return False
                m = self._bbox_margin
                return (bbox[0]-m) <= px <= (bbox[2]+m) and (bbox[1]-m) <= py <= (bbox[3]+m)

            for px, py in best["mcps"]:
                in_bbox = _in_expanded(px, py) if bbox else False
                color = (0, 255, 0) if in_bbox else (128, 128, 128)
                cv2.circle(display, (px, py), 4, color, -1)

            # TIPs (绿/灰)
            for px, py in best["tips"]:
                in_bbox = _in_expanded(px, py) if bbox else False
                color = (0, 255, 0) if in_bbox else (128, 128, 128)
                cv2.circle(display, (px, py), 3, color, -1)

            # handedness + conf + finger dists
            fd = best.get("finger_dists", [])
            fd_str = " ".join(f"{d:.0f}" for d in fd)
            cv2.putText(display, f"{best['handedness']}({best['conf']:.2f}) fingers:[{fd_str}]px",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

        return display

    # ------------------------------------------------------------
    #  图片路径收集
    # ------------------------------------------------------------

    @staticmethod
    def _collect_image_paths(input_path: str) -> list:
        """收集图片路径: 支持单个文件或文件夹.

        支持的扩展名: .jpg .jpeg .png .bmp .tiff .webp
        """
        _exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}
        if os.path.isfile(input_path):
            return [input_path]
        if os.path.isdir(input_path):
            paths = []
            for fname in sorted(os.listdir(input_path)):
                ext = os.path.splitext(fname)[1].lower()
                if ext in _exts:
                    paths.append(os.path.join(input_path, fname))
            return paths
        print(f"[WARN] 路径不存在: {input_path}")
        return []

    # ------------------------------------------------------------
    #  诊断核心 (帧级, 供图片和实时复用)
    # ------------------------------------------------------------

    def _diagnose_frame(self, color_image: np.ndarray, name: str = "",
                        log_file: str = ""):
        """对单帧运行诊断, 输出到日志文件 (避免控制台刷屏).

        Args:
            color_image: BGR 图像.
            name:        帧标识 (用于日志).
            log_file:    日志路径, 为空则自动生成.

        Returns:
            带标注的 BGR 图像.
        """
        import datetime
        if not log_file:
            log_file = os.path.join(_project_root, "logs",
                                    f"handincup_diag_{datetime.date.today():%Y%m%d}.log")

        # 确保 logs 目录存在
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        h, w = color_image.shape[:2]
        lines = []
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        lines.append(f"\n{'='*60}")
        lines.append(f"  [{ts}] HandInCup 诊断: {name}  ({w}x{h})")
        lines.append(f"{'='*60}")

        # Cup 模型 — class 41
        lines.append(f"\n--- Cup (class=41, conf>={self._conf_threshold}) ---")
        results = self._cup_model(color_image, classes=[41], verbose=False)
        cup_bboxes = []
        if results and len(results[0].boxes) > 0:
            for i in range(len(results[0].boxes)):
                conf = float(results[0].boxes.conf[i])
                xyxy = [int(x) for x in results[0].boxes.xyxy[i].cpu().numpy()]
                cup_bboxes.append((conf, xyxy))
                lines.append(f"  [{i}] bbox={xyxy}  conf={conf:.3f}")
        if not cup_bboxes:
            lines.append("  (无)")

        # Cup 模型 — 所有类别低阈值
        results_all = self._cup_model(color_image, verbose=False)
        if results_all and len(results_all[0].boxes) > 0:
            names = results_all[0].names
            found = []
            for i in range(len(results_all[0].boxes)):
                cls_id = int(results_all[0].boxes.cls[i])
                conf = float(results_all[0].boxes.conf[i])
                if conf >= 0.3:
                    name_str = names.get(cls_id, str(cls_id))
                    xyxy = [int(x) for x in results_all[0].boxes.xyxy[i].cpu().numpy()]
                    found.append((name_str, cls_id, conf, xyxy))
            if found:
                lines.append("\n  全部类别 (conf>=0.3):")
                for n, cid, conf, xyxy in sorted(found, key=lambda x: -x[2]):
                    lines.append(f"  class={cid}({n}) conf={conf:.3f}  bbox={xyxy}")

        # Hand 模型 (MediaPipe)
        lines.append("\n--- MediaPipe Hands ---")
        hands = self._detect_hands(color_image)
        if hands:
            for i, h in enumerate(hands):
                wx, wy = h["wrist"]
                lines.append(f"  [{i}] {h['handedness']}({h['conf']:.2f}) "
                             f"wrist=({wx},{wy})  mcps={len(h['mcps'])} tips={len(h['tips'])}")
        else:
            lines.append("  (无)")

        # ---- HandInCup 判定分析 ----
        lines.append("\n--- HandInCup 判定 ---")
        if cup_bboxes and hands:
            best_cup = max(cup_bboxes, key=lambda x: x[0])
            best_cup_conf, (x1,y1,x2,y2) = best_cup
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bbox_diag = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            lines.append(f"  杯心=({cx},{cy})  bbox={x2-x1}x{y2-y1}  对角线={bbox_diag:.0f}px")
            lines.append(f"  bbox_margin={self._bbox_margin}  score_thresh=2")

            # 找最近手并计分
            best_hand = min(hands, key=lambda h: (
                (h["wrist"][0]-cx)**2 + (h["wrist"][1]-cy)**2)**0.5)
            wx, wy = best_hand["wrist"]
            m = self._bbox_margin
            mx1, my1 = max(0, x1-m), max(0, y1-m)
            mx2, my2 = min(w, x2+m), min(h, y2+m)

            score = 0
            if mx1 <= wx <= mx2 and my1 <= wy <= my2:
                score += 2
                lines.append(f"  wrist({wx},{wy}) in bbox → +2")
            else:
                lines.append(f"  wrist({wx},{wy}) OUT")
            for (px, py) in best_hand["mcps"]:
                if mx1 <= px <= mx2 and my1 <= py <= my2:
                    score += 1
                    lines.append(f"  mcp({px},{py}) in bbox → +1")
            for (px, py) in best_hand["tips"]:
                if mx1 <= px <= mx2 and my1 <= py <= my2:
                    score += 1
                    lines.append(f"  tip({px},{py}) in bbox → +1")
            hit = score >= 2
            lines.append(f"  score={score} → {'✓ HIT' if hit else '✗ MISS'}")
        else:
            lines.append(f"  cup={'✓' if cup_bboxes else '✗'}  hands={'✓' if hands else '✗'} — 无法判定")

        # 写入日志
        text = "\n".join(lines)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[DIAG] 诊断结果 → {log_file}")

        # 可视化: 复用 draw_visualization + 添加 YOLO cup 标注
        display = self.draw_visualization(color_image)
        if cup_bboxes:
            best = max(cup_bboxes, key=lambda x: x[0])
            x1, y1, x2, y2 = best[1]
            cv2.putText(display, f"cup {best[0]:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 3)
        return display

    # ------------------------------------------------------------
    #  诊断模式 — 图片
    # ------------------------------------------------------------

    def diagnose_image(self, image_path: str):
        """对单张图片运行诊断."""
        if not os.path.exists(image_path):
            print(f"[ERROR] 图片不存在: {image_path}")
            return
        color = cv2.imread(image_path)
        if color is None:
            print(f"[ERROR] 无法读取: {image_path}")
            return
        display = self._diagnose_frame(color, os.path.basename(image_path))
        cv2.imshow("HandInCup - Diagnose", display)
        print("[INFO] 按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def diagnose_images(self, paths: list):
        """对多张图片依次诊断."""
        for p in paths:
            self.diagnose_image(p)

    # ------------------------------------------------------------
    #  诊断模式 — 实时 (Camera B)
    # ------------------------------------------------------------

    def diagnose_live(self):
        """实时诊断模式: UGREEN Camera, 按 'd' 诊断当前帧, 'q' 退出."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(_cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)

        _ugreen = _cfg.get("ugreen", {})
        cam = UGreenCamera(
            camera_id=_ugreen.get("camera_id", 1),
            width=_ugreen.get("width", 1920),
            height=_ugreen.get("height", 1080),
            fps=_ugreen.get("fps", 30),
        )
        print("=" * 55)
        print("  HandInCup 实时诊断 (纯 RGB)")
        print("  按 'd' 诊断当前帧 | 'q' 退出")
        print("=" * 55)

        try:
            while True:
                color = cam.read()[0]
                if color is None:
                    continue
                cv2.imshow("HandInCup - Live", color)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    display = self._diagnose_frame(color, f"live_{int(time.time())}")
                    cv2.imshow("HandInCup - Diagnose", display)
                    print("[INFO] 按任意键返回实时画面...")
                    cv2.waitKey(0)
                    cv2.destroyWindow("HandInCup - Diagnose")
        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            cam.release()
            cv2.destroyAllWindows()
            print("[INFO] 已退出")

    # ------------------------------------------------------------
    #  静态图片测试
    # ------------------------------------------------------------

    def test_image(self, image_path: str, output_path: str = "",
                   required_stable_override: int = 1):
        """对单张图片运行检测并显示/保存结果.

        因为图片是单帧, 无连续帧概念, 默认 required_stable=1
        (检测到即为判定结果).

        Args:
            image_path:  输入图片路径.
            output_path: 输出图片路径 (为空则仅显示窗口).
            required_stable_override: 覆盖连续帧数 (默认 1).
        """
        if not os.path.exists(image_path):
            print(f"[ERROR] 图片不存在: {image_path}")
            return

        # 覆盖连续帧参数 (单帧模式)
        original_stable = self._required_stable
        self._required_stable = required_stable_override

        color = cv2.imread(image_path)
        if color is None:
            print(f"[ERROR] 无法读取图片: {image_path}")
            self._required_stable = original_stable
            return

        print(f"[INFO] 检测图片: {image_path}  ({color.shape[1]}x{color.shape[0]})")

        # 运行检测 (无深度)
        self.update(color)
        display = self.draw_visualization(color)

        # 输出判定
        verdict = "TRUE  (人手在杯上)" if self._on_cup else "FALSE (无人手/不在杯上)"
        print(f"[RESULT] HandInCup = {verdict}")
        print(f"         cup_bbox={self._last_cup_bbox}")
        print(f"         score={self._last_score}")
        if self._last_best_hand:
            h = self._last_best_hand
            print(f"         best_hand={h['handedness']}({h['conf']:.2f}) wrist={h['wrist']} mcps={len(h['mcps'])} tips={len(h['tips'])}")

        # 恢复
        self._required_stable = original_stable

        if output_path:
            cv2.imwrite(output_path, display)
            print(f"[SAVE] {output_path}")

        cv2.imshow("HandInCup - Image Test", display)
        print("[INFO] 按任意键关闭窗口...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def test_images(self, image_paths: list, output_dir: str = ""):
        """对多张图片依次运行检测.

        Args:
            image_paths: 图片路径列表.
            output_dir:  输出目录 (为空则仅显示).
        """
        for path in image_paths:
            fname = os.path.basename(path)
            out = os.path.join(output_dir, f"handincup_{fname}") if output_dir else ""
            self.test_image(path, out)
            self.reset()  # 重置状态, 每张图独立判定

    # ------------------------------------------------------------
    #  交互式实时测试
    # ------------------------------------------------------------

    def run(self):
        """交互式测试主循环 (纯 RGB, UGREEN Camera)."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(_cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)

        _ugreen = _cfg.get("ugreen", {})
        cam = UGreenCamera(
            camera_id=_ugreen.get("camera_id", 1),
            width=_ugreen.get("width", 1920),
            height=_ugreen.get("height", 1080),
            fps=_ugreen.get("fps", 30),
        )
        print("=" * 55)
        print("  Hand-in-cup 检测测试 (纯 RGB)")
        print("  YOLOv8m (cup) + MediaPipe Hands (21 landmarks)")
        print("  landmarks-in-bbox 计分制")
        print("  按 'q' 退出")
        print("=" * 55)

        try:
            while True:
                color = cam.read()[0]
                if color is None:
                    continue

                self.update(color)
                display = self.draw_visualization(color)

                cv2.imshow("HandInCup Test", display)

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
            cam.release()
            cv2.destroyAllWindows()
            print("[INFO] 已退出")


# ============================================================
#  主入口
# ============================================================

# ============================================================
#  杯 ROI 标定 (鼠标框选 → 写入 config.yaml)
# ============================================================

def calibrate_cup_roi():
    """UGREEN Camera 实时画面 + 鼠标框选杯区域.

    操作:
      鼠标拖拽框选 ROI → 按 s 保存 → 按 r 重选 → 按 q 退出.
    保存内容:
      - cup_roi_fallback: 框选的 bbox
      (纯 RGB, 不测量深度 — cup_depth_expected 需手动填入 config)
    """
    _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(_cfg_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    _ugreen = _cfg.get("ugreen", {})
    cam = UGreenCamera(
        camera_id=_ugreen.get("camera_id", 1),
        width=_ugreen.get("width", 1920),
        height=_ugreen.get("height", 1080),
        fps=_ugreen.get("fps", 30),
    )

    roi = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "drawing": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            roi["drawing"] = True
            roi["x1"], roi["y1"] = x, y
            roi["x2"], roi["y2"] = x, y
        elif event == cv2.EVENT_MOUSEMOVE:
            if roi["drawing"]:
                roi["x2"], roi["y2"] = x, y
        elif event == cv2.EVENT_LBUTTONUP:
            roi["drawing"] = False
            roi["x2"], roi["y2"] = x, y

    cv2.namedWindow("Cup ROI Calibration")
    cv2.setMouseCallback("Cup ROI Calibration", on_mouse)

    print("=" * 55)
    print("  杯 ROI 标定 (纯 RGB)")
    print("  鼠标拖拽框选杯区域 → s 保存 → r 重选 → q 退出")
    print("=" * 55)

    saved = False
    try:
        while True:
            color = cam.read()[0]
            if color is None:
                continue
            h, w = color.shape[:2]
            display = color.copy()

            x1, y1 = roi["x1"], roi["y1"]
            x2, y2 = roi["x2"], roi["y2"]
            if x1 != x2 or y1 != y2:
                rx1, ry1 = min(x1, x2), min(y1, y2)
                rx2, ry2 = max(x1, x2), max(y1, y2)
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

                cx, cy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
                cv2.drawMarker(display, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 8, 1)
                cv2.putText(display,
                            f"ROI:({rx1},{ry1})-({rx2},{ry2}) {rx2-rx1}x{ry2-ry1}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                cv2.putText(display, "Drag mouse to select cup ROI", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

            cv2.putText(display, "s:save  r:reset  q:quit", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 128, 128), 1)

            cv2.imshow("Cup ROI Calibration", display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                roi["x1"] = roi["y1"] = roi["x2"] = roi["y2"] = 0
                print("[INFO] ROI 已重置")
            elif key == ord("s"):
                rx1, ry1 = min(roi["x1"], roi["x2"]), min(roi["y1"], roi["y2"])
                rx2, ry2 = max(roi["x1"], roi["x2"]), max(roi["y1"], roi["y2"])
                if rx1 == rx2 or ry1 == ry2:
                    print("[WARN] 未框选 ROI, 请先用鼠标拖拽框选杯区域")
                    continue

                # 写入 config.yaml (仅 bbox, 无深度)
                _cfg["cup_roi_fallback"] = {
                    "enabled": True,
                    "x1": rx1, "y1": ry1, "x2": rx2, "y2": ry2,
                }
                with open(_cfg_path, "w", encoding="utf-8") as f:
                    yaml.dump(_cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                print(f"[SAVE] cup_roi_fallback: ({rx1},{ry1})-({rx2},{ry2})")
                print(f"[SAVE] → {_cfg_path}")
                saved = True
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam.release()
        cv2.destroyAllWindows()
        if saved:
            print("[INFO] 标定完成!")
        else:
            print("[INFO] 已退出 (未保存)")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--calibrate-roi":
        calibrate_cup_roi()
        sys.exit(0)

    detector = HandInCupDetector()

    if len(sys.argv) > 1:
        args = sys.argv[1:]
        mode = "test"  # default

        # 解析 mode flag
        if args[0] == "--diagnose":
            mode = "diagnose"
            args = args[1:]
        elif args[0] == "--diagnose-live":
            detector.diagnose_live()
            sys.exit(0)

        if not args:
            print("用法:")
            print("  杯ROI标定:    python hand_in_cup.py --calibrate-roi")
            print("  实时测试:     python hand_in_cup.py")
            print("  图片测试:     python hand_in_cup.py <image_or_folder> ...")
            print("  图片诊断:     python hand_in_cup.py --diagnose <image_or_folder> ...")
            print("  实时诊断:     python hand_in_cup.py --diagnose-live")
            sys.exit(1)

        # 收集所有图片 (支持文件+文件夹混用)
        all_paths = []
        for a in args:
            all_paths.extend(detector._collect_image_paths(a))
        if not all_paths:
            print("[ERROR] 未找到任何图片")
            sys.exit(1)

        print(f"[INFO] {'诊断' if mode == 'diagnose' else '测试'}模式: {len(all_paths)} 张图片")
        if mode == "diagnose":
            detector.diagnose_images(all_paths)
        else:
            detector.test_images(all_paths)
    else:
        # 实时模式: 连接 Camera B
        detector.run()
