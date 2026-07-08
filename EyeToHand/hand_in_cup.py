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
import time

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
    def _load_default_config():
        """从 EyeToHand/config.yaml 的 hand_in_cup 段读取所有默认参数."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        try:
            with open(_cfg_path, "r", encoding="utf-8") as f:
                _cfg = yaml.safe_load(f)
        except Exception:
            _cfg = {}
        hic = _cfg.get("hand_in_cup", {})
        return {
            "cup_model": hic.get("cup_model", os.path.join(_project_root, "model", "yolov8m.pt")),
            "pose_model": hic.get("pose_model", os.path.join(_project_root, "model", "yolov8n-pose.pt")),
            "conf_threshold": float(hic.get("conf_threshold", 0.5)),
            "required_stable": int(hic.get("required_stable", 5)),
            "dist_factor": float(hic.get("dist_factor", 1.2)),
            "depth_tolerance": float(hic.get("depth_tolerance", 0.08)),
        }

    # ------------------------------------------------------------

    def __init__(self,
                 cup_model_path: str = "",
                 pose_model_path: str = "",
                 device: str = "cuda",
                 conf_threshold: float = 0.0,
                 required_stable: int = 0,
                 dist_factor: float = 0.0,
                 depth_tolerance: float = 0.0):
        """
        Args:
            cup_model_path:  YOLO 杯检测模型路径 (空=从 config 读).
            pose_model_path: YOLO-Pose 模型路径 (空=从 config 读).
            device:          "cuda" | "cpu".
            conf_threshold:  检测置信度阈值 (0=从 config 读).
            required_stable: 连续确认帧数 (0=从 config 读).
            dist_factor:     距离阈值 = bbox 对角线 × dist_factor (0=从 config 读).
            depth_tolerance: 手腕-杯心深度差阈值 (米) (0=从 config 读).
        """
        cfg = self._load_default_config()
        if not cup_model_path:
            cup_model_path = cfg["cup_model"]
        if not pose_model_path:
            pose_model_path = cfg["pose_model"]

        self._conf_threshold = conf_threshold if conf_threshold > 0 else cfg["conf_threshold"]
        self._required_stable = required_stable if required_stable > 0 else cfg["required_stable"]
        self._dist_factor = dist_factor if dist_factor > 0 else cfg["dist_factor"]
        self._depth_tolerance = depth_tolerance if depth_tolerance > 0 else cfg["depth_tolerance"]
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
    #  可视化绘制 (复用)
    # ------------------------------------------------------------

    def draw_visualization(self, color_image: np.ndarray) -> np.ndarray:
        """在图像上绘制检测结果可视化.

        Returns:
            带标注的 BGR 图像副本.
        """
        display = color_image.copy()

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
                        depth_frame=None, log_file: str = ""):
        """对单帧运行诊断, 输出到日志文件 (避免控制台刷屏).

        Args:
            color_image: BGR 图像.
            name:        帧标识 (用于日志).
            depth_frame: rs.depth_frame (可选, 用于深度诊断).
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
        lines.append(f"  depth_frame={'OK' if depth_frame is not None else 'None'}")
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

        # Pose 模型
        lines.append("\n--- Pose ---")
        ks_names = {5:"L_SHOU",6:"R_SHOU",7:"L_ELBOW",8:"R_ELBOW",9:"L_WRIST",10:"R_WRIST"}
        results = self._pose_model(color_image, verbose=False)
        pose_kpts = None
        if results and results[0].keypoints is not None:
            kpts = results[0].keypoints
            if kpts.conf is not None:
                pose_kpts = kpts
                n_p, n_k = kpts.conf.shape[0], kpts.conf.shape[1]
                lines.append(f"  检测到 {n_p} 人, {n_k} 关键点")
                for p in range(n_p):
                    pconf = kpts.conf[p].cpu().numpy()
                    best_k = pconf.argmax()
                    lines.append(f"\n  Person {p}: best={ks_names.get(best_k,str(best_k))} conf={pconf[best_k]:.3f}")
                    for idx in [5,6,7,8,9,10]:
                        if idx < n_k:
                            xy = kpts.xy[p,idx].cpu().numpy()
                            lines.append(f"    {ks_names.get(idx,str(idx)):8s} conf={pconf[idx]:.3f}  xy=({int(xy[0]):4d},{int(xy[1]):4d})")
        if pose_kpts is None:
            lines.append("  (无)")

        # ---- HandInCup 判定分析 ----
        lines.append("\n--- HandInCup 判定 ---")
        if cup_bboxes and pose_kpts is not None:
            best_cup_conf, (x1,y1,x2,y2) = max(cup_bboxes, key=lambda x: x[0])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            bbox_diag = ((x2-x1)**2 + (y2-y1)**2) ** 0.5
            dist_thresh = bbox_diag * self._dist_factor
            lines.append(f"  杯心=({cx},{cy})  bbox={x2-x1}x{y2-y1}  对角线={bbox_diag:.0f}px  阈值={dist_thresh:.0f}px")

            wrists = []
            for p in range(pose_kpts.conf.shape[0]):
                for idx in [9, 10]:
                    if float(pose_kpts.conf[p,idx]) >= self._conf_threshold:
                        xy = pose_kpts.xy[p,idx].cpu().numpy()
                        d = ((xy[0]-cx)**2 + (xy[1]-cy)**2) ** 0.5
                        wrists.append((p, idx, int(xy[0]), int(xy[1]), d))
            if wrists:
                for p, idx, wx, wy, d in wrists:
                    hit = "✓ HIT" if d < dist_thresh else "✗ MISS"
                    lines.append(f"  Person{p} {'L_WRIST' if idx==9 else 'R_WRIST'} ({wx},{wy}) → {d:.0f}px  [{hit}]")

            if depth_frame is not None:
                cup_depths = []
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        py, px = cy+dy, cx+dx
                        if 0 <= py < h and 0 <= px < w:
                            d = depth_frame.get_distance(px, py)
                            if 0 < d < 2.0:
                                cup_depths.append(d)
                if cup_depths:
                    cup_z = np.median(cup_depths)
                    lines.append(f"  杯深度: {cup_z:.3f}m (中值 {len(cup_depths)} 点)")
                    for p, idx, wx, wy, _ in wrists:
                        wz = depth_frame.get_distance(wx, wy)
                        ok = "OK" if 0 < wz < 2.0 and abs(wz - cup_z) < self._depth_tolerance else "FAIL"
                        lines.append(f"  手腕深度: {wz:.3f}m  diff={abs(wz-cup_z):.3f}m  [{ok}]")
                else:
                    lines.append("  ⚠️ 杯深度无效 (杯心 5x5 区域无有效深度值)")
            else:
                lines.append("  depth_frame=None (仅2D判定)")
        else:
            lines.append(f"  cup={'✓' if cup_bboxes else '✗'}  pose={'✓' if pose_kpts else '✗'} — 无法判定")

        # 写入日志
        text = "\n".join(lines)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"[DIAG] 诊断结果 → {log_file}")

        # 可视化
        display = color_image.copy()
        if cup_bboxes:
            best = max(cup_bboxes, key=lambda x: x[0])
            x1,y1,x2,y2 = best[1]
            cv2.rectangle(display, (x1,y1), (x2,y2), (255,0,255), 3)
            cv2.putText(display, f"cup {best[0]:.2f}", (x1,y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
        if pose_kpts is not None:
            for p in range(pose_kpts.conf.shape[0]):
                for idx in range(min(pose_kpts.conf.shape[1], 17)):
                    if float(pose_kpts.conf[p,idx]) >= 0.3:
                        xy = pose_kpts.xy[p,idx].cpu().numpy()
                        cv2.drawMarker(display, (int(xy[0]),int(xy[1])),
                                     (0,255,255), cv2.MARKER_CROSS, 8, 1)
                        cv2.putText(display, str(idx), (int(xy[0])+3,int(xy[1])-3),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,255,255), 1)
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
        """实时诊断模式: 连接 Camera B, 按 'd' 诊断当前帧, 'q' 退出."""
        _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        with open(_cfg_path, "r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)

        class _DiagRunner(BaseRealSenseDetector):
            USE_YOLO_WORLD = False; USE_SAM = False
            LOG_NAME = "hand_in_cup_diag.log"; TARGET_CLASS_ID = 0

        runner = _DiagRunner(config_dict=_cfg)
        print("=" * 55)
        print("  HandInCup 实时诊断")
        print("  按 'd' 诊断当前帧 | 'q' 退出")
        print("=" * 55)

        try:
            while True:
                color, depth_img, depth_cmap, depth_frame = runner.get_frames()
                if color is None:
                    continue
                cv2.imshow("HandInCup - Live", color)
                key = cv2.waitKey(30) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("d"):
                    display = self._diagnose_frame(color, f"live_{int(time.time())}",
                                                   depth_frame=depth_frame)
                    cv2.imshow("HandInCup - Diagnose", display)
                    print("[INFO] 按任意键返回实时画面...")
                    cv2.waitKey(0)
                    cv2.destroyWindow("HandInCup - Diagnose")
        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            runner.stop()
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
        self.update(color, depth_frame=None)
        display = self.draw_visualization(color)

        # 输出判定
        verdict = "TRUE  (人手在杯上)" if self._on_cup else "FALSE (无人手/不在杯上)"
        print(f"[RESULT] HandInCup = {verdict}")
        print(f"         cup_bbox={self._last_cup_bbox}")
        print(f"         wrists={self._last_wrists}")
        print(f"         distances={[f'{d:.0f}px' for d in self._last_distances]}")

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
                display = self.draw_visualization(color)

                # ---- 深度状态 (仅实时模式) ----
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
            print("  实时测试:   python hand_in_cup.py")
            print("  图片测试:   python hand_in_cup.py <image_or_folder> ...")
            print("  图片诊断:   python hand_in_cup.py --diagnose <image_or_folder> ...")
            print("  实时诊断:   python hand_in_cup.py --diagnose-live")
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
