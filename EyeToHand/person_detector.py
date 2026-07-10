# -*- coding: utf-8 -*-
"""
EyeToHand 人体位置检测
=======================
使用标准 YOLOv8m COCO 模型检测人体 (class_id=0),
输出 bbox 中心 3D 相机坐标.

检测流程:
  1. YOLO 检测 person bbox (取置信度最高者)
  2. bbox 中心彩色像素 → 映射到深度图坐标
  3. 深度局部中值滤波
  4. 彩色内参反投影 → 相机系 3D (x_cam, y_cam, z_cam)

使用方法:
  python EyeToHand/person_detector.py
  按 'q' 退出

参考:
  - 继承模式: RealSense/test_3D.py (直接继承 BaseRealSenseDetector)
  - 配置入口: EyeToHand/config.yaml
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np

from RealSense.realsense_shared import BaseRealSenseDetector, PositionSmoother


# ============================================================
#  1. PersonStabilizer (参考 estun_handle_cup_run.py PoseStabilizer)
# ============================================================

class PersonStabilizer:
    """
    人体 3D 位置稳定器.

    缓存最近 N 帧位置数据; 当位置变化低于阈值时, 认为稳定,
    返回缓冲区均值作为稳定结果.

    与 PoseStabilizer 的差异:
      - 仅稳定 3D 位置 (x, y, z), 无角度
      - 无代际计数器 (人体检测不涉及 TCP 连接周期)
      - 阈值较大 (人体检测精度低于物体 6D 位姿)
    """

    def __init__(self, window=10, pos_thresh=0.03):
        """
        参数:
            window      : int, 缓冲区帧数 (默认 10, ~0.3s@30fps)
            pos_thresh  : float, 位置波动阈值 (米), 默认 0.03m (3cm)
        """
        self.window = window
        self.pos_thresh = pos_thresh
        self._buffer = []           # [(x, y, z), ...]

    def add(self, pos):
        """添加一帧位置到缓冲区."""
        if pos is None:
            return
        self._buffer.append(tuple(pos))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self):
        """
        判断位置是否稳定 (缓冲区满 + 各轴波动 < 阈值).

        返回:
            bool
        """
        if len(self._buffer) < self.window:
            return False
        arr = np.array(self._buffer)          # (N, 3)
        pos_range = np.ptp(arr, axis=0).max()  # 三轴中最大峰峰值
        return pos_range < self.pos_thresh

    def get_mean(self):
        """
        返回缓冲区均值.

        返回:
            (x, y, z) 或 None (缓冲区为空时)
        """
        if not self._buffer:
            return None
        arr = np.array(self._buffer)
        return tuple(arr.mean(axis=0))

    def reset(self):
        """清空缓冲区."""
        self._buffer = []

    @property
    def count(self):
        """当前缓冲区帧数."""
        return len(self._buffer)


# ============================================================
#  2. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  3. PersonDetector
# ============================================================

class PersonDetector(BaseRealSenseDetector):
    """
    人体位置检测器.

    直接继承 BaseRealSenseDetector, 与 test_3D.py 的 RealSenseYOLO3DDetector 模式一致.
    使用标准 YOLO COCO person class (class_id=0).
    """

    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "person_detect.log"
    TARGET_CLASS_ID = 0             # COCO person
    CHAIR_CLASS_ID = 56             # COCO chair

    def __init__(self, config_dict=None, serial_number=None):
        super().__init__(config_dict=config_dict, serial_number=serial_number)

        # EMA 平滑器 (实时显示用, 使用基类 _apply_config 已设置的 SMOOTH_ALPHA)
        self._smoother = PositionSmoother(alpha=self.SMOOTH_ALPHA)

        # 窗口稳定器 (参数从 config.yaml stabilizer 段读取)
        _stab_cfg = _MODULE_CFG.get("stabilizer", {})
        self._stabilizer = PersonStabilizer(
            window=_stab_cfg.get("window", 10),
            pos_thresh=_stab_cfg.get("pos_thresh", 0.03),
        )

        # ---- 椅子标定 + 人就位检测 ----
        _chair_cfg = _MODULE_CFG.get("chair", {})
        self._chair_conf_threshold = float(_chair_cfg.get("conf_threshold", 0.4))
        self._chair_stabilize_frames = int(_chair_cfg.get("stabilize_frames", 10))
        self._seated_overlap_threshold = float(_chair_cfg.get("overlap_threshold", 0.4))
        self._seated_stable_required = int(_chair_cfg.get("seated_stable_frames", 10))

        self._chair_bbox = None           # [x1, y1, x2, y2] 标定后固定
        self._chair_calibrated = False
        self._chair_buffer = []           # 椅子 bbox 稳定缓冲区
        self._seated_stable_count = 0
        self._is_seated = False

    # ------------------------------------------------------------
    def get_person_3d(self, bbox, depth_frame):
        """
        bbox 中心像素反投影到相机系 3D 坐标.

        参数:
            bbox        : [x1, y1, x2, y2]
            depth_frame : rs.depth_frame

        返回:
            (x, y, z) 或 None (映射失败时)
        """
        cx = (int(bbox[0]) + int(bbox[2])) // 2
        cy = (int(bbox[1]) + int(bbox[3])) // 2
        return self._color_pixel_to_3d(cx, cy, depth_frame)

    # ------------------------------------------------------------
    def detect_person(self, color_image, depth_frame):
        """
        检测人体: YOLO bbox + bbox 中心 3D 相机坐标.

        参数:
            color_image : np.ndarray (H, W, 3), BGR 图像
            depth_frame : rs.depth_frame

        返回:
            dict 或 None:
              {
                  "bbox":         [x1, y1, x2, y2],
                  "conf":         float,
                  "pos_3d_cam":   (x, y, z) 或 None,
              }
        """
        detection = self._detect_targets(color_image)
        if detection is None:
            return None

        bbox = detection["bbox"]
        pos_3d = self.get_person_3d(bbox, depth_frame)

        return {
            "bbox": bbox,
            "conf": detection["conf"],
            "pos_3d_cam": pos_3d,
        }

    # ------------------------------------------------------------
    def detect_person_stable(self, color_image, depth_frame):
        """
        检测人体, 经 PersonStabilizer 稳定后返回.

        每帧调用, 内部累积到 _stabilizer 缓冲区.
        仅在稳定时返回均值, 否则返回 None.

        返回:
            dict 或 None:
              {
                  "bbox":         [x1, y1, x2, y2],
                  "conf":         float,
                  "pos_3d_cam":   (x_s, y_s, z_s),  # 稳定均值
                  "pos_raw":      (x, y, z),          # 当前帧原始值
              }
        """
        result = self.detect_person(color_image, depth_frame)
        if result is None or result["pos_3d_cam"] is None:
            self._stabilizer.reset()
            return None

        pos_raw = result["pos_3d_cam"]
        self._stabilizer.add(pos_raw)

        if not self._stabilizer.is_stable():
            return None

        pos_stable = self._stabilizer.get_mean()
        return {
            "bbox": result["bbox"],
            "conf": result["conf"],
            "pos_3d_cam": pos_stable,
            "pos_raw": pos_raw,
        }

    # ------------------------------------------------------------
    #  椅子检测 + 标定
    # ------------------------------------------------------------

    def _detect_chairs(self, color_image):
        """YOLO 检测椅子 (COCO class 56).

        Returns:
            按 conf 降序的 bbox 列表 [[x1,y1,x2,y2], ...], 或 []
        """
        if self.model is None:
            return []
        results = self.model(color_image, classes=[self.CHAIR_CLASS_ID], verbose=False)
        if not results or len(results[0].boxes) == 0:
            return []
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()
        # 按置信度降序
        idx = np.argsort(-confs)
        out = []
        for i in idx:
            if float(confs[i]) >= self._chair_conf_threshold:
                out.append([int(x) for x in xyxy[i]])
        return out

    def _pick_center_chair(self, bboxes, h, w):
        """从 bbox 列表中选离画面中心最近的."""
        if not bboxes:
            return None
        cx0, cy0 = w / 2.0, h / 2.0
        best = min(bboxes, key=lambda b:
            abs((b[0] + b[2]) / 2.0 - cx0) + abs((b[1] + b[3]) / 2.0 - cy0))
        return list(best)

    def calibrate_chair(self, color_image):
        """每帧调用, 标定椅子 bbox. 稳定 N 帧后记录并锁定.

        Returns:
            bool: True = 标定完成 (后续调用直接返回 True)
        """
        if self._chair_calibrated:
            return True

        bboxes = self._detect_chairs(color_image)
        if not bboxes:
            self._chair_buffer = []
            return False

        h, w = color_image.shape[:2]
        best = self._pick_center_chair(bboxes, h, w)
        if best is None:
            self._chair_buffer = []
            return False

        self._chair_buffer.append(best)
        if len(self._chair_buffer) > self._chair_stabilize_frames + 5:
            self._chair_buffer.pop(0)

        if len(self._chair_buffer) >= self._chair_stabilize_frames:
            arr = np.array(self._chair_buffer[-self._chair_stabilize_frames:], dtype=float)
            # 各坐标波动 < 25px 即稳定
            if np.ptp(arr, axis=0).max() < 25.0:
                self._chair_bbox = arr.mean(axis=0).astype(int).tolist()
                self._chair_calibrated = True
                self._chair_buffer = []
                print(f"[Chair] 标定完成: bbox={self._chair_bbox}")
                return True

        return False

    # ------------------------------------------------------------
    #  人就位检测
    # ------------------------------------------------------------

    def _compute_seated_overlap(self, person_bbox):
        """计算 person ∩ chair / chair_area.

        Returns:
            float: 0.0 ~ 1.0
        """
        if self._chair_bbox is None:
            return 0.0
        ix1 = max(person_bbox[0], self._chair_bbox[0])
        iy1 = max(person_bbox[1], self._chair_bbox[1])
        ix2 = min(person_bbox[2], self._chair_bbox[2])
        iy2 = min(person_bbox[3], self._chair_bbox[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        chair_area = (self._chair_bbox[2] - self._chair_bbox[0]) * (self._chair_bbox[3] - self._chair_bbox[1])
        if chair_area <= 0:
            return 0.0
        return inter / chair_area

    def detect_person_seated(self, color_image):
        """检测人体 + 判定是否就位 (需先 calibrate_chair).

        每帧调用. 内部累积 seated_stable_count.

        Returns:
            dict 或 None:
              {
                  "bbox": [x1,y1,x2,y2],
                  "conf": float,
                  "overlap_ratio": float,   # person∩chair/chair_area
                  "is_seated": bool,
                  "seated_stable": int,     # 当前累积帧数
                  "seated_required": int,   # 需要帧数
              }
        """
        if not self._chair_calibrated:
            return None

        detection = self._detect_targets(color_image)
        if detection is None:
            # 无人 → 退避
            self._seated_stable_count = max(0, self._seated_stable_count - 1)
            self._is_seated = self._seated_stable_count >= self._seated_stable_required
            return None

        overlap = self._compute_seated_overlap(detection["bbox"])

        if overlap >= self._seated_overlap_threshold:
            self._seated_stable_count += 1
        else:
            self._seated_stable_count = max(0, self._seated_stable_count - 1)

        self._is_seated = self._seated_stable_count >= self._seated_stable_required

        return {
            "bbox": detection["bbox"],
            "conf": detection["conf"],
            "overlap_ratio": round(overlap, 3),
            "is_seated": self._is_seated,
            "seated_stable": self._seated_stable_count,
            "seated_required": self._seated_stable_required,
        }

    # ------------------------------------------------------------
    #  属性
    # ------------------------------------------------------------

    @property
    def is_person_stable(self):
        """人体位置是否已稳定."""
        return self._stabilizer.is_stable()

    @property
    def is_chair_calibrated(self):
        """椅子 bbox 是否已标定."""
        return self._chair_calibrated

    @property
    def chair_bbox(self):
        """已标定的椅子 bbox [x1, y1, x2, y2] 或 None."""
        return self._chair_bbox

    @property
    def is_person_seated(self):
        """人是否已就位 (连续 N 帧满足)."""
        return self._is_seated

    @property
    def seated_stable_count(self):
        """就位累积帧数."""
        return self._seated_stable_count

    def reset_seated(self):
        """清零就位累积 (状态转换时调用)."""
        self._seated_stable_count = 0
        self._is_seated = False

    # ------------------------------------------------------------
    def run(self):
        """主循环: 椅子标定 → 人就位检测 + 3D 位置显示."""
        print("=" * 55)
        print("  EyeToHand 椅子标定 + 人就位检测")
        print("  椅子稳定: %d帧 / 人就位: %d帧 / overlap≥%.1f" % (
            self._chair_stabilize_frames, self._seated_stable_required,
            self._seated_overlap_threshold))
        print("  按 'q' 键退出")
        print("=" * 55)

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None:
                    continue

                # ---- 椅子标定 (仅一次) ----
                chair_done = self.calibrate_chair(color_image)

                # ---- 人体检测 (原始 + 稳定) ----
                result = self.detect_person(color_image, depth_frame)
                stable_result = self.detect_person_stable(color_image, depth_frame)

                # ---- 人就位检测 (椅子标定完成后) ----
                seated = self.detect_person_seated(color_image)

                # ---- 可视化 ----
                display_img = color_image.copy()
                h, w = display_img.shape[:2]

                # 椅子标定状态
                if self._chair_calibrated and self._chair_bbox is not None:
                    cb = self._chair_bbox
                    cv2.rectangle(display_img, (cb[0], cb[1]), (cb[2], cb[3]),
                                  (255, 200, 0), 2)
                    cv2.putText(display_img, "chair", (cb[0], cb[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                    cv2.putText(display_img, "Chair: OK",
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 200, 0), 1)
                else:
                    cv2.putText(display_img,
                                "Chair: calibrating... %d/%d" % (
                                    len(self._chair_buffer), self._chair_stabilize_frames),
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 165, 255), 1)

                if result is not None:
                    bbox = result["bbox"]
                    conf = result["conf"]
                    pos_3d = result["pos_3d_cam"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    # bbox 颜色: 就位=绿, 稳定=黄, 否则=橙
                    if seated and seated["is_seated"]:
                        bbox_color = (0, 255, 0)
                    elif stable_result:
                        bbox_color = (0, 255, 255)
                    else:
                        bbox_color = (0, 165, 255)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), bbox_color, 2)
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    cv2.drawMarker(display_img, (cx, cy), (0, 0, 255),
                                   cv2.MARKER_CROSS, 12, 2)
                    cv2.putText(display_img, "person %.2f" % conf, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                    # 3D 位置
                    if pos_3d is not None and stable_result is not None:
                        sp = stable_result["pos_3d_cam"]
                        cv2.putText(display_img,
                                    "live: (%.3f,%.3f,%.3f)m" % pos_3d,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
                        cv2.putText(display_img,
                                    "stable: (%.3f,%.3f,%.3f)m" % sp,
                                    (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
                    elif pos_3d is not None:
                        cv2.putText(display_img,
                                    "live: (%.3f,%.3f,%.3f)m" % pos_3d,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)

                    # 就位状态
                    if seated is not None:
                        ov = seated["overlap_ratio"]
                        st = seated["seated_stable"]
                        sr = seated["seated_required"]
                        status = "SEATED" if seated["is_seated"] else "waiting"
                        s_color = (0, 255, 0) if seated["is_seated"] else (0, 165, 255)
                        cv2.putText(display_img,
                                    "Seated: %s  ov=%.2f  %d/%d" % (status, ov, st, sr),
                                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)
                    else:
                        cv2.putText(display_img, "Seated: n/a (no chair)",
                                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (128, 128, 128), 1)
                else:
                    cv2.putText(display_img, "no person detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    # 就位退避显示
                    if self._chair_calibrated:
                        sc = self._seated_stable_count
                        cv2.putText(display_img,
                                    "Seated: counting down  %d/%d" % (sc, self._seated_stable_required),
                                    (10, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                    (0, 165, 255), 2)

                # ---- 并排显示 ----
                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img

                cv2.imshow("EyeToHand Person Detection", images)

                if cv2.waitKey(25) & 0xFF == ord("q"):
                    print("\n[INFO] 用户按 q 退出")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断 (Ctrl+C)")
        except Exception as e:
            print("\n[ERROR] 运行异常: %s" % str(e))
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
            cv2.destroyAllWindows()
            print("[INFO] 程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = PersonDetector(config_dict=_MODULE_CFG)
    detector.run()
