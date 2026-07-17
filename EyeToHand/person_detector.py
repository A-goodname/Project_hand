# -*- coding: utf-8 -*-
"""
EyeToHand 人体位置检测 (纯 RGB)
===============================
使用标准 YOLOv8m COCO 模型检测人体 (class_id=0) 和椅子 (class_id=56),
判定人就位状态. 纯 RGB 流程, 无深度依赖.

检测流程:
  1. 椅子实时检测: 每帧 YOLO 检测, 取画面中心最近的椅子
  2. 人就位: person bbox ∩ chair / chair_area ≥ threshold → 累积确认
     - 进入阈值: 连续 N 帧满足 → True
     - 退出阈值: 连续 N 帧无人或重叠不足 → 立刻 False

使用方法:
  python EyeToHand/person_detector.py
  按 'q' 退出

依赖:
  - EyeToHand/camera.py  (UGreenCamera)
  - YOLOv8m COCO 模型
  - EyeToHand/config.yaml (chair 段)
"""

import sys
import os
from typing import Optional

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np
from ultralytics import YOLO

from EyeToHand.camera import UGreenCamera


# ============================================================
#  PersonDetector
# ============================================================

class PersonDetector:
    """
    人体位置检测器 (纯 RGB).

    椅子实时检测 → 人就位判定 全链路, 无深度依赖.
    使用标准 YOLO COCO person (class=0) 和 chair (class=56).
    """

    TARGET_CLASS_ID = 0             # COCO person
    CHAIR_CLASS_ID = 56             # COCO chair

    def __init__(self, camera_id: int = 1, config_dict: Optional[dict] = None,
                 start_camera: bool = True):
        """
        Args:
            camera_id:     OpenCV 相机索引 (UGreenCamera).
            config_dict:   配置字典 (EyeToHand/config.yaml). None=从文件加载.
            start_camera:  True=立即打开相机, False=需手动调 start_camera().
        """
        # ---- 加载配置 ----
        if config_dict is None:
            _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(_cfg_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        assert config_dict is not None

        _model_cfg = config_dict.get("model", {})
        _chair_cfg = config_dict.get("chair", {})

        # ---- 椅子检测参数 ----
        self._chair_conf_threshold = float(_chair_cfg.get("conf_threshold", 0.4))

        # ---- 人就位参数 (非对称阈值) ----
        self._seated_overlap_threshold = float(_chair_cfg.get("overlap_threshold", 0.4))
        self._seated_stable_required = int(_chair_cfg.get("seated_stable_frames", 10))  # 进入
        self._leave_frames = int(_chair_cfg.get("leave_frames", 5))                     # 退出

        # ---- YOLO 模型 ----
        model_name = _model_cfg.get("name", os.path.join(_project_root, "model", "yolov8m"))
        model_path_pt = model_name + ".pt" if not model_name.endswith(".pt") else model_name
        if not os.path.isabs(model_path_pt):
            model_path_pt = os.path.join(_project_root, model_path_pt)
        if not os.path.exists(model_path_pt):
            raise FileNotFoundError(f"YOLO 模型不存在: {model_path_pt}")

        self._conf_threshold = float(_model_cfg.get("conf_threshold", 0.5))
        device_pref = _model_cfg.get("device", "gpu")
        import torch
        device = "cuda:0" if (device_pref == "gpu" and torch.cuda.is_available()) else "cpu"

        print(f"[PersonDetector] YOLO: {model_path_pt}  device={device}")
        self.model = YOLO(model_path_pt)
        self.model.to(device)

        # ---- 相机 (可选延迟启动) ----
        _ugreen_cfg = config_dict.get("ugreen", {})
        self._camera_id = _ugreen_cfg.get("camera_id", camera_id)
        self._camera_name = _ugreen_cfg.get("camera_name", "")

        self._camera_cfg = {
            "width": _ugreen_cfg.get("width", 1280),
            "height": _ugreen_cfg.get("height", 720),
            "fps": _ugreen_cfg.get("fps", 30),
        }
        self._camera = None
        if start_camera:
            self._open_camera()

        # ---- 椅子实时检测状态 ----
        self._has_chair = False         # 至少检测到过一次椅子
        self._chair_ref = None          # 用于 overlap 计算 / 可视化
        self._chair_baseline = None     # 尺寸锚点 (仅 not person_near 时更新)

        # ---- 人就位状态 ----
        self._seated_stable_count = 0
        self._absent_count = 0
        self._is_seated = False

    # ------------------------------------------------------------
    #  相机管理
    # ------------------------------------------------------------

    def start_camera(self):
        """手动启动相机 (start_camera=False 时使用)."""
        if self._camera is None:
            self._open_camera()

    def _open_camera(self):
        self._camera = UGreenCamera(
            camera_id=self._camera_id,
            camera_name=self._camera_name,
            width=self._camera_cfg["width"],
            height=self._camera_cfg["height"],
            fps=self._camera_cfg["fps"],
        )

    def get_frames(self):
        """读取一帧.

        Returns:
            (color_image,) 单元素元组, 或 (None,) 读取失败时.
        """
        if self._camera is None:
            return (None,)
        return self._camera.read()

    # ------------------------------------------------------------
    #  YOLO 检测
    # ------------------------------------------------------------

    def _detect_targets(self, color_image):
        """YOLO 检测 person (class=TARGET_CLASS_ID), 返回置信度最高的 bbox.

        Returns:
            {"bbox": [x1,y1,x2,y2], "conf": float} 或 None.
        """
        if self.model is None:
            return None
        results = self.model(
            color_image, conf=self._conf_threshold,
            classes=[self.TARGET_CLASS_ID], verbose=False,
        )
        best = None
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf_val in zip(boxes, confs):
                if best is None or conf_val > best["conf"]:
                    best = {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "conf": float(conf_val),
                    }
        return best

    # ------------------------------------------------------------
    #  椅子实时检测 (每帧运行, 不锁定)
    # ------------------------------------------------------------

    def _detect_chair_live(self, color_image):
        """每帧 YOLO 检测椅子 (COCO class 56), 取画面中心最近的 bbox.

        Returns:
            [x1,y1,x2,y2] 或 None.
        """
        if self.model is None:
            return None
        results = self.model(color_image, classes=[self.CHAIR_CLASS_ID], verbose=False)
        if not results or len(results[0].boxes) == 0:
            return None
        boxes = results[0].boxes
        confs = boxes.conf.cpu().numpy()
        xyxy = boxes.xyxy.cpu().numpy()

        # 过滤置信度
        valid = []
        for i in range(len(confs)):
            if float(confs[i]) >= self._chair_conf_threshold:
                valid.append([int(x) for x in xyxy[i]])
        if not valid:
            return None

        # 取离画面中心最近的
        h, w = color_image.shape[:2]
        cx0, cy0 = w / 2.0, h / 2.0
        best = min(valid, key=lambda b:
            abs((b[0] + b[2]) / 2.0 - cx0) + abs((b[1] + b[3]) / 2.0 - cy0))
        return best

    # ------------------------------------------------------------
    #  纯人检测 (only_person_detect 分支, 仅 YOLO class 0)
    # ------------------------------------------------------------

    def detect_person_only(self, color_image):
        """仅检测画面中是否有人 (class 0). 复用已有 YOLO model.
        不做椅子标定, 不走 overlap 计算.

        Returns:
            True 或 False.
        """
        detection = self._detect_targets(color_image)
        return detection is not None

    # ------------------------------------------------------------
    #  合并检测: person + chair 单次推理 (GPU 加速后就位判定主力)
    # ------------------------------------------------------------

    def _detect_person_and_chair(self, color_image):
        """单次 YOLO 推理同时检测 person (class 0) + chair (class 56).

        替代两次独立 self.model() 调用, 推理时间减半.
        在 GPU 上优势尤为明显 (kernel launch overhead 合并).

        Returns:
            (person_dict, chair_bbox) —
            person_dict: {"bbox": [x1,y1,x2,y2], "conf": float} 或 None
            chair_bbox:  [x1,y1,x2,y2] 或 None
        """
        if self.model is None:
            return None, None
        results = self.model(
            color_image,
            conf=self._conf_threshold,
            classes=[self.TARGET_CLASS_ID, self.CHAIR_CLASS_ID],
            verbose=False,
        )
        best_person = None
        best_chair = None
        for result in results:
            if result.boxes is None:
                continue
            cls_ids = result.boxes.cls.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            xyxy = result.boxes.xyxy.cpu().numpy()

            for i in range(len(cls_ids)):
                cid = int(cls_ids[i])
                cf = float(confs[i])
                box = [int(x) for x in xyxy[i]]
                if cid == self.TARGET_CLASS_ID:
                    if best_person is None or cf > best_person["conf"]:
                        best_person = {"bbox": box, "conf": cf}
                elif cid == self.CHAIR_CLASS_ID:
                    if cf >= self._chair_conf_threshold:
                        if best_chair is None:
                            best_chair = box
                        else:
                            # 取离画面中心最近的椅子
                            h, w = color_image.shape[:2]
                            cx0, cy0 = w / 2.0, h / 2.0
                            def _dist(b):
                                return abs((b[0]+b[2])/2.0-cx0)+abs((b[1]+b[3])/2.0-cy0)
                            if _dist(box) < _dist(best_chair):
                                best_chair = box
        return best_person, best_chair

    # ------------------------------------------------------------
    #  人就位检测
    # ------------------------------------------------------------

    @staticmethod
    def _bbox_intersects(a, b):
        """两 bbox 是否有交集."""
        if a is None or b is None:
            return False
        return not (a[2] <= b[0] or b[2] <= a[0]
                    or a[3] <= b[1] or b[3] <= a[1])

    def _compute_seated_overlap(self, person_bbox):
        """计算 person ∩ chair / chair_area.

        Returns:
            float: 0.0 ~ 1.0
        """
        if self._chair_ref is None:
            return 0.0
        ix1 = max(person_bbox[0], self._chair_ref[0])
        iy1 = max(person_bbox[1], self._chair_ref[1])
        ix2 = min(person_bbox[2], self._chair_ref[2])
        iy2 = min(person_bbox[3], self._chair_ref[3])
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        inter = (ix2 - ix1) * (iy2 - iy1)
        chair_area = (self._chair_ref[2] - self._chair_ref[0]) * (self._chair_ref[3] - self._chair_ref[1])
        if chair_area <= 0:
            return 0.0
        return inter / chair_area

    def detect_person_seated(self, color_image):
        """检测人体 + 判定是否就位 (椅子实时检测, 非对称阈值).

        椅子参考更新策略:
          - 无人靠近椅子 → 安全更新 _chair_ref (椅子完整可见)
          - 有人靠近椅子 → 锁定 _chair_ref, 计算 overlap

        每帧调用.
          进入: 连续 seated_stable_required 帧满足 → True
          退出: 连续 leave_frames 帧无人靠近或重叠不足 → 立刻 False + 清零

        Returns:
            dict 或 None:
              {
                  "bbox": [x1,y1,x2,y2],
                  "conf": float,
                  "overlap_ratio": float,
                  "is_seated": bool,
                  "seated_stable": int,
                  "seated_required": int,
              }
        """
        # ---- 单次推理同时检测椅子 + 人体 (GPU 合并, 推理时间减半) ----
        detection, chair_live = self._detect_person_and_chair(color_image)

        # ---- 判断人是否在椅子附近 ----
        person_near = (detection is not None and self._chair_ref is not None
                       and self._bbox_intersects(detection["bbox"], self._chair_ref))

        if not person_near:
            # 无人靠近 → 安全更新 baseline + ref (椅子完整可见)
            if chair_live is not None:
                was_new = not self._has_chair
                self._chair_baseline = chair_live   # 尺寸锚点
                self._chair_ref = chair_live
                self._has_chair = True
                if was_new:
                    _p = chair_live
                    print(f"[PersonDetector] 椅子已检测: bbox=[{_p[0]},{_p[1]},{_p[2]},{_p[3]}] "
                          f"size={(_p[2]-_p[0])}x{(_p[3]-_p[1])}")

            # 退出退避
            self._absent_count += 1
            if self._absent_count >= self._leave_frames:
                if self._is_seated:
                    print(f"[PersonDetector] 人就位: False (连续{self._absent_count}帧未检测到)")
                self._seated_stable_count = 0
                self._is_seated = False
                self._absent_count = 0
            return None

        # 有人在椅子附近
        assert detection is not None

        # chair_live vs _chair_baseline 面积差异 ≤30% → 信任, 更新 _chair_ref
        if chair_live is not None and self._chair_baseline is not None:
            b_area = (self._chair_baseline[2] - self._chair_baseline[0]) * (self._chair_baseline[3] - self._chair_baseline[1])
            l_area = (chair_live[2] - chair_live[0]) * (chair_live[3] - chair_live[1])
            if b_area > 0 and abs(l_area - b_area) / b_area <= 0.30:
                self._chair_ref = chair_live

        overlap = self._compute_seated_overlap(detection["bbox"])
        was_seated = self._is_seated

        if overlap >= self._seated_overlap_threshold:
            self._seated_stable_count += 1
            self._absent_count = 0
        else:
            self._absent_count += 1
            if self._absent_count >= self._leave_frames:
                self._seated_stable_count = 0
                self._absent_count = 0

        self._is_seated = self._seated_stable_count >= self._seated_stable_required
        if self._is_seated and not was_seated:
            print(f"[PersonDetector] 人就位: True "
                  f"(stable={self._seated_stable_count}/{self._seated_stable_required} "
                  f"overlap={overlap:.2f})")

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
    def has_chair(self):
        """是否至少检测到过一次椅子 (用于 Idle → GREETING 门控)."""
        return self._has_chair

    @property
    def chair_bbox(self):
        """当前帧实时检测的椅子 bbox [x1, y1, x2, y2] 或 None."""
        return self._chair_ref

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
        self._absent_count = 0
        self._is_seated = False

    # ------------------------------------------------------------
    #  清理
    # ------------------------------------------------------------

    def stop(self):
        """释放相机."""
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            print("[PersonDetector] 已停止")

    # ------------------------------------------------------------
    #  独立测试主循环
    # ------------------------------------------------------------

    def run(self):
        """主循环: 椅子实时检测 + 人就位判定 (纯 RGB)."""
        print("=" * 55)
        print("  EyeToHand 椅子实时检测 + 人就位判定 (纯 RGB)")
        print("  椅子: 每帧检测, 取画面中心最近")
        print("  就位: 进 %d帧 / 出 %d帧 / overlap>=%.1f" % (
            self._seated_stable_required, self._leave_frames,
            self._seated_overlap_threshold))
        print("  按 'q' 键退出")
        print("=" * 55)

        try:
            while True:
                color_image = self.get_frames()[0]
                if color_image is None:
                    continue

                # ---- 人就位检测 (内含椅子实时检测) ----
                seated = self.detect_person_seated(color_image)

                # ---- 可视化 ----
                display_img = color_image.copy()
                h, w = display_img.shape[:2]

                # 椅子 (实时)
                if self._chair_ref is not None:
                    cb = self._chair_ref
                    cv2.rectangle(display_img, (cb[0], cb[1]), (cb[2], cb[3]),
                                  (255, 200, 0), 2)
                    cv2.putText(display_img, "chair", (cb[0], cb[1] - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
                else:
                    cv2.putText(display_img, "Chair: --",
                                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (128, 128, 128), 1)

                # 人体 bbox
                detection = self._detect_targets(color_image)
                if detection is not None:
                    bbox = detection["bbox"]
                    conf = detection["conf"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    if seated and seated["is_seated"]:
                        bbox_color = (0, 255, 0)
                    else:
                        bbox_color = (0, 165, 255)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), bbox_color, 2)
                    cv2.drawMarker(display_img, ((x1 + x2) // 2, (y1 + y2) // 2),
                                   (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
                    cv2.putText(display_img, "person %.2f" % conf, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                    # 就位状态
                    if seated is not None:
                        ov = seated["overlap_ratio"]
                        st = seated["seated_stable"]
                        sr = seated["seated_required"]
                        status = "SEATED" if seated["is_seated"] else "waiting"
                        s_color = (0, 255, 0) if seated["is_seated"] else (0, 165, 255)
                        cv2.putText(display_img,
                                    "Seated: %s  ov=%.2f  %d/%d" % (status, ov, st, sr),
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)
                else:
                    cv2.putText(display_img, "no person detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(display_img,
                                "Seated: absent %d/%d" % (self._absent_count, self._leave_frames),
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 165, 255), 2)

                cv2.imshow("EyeToHand Person Detection", display_img)

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
#  主入口
# ============================================================

if __name__ == "__main__":
    detector = PersonDetector()
    detector.run()
