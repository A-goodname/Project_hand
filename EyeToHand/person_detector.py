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
    @property
    def is_person_stable(self):
        """人体位置是否已稳定."""
        return self._stabilizer.is_stable()

    # ------------------------------------------------------------
    def run(self):
        """主循环: 实时检测人体并显示 bbox 中心 3D 位置 + 稳定状态."""
        print("=" * 50)
        print("  EyeToHand 人体位置检测 (窗口稳定: %d帧 / %.0fcm)" % (
            self._stabilizer.window, self._stabilizer.pos_thresh * 100))
        print("  按 'q' 键退出")
        print("=" * 50)

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- 人体检测 (原始 + 稳定) ----
                result = self.detect_person(color_image, depth_frame)
                stable_result = self.detect_person_stable(color_image, depth_frame)

                # ---- 可视化 ----
                display_img = color_image.copy()

                if result is not None:
                    bbox = result["bbox"]
                    conf = result["conf"]
                    pos_3d = result["pos_3d_cam"]

                    x1, y1, x2, y2 = [int(v) for v in bbox]

                    # bbox 矩形框 + 中心十字 (稳定时绿色, 否则黄色)
                    bbox_color = (0, 255, 0) if stable_result else (0, 255, 255)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), bbox_color, 2)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    cv2.drawMarker(display_img, (cx, cy), (0, 0, 255),
                                   cv2.MARKER_CROSS, 12, 2)

                    # 置信度标签
                    cv2.putText(display_img, "person %.2f" % conf, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

                    # ---- 终端 + 画面输出 ----
                    sc = self._stabilizer.count
                    if pos_3d is not None and stable_result is not None:
                        sp = stable_result["pos_3d_cam"]
                        # 画面: 左上角 live + stable
                        cv2.putText(display_img,
                                    "live: (%.3f, %.3f, %.3f) m" % pos_3d,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(display_img,
                                    "stable: (%.3f, %.3f, %.3f) m" % sp,
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # 终端: 单行
                        msg = "[Person] live=(%.3f,%.3f,%.3f)m  conf=%.2f  stable=(%.3f,%.3f,%.3f)m" % (
                            pos_3d[0], pos_3d[1], pos_3d[2], conf, sp[0], sp[1], sp[2])
                        self._log_write("[Person]", "stable=(%.3f,%.3f,%.3f)m conf=%.2f" % (
                            sp[0], sp[1], sp[2], conf))
                    elif pos_3d is not None:
                        cv2.putText(display_img,
                                    "live: (%.3f, %.3f, %.3f) m" % pos_3d,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(display_img,
                                    "stable: %d/%d" % (sc, self._stabilizer.window),
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        msg = "[Person] live=(%.3f,%.3f,%.3f)m  conf=%.2f  stable=%d/%d" % (
                            pos_3d[0], pos_3d[1], pos_3d[2], conf, sc, self._stabilizer.window)
                    else:
                        cv2.putText(display_img, "person %.2f | depth: --" % conf,
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        msg = "[Person] conf=%.2f  depth: --" % conf
                else:
                    cv2.putText(display_img, "no person detected",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    msg = "[Person] 未检测到人体"

                print("\r%-90s" % msg, end="")

                # ---- 并排显示: 彩色 | 深度热力图 ----
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
