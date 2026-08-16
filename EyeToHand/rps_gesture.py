# -*- coding: utf-8 -*-
"""
Camera B 剪刀石头布手势识别 (MediaPipe 几何特征)
==================================================
基于 MediaPipe Hands 21 个 3D landmarks, 通过关节角度 + 距离比值
判断手指伸直/弯曲状态, 分类为 石头/布/剪刀.

特性:
  - 尺���不变: 3D 归一化坐标, 不受分辨率/ROI/距离影响
  - 多人多手: 最多同时检测 4 只手, 每手独立稳定跟踪
  - 几何方法: PIP关节角度(<=35°=伸直) + 距离比值(>1.12=伸直)

用法:
  python EyeToHand/rps_gesture.py [--debug]
  按 'q' 退出

依赖:
  - EyeToHand/camera.py + config.yaml (UGreenCamera 配置)
  - model/hand_landmarker.task (MediaPipe Hands, 项目已有)
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# 禁 MediaPipe clearcut 遥测 (必须在 import mediapipe 之前!)
# GLOG_logtostderr=1: 日志直出 stderr, 不写磁盘 → clearcut 无文件可扫描上传
# GLOG_minloglevel=5: 仅 FATAL → 最小化事件量
os.environ["GLOG_logtostderr"] = "1"
os.environ["GLOG_minloglevel"] = "5"
os.environ["GLOG_stderrthreshold"] = "5"
os.environ["GRPC_VERBOSITY"] = "NONE"

import yaml
import cv2
import numpy as np
import time

from EyeToHand.camera import UGreenCamera, find_camera_id

# MediaPipe
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp

# ============================================================
#  MediaPipe landmark 索引
# ============================================================
WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# 4 指: (name, tip, pip, mcp)
FINGERS = [
    ("index",  INDEX_TIP,  INDEX_PIP,  INDEX_MCP),
    ("middle", MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    ("ring",   RING_TIP,   RING_PIP,   RING_MCP),
    ("pinky",  PINKY_TIP,  PINKY_PIP,  PINKY_MCP),
]

CLASS_NAMES = {0: "rock", 1: "paper", 2: "scissors"}
CLASS_CN    = {0: "石头", 1: "布", 2: "剪刀"}
COLORS = {0: (0, 0, 255), 1: (0, 255, 0), 2: (255, 0, 0)}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


# ============================================================
#  几何特征计算 (尺度不变: 归一化3D坐标)
# ============================================================

def _landmark_xyz(hand_lms, idx):
    lm = hand_lms[idx]
    return np.array([lm.x, lm.y, lm.z], dtype=np.float64)


def _angle_between(v1, v2):
    """两向量夹角 (度). 仅取决于方向, 与向量长度无关."""
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-9:
        return 0.0
    cos = np.clip(dot / norm, -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def _finger_angle(hand_lms, tip_id, pip_id, mcp_id):
    """PIP 关节角度 (3D). 伸直≈180°, 弯曲变小."""
    v1 = _landmark_xyz(hand_lms, pip_id) - _landmark_xyz(hand_lms, mcp_id)
    v2 = _landmark_xyz(hand_lms, tip_id) - _landmark_xyz(hand_lms, pip_id)
    return _angle_between(v1, v2)


def _finger_dist_ratio(hand_lms, tip_id, pip_id):
    """tip→wrist / pip→wrist 比值. 伸直>1.0, 弯曲<1.0.

    尺度不变: 手近/远时分子分母同比例缩放.
    """
    tip = _landmark_xyz(hand_lms, tip_id)
    pip = _landmark_xyz(hand_lms, pip_id)
    wrist = _landmark_xyz(hand_lms, WRIST)

    tip_dist = np.linalg.norm(tip - wrist)
    pip_dist = np.linalg.norm(pip - wrist)
    if pip_dist < 1e-9:
        return 1.0
    return float(tip_dist / pip_dist)


def is_finger_extended(hand_lms, tip_id, pip_id, mcp_id):
    """手指是否伸直. 角度+比值 ≥1票.

    伸直: MCP→PIP 与 PIP→TIP 同向 → 夹角≈0°
    弯曲: 两向量分叉 → 夹角 60°~180°
    """
    angle = _finger_angle(hand_lms, tip_id, pip_id, mcp_id)
    ratio = _finger_dist_ratio(hand_lms, tip_id, pip_id)
    # 角度小=同向=伸直, 比值大=指尖远离手腕=伸直
    extended = (angle <= 35.0) or (ratio > 1.12)
    return extended, angle, ratio


def is_thumb_extended(hand_lms):
    """拇指是否伸出 (仅用于显示, 不参与分类)."""
    angle = _finger_angle(hand_lms, THUMB_TIP, THUMB_IP, THUMB_MCP)
    # 拇指关节活动范围更大, 阈值放宽
    thumb_pinky = np.linalg.norm(
        _landmark_xyz(hand_lms, THUMB_TIP) - _landmark_xyz(hand_lms, PINKY_MCP))
    palm_ref = np.linalg.norm(
        _landmark_xyz(hand_lms, WRIST) - _landmark_xyz(hand_lms, MIDDLE_MCP))
    if palm_ref < 1e-9:
        palm_ref = 1.0
    ratio = float(thumb_pinky / palm_ref)
    extended = (angle <= 45.0) or (ratio > 1.0)
    return extended, angle, ratio


def classify_gesture(hand_lms):
    """分类单手手势 → (gesture_id, extended_count, details)."""
    extended_fingers = []
    details = {}

    for name, tip_id, pip_id, mcp_id in FINGERS:
        ext, angle, ratio = is_finger_extended(hand_lms, tip_id, pip_id, mcp_id)
        extended_fingers.append(ext)
        details[name] = {"extended": ext, "angle": round(angle, 1), "ratio": round(ratio, 3)}

    # 拇指仅记录不参与
    thumb_ext, thumb_angle, thumb_ratio = is_thumb_extended(hand_lms)
    details["thumb"] = {"extended": thumb_ext, "angle": round(thumb_angle, 1), "ratio": round(thumb_ratio, 3)}

    count = sum(extended_fingers)

    if count <= 1:
        gid = 0       # 石头
    elif count >= 4:
        gid = 1       # 布
    elif count == 2 and extended_fingers[0] and extended_fingers[1]:
        gid = 2       # 剪刀: 仅食指+中指
    else:
        gid = -1      # 模糊

    return gid, count, details


# ============================================================
#  RPSGestureDetector
# ============================================================

class RPSGestureDetector:
    """剪刀石头布手势识别 (MediaPipe Hands 几何特征).

    支持多人多手 (最多4只手), 每只手独立跟踪稳定计数.
    几何特征尺度不变, 不受分辨率/ROI/距离影响.
    """

    HAND_MODEL_PATH = os.path.join(_project_root, "model", "hand_landmarker.task")

    def __init__(self, start_camera: bool = True, debug: bool = False,
                 config_dict: "dict | None" = None):
        self.debug = debug

        # ---- 加载配置 ----
        if config_dict is None:
            _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            with open(_cfg_path, "r", encoding="utf-8") as f:
                config_dict = yaml.safe_load(f)
        assert config_dict is not None
        self._cfg = config_dict

        # ---- MediaPipe Hands ----
        print(f"[RPS] MediaPipe Hands: {self.HAND_MODEL_PATH}")
        _options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self.HAND_MODEL_PATH),
            running_mode=RunningMode.VIDEO,
            num_hands=4,   # 支持多人多手
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._hand_landmarker = HandLandmarker.create_from_options(_options)
        self._mp_timestamp = 0

        # ---- 每只手独立稳定跟踪 ----
        # key = "Left" / "Right", value = {"gesture_id", "stable_count"}
        self._hand_state = {}
        self._stable_required = 4     # CPU 模式减少确认帧数
        self._absent_decay = 5       # 每帧衰减量, 消失约 1 帧清零

        # ---- 相机 ----
        self._camera = None
        if start_camera:
            self._open_camera()

    # ------------------------------------------------------------
    #  相机 (从 config.yaml 读分辨率)
    # ------------------------------------------------------------

    def _open_camera(self):
        ug_cfg = self._cfg.get("ugreen", {})
        cam_id = ug_cfg.get("camera_id", -1)
        cam_name = ug_cfg.get("camera_name", "UGREEN Camera 1080P")
        cam_path = ug_cfg.get("camera_path", "")
        width = ug_cfg.get("width", 1280)
        height = ug_cfg.get("height", 720)
        fps = ug_cfg.get("fps", 30)

        if cam_id == -1:
            cam_id = find_camera_id([cam_name], path_keyword=cam_path or None)
            if cam_id is None:
                raise RuntimeError(f"未找到摄像头: {cam_name} (path={cam_path})")

        self._camera = UGreenCamera(
            camera_id=cam_id, width=width, height=height, fps=fps)

    def start_camera(self):
        if self._camera is None:
            self._open_camera()

    def get_frame(self):
        if self._camera is None:
            return (None,)
        return self._camera.read()

    # ------------------------------------------------------------
    #  手势检测
    # ------------------------------------------------------------

    def _hand_key(self, handedness):
        """用手性作为跟踪 key."""
        return handedness  # "Left" / "Right"

    def _update_stable(self, key, gesture_id):
        """更新单手的稳定计数."""
        if key not in self._hand_state:
            self._hand_state[key] = {"gesture_id": -1, "stable_count": 0}

        state = self._hand_state[key]

        if gesture_id == -1:
            # 模糊手势 → 衰减, 保持上次确认的 gesture_id
            state["stable_count"] = max(0, state["stable_count"] - 1)
        elif gesture_id == state["gesture_id"]:
            state["stable_count"] += 1
        else:
            state["gesture_id"] = gesture_id
            state["stable_count"] = 1

        return state["stable_count"] >= self._stable_required

    def _decay_absent(self):
        """衰减未出现的手."""
        gone = []
        for key, state in self._hand_state.items():
            state["stable_count"] = max(0, state["stable_count"] - self._absent_decay)
            if state["stable_count"] <= 0:
                gone.append(key)
        for key in gone:
            del self._hand_state[key]

    def detect_all(self, color_image):
        """检测所有手, 每只手独立分类+稳定.

        Returns:
            list[dict]: 每只手一个结果, 按置信度降序.
            每个 dict:
              {"gesture_id", "gesture_name", "gesture_cn",
               "extended_count", "confidence", "stable",
               "hand_bbox", "handedness", "hand_conf",
               "details", "landmarks"}
        """
        h, w = color_image.shape[:2]
        rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._mp_timestamp += 33

        result = self._hand_landmarker.detect_for_video(mp_image, self._mp_timestamp)
        if not result.hand_landmarks:
            self._decay_absent()
            return []

        # 收集当前帧出现的手
        seen_keys = set()
        detections = []

        for i, hand_lms in enumerate(result.hand_landmarks):
            handedness = result.handedness[i][0].category_name if result.handedness else "?"
            hand_conf = result.handedness[i][0].score if result.handedness else 0.0
            key = self._hand_key(handedness)
            seen_keys.add(key)

            # 分类
            gesture_id, ext_count, details = classify_gesture(hand_lms)
            stable = self._update_stable(key, gesture_id)

            # 手势用稳定后的 (如果是衰减中, 用最后一次确认的)
            confirmed_id = self._hand_state[key]["gesture_id"]
            if confirmed_id == -1:
                confirmed_id = gesture_id
            stable_count = self._hand_state[key]["stable_count"]

            if self.debug:
                f_status = " ".join(f"{n}={d['extended']}" for n, d in details.items())
                name = CLASS_CN.get(gesture_id, "?")
                print(f"[RPS][{handedness}] {name} | ext={ext_count} | {f_status} | "
                      f"stable={stable_count}/{self._stable_required} {'✓' if stable else ''}")

            # bbox
            xs = [lm.x for lm in hand_lms]
            ys = [lm.y for lm in hand_lms]
            bbox = [
                max(0, int(min(xs) * w)), max(0, int(min(ys) * h)),
                min(w, int(max(xs) * w)), min(h, int(max(ys) * h)),
            ]

            confidence = min(1.0, stable_count / self._stable_required)

            detections.append({
                "gesture_id": confirmed_id,
                "gesture_name": CLASS_NAMES.get(confirmed_id, "unknown"),
                "gesture_cn": CLASS_CN.get(confirmed_id, "?"),
                "raw_gesture_id": gesture_id,
                "extended_count": ext_count,
                "confidence": round(confidence, 2),
                "stable": stable,
                "stable_count": stable_count,
                "hand_bbox": bbox,
                "handedness": handedness,
                "hand_conf": round(hand_conf, 3),
                "details": details,
                "landmarks": hand_lms,
            })

        # 衰减未出现的手
        for key in list(self._hand_state.keys()):
            if key not in seen_keys:
                self._hand_state[key]["stable_count"] = max(
                    0, self._hand_state[key]["stable_count"] - self._absent_decay)
                if self._hand_state[key]["stable_count"] <= 0:
                    del self._hand_state[key]

        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return detections

    def detect(self, color_image):
        """检测最佳手势 (最高稳定度). 兼容单结果接口.

        Returns:
            dict 或 None.
        """
        all_det = self.detect_all(color_image)
        if not all_det:
            return None
        return all_det[0]

    # ------------------------------------------------------------

    def reset(self):
        self._hand_state.clear()

    def close(self):
        """释放资源. 不显式关闭 MediaPipe HandLandmarker (其 close() 可能阻塞),
        交由进程退出时回收 — 与 hand_in_cup.py 策略一致.
        """
        if self._camera:
            self._camera.release()
            self._camera = None

    def stop(self):
        self.close()
        print("[RPS] 已停止")


# ============================================================
#  独立测试主循环
# ============================================================

def run_test(debug=False):
    # 加载配置
    _cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(_cfg_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
    _ugreen = _cfg.get("ugreen", {})

    # 独立创建相机 (与 hand_in_cup.run() 完全相同模式)
    cam = UGreenCamera(
        camera_id=_ugreen.get("camera_id", -1),
        camera_name=_ugreen.get("camera_name", "UGREEN Camera 1080P"),
        camera_path=_ugreen.get("camera_path", ""),
        width=_ugreen.get("width", 1280),
        height=_ugreen.get("height", 720),
        fps=_ugreen.get("fps", 30),
    )

    # detector 不自己开相机
    detector = RPSGestureDetector(start_camera=False, debug=debug,
                                  config_dict=_cfg)

    print("=" * 55)
    print("  Camera B 剪刀石头布 (MediaPipe 几何特征)")
    print(f"  分辨率: {_ugreen.get('width', 1280)}x{_ugreen.get('height', 720)}")
    print("  方法: PIP关节角度 + 距离比值 (尺度不变)")
    print("  多人多手独立跟踪 (最多4只手), 按 'q' 退出")
    print("=" * 55)

    frame_count = 0
    fps_t0 = time.time()
    fps = 0.0

    try:
        while True:
            color = cam.read()[0]
            if color is None:
                continue

            frame_count += 1
            detections = detector.detect_all(color)
            display = color.copy()
            h, w = display.shape[:2]

            # ---- 绘制每只手 ----
            for det in detections:
                hand_lms = det["landmarks"]
                details = det["details"]
                gid = det["gesture_id"]
                stable = det["stable"]
                handedness = det["handedness"]
                color = COLORS.get(gid, (128, 128, 128))

                # 骨架
                for (a, b) in HAND_CONNECTIONS:
                    pa = (int(hand_lms[a].x * w), int(hand_lms[a].y * h))
                    pb = (int(hand_lms[b].x * w), int(hand_lms[b].y * h))
                    cv2.line(display, pa, pb, (180, 180, 180), 1)

                # 关节点: 绿=伸直, 红=弯曲
                for name, tip_id, pip_id, mcp_id in FINGERS:
                    finfo = details.get(name, {})
                    pt_color = (0, 255, 0) if finfo.get("extended") else (0, 0, 255)
                    for lid in [mcp_id, pip_id, tip_id]:
                        px, py = int(hand_lms[lid].x * w), int(hand_lms[lid].y * h)
                        cv2.circle(display, (px, py), 4, pt_color, -1)
                # 拇指 (灰色, 不参与分类)
                for lid in [THUMB_MCP, THUMB_IP, THUMB_TIP]:
                    px, py = int(hand_lms[lid].x * w), int(hand_lms[lid].y * h)
                    cv2.circle(display, (px, py), 3, (180, 180, 180), -1)
                # 手腕
                wx, wy = int(hand_lms[WRIST].x * w), int(hand_lms[WRIST].y * h)
                cv2.circle(display, (wx, wy), 6, (255, 255, 0), -1)

                # bbox + 标签
                x1, y1, x2, y2 = det["hand_bbox"]
                thickness = 3 if stable else 1
                cv2.rectangle(display, (x1, y1), (x2, y2), color, thickness)
                # putText 只支持 ASCII: 手势名用 gesture_name (英文), ✓/箭头等符号禁用
                status = "OK" if stable else f"{det['stable_count']}/{detector._stable_required}"
                label = f"[{handedness}] {det['gesture_name']} [{status}]"
                cv2.putText(display, label, (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not detections:
                cv2.putText(display, "No hand detected",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)

            # 手指详情 (仅显示第一只手)
            if detections:
                det = detections[0]
                details = det["details"]
                y_bar = 30
                for fname in ["thumb", "index", "middle", "ring", "pinky"]:
                    finfo = details.get(fname, {})
                    ext = "UP" if finfo.get("extended") else "DN"
                    ang = finfo.get("angle", 0)
                    text = f"{fname}: {ext} ({ang:.0f}deg)"
                    bar_color = (0, 255, 0) if finfo.get("extended") else (0, 0, 255)
                    cv2.putText(display, text, (10, y_bar),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, bar_color, 1)
                    y_bar += 16

            # FPS
            if frame_count % 15 == 0:
                t1 = time.time()
                fps = 15.0 / max(t1 - fps_t0, 0.001)
                fps_t0 = t1
            cv2.putText(display, f"FPS: {fps:.0f}", (w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            cv2.imshow("RPS Gesture - MediaPipe Geometry (Camera B)", display)

            # 与 hand_in_cup.py 完全相同的退出模式
            if cv2.waitKey(25) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
    finally:
        cam.release()
        detector.close()
        cv2.destroyAllWindows()
        print("[INFO] 程序已退出")
        # clearcut C++ 析构上传阻塞 (连不上 Google → gRPC 超时).
        # GLOG env vars 已尽力抑制, 若仍慢则 os._exit 跳过 C++ 清理.
        os._exit(0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    run_test(debug=args.debug)
