# -*- coding: utf-8 -*-
"""
MediaPipe Hands 手指关节检测测试 (Task API, v0.10.x)
=====================================================
对比 YOLOv8n-Pose (仅 2 个手腕点), MediaPipe 提供 21 个手部 landmarks.

操作: 按 'q' 退出
"""

import sys
import os
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

import cv2
import yaml
import time

# ---- Camera B ----
from RealSense.realsense_shared import BaseRealSenseDetector

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

class _TestCamera(BaseRealSenseDetector):
    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "mediapipe_test.log"
    TARGET_CLASS_ID = 0

# ---- MediaPipe Task API ----
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
import mediapipe as mp

MODEL_PATH = os.path.join(_project_root, "model", "hand_landmarker.task")

# 手腕 + 指尖 + MCP 关键点 (索引 → 名称)
KEY_LANDMARKS = {
    0:  "WRIST",
    4:  "THUMB_TIP",
    8:  "INDEX_TIP",
    12: "MIDDLE_TIP",
    16: "RING_TIP",
    20: "PINKY_TIP",
    5:  "INDEX_MCP",
    9:  "MIDDLE_MCP",
    13: "RING_MCP",
    17: "PINKY_MCP",
}

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),        # thumb
    (0,5),(5,6),(6,7),(7,8),        # index
    (0,9),(9,10),(10,11),(11,12),   # middle
    (0,13),(13,14),(14,15),(15,16), # ring
    (0,17),(17,18),(18,19),(19,20), # pinky
    (5,9),(9,13),(13,17),           # palm
]

# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 55)
    print("  MediaPipe Hands 手指关节检测测试")
    print("  21 landmarks / 手")
    print("  按 'q' 退出")
    print("=" * 55)

    camera = _TestCamera(config_dict=_cfg)

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = HandLandmarker.create_from_options(options)

    try:
        while True:
            color, _, _, depth_frame = camera.get_frames()
            if color is None:
                continue

            h, w = color.shape[:2]
            display = color.copy()

            # MediaPipe Image (RGB)
            rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(time.time() * 1000)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for i, hand_lms in enumerate(result.hand_landmarks):
                    handedness = result.handedness[i][0].category_name if result.handedness else "?"
                    score = result.handedness[i][0].score if result.handedness else 0.0

                    # 绘制手掌骨架连线
                    for a, b in HAND_CONNECTIONS:
                        ax = int(hand_lms[a].x * w)
                        ay = int(hand_lms[a].y * h)
                        bx = int(hand_lms[b].x * w)
                        by = int(hand_lms[b].y * h)
                        cv2.line(display, (ax, ay), (bx, by), (0, 255, 0), 1, cv2.LINE_AA)

                    # 绘制所有 21 个 landmarks
                    for j, lm in enumerate(hand_lms):
                        px, py = int(lm.x * w), int(lm.y * h)
                        if j == 0:
                            # 手腕: 大红点
                            cv2.circle(display, (px, py), 7, (0, 0, 255), -1)
                        elif j in KEY_LANDMARKS:
                            # 指尖+MCP: 黄点 + 标签
                            cv2.circle(display, (px, py), 4, (0, 255, 255), -1)
                            cv2.putText(display, KEY_LANDMARKS[j], (px + 4, py - 4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                        else:
                            # 其他关节: 小绿点
                            cv2.circle(display, (px, py), 2, (0, 255, 0), -1)

                    # 手腕坐标 + 深度
                    wrist = hand_lms[0]
                    wx, wy = int(wrist.x * w), int(wrist.y * h)
                    wz = 0.0
                    if depth_frame is not None:
                        wz = depth_frame.get_distance(wx, wy)
                    label = f"{handedness}({score:.2f}) wrist:({wx},{wy}) z={wz:.3f}m"
                    cv2.putText(display, label, (10, 30 + i * 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    # 终端
                    print(f"\r[{handedness}] wrist=({wx},{wy}) z={wz:.3f}m  "
                          f"conf={score:.2f}  ", end="")
            else:
                cv2.putText(display, "No hand detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                print("\r[检测中...]                              ", end="")

            cv2.imshow("MediaPipe Hands Test", display)
            if cv2.waitKey(25) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出")
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        camera.stop()
        landmarker.close()
        cv2.destroyAllWindows()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
