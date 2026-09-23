# -*- coding: utf-8 -*-
"""只读预览两只 Windows UGREEN 相机，用于确认 Phase0 / PhaseD 物理对应。

不会连接 D435i、灵巧手或机械臂。按 q 或 Esc 退出。
"""
import os
import sys
import time

import cv2

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from EyeToHand.camera import UGreenCamera, find_camera_id  # noqa: E402


CAMERA_NAMES = ("UGREEN Camera", "UGREEN Camera 1080P")


def main() -> int:
    cameras = []
    try:
        for name in CAMERA_NAMES:
            camera_id = find_camera_id([name])
            if camera_id is None:
                raise RuntimeError(f"未找到相机: {name}")
            camera = UGreenCamera(
                camera_id=camera_id, camera_name=name,
                width=1280, height=720, fps=30,
            )
            cameras.append((name, camera))

        print("两只 UGREEN 已打开。根据画面确认物理位置；按 q 或 Esc 退出。")
        last_ok = time.monotonic()
        while True:
            any_frame = False
            for name, camera in cameras:
                frame = camera.read()[0]
                if frame is None:
                    continue
                any_frame = True
                last_ok = time.monotonic()
                cv2.putText(
                    frame, f"[{camera.camera_id}] {name}", (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2,
                )
                cv2.putText(
                    frame, "Identify: Phase0 welcome or PhaseD handover", (18, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2,
                )
                cv2.imshow(f"[{camera.camera_id}] {name}", frame)

            if not any_frame and time.monotonic() - last_ok > 3.0:
                raise RuntimeError("两只相机连续 3 秒均无画面")
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        return 0
    finally:
        for _name, camera in cameras:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(main())
