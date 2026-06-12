# -*- coding: utf-8 -*-
"""
RealSense 水壶数据集采集脚本
============================
采集水壶不同角度的彩色图像, 保存至 KettleTraining/images/.
多角度要求: 旋转 0-360 度, 俯仰 ±30 度, 不同距离 0.3-1.0m.

使用方法:
1. 连接 Intel RealSense 相机
2. 运行: python KettleTraining/collect_kettle_data.py
3. [Space] 保存当前帧, [q] 退出
"""

import sys
import os

# ---- 将项目根目录加入 sys.path, 确保能导入 pyrealsense2 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pyrealsense2 as rs

# 保存目录: KettleTraining/images/
SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")
os.makedirs(SAVE_DIR, exist_ok=True)

# 初始化 RealSense
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
pipeline.start(config)

print("=" * 60)
print("  水壶数据集采集工具")
print("  保存目录: %s" % SAVE_DIR)
print("  [Space] 保存当前帧")
print("  [q]     退出")
print("=" * 60)

counter = 0
try:
    while True:
        frames = pipeline.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())

        # 显示计数
        info = "Count: %d | [Space] Save | [q] Quit" % counter
        display = color.copy()
        cv2.putText(display, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("Kettle Data Collection", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            fname = "kettle_%03d.jpg" % counter
            cv2.imwrite(os.path.join(SAVE_DIR, fname), color)
            print("[SAVE] %s saved." % fname)
            counter += 1
        elif key == ord('q'):
            break
except KeyboardInterrupt:
    pass
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("[INFO] 采集完成, 共保存 %d 张图片." % counter)