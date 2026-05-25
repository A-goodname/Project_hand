# -*- coding: utf-8 -*-
"""
RealSense YOLOv8m COCO 全类别目标检测 (彩色流)
==========================================
功能: 使用 YOLOv8m 检测 COCO 数据集 80 类物体, 仅开启彩色相机流,
      实时显示检测框 + 类别英文名称 + 置信度.

流程:
  1. 从 config.yaml 加载模型路径 / 阈值 / 分辨率
  2. 初始化 RealSense (仅彩色流, 不开启深度)
  3. 初始化 YOLO (全类别检测, 不限制 class_id)
  4. 主循环: 采集帧 -> YOLO 检测 -> 绘制 (英文标签) -> 显示
  5. Cleanup

与 config.yaml 对接:
  - model.name              : YOLO 模型路径
  - model.conf_threshold    : 检测置信度阈值
  - model.target_class_id   : 忽略 (检测全部类别)
  - resolution.width/height/fps : 彩色流分辨率
  - depth.*                 : 忽略 (纯彩色检测)
  - smoothing.*             : 忽略

输入:
  无 (实时相机流)

输出:
  OpenCV 窗口显示实时检测结果
  按 'q' 退出
"""

import os
import sys

# ---- 将项目根目录加入 sys.path, 以导入 pyrealsense2 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import yaml
import torch
import pyrealsense2 as rs
from ultralytics import YOLO


# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

MODEL_NAME = os.path.join(_project_root, _cfg["model"]["name"])
CONF_THRESHOLD = _cfg["model"]["conf_threshold"]
FRAME_W = _cfg["resolution"]["width"]
FRAME_H = _cfg["resolution"]["height"]
FRAME_FPS = _cfg["resolution"]["fps"]

# ---- 设备选择: config device="gpu" 时优先 GPU, 否则回退 CPU ----
_device_pref = _cfg["model"].get("device", "cpu")
if _device_pref == "gpu" and torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print("[INFO] 运行设备: %s" % DEVICE)


# =============================================================
# 辅助函数
# =============================================================

def _init_colors(num_classes, seed=42):
    """
    为每个类别分配固定颜色 (HSV 色调均匀采样).

    参数:
        num_classes : int, 类别数量
        seed        : int, 随机种子, 保证每次运行颜色一致

    返回:
        list[tuple[int,int,int]], 每个元素为 (B, G, R) 颜色值
    """
    rng = np.random.RandomState(seed)
    np.random.seed(seed)
    colors = []
    for i in range(num_classes):
        # 色调在 [0, 179] 范围内均匀分布 (OpenCV Hue 范围)
        step = max(1, 180 // num_classes)
        hue = (i * step) % 180
        hsv = np.uint8([[[hue, 200, 240]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        colors.append((int(bgr[0]), int(bgr[1]), int(bgr[2])))
    return colors


# =============================================================
# 功能模块: RealSense YOLO 全类别检测器
# =============================================================

class RealSenseYOLODetector:
    """
    RealSense YOLOv8m 全类别目标检测器.

    仅开启彩色相机流, 使用 YOLOv8m 检测 COCO 所有 80 类物体,
    以 OpenCV 窗口实时显示检测框 + 类别英文名称 + 置信度.

    属性:
        pipeline         : rs.pipeline, RealSense 数据管道
        model            : YOLO, 加载的 YOLOv8m 模型
        class_names      : dict[int, str], COCO 类别名称映射
        colors           : list[tuple], 每个类别的固定颜色
    """

    def __init__(self, model_name=MODEL_NAME, conf_threshold=CONF_THRESHOLD):
        """
        初始化检测器.

        参数:
            model_name     : str, YOLO 模型路径 (不含 .pt 后缀)
            conf_threshold : float, 检测置信度阈值
        """
        self.conf_threshold = conf_threshold

        # ---- [Step 1] 初始化 RealSense (仅彩色流) ----
        print("[INFO] 初始化 RealSense 彩色流...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
        # 不开启深度流

        self.profile = self.pipeline.start(config)
        print("[INFO] RealSense 彩色流已启动 (%dx%d @ %dfps)" % (FRAME_W, FRAME_H, FRAME_FPS))

        # ---- [Step 2] 初始化 YOLO (全类别检测) ----
        print("[INFO] 加载 YOLO 模型: %s" % model_name)
        model_path = "%s.pt" % model_name
        if not os.path.exists(model_path):
            model_path = model_name
        self.model = YOLO(model_path).to(DEVICE)
        self.class_names = self.model.names  # dict[int, str], 如 {0: 'person', ...}
        print("[INFO] 模型加载完成, 支持 %d 种类别" % len(self.class_names))

        # ---- [Step 3] 生成颜色表 ----
        self.colors = _init_colors(len(self.class_names))
        print("[INFO] 颜色表已生成 (%d 种颜色)" % len(self.colors))

    # ----------------------------------------------------------

    def get_color_frame(self):
        """
        获取一帧原始彩色图像.

        返回:
            color_image : np.ndarray, BGR 彩色图像, 或 None (获取失败时)
        """
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return None
            return np.asanyarray(color_frame.get_data())
        except Exception as e:
            print("[ERROR] 获取彩色帧失败: %s" % str(e))
            return None

    # ----------------------------------------------------------

    def detect_all(self, color_image):
        """
        对彩色图像进行 YOLO 全类别检测.

        不限制 class_id, 检测所有 COCO 类别.

        参数:
            color_image : np.ndarray, BGR 彩色图像

        返回:
            list[list], 每个元素为 [x1, y1, x2, y2, conf, class_id]
            若无可检测目标, 返回空列表
        """
        results = self.model(color_image, conf=self.conf_threshold, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections.append([int(box[0]), int(box[1]),
                                   int(box[2]), int(box[3]),
                                   float(conf), int(cls_id)])

        return detections

    # ----------------------------------------------------------

    def draw_detections(self, image, detections):
        """
        在图像上绘制所有检测结果.

        每个检测物绘制:
          - 彩色检测框 (颜色根据类别固定)
          - 标签文字: "ClassName confidence" (英文, 避免 cv2 中文显示问题)

        参数:
            image       : np.ndarray, 待绘制的 BGR 图像 (会被修改)
            detections  : list[list], detect_all() 返回值

        返回:
            np.ndarray, 绘制后的图像
        """
        for det in detections:
            x1, y1, x2, y2 = det[:4]
            conf = det[4]
            cls_id = det[5]

            # 获取该类别对应的颜色
            color = self.colors[cls_id % len(self.colors)]

            # 绘制检测框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

            # 英文标签: "ClassName conf"
            class_name = self.class_names.get(cls_id, "unknown")
            label = "%s %.2f" % (class_name, conf)

            # 标签背景 + 文字
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(image, (x1, y1 - label_h - baseline - 4),
                          (x1 + label_w, y1), color, -1)
            cv2.putText(image, label, (x1, y1 - baseline - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return image

    # ----------------------------------------------------------

    def run(self):
        """
        主循环: 采集 -> 检测 -> 绘制 -> 显示.

        按 'q' 或 Ctrl+C 退出.
        """
        print("[INFO] 按 'q' 键退出程序...")

        try:
            while True:
                # ---- [Step 1] 获取彩色帧 ----
                color_image = self.get_color_frame()
                if color_image is None:
                    continue

                # ---- [Step 2] YOLO 全类别检测 ----
                detections = self.detect_all(color_image)

                # ---- [Step 3] 绘制检测结果 ----
                display_img = color_image.copy()
                if detections:
                    display_img = self.draw_detections(display_img, detections)

                # ---- [Step 4] 显示 ----
                cv2.imshow("RealSense YOLOv8 COCO Detection (RealSense_R/yolo_detect)", display_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\n[INFO] 用户按 q 退出")
                    break

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断 (Ctrl+C)")
        except Exception as e:
            print("\n[ERROR] 运行异常: %s" % str(e))
        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("[INFO] 程序已退出")


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  RealSense YOLOv8 COCO 全类别目标检测")
    print("  模型: %s" % MODEL_NAME)
    print("  分辨率: %dx%d @ %dfps" % (FRAME_W, FRAME_H, FRAME_FPS))
    print("  置信度阈值: %.2f" % CONF_THRESHOLD)
    print("=" * 60)

    detector = RealSenseYOLODetector(model_name=MODEL_NAME,
                                     conf_threshold=CONF_THRESHOLD)
    detector.run()


if __name__ == "__main__":
    main()