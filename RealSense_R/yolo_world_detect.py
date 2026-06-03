# -*- coding: utf-8 -*-
"""
RealSense YOLO-World 零样本水壶检测
=====================================
使用 YOLO-World v2 (yolov8m-worldv2.pt) 零样本检测水壶/茶壶/水罐本体.

核心机制:
  - 加载 YOLO-World 模型后调用 set_classes() 设置自定义文本描述
  - 模型通过 CLIP 文本编码器将描述转为 embedding 进行匹配
  - 仅开启彩色相机流, 不需要深度

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/yolo_world_detect.py
  3. 按 'q' 键退出

检测策略:
  - set_classes 限定了只有水壶本体类, 所有检测结果均为水壶
  - 同一场景若检测到多个水壶, 只返回最高置信度的一个
  - 检测不到水壶时不回退, 直接返回 None
  - 本阶段仅检测水壶本体 bbox, 把手分离在后续 test_SAM_handle.py 中完成
"""

import sys
import os
import yaml

# ---- 将项目根目录加入 sys.path, 以导入 pyrealsense2 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO


# ============================================================
#  配置加载 (从 RealSense_R/config.yaml 读取)
# ============================================================

_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---- 从 yolo_world 配置段读取 ----
_yw_cfg = _cfg.get("yolo_world", {})
if not _yw_cfg:
    raise RuntimeError("config.yaml 缺少 yolo_world 配置段")

KETTLE_CLASSES = _yw_cfg.get("classes", [
    "kettle", "electric kettle", "water jug", "teapot"
])

MODEL_PATH = os.path.join(_project_root, _yw_cfg.get("model", "./model/yolov8m-worldv2.pt"))
CONF_THRESHOLD = _yw_cfg.get("conf_threshold", 0.25)

# ---- 设备选择 ----
_device_pref = _yw_cfg.get("device", "cpu")
if _device_pref == "gpu" and torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print("[INFO] YOLO-World 运行设备: %s" % DEVICE)

# ---- 分辨率 / 帧率沿用 config.yaml 中的配置 ----
FRAME_W = _cfg["resolution"]["width"]
FRAME_H = _cfg["resolution"]["height"]
FRAME_FPS = _cfg["resolution"]["fps"]


# ============================================================
#  主检测器类
# ============================================================

class YOLOWorldKettleDetector:
    """
    RealSense + YOLO-World 水壶本体零样本检测器.

    使用 YOLO-World v2 模型 + set_classes 自定义词汇表,
    检测水壶/茶壶/水罐本体, 不检测把手.

    属性:
        pipeline       : rs.pipeline, RealSense 数据管道 (仅彩色流)
        model          : YOLOWorld, 加载的 YOLO-World 模型
        kettle_classes : list[str], 当前设置的水壶词汇表
    """

    def __init__(self, model_path=MODEL_PATH,
                 classes=None, conf_threshold=CONF_THRESHOLD):
        """
        初始化检测器.

        参数:
            model_path     : str, YOLO-World 模型路径 (含 .pt 后缀)
            classes        : list[str] or None, 自定义词汇表 (默认 KETTLE_CLASSES)
            conf_threshold : float, 检测置信度阈值
        """
        self.conf_threshold = conf_threshold
        self.kettle_classes = classes if classes is not None else KETTLE_CLASSES

        # ---- [Step 1] 初始化 RealSense (仅彩色流) ----
        print("[INFO] 初始化 RealSense 彩色流...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
        # 不开启深度流

        self.profile = self.pipeline.start(config)
        print("[INFO] RealSense 彩色流已启动 (%dx%d @ %dfps)" % (FRAME_W, FRAME_H, FRAME_FPS))

        # ---- [Step 2] 加载 YOLO-World 模型 ----
        print("[INFO] 加载 YOLO-World 模型: %s" % model_path)
        if not os.path.exists(model_path):
            raise FileNotFoundError("模型文件不存在: %s" % model_path)

        self.model = YOLO(model_path)
        self.model.to(DEVICE)

        # ---- [Step 3] 设置自定义词汇表 ----
        print("[INFO] 设置自定义词汇表 (%d 类):" % len(self.kettle_classes))
        for i, cls_name in enumerate(self.kettle_classes):
            print("       id=%d: %s" % (i, cls_name))
        self.model.set_classes(self.kettle_classes)

        # 确认 names 已更新
        print("[INFO] 模型类别映射: %s" % str(self.model.names))
        print("[INFO] 初始化完成, 仅检测水壶本体.")

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

    def detect_best_kettle(self, color_image):
        """
        检测水壶, 返回置信度最高的一个 (或 None).

        所有检测结果均属于水壶类别 (因为 set_classes 限定了词汇表),
        不需要额外过滤 class_id.

        参数:
            color_image : np.ndarray, BGR 彩色图像

        返回:
            dict or None:
                {
                    "bbox": [x1, y1, x2, y2],   # 像素坐标
                    "conf": float,               # 置信度
                    "class_id": int,             # 类别索引
                    "label": str                 # 类别名称
                }
            若未检测到任何水壶, 返回 None
        """
        results = self.model(color_image, conf=self.conf_threshold, verbose=False)

        best = None
        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                if best is None or conf > best["conf"]:
                    best = {
                        "bbox": [int(box[0]), int(box[1]),
                                 int(box[2]), int(box[3])],
                        "conf": float(conf),
                        "class_id": int(cls_id),
                        "label": self.model.names[cls_id]
                    }

        return best

    # ----------------------------------------------------------

    @staticmethod
    def draw_detection(image, detection):
        """
        在图像上绘制检测框 + 标签文字.

        参数:
            image     : np.ndarray, 待绘制的 BGR 图像 (会被修改)
            detection : dict, detect_best_kettle() 返回值, 或 None

        返回:
            np.ndarray, 绘制后的图像
        """
        if detection is None:
            cv2.putText(image, "No kettle detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return image

        x1, y1, x2, y2 = detection["bbox"]
        conf = detection["conf"]
        label = detection["label"]

        # 绘制检测框 (绿色)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 标签: "kettle 0.85"
        text = "%s %.2f" % (label, conf)
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
        )
        cv2.rectangle(image, (x1, y1 - text_h - baseline - 4),
                      (x1 + text_w, y1), (0, 255, 0), -1)
        cv2.putText(image, text, (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # 显示检测信息
        info = "Best kettle: %.2f" % conf
        cv2.putText(image, info, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        return image

    # ----------------------------------------------------------

    def run(self):
        """
        主循环: 采集 -> YOLO-World 检测 -> 绘制 -> 显示.

        按 'q' 或 Ctrl+C 退出.
        检测到水壶时打印结果到控制台.
        """
        print("=" * 60)
        print("  RealSense YOLO-World 零样本水壶检测")
        print("  模型: %s" % MODEL_PATH)
        print("  词汇表: %s" % str(self.kettle_classes))
        print("  分辨率: %dx%d @ %dfps" % (FRAME_W, FRAME_H, FRAME_FPS))
        print("  置信度阈值: %.2f" % self.conf_threshold)
        print("=" * 60)
        print("[INFO] 按 'q' 键退出程序...")

        try:
            while True:
                # ---- [Step 1] 获取彩色帧 ----
                color_image = self.get_color_frame()
                if color_image is None:
                    continue

                # ---- [Step 2] YOLO-World 水壶检测 ----
                best = self.detect_best_kettle(color_image)

                # ---- [Step 3] 绘制检测结果 ----
                display_img = color_image.copy()
                display_img = self.draw_detection(display_img, best)

                # ---- [Step 4] 控制台输出 ----
                if best is not None:
                    print("\r[KETTLE] %s | conf=%.3f | "
                          "bbox=(%d,%d,%d,%d)     " % (
                              best["label"], best["conf"],
                              best["bbox"][0], best["bbox"][1],
                              best["bbox"][2], best["bbox"][3]
                          ), end="", flush=True)
                else:
                    print("\r[KETTLE] No kettle detected     ", end="", flush=True)

                # ---- [Step 5] 显示 ----
                cv2.imshow("YOLO-World Kettle Detection", display_img)

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


# ============================================================
#  主函数
# ============================================================

def main():
    print("[INFO] 正在初始化 YOLO-World 水壶检测器...")
    print("[INFO] 首次加载会自动下载 CLIP 权重 (~338MB), 请耐心等待...")
    print("[INFO] CLIP 缓存位置: C:\\Users\\<username>\\.cache\\clip\\ViT-B-32.pt")

    detector = YOLOWorldKettleDetector(
        model_path=MODEL_PATH,
        classes=KETTLE_CLASSES,
        conf_threshold=CONF_THRESHOLD
    )
    detector.run()


if __name__ == "__main__":
    main()