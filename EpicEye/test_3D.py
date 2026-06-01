# -*- coding: utf-8 -*-
"""
EpicEye YOLOv8n 水瓶3D定位抓拍
==============================
使用 EpicEye 3D 结构光相机 + YOLO 实现水瓶检测与 3D 定位

工作模式:
  online=true (默认): 连接相机触发抓拍，检测后保存到 captures/
  online=false:      从 captures/ 加载已有数据进行3D定位（无需相机）

使用方法:
  1. 修改 EpicEye/epiceye_config.yaml 中的 mode.online 设置
  2. 运行: python EpicEye/test_3D.py
  3. 按键操作:
        空格/p  →  触发抓拍（online模式）
        n       →  加载下一组数据（offline模式）
        q       →  退出

输入:
  - EpicEye/epiceye_config.yaml   : 相机/YOLO/深度配置

输出:
  - 显示窗口: 灰度图 + 深度伪彩色（带检测框和3D坐标）
  - 控制台:   (x, y, z) 坐标
"""

import os
import re
import sys
import glob
import yaml
from datetime import datetime

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from typing import Optional
from EpicEye.epiceye_camera import EpicEyeCamera


# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "epiceye_config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 提取配置项
CAMERA_IP = _cfg.get("camera", {}).get("ip", "")
AUTO_SEARCH = _cfg.get("camera", {}).get("auto_search", True)
MODEL_NAME = _cfg.get("model", {}).get("path", "../model/yolov8n")
CLASS_ID = _cfg.get("model", {}).get("class_id", 39)
CONF_THRESHOLD = _cfg.get("model", {}).get("conf_threshold", 0.5)
DEVICE_CFG = _cfg.get("model", {}).get("device", "gpu")
DEPTH_MIN = _cfg.get("depth", {}).get("min", 0.15)
DEPTH_MAX = _cfg.get("depth", {}).get("max", 1.5)
DEPTH_Z_THRESHOLD = _cfg.get("depth", {}).get("z_threshold", 0.05)
MODE_ONLINE = _cfg.get("mode", {}).get("online", True)
LOAD_TIMESTAMP = _cfg.get("mode", {}).get("load_timestamp", "")
SAVE_DIR = _cfg.get("save", {}).get("dir", "captures")

# 设备选择
if DEVICE_CFG == "gpu" and torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"[INFO] Running on device: {DEVICE} (config: {DEVICE_CFG})")
print(f"[INFO] Mode: {'online' if MODE_ONLINE else 'offline'}")


# =============================================================
# 辅助函数
# =============================================================

def _get_captures_dir():
    """获取 captures 目录绝对路径"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), SAVE_DIR)


def _timestamp_str():
    """生成用于文件命名的时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _list_captures(captures_dir):
    """
    列出 captures 目录下的所有抓拍时间戳（按时间排序）

    返回:
        list of str, 时间戳列表（按时间升序）
    """
    image_files = sorted(glob.glob(os.path.join(captures_dir, "image_*.png")))
    timestamps = []
    for f in image_files:
        basename = os.path.basename(f)
        match = re.match(r"image_(\d{8}_\d{6}_\d{3})\.png", basename)
        if match:
            timestamps.append(match.group(1))
    return timestamps


# =============================================================
# 功能模块: EpicEye YOLO 3D 检测器
# =============================================================

class EpicEyeYOLO3DDetector:
    """
    EpicEye YOLO 3D 检测器

    online 模式连接相机抓拍，offline 模式从 captures/ 加载数据。
    """

    def __init__(self):
        """初始化检测器"""
        self.is_online = MODE_ONLINE

        # ---- 加载 YOLO 模型 ----
        model_path = f"{MODEL_NAME}.pt"
        if not os.path.exists(model_path):
            model_path = MODEL_NAME
        self.model = YOLO(model_path).to(DEVICE)
        print(f"[INFO] YOLO 模型已加载: {model_path} | 类别: {CLASS_ID} | 阈值: {CONF_THRESHOLD}")

        # ---- 在线模式: 初始化相机 ----
        self.camera: Optional[EpicEyeCamera] = None
        if self.is_online:
            self.camera = EpicEyeCamera(ip=CAMERA_IP, auto_search=AUTO_SEARCH)
            self._cache_intrinsics()
            print("[INFO] 按键: 空格/p=抓拍  q=退出")
        else:
            self.fx = self.fy = self.cx = self.cy = None
            self.camera_matrix = None
            print("[INFO] 按键: n=下一组  q=退出")

    # ----------------------------------------------------------
    # 相机内参缓存
    # ----------------------------------------------------------

    def _cache_intrinsics(self):
        """缓存相机内参（仅 online 模式调用）"""
        assert self.camera is not None
        self.camera_matrix = self.camera.get_camera_matrix()
        if self.camera_matrix is not None:
            self.fx = self.camera_matrix[0]
            self.fy = self.camera_matrix[4]
            self.cx = self.camera_matrix[2]
            self.cy = self.camera_matrix[5]
            print(f"[INFO] 相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        else:
            print("[ERROR] 无法获取相机内参矩阵，3D 定位将不可用")
            self.fx = self.fy = self.cx = self.cy = None

    # ----------------------------------------------------------
    # Online 模式: 帧获取
    # ----------------------------------------------------------

    def get_frames_online(self):
        """
        online 模式触发抓拍

        返回:
            color_image     : 3通道图像
            depth_data      : 原始深度数据 (H x W) 或 None
            depth_colormap  : 深度伪彩色图
        """
        assert self.camera is not None
        self.camera.logger.info("触发抓拍...")
        frame_id = self.camera.trigger_frame(pointcloud=False)
        if frame_id is None:
            print("[ERROR] 触发拍照失败")
            return None, None, None

        image_raw = self.camera.get_image(frame_id)
        if image_raw is None:
            print("[ERROR] 获取图像失败")
            return None, None, None

        # 灰度 → 3通道
        if len(image_raw.shape) == 2:
            color_image = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image_raw

        depth_data = self.camera.get_depth(frame_id)
        depth_colormap = None
        if depth_data is not None:
            depth_valid = (depth_data > DEPTH_MIN).astype(np.uint8) * 255
            depth_norm = cv2.normalize(depth_data, np.zeros_like(depth_data), 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
            depth_colormap[depth_valid == 0] = 0

        return color_image, depth_data, depth_colormap

    # ----------------------------------------------------------
    # Online 模式: 保存数据
    # ----------------------------------------------------------

    def _save_capture_online(self, color_image, depth_colormap):
        """
        online 模式下保存抓拍结果

        保存内容:
          - image_<timestamp>.png       : 灰度图
          - depth_<timestamp>.png       : 深度伪彩色图

        参数:
            color_image     : 3通道图像
            depth_colormap  : 深度伪彩色图或 None

        返回:
            str, 时间戳
        """
        captures_dir = _get_captures_dir()
        os.makedirs(captures_dir, exist_ok=True)
        ts = _timestamp_str()

        # 保存灰度图
        if len(color_image.shape) == 3:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = color_image
        img_path = os.path.join(captures_dir, f"image_{ts}.png")
        cv2.imwrite(img_path, gray)

        # 保存深度伪彩色图
        if depth_colormap is not None:
            depth_path = os.path.join(captures_dir, f"depth_{ts}.png")
            cv2.imwrite(depth_path, depth_colormap)

        print(f"  [SAVE] 已保存到 captures/: {ts}")
        return ts

    # ----------------------------------------------------------
    # Offline 模式: 加载数据
    # ----------------------------------------------------------

    def _load_capture_offline(self, timestamp):
        """
        offline 模式加载指定时间戳的抓拍数据

        参数:
            timestamp : str, 时间戳 (YYYYMMDD_HHMMSS_fff)

        返回:
            color_image     : 3通道图像
            depth_colormap  : 深度伪彩色图或 None
        """
        captures_dir = _get_captures_dir()

        # 加载灰度图
        img_path = os.path.join(captures_dir, f"image_{timestamp}.png")
        if not os.path.exists(img_path):
            print(f"[ERROR] 图像不存在: {img_path}")
            return None, None
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print(f"[ERROR] 图像加载失败: {img_path}")
            return None, None
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 加载深度伪彩色图
        depth_colormap = None
        depth_path = os.path.join(captures_dir, f"depth_{timestamp}.png")
        if os.path.exists(depth_path):
            depth_colormap = cv2.imread(depth_path)

        return color_image, depth_colormap

    # ----------------------------------------------------------
    # YOLO 检测
    # ----------------------------------------------------------

    def detect_targets(self, color_image):
        """
        YOLO 检测水瓶目标

        参数:
            color_image : numpy.ndarray, 3通道图像 (H x W x 3)

        返回:
            list, 每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(color_image, conf=CONF_THRESHOLD, classes=[CLASS_ID])

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if cls_id == CLASS_ID:
                    detections.append([*box, conf, cls_id])

        return detections

    # ----------------------------------------------------------
    # 3D 位置计算
    # ----------------------------------------------------------

    def get_3d_position(self, bbox, depth_data):
        """
        计算目标 3D 位置（点云均值方案）

        算法:
          1. 锚点深度: bbox 中心附近取深度中值
          2. bbox 区域多点采样
          3. 深度离群过滤
          4. 反投影为 3D 点云
          5. 点云均值作为最终位置

        参数:
            bbox       : [x1, y1, x2, y2, conf, cls_id]
            depth_data : numpy.ndarray, 深度图（米），已对齐到彩色坐标系

        返回:
            (x, y, z) or None, 3D 位置（米）
        """
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None or depth_data is None:
            return None

        fx: float = self.fx
        fy: float = self.fy
        cx: float = self.cx
        cy: float = self.cy

        # ---- Step 1: 锚点深度 ----
        cx_bbox = int((bbox[0] + bbox[2]) / 2)
        cy_bbox = int((bbox[1] + bbox[3]) / 2)
        cx_bbox = np.clip(cx_bbox, 0, depth_data.shape[1] - 1)
        cy_bbox = np.clip(cy_bbox, 0, depth_data.shape[0] - 1)

        half = 3
        x_s = max(0, cx_bbox - half)
        y_s = max(0, cy_bbox - half)
        x_e = min(depth_data.shape[1], cx_bbox + half + 1)
        y_e = min(depth_data.shape[0], cy_bbox + half + 1)

        patch = depth_data[y_s:y_e, x_s:x_e]
        valid = patch[(patch > DEPTH_MIN) & (patch < DEPTH_MAX)]
        if len(valid) < 3:
            return None
        center_depth = float(np.median(valid))

        # ---- Step 2: bbox 区域采样 + 深度过滤 + 反投影 ----
        x1 = max(0, min(int(bbox[0]), depth_data.shape[1] - 1))
        y1 = max(0, min(int(bbox[1]), depth_data.shape[0] - 1))
        x2 = max(0, min(int(bbox[2]), depth_data.shape[1] - 1))
        y2 = max(0, min(int(bbox[3]), depth_data.shape[0] - 1))

        points = []
        for vy in range(y1, y2 + 1, 2):
            for vx in range(x1, x2 + 1, 2):
                d = depth_data[vy, vx]
                if not (DEPTH_MIN < d < DEPTH_MAX):
                    continue
                if abs(d - center_depth) > DEPTH_Z_THRESHOLD:
                    continue
                depth_x = (vx - cx) * d / fx
                depth_y = (vy - cy) * d / fy
                points.append([depth_x, depth_y, d])

        if len(points) < 10:
            # 点太少，不做离群过滤再试一次
            points_wide = []
            for vy in range(y1, y2 + 1, 2):
                for vx in range(x1, x2 + 1, 2):
                    d = depth_data[vy, vx]
                    if DEPTH_MIN < d < DEPTH_MAX:
                        depth_x = (vx - cx) * d / fx
                        depth_y = (vy - cy) * d / fy
                        points_wide.append([depth_x, depth_y, d])
            if len(points_wide) < 10:
                return None
            point_cloud = np.array(points_wide)
        else:
            point_cloud = np.array(points)

        # ---- Step 3: 点云均值 ----
        center = np.mean(point_cloud, axis=0)
        return (center[0], center[1], center[2])

    # ----------------------------------------------------------
    # 核心处理: 检测 + 3D定位 + 可视化
    # ----------------------------------------------------------

    def process_and_display(self, color_image, depth_data, depth_colormap, win_name, display_label=""):
        """
        执行 YOLO 检测 → 3D定位 → 绘制结果 → 显示

        参数:
            color_image    : 3通道图像
            depth_data     : 深度数据或 None（offline模式无原始深度）
            depth_colormap : 深度伪彩色图或 None
            win_name       : OpenCV 窗口名
            display_label  : 显示标签（如时间戳）

        返回:
            display_combined : 拼接后的显示图像
            best_position    : (x, y, z) 或 None
        """
        # YOLO 检测
        detections = self.detect_targets(color_image)
        print(f"  YOLO 检测到 {len(detections)} 个目标")

        display_img = color_image.copy()
        best_position = None

        for det in detections:
            bbox = det[:4]

            if depth_data is not None:
                position = self.get_3d_position(bbox, depth_data)
            else:
                position = None

            if position is not None:
                best_position = position
                print(f"  3D位置: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} m")

            # 绘制检测框
            x1, y1, x2, y2 = map(int, bbox)
            conf = det[4]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 中心点
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

            # 显示 3D 坐标
            if position is not None:
                text_pos = f"({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                cv2.putText(display_img, text_pos, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                text_conf = f"conf: {conf:.2f}"
                cv2.putText(display_img, text_conf, (x1, y1 - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 缩放 + 深度并排
        h, w = display_img.shape[:2]
        display_w = 640
        display_h = int(h * display_w / w)
        img_display = cv2.resize(display_img, (display_w, display_h))

        if depth_colormap is not None:
            depth_display = cv2.resize(depth_colormap, (display_w, display_h))
            display_combined = np.hstack((img_display, depth_display))
        else:
            display_combined = img_display

        # 显示标签
        if display_label:
            cv2.putText(display_combined, display_label, (10, display_combined.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if best_position is not None:
            pos_text = f"3D: ({best_position[0]:.3f}, {best_position[1]:.3f}, {best_position[2]:.3f}) m"
            cv2.putText(display_combined, pos_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(win_name, display_combined)
        return display_combined, best_position

    # ----------------------------------------------------------
    # Online 主循环
    # ----------------------------------------------------------

    def run_online(self):
        """online 模式主循环"""
        WIN_NAME = "EpicEye YOLO 3D Detection (Online)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 600)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 3D 水瓶检测 [Online]")
        print("  " + "-" * 56)
        print("  [空格/p] 触发抓拍并保存  [q] 退出")
        print("=" * 60)

        hint_img = np.zeros((600, 1280, 3), dtype=np.uint8)
        cv2.putText(hint_img, "Online Mode - Press SPACE or P to capture", (100, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(hint_img, "Press Q to quit", (100, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow(WIN_NAME, hint_img)

        try:
            while True:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(50) & 0xFF

                if key == ord("q") or key == ord("Q") or key == 27:
                    print("\n[INFO] 用户退出")
                    break

                if key == ord(" ") or key == ord("p") or key == ord("P"):
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 触发抓拍...")
                    print("  等待相机响应...", end=" ", flush=True)

                    color_image, depth_data, depth_colormap = self.get_frames_online()

                    if color_image is None:
                        print("[失败]")
                        continue
                    print("完成")

                    # 检测 + 3D定位 + 显示
                    display_combined, best_position = self.process_and_display(
                        color_image, depth_data, depth_colormap, WIN_NAME,
                        display_label="[SPACE/p] Capture  [q] Quit"
                    )

                    # 保存到 captures/
                    self._save_capture_online(color_image, depth_colormap)

                    print("  就绪，等待按键...")

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            cv2.destroyAllWindows()

    # ----------------------------------------------------------
    # Offline 主循环
    # ----------------------------------------------------------

    def run_offline(self):
        """offline 模式主循环"""
        captures_dir = _get_captures_dir()
        if not os.path.exists(captures_dir):
            print(f"[ERROR] captures 目录不存在: {captures_dir}")
            return

        timestamps = _list_captures(captures_dir)
        if not timestamps:
            print("[ERROR] captures 目录中没有找到抓拍数据")
            return
        print(f"[INFO] 找到 {len(timestamps)} 组抓拍数据")

        if LOAD_TIMESTAMP:
            if LOAD_TIMESTAMP in timestamps:
                current_idx = timestamps.index(LOAD_TIMESTAMP)
            else:
                print(f"[WARN] 指定时间戳 {LOAD_TIMESTAMP} 未找到，加载最新")
                current_idx = len(timestamps) - 1
        else:
            current_idx = len(timestamps) - 1

        WIN_NAME = "EpicEye YOLO 3D Detection (Offline)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 600)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 3D 水瓶检测 [Offline]")
        print("  " + "-" * 56)
        print("  [n] 下一组  [p] 上一组  [q] 退出")
        print("=" * 60)

        def load_and_show(idx):
            """加载并显示指定索引的数据"""
            ts = timestamps[idx]
            print(f"\n[{ts}] 加载数据 ({idx + 1}/{len(timestamps)})...")
            color_image, depth_colormap = self._load_capture_offline(ts)
            if color_image is None:
                print("[ERROR] 加载失败")
                return

            display_combined, best_position = self.process_and_display(
                color_image, None, depth_colormap, WIN_NAME,
                display_label=f"[{ts}]  [n] Next  [p] Prev  [q] Quit"
            )
            print("  就绪")

        hint_img = np.zeros((600, 1280, 3), dtype=np.uint8)
        cv2.putText(hint_img, f"Offline Mode - {len(timestamps)} captures found", (100, 250),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        cv2.putText(hint_img, "Press N to load the latest capture", (100, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(hint_img, "Press Q to quit", (100, 350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.imshow(WIN_NAME, hint_img)

        try:
            while True:
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(50) & 0xFF

                if key == ord("q") or key == ord("Q") or key == 27:
                    print("\n[INFO] 用户退出")
                    break

                if key == ord("n") or key == ord("N"):
                    current_idx = (current_idx + 1) % len(timestamps)
                    load_and_show(current_idx)

                if key == ord("p") or key == ord("P"):
                    current_idx = (current_idx - 1) % len(timestamps)
                    load_and_show(current_idx)

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            cv2.destroyAllWindows()

    # ----------------------------------------------------------
    # 统一运行入口
    # ----------------------------------------------------------

    def run(self):
        """根据 mode.online 配置选择运行模式"""
        if self.is_online:
            self.run_online()
        else:
            self.run_offline()
        self.close()

    def close(self):
        """清理资源"""
        if self.camera is not None:
            self.camera.close()
        print("[INFO] 资源已清理")


# =============================================================
# 主函数
# =============================================================

if __name__ == "__main__":
    detector = EpicEyeYOLO3DDetector()
    try:
        detector.run()
    except Exception as e:
        print(f"[FATAL] 运行异常: {e}")
        detector.close()