# -*- coding: utf-8 -*-
"""
EpicEye YOLOv8n 水瓶3D定位抓拍
==============================
使用 EpicEye 3D 结构光相机 + YOLO 实现水瓶检测与 3D 定位

工作模式:
  本相机为 HTTP 触发式抓拍，非实时视频流。
  每次触发拍照约耗时 3~5 秒（含网络传输），检测结果保持显示在窗口。

使用方法:
  1. 连接 EpicEye 相机（需在同一网段）
  2. 运行: python EpicEye/test_3D.py
  3. 按键操作:
       空格/回车  →  触发下一次抓拍
       s          →  保存本次结果到 captures/
       q / ESC    →  退出

输入:
  - EpicEye/epiceye_config.yaml   : 相机/YOLO/深度配置

输出:
  - 显示窗口: 灰度图 + 深度伪彩色（带检测框和3D坐标）
  - 控制台:   (x, y, z) 坐标
  - EpicEye/captures/: 保存 image/point_cloud/depth

架构:
  - EpicEyeYOLO3DDetector: 封装相机操作 + YOLO 检测 + 3D 定位
  - 主循环: 抓拍 → 检测 → 显示 → 等待按键
"""

import os
import sys
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
SMOOTH_ALPHA = _cfg.get("smoothing", {}).get("alpha", 0.4)
SAVE_DIR = _cfg.get("save", {}).get("dir", "captures")

# 设备选择
if DEVICE_CFG == "gpu" and torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"[INFO] Running on device: {DEVICE} (config: {DEVICE_CFG})")


# =============================================================
# 辅助函数: 时间戳 / 保存目录
# =============================================================

def _timestamp_str():
    """生成用于文件命名的时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _get_save_dir():
    """获取保存目录（EpicEye/captures/）"""
    epiceye_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(epiceye_dir, SAVE_DIR)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


# =============================================================
# 功能模块: 时域平滑
# =============================================================

class PositionSmoother:
    """EMA 时域平滑器"""
    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered = None

    def update(self, pos):
        if pos is None:
            return None
        if self.filtered is None:
            self.filtered = list(pos)
        else:
            for i in range(3):
                self.filtered[i] = (1 - self.alpha) * self.filtered[i] + self.alpha * pos[i]
        return tuple(self.filtered)

    def reset(self):
        self.filtered = None


# =============================================================
# 功能模块: EpicEye YOLO 3D 检测器
# =============================================================

class EpicEyeYOLO3DDetector:
    """
    EpicEye YOLO 3D 检测器

    封装相机触发、YOLO 检测、3D 定位、显示、保存等完整流程。
    每次触发拍照约 3~5 秒，结果保持显示在窗口。

    使用示例:
        detector = EpicEyeYOLO3DDetector()
        detector.run()
    """

    def __init__(self, ip="", auto_search=True):
        """
        初始化检测器

        参数:
            ip          : str, 相机 IP，为空则根据配置或自动搜索决定
            auto_search : bool, 是否自动搜索相机
        """
        # ---- 初始化相机 ----
        self.camera = EpicEyeCamera(ip=ip or CAMERA_IP, auto_search=auto_search or AUTO_SEARCH)

        # ---- 加载 YOLO 模型 ----
        model_path = f"{MODEL_NAME}.pt"
        if not os.path.exists(model_path):
            model_path = MODEL_NAME
        self.model = YOLO(model_path).to(DEVICE)
        print(f"[INFO] YOLO 模型已加载: {model_path} | 类别: {CLASS_ID} | 阈值: {CONF_THRESHOLD}")

        # ---- 缓存相机内参 ----
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

        # ---- 时域平滑 ----
        self._smoother = PositionSmoother(alpha=SMOOTH_ALPHA)

        # ---- 上次抓拍结果缓存（用于显示和保存） ----
        self._last_color = None
        self._last_depth = None
        self._last_depth_colormap = None
        self._last_frame_id = None
        self._last_position = None

        print("[INFO] EpicEye YOLO 3D 检测器初始化完成")
        print("[INFO] 按键: 空格=抓拍  s=保存  q=退出")

    # ----------------------------------------------------------
    # 帧获取
    # ----------------------------------------------------------

    def get_frames(self):
        """
        触发抓拍并获取 2D 图像和深度图

        返回:
            color_image     : numpy.ndarray, 8bit 灰度图 (H x W)，已转为3通道用于YOLO
            depth_data      : numpy.ndarray or None, 原始深度数据 (H x W)
            depth_colormap  : numpy.ndarray or None, 深度伪彩色图
        """
        self.camera.logger.info("触发抓拍...")

        # 触发拍照
        frame_id = self.camera.trigger_frame(pointcloud=False)
        if frame_id is None:
            print("[ERROR] 触发拍照失败")
            return None, None, None, None

        # 获取 2D 图像（10bit raw → 8bit）
        image_raw = self.camera.get_image(frame_id)
        if image_raw is None:
            print("[ERROR] 获取图像失败")
            return None, None, None, None

        # 判断是否为单通道灰度图
        if len(image_raw.shape) == 2:
            # 单通道灰度 → 复制为3通道（YOLO需要3通道输入）
            color_image = cv2.cvtColor(image_raw, cv2.COLOR_GRAY2BGR)
        else:
            color_image = image_raw

        # 获取深度图
        depth_data = self.camera.get_depth(frame_id)
        depth_colormap = None
        if depth_data is not None:
            # 深度伪彩色图
            depth_valid = (depth_data > DEPTH_MIN).astype(np.uint8) * 255
            depth_norm = cv2.normalize(depth_data, np.zeros_like(depth_data), 0, 255, cv2.NORM_MINMAX)
            depth_colormap = cv2.applyColorMap(depth_norm.astype(np.uint8), cv2.COLORMAP_JET)
            depth_colormap[depth_valid == 0] = 0
        else:
            print("[WARN] 获取深度图失败")

        return color_image, depth_data, depth_colormap, frame_id

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

        与 test_3D.py / Orbbec/test_3D.py 相同的算法:
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
    # 保存结果
    # ----------------------------------------------------------

    def save_results(self, color_image, depth_data, depth_colormap, frame_id, save_dir=None):
        """
        保存抓拍结果到文件

        保存内容:
          - image_<timestamp>.png      : 灰度图
          - depth_<timestamp>.png      : 深度伪彩色图
          - position_<timestamp>.txt   : 3D 位置坐标

        参数:
            color_image     : numpy.ndarray, 3通道彩色图
            depth_data      : numpy.ndarray, 深度数据
            depth_colormap  : numpy.ndarray, 深度伪彩色图
            frame_id        : str, 帧ID
            save_dir        : str or None, 保存目录

        返回:
            dict, 保存的文件路径
        """
        save_dir = save_dir or _get_save_dir()
        ts = _timestamp_str()

        # 保存灰度图（将3通道转回单通道）
        if len(color_image.shape) == 3 and color_image.shape[2] == 3:
            gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = color_image
        img_path = os.path.join(save_dir, f"image_{ts}.png")
        cv2.imwrite(img_path, gray)

        # 保存深度伪彩色图
        depth_path = None
        if depth_colormap is not None:
            depth_path = os.path.join(save_dir, f"depth_{ts}.png")
            cv2.imwrite(depth_path, depth_colormap)

        # 保存 3D 位置
        pos_path = None
        if self._last_position is not None:
            pos_path = os.path.join(save_dir, f"position_{ts}.txt")
            with open(pos_path, "w", encoding="utf-8") as f:
                f.write(f"x={self._last_position[0]:.4f}\n")
                f.write(f"y={self._last_position[1]:.4f}\n")
                f.write(f"z={self._last_position[2]:.4f}\n")

        print(f"\n[SAVE] 已保存:")
        print(f"  图像:     {img_path}")
        if depth_path:
            print(f"  深度图:   {depth_path}")
        if pos_path:
            print(f"  位置:     {pos_path}")

        return {
            "image": img_path,
            "depth": depth_path,
            "position": pos_path,
        }

    # ----------------------------------------------------------
    # 运行主循环
    # ----------------------------------------------------------

    def run(self):
        """
        运行主循环

        流程:
          1. 等待用户按键
          2. 按空格 → 触发抓拍 → YOLO检测 → 3D定位 → 显示
          3. 按 s → 保存结果
          4. 按 q → 退出
        """
        WIN_NAME = "EpicEye YOLO 3D Detection"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 1280, 600)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 3D 水瓶检测")
        print("  " + "-" * 56)
        print("  [空格] 触发抓拍  [s] 保存结果  [q] 退出")
        print("=" * 60)

        # 初始显示提示界面
        hint_img = np.zeros((600, 1280, 3), dtype=np.uint8)
        cv2.putText(hint_img, "EpicEye YOLO 3D Detection", (200, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        cv2.putText(hint_img, "Press SPACE to capture", (200, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.putText(hint_img, "Press S to save, Q to quit", (200, 330),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
        cv2.imshow(WIN_NAME, hint_img)

        try:
            while True:
                # 检查窗口是否被关闭
                if cv2.getWindowProperty(WIN_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    break

                key = cv2.waitKey(50) & 0xFF

                # ---- 退出 ----
                if key == ord("q") or key == ord("Q") or key == 27:
                    print("\n[INFO] 用户退出")
                    break

                # ---- 保存 ----
                if key == ord("s") or key == ord("S"):
                    if self._last_color is not None:
                        self.save_results(
                            self._last_color,
                            self._last_depth,
                            self._last_depth_colormap,
                            self._last_frame_id,
                        )
                    else:
                        print("[WARN] 尚无抓拍结果，请先按空格触发抓拍")
                    continue

                # ---- 触发抓拍 ----
                if key == ord(" ") or key == 13:  # 空格或回车
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 触发抓拍...")
                    print("  等待相机响应...", end=" ", flush=True)

                    color_image, depth_data, depth_colormap, frame_id = self.get_frames()

                    if color_image is None:
                        print("[失败]")
                        continue
                    print("完成")

                    # ---- YOLO 检测 ----
                    print("  YOLO 检测...", end=" ", flush=True)
                    detections = self.detect_targets(color_image)
                    print(f"检测到 {len(detections)} 个目标")

                    # ---- 3D 定位与可视化 ----
                    display_img = color_image.copy()
                    best_position = None

                    for det in detections:
                        bbox = det[:4]

                        # 计算 3D 位置
                        position = self.get_3d_position(bbox, depth_data)
                        if position is not None:
                            smoothed = self._smoother.update(position)
                            if smoothed is not None:
                                best_position = smoothed
                                print(f"  3D位置: x={smoothed[0]:.3f}, y={smoothed[1]:.3f}, z={smoothed[2]:.3f} m")

                        # 绘制检测框
                        x1, y1, x2, y2 = map(int, bbox)
                        conf = det[4]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        # 中心点
                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)
                        cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)

                        # 显示 3D 坐标
                        if position is not None and smoothed is not None:
                            text_pos = f"({smoothed[0]:.2f}, {smoothed[1]:.2f}, {smoothed[2]:.2f})"
                            cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            text_conf = f"conf: {conf:.2f}"
                            cv2.putText(display_img, text_conf, (x1, y1 - 30),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    self._last_position = best_position

                    # ---- 更新显示 ----
                    # 缩放至合适显示尺寸
                    h, w = display_img.shape[:2]
                    display_w = 640
                    display_h = int(h * display_w / w)
                    img_display = cv2.resize(display_img, (display_w, display_h))

                    if depth_colormap is not None:
                        depth_display = cv2.resize(depth_colormap, (display_w, display_h))
                        # 并排显示
                        display_combined = np.hstack((img_display, depth_display))
                    else:
                        display_combined = img_display

                    # 添加提示文字
                    cv2.putText(display_combined, "[SPACE] Capture  [s] Save  [q] Quit",
                                (10, display_combined.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    if best_position is not None:
                        pos_text = f"3D: ({best_position[0]:.3f}, {best_position[1]:.3f}, {best_position[2]:.3f}) m"
                        cv2.putText(display_combined, pos_text, (10, 25),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    cv2.imshow(WIN_NAME, display_combined)

                    # 缓存当前结果
                    self._last_color = color_image
                    self._last_depth = depth_data
                    self._last_depth_colormap = depth_colormap
                    self._last_frame_id = frame_id

                    print("  就绪，等待按键...")

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            cv2.destroyAllWindows()
            self.close()

    # ----------------------------------------------------------
    # 资源清理
    # ----------------------------------------------------------

    def close(self):
        """清理相机资源"""
        if hasattr(self, "camera") and self.camera is not None:
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