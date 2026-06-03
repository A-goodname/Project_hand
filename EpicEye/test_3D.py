# -*- coding: utf-8 -*-
"""
EpicEye YOLO 水瓶3D定位 (灰度版)
===============================
使用 EpicEye 3D 结构光相机 + YOLO (灰度模型)实现水瓶检测与 3D 定位

工作模式:
  online=true (默认): 连接相机触发抓拍,检测后保存到 captures/
  online=false:      从 captures/ 加载已有数据进行3D定位 (无需相机)

数据文件 (一组数据):
  - image_<ts>.png  : 灰度图
  - depth_<ts>.png  : 深度伪彩色图 (用于可视化显示)
  - cloud_<ts>.ply  : 点云 (ASCII PLY, N×3, 单位毫米)

3D 定位方案:
  - 统一使用 PLY 点云投影筛选 bbox 内点取均值
  - online 模式: 抓拍后保存为 PLY, 再按 offline 流程加载处理
  - offline 模式: 直接从 captures/ 加载 PLY 点云

使用方法:
  1. 修改 EpicEye/epiceye_config.yaml 中的 mode.online 设置
  2. 运行: python EpicEye/test_3D.py
  3. 按键操作:
        空格/p  →  触发抓拍 (online模式)
        n       →  加载下一组数据 (offline模式)
        p       →  加载上一组数据 (offline模式)
        q       →  退出

输入:
  - EpicEye/epiceye_config.yaml   : 相机/YOLO/深度配置

输出:
  - 显示窗口: 图像 (带检测框和3D坐标)
  - 控制台:   (x, y, z) 坐标 (单位米)
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
GRAY_MODEL_NAME = _cfg.get("model", {}).get("gray_path", "../model/yolo11n-grayscale")
CLASS_ID = _cfg.get("model", {}).get("class_id", 39)
CONF_THRESHOLD = _cfg.get("model", {}).get("conf_threshold", 0.5)
DEVICE_CFG = _cfg.get("model", {}).get("device", "gpu")
DEPTH_MIN = _cfg.get("depth", {}).get("min", 0.15)
DEPTH_MAX = _cfg.get("depth", {}).get("max", 1.5)
DEPTH_Z_THRESHOLD = _cfg.get("depth", {}).get("z_threshold", 0.05)
MODE_ONLINE = _cfg.get("mode", {}).get("online", True)
LOAD_TIMESTAMP = _cfg.get("mode", {}).get("load_timestamp", "")
SAVE_DIR = _cfg.get("save", {}).get("dir", "captures")
CAMERA_INTRINSICS = _cfg.get("camera_intrinsics", {})

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


def _list_captures(captures_dir):
    """
    列出 captures 目录下的所有抓拍时间戳 (按时间排序)

    返回:
        list of str, 时间戳列表 (按时间升序)
    """
    image_files = sorted(glob.glob(os.path.join(captures_dir, "image_*.png")))
    timestamps = []
    for f in image_files:
        basename = os.path.basename(f)
        match = re.match(r"image_(\d{8}_\d{6}_\d{3})\.png", basename)
        if match:
            timestamps.append(match.group(1))
    return timestamps


def _load_depth_png(filepath):
    """
    从 PNG 加载深度数据 (转换为米)

    支持两种格式:
      - 3通道伪彩色图 (由 epiceye_camera.save_depth 保存, scale=10.0)
      - 单通道 16-bit 原始深度 (由 _save_depth_png 保存, 单位毫米)

    参数:
        filepath : str, 文件路径

    返回:
        numpy.ndarray or None, 深度数据 (米)
    """
    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # 3通道 → 伪彩色深度图 (还原为米)
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray.astype(np.float32) / 10.0

    # 单通道 → 16-bit 原始深度 (毫米转米)
    return img.astype(np.float32) / 1000.0


def _parse_ply(filepath):
    """
    解析 ASCII PLY 文件为 N×3 numpy 数组

    参数:
        filepath : str, PLY 文件路径

    返回:
        numpy.ndarray or None, N×3 点云数据 (单位毫米)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    header_end = 0
    vertex_count = 0
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("element vertex"):
            vertex_count = int(line.split()[-1])
        elif line == "end_header":
            header_end = i + 1
            break

    if vertex_count == 0 or header_end == 0:
        print(f"[ERROR] PLY 文件格式错误: {filepath}")
        return None

    vertices = []
    for line in lines[header_end:header_end + vertex_count]:
        parts = line.strip().split()
        if len(parts) >= 3:
            try:
                vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
            except ValueError:
                continue

    if len(vertices) < 10:
        print(f"[WARN] PLY 文件有效点数不足: {len(vertices)}")
        return None

    return np.array(vertices, dtype=np.float64)


def _filter_ply_by_bbox(ply_points, bbox, K, img_shape):
    """
    将 PLY 点云投影回图像,筛选出落在 bbox 内的点 (单位转换)

    参数:
        ply_points : N×3 点云 (相机坐标系,单位毫米)
        bbox       : [x1, y1, x2, y2]
        K          : 3×3 相机内参矩阵
        img_shape  : (H, W) 图像尺寸

    返回:
        N'×3 点云 (单位米),仅 bbox 内且在深度范围内的点
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x1, y1, x2, y2 = map(int, bbox[:4])
    H, W = img_shape

    x_vals = ply_points[:, 0]
    y_vals = ply_points[:, 1]
    z_vals = ply_points[:, 2]

    z_vals_m = z_vals / 1000.0

    u = np.round(fx * x_vals / z_vals + cx).astype(int)
    v = np.round(fy * y_vals / z_vals + cy).astype(int)

    mask = (
        (u >= x1) & (u < x2) &
        (v >= y1) & (v < y2) &
        (u >= 0) & (u < W) &
        (v >= 0) & (v < H) &
        (z_vals_m > DEPTH_MIN) & (z_vals_m < DEPTH_MAX)
    )

    # ---- 离群过滤: 仅使用 bbox 中心 20% 区域 z_median 作为锚点 ----
    if np.any(mask):
        z_masked = z_vals_m[mask]
        u_masked = u[mask]
        v_masked = v[mask]

        cx_bbox = (x1 + x2) / 2
        cy_bbox = (y1 + y2) / 2
        half_w = (x2 - x1) * 0.1
        half_h = (y2 - y1) * 0.1

        center_mask = (
            (u_masked >= cx_bbox - half_w) & (u_masked <= cx_bbox + half_w) &
            (v_masked >= cy_bbox - half_h) & (v_masked <= cy_bbox + half_h)
        )

        if np.any(center_mask):
            anchor_depth = np.median(z_masked[center_mask])
            z_dev = np.abs(z_masked - anchor_depth)
            valid_z = z_dev < DEPTH_Z_THRESHOLD
            sub_indices = np.where(mask)[0]
            outlier_mask = np.ones(len(mask), dtype=bool)
            outlier_mask[sub_indices[~valid_z]] = False
            mask = mask & outlier_mask

    # 返回米单位
    result = ply_points[mask].astype(np.float64)
    result[:, 2] = result[:, 2] / 1000.0
    result[:, 0] = result[:, 0] / 1000.0
    result[:, 1] = result[:, 1] / 1000.0
    return result


def _load_capture_offline(timestamp):
    """
    加载指定时间戳的抓拍数据 (与 test_6D 一致)

    参数:
        timestamp : str, 时间戳 (YYYYMMDD_HHMMSS_fff)

    返回:
        color_image : 3通道图像
        point_cloud : N×3 点云 (单位毫米)或 None
        depth_data  : 深度伪彩色图 (H x W) 或 None
    """
    captures_dir = _get_captures_dir()

    # 加载灰度图
    img_path = os.path.join(captures_dir, f"image_{timestamp}.png")
    if not os.path.exists(img_path):
        print(f"[ERROR] 图像不存在: {img_path}")
        return None, None, None
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        print(f"[ERROR] 图像加载失败: {img_path}")
        return None, None, None
    color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 加载点云 (PLY → N×3, 单位毫米)
    point_cloud = None
    ply_path = os.path.join(captures_dir, f"cloud_{timestamp}.ply")
    if os.path.exists(ply_path):
        point_cloud = _parse_ply(ply_path)

    # 加载深度伪彩色图
    depth_data = None
    depth_path = os.path.join(captures_dir, f"depth_{timestamp}.png")
    if os.path.exists(depth_path):
        depth_data = _load_depth_png(depth_path)

    return color_image, point_cloud, depth_data


# =============================================================
# 功能模块: EpicEye YOLO 3D 检测器
# =============================================================

class EpicEyeYOLO3DDetector:
    """
    EpicEye YOLO 3D 检测器

    online 模式连接相机抓拍,offline 模式从 captures/ 加载数据.
    均使用 PLY 点云 + _filter_ply_by_bbox 投影筛选流程.
    """

    def __init__(self):
        """初始化检测器"""
        self.is_online = MODE_ONLINE

        # ---- 加载 YOLO 模型 (固定灰度模型) ----
        model_path = f"{GRAY_MODEL_NAME}.pt"
        if not os.path.exists(model_path):
            model_path = GRAY_MODEL_NAME
        self.model = YOLO(model_path).to(DEVICE)
        print(f"[INFO] YOLO 灰度模型已加载: {model_path} | 类别: {CLASS_ID} | 阈值: {CONF_THRESHOLD}")

        # ---- 在线模式: 初始化相机 ----
        self.camera: Optional[EpicEyeCamera] = None
        if self.is_online:
            self.camera = EpicEyeCamera(ip=CAMERA_IP, auto_search=AUTO_SEARCH)
            self._cache_intrinsics()
            print("[INFO] 按键: 空格/p=抓拍  q=退出")
        else:
            # offline 模式从配置文件读取内参
            ci = CAMERA_INTRINSICS
            self.fx = ci.get("fx")
            self.fy = ci.get("fy")
            self.cx = ci.get("cx")
            self.cy = ci.get("cy")
            if None in (self.fx, self.fy, self.cx, self.cy):
                print("[WARN] 配置文件缺少相机内参,3D 定位将不可用")
                self.camera_matrix = None
            else:
                self.camera_matrix = np.array(
                    [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]],
                    dtype=float,
                )
            print("[INFO] 按键: n=下一组  q=退出")

    # ----------------------------------------------------------
    # 相机内参缓存
    # ----------------------------------------------------------

    def _cache_intrinsics(self):
        """缓存相机内参 (仅 online 模式调用)"""
        assert self.camera is not None
        camera_matrix = self.camera.get_camera_matrix()
        if camera_matrix is not None:
            self.camera_matrix = np.array(camera_matrix, dtype=float).reshape(3, 3)
            self.fx = self.camera_matrix[0, 0]
            self.fy = self.camera_matrix[1, 1]
            self.cx = self.camera_matrix[0, 2]
            self.cy = self.camera_matrix[1, 2]
            print(f"[INFO] 相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        else:
            print("[ERROR] 无法获取相机内参矩阵,3D 定位将不可用")
            self.fx = self.fy = self.cx = self.cy = None

    # ----------------------------------------------------------
    # 帧获取 (通过 camera.save_all 保存,再加载 PLY)
    # ----------------------------------------------------------

    def trigger_and_save(self):
        """
        触发相机抓拍并保存到 captures/, 返回数据

        使用 camera.save_all() 保存图像/点云/深度,
        再按 offline 流程加载 PLY 点云, 统一处理流程.

        返回:
            color_image : 3通道图像
            point_cloud : N×3 PLY 点云 (单位毫米),或 None
            depth_data  : 深度图,或 None
        """
        assert self.camera is not None
        self.camera.logger.info("触发抓拍...")
        frame_id = self.camera.trigger_frame(pointcloud=True)
        if frame_id is None:
            print("[ERROR] 触发拍照失败")
            return None, None, None

        paths = self.camera.save_all(frame_id)
        if paths["image"] is None:
            print("[ERROR] 图像保存失败")
            return None, None, None

        img_path = paths["image"]
        ply_path = paths["point_cloud"]
        depth_path = paths["depth"]

        # 加载灰度图
        gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            print("[ERROR] 图像加载失败")
            return None, None, None
        color_image = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # 加载 PLY 点云
        point_cloud = None
        if ply_path is not None and os.path.exists(ply_path):
            point_cloud = _parse_ply(ply_path)

        # 加载深度图
        depth_data = None
        if depth_path is not None and os.path.exists(depth_path):
            depth_data = _load_depth_png(depth_path)

        print(f"  [SAVE] 图像: {os.path.basename(img_path)}")
        if ply_path:
            print(f"  [SAVE] 点云: {os.path.basename(ply_path)} ({point_cloud.shape[0]} 点)" if point_cloud is not None else "  [SAVE] 点云: 空")
        if depth_path:
            print(f"  [SAVE] 深度: {os.path.basename(depth_path)}")

        return color_image, point_cloud, depth_data

    # ----------------------------------------------------------
    # YOLO 检测
    # ----------------------------------------------------------

    def detect_targets(self, color_image):
        """
        YOLO 检测水瓶目标 (固定灰度输入)

        参数:
            color_image : numpy.ndarray, 3通道图像 (H x W x 3)

        返回:
            list, 每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        # 转灰度输入
        gray_input = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
        results = self.model(gray_input, conf=CONF_THRESHOLD, classes=[CLASS_ID])

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
    # 3D 位置计算 (统一使用 _filter_ply_by_bbox)
    # ----------------------------------------------------------

    def get_3d_position(self, bbox, point_cloud):
        """
        从点云计算目标 3D 位置 (投影筛选 bbox 内点后取均值)

        参数:
            bbox        : [x1, y1, x2, y2, conf, cls_id]
            point_cloud : N×3 点云 (单位毫米)

        返回:
            (x, y, z) or None, 3D 位置 (米)
        """
        if point_cloud is None or self.camera_matrix is None:
            return None

        points = _filter_ply_by_bbox(point_cloud, bbox, self.camera_matrix, (1200, 1920))

        if len(points) < 10:
            return None

        center = np.mean(points, axis=0)
        return (float(center[0]), float(center[1]), float(center[2]))

    # ----------------------------------------------------------
    # 核心处理: 检测 + 3D定位 + 可视化
    # ----------------------------------------------------------

    def process_and_display(self, color_image, point_data, win_name, display_label=""):
        """
        执行 YOLO 检测 → 3D定位 → 绘制结果 → 显示

        参数:
            color_image   : 3通道图像
            point_data    : N×3 PLY 点云 (单位毫米)或 None
            win_name      : OpenCV 窗口名
            display_label : 显示标签 (如时间戳)

        返回:
            display_img   : 显示图像
            best_position : (x, y, z) 或 None
        """
        # YOLO 检测
        detections = self.detect_targets(color_image)
        print(f"  YOLO 检测到 {len(detections)} 个目标")

        display_img = color_image.copy()
        best_position = None

        for det in detections:
            bbox = det[:4]
            conf = det[4]

            position = self.get_3d_position(bbox, point_data) if point_data is not None else None

            if position is not None:
                best_position = position
                print(f"  3D位置: x={position[0]:.3f}, y={position[1]:.3f}, z={position[2]:.3f} m")

            # 绘制检测框
            x1, y1, x2, y2 = map(int, bbox)
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

        # 缩放显示
        h, w = display_img.shape[:2]
        display_w = 960
        display_h = int(h * display_w / w)
        display_img = cv2.resize(display_img, (display_w, display_h))

        # 显示标签
        if display_label:
            cv2.putText(display_img, display_label, (10, display_img.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if best_position is not None:
            pos_text = f"3D: ({best_position[0]:.3f}, {best_position[1]:.3f}, {best_position[2]:.3f}) m"
            cv2.putText(display_img, pos_text, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow(win_name, display_img)
        return display_img, best_position

    # ----------------------------------------------------------
    # Online 主循环
    # ----------------------------------------------------------

    def run_online(self):
        """online 模式主循环"""
        WIN_NAME = "EpicEye YOLO 3D Detection (Online)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 720)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 3D 水瓶检测 [Online]")
        print("  " + "-" * 56)
        print("  [空格/p] 触发抓拍并保存  [q] 退出")
        print("=" * 60)

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

                    # 触发拍照 → 保存到 captures/ → 加载 PLY 点云
                    color_image, point_cloud, depth_data = self.trigger_and_save()

                    if color_image is None:
                        print("[失败]")
                        continue
                    print("完成")

                    # 检测 + 3D定位 + 显示
                    display_img, best_position = self.process_and_display(
                        color_image, point_cloud, WIN_NAME,
                        display_label="[SPACE/p] Capture  [q] Quit"
                    )

                    print("  就绪,等待按键...")

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
                print(f"[WARN] 指定时间戳 {LOAD_TIMESTAMP} 未找到,加载最新")
                current_idx = len(timestamps) - 1
        else:
            current_idx = len(timestamps) - 1

        WIN_NAME = "EpicEye YOLO 3D Detection (Offline)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 720)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 3D 水瓶检测 [Offline]")
        print("  " + "-" * 56)
        print("  [n] 下一组  [p] 上一组  [q] 退出")
        print("=" * 60)

        def load_and_show(idx):
            """加载并显示指定索引的数据"""
            ts = timestamps[idx]
            print(f"\n[{ts}] 加载数据 ({idx + 1}/{len(timestamps)})...")
            color_image, point_cloud, depth_data = _load_capture_offline(ts)
            if color_image is None:
                print("[ERROR] 加载失败")
                return

            display_img, best_position = self.process_and_display(
                color_image, point_cloud, WIN_NAME,
                display_label=f"[{ts}]  [n] Next  [p] Prev  [q] Quit"
            )
            print("  就绪")

        # 显示初始数据
        load_and_show(current_idx)

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