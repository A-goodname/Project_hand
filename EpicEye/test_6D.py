# -*- coding: utf-8 -*-
"""
EpicEye YOLO 水瓶6D位姿估计 (灰度版)
=====================================
使用 EpicEye 3D 结构光相机 + YOLO (灰度模型)+ PCA 实现水瓶的 6D 位姿估计

工作模式:
  online=true (默认): 连接相机触发抓拍,检测后保存到 captures/
  online=false:      从 captures/ 加载已有数据进行位姿估计 (无需相机)

数据文件 (一组数据):
  - image_<ts>.png  : 灰度图
  - depth_<ts>.png  : 深度伪彩色图 (用于可视化显示)
  - cloud_<ts>.ply  : 点云 (ASCII PLY, N×3, 单位毫米)

6D 位姿方案 (PCA):
  - online 模式: 用点云 H×W×3 直接 bbox 区域采样
  - offline 模式: 用 PLY 点云投影筛选 bbox 内点
  - PCA 主成分分析求主方向 → 构建右手系旋转矩阵

使用方法:
  1. 修改 EpicEye/epiceye_config.yaml 中的 mode.online 设置
  2. 运行: python EpicEye/test_6D.py
  3. 按键操作:
        空格/p  →  触发抓拍 (online模式)
        n       →  加载下一组数据 (offline模式)
        p       →  加载上一组数据 (offline模式)
        q       →  退出

输入:
  - EpicEye/epiceye_config.yaml   : 相机/YOLO/深度/mode配置

输出:
  - online模式: 显示窗口 + 保存到 captures/
  - offline模式: 从 captures/ 加载并显示
  - 控制台:     (x, y, z, A, B, C) 位姿 (位置单位米,角度单位度)
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


def _timestamp_str():
    """生成用于文件命名的时间戳字符串"""
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _parse_ply(filepath):
    """
    解析 ASCII PLY 文件为 N×3 numpy 数组

    参数:
        filepath : str, PLY 文件路径

    返回:
        numpy.ndarray or None, N×3 点云数据
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


def _save_ply(filepath, points):
    """
    将 N×3 点云保存为 ASCII PLY 文件

    参数:
        filepath : str, 保存路径
        points   : numpy.ndarray, N×3 点云
    """
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")


def _save_depth_png(filepath, depth_meters):
    """
    将深度数据 (米)保存为 16-bit PNG (毫米)

    参数:
        filepath     : str, 保存路径
        depth_meters : numpy.ndarray, 深度数据 (米)
    """
    depth_mm = np.clip(depth_meters * 1000, 0, 65535).astype(np.uint16)
    cv2.imwrite(filepath, depth_mm)


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

    # 返回米单位
    result = ply_points[mask].astype(np.float64)
    result[:, 2] = result[:, 2] / 1000.0
    result[:, 0] = result[:, 0] / 1000.0
    result[:, 1] = result[:, 1] / 1000.0
    return result


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


# =============================================================
# 功能模块: EpicEye YOLO 6D 位姿检测器
# =============================================================

class EpicEyeYOLO6DPoseDetector:
    """
    EpicEye YOLO 6D 位姿检测器

    online 模式连接相机抓拍,offline 模式从 captures/ 加载数据.
    6D 位姿方案: PCA 主成分分析 → 构建右手系旋转矩阵
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
                print("[WARN] 配置文件缺少相机内参,使用硬编码默认值")
                self.fx, self.fy = 2082.12, 2082.32
                self.cx, self.cy = 975.08, 581.07
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
        self.camera_matrix = self.camera.get_camera_matrix()
        if self.camera_matrix is not None:
            self.fx = self.camera_matrix[0]
            self.fy = self.camera_matrix[4]
            self.cx = self.camera_matrix[2]
            self.cy = self.camera_matrix[5]
            print(f"[INFO] 相机内参: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}")
        else:
            print("[ERROR] 无法获取相机内参矩阵,6D 位姿可视化将使用默认内参")
            self.fx = self.fy = self.cx = self.cy = None

    # ----------------------------------------------------------
    # Online 模式: 帧获取
    # ----------------------------------------------------------

    def get_frames_online(self):
        """
        online 模式触发抓拍

        返回:
            color_image     : 3通道图像
            point_map       : 点云数据 (H×W×3) 或 None
            depth_data      : 原始深度数据 (H x W) 或 None
        """
        assert self.camera is not None
        self.camera.logger.info("触发抓拍...")
        frame_id = self.camera.trigger_frame(pointcloud=True)
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

        point_map = self.camera.get_point_cloud(frame_id)
        depth_data = self.camera.get_depth(frame_id)

        return color_image, point_map, depth_data

    # ----------------------------------------------------------
    # Online 模式: 保存数据
    # ----------------------------------------------------------

    def _save_capture_online(self, color_image, point_map, depth_data):
        """
        online 模式下保存抓拍结果

        保存内容:
          - image_<timestamp>.png  : 灰度图
          - cloud_<timestamp>.ply  : 点云 (ASCII PLY)
          - depth_<timestamp>.png  : 深度伪彩色图

        所有文件使用同一个时间戳,确保数据组一致性.

        参数:
            color_image : 3通道图像
            point_map   : 点云 (H×W×3) 或 None
            depth_data  : 原始深度数据 (H x W) 或 None

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

        # 保存点云 (PLY)— 动态单位判断
        if point_map is not None:
            points = point_map.reshape(-1, 3)
            finite_mask = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
            finite_points = points[finite_mask]

            if len(finite_points) > 0:
                z_valid = finite_points[:, 2]
                z_median = float(np.median(z_valid))
                if z_median > 100:
                    valid_mask = (finite_points[:, 2] > z_median * 0.1) & (finite_points[:, 2] < z_median * 5.0)
                else:
                    valid_mask = (finite_points[:, 2] > 0.01) & (finite_points[:, 2] < 10.0)
                valid_points = finite_points[valid_mask]

                if len(valid_points) > 0:
                    ply_path = os.path.join(captures_dir, f"cloud_{ts}.ply")
                    _save_ply(ply_path, valid_points)

        # 保存深度伪彩色图
        if depth_data is not None:
            depth_path = os.path.join(captures_dir, f"depth_{ts}.png")
            pseudo = cv2.applyColorMap(
                (depth_data / 10.0).astype("uint8"), cv2.COLORMAP_JET
            )
            cv2.imwrite(depth_path, pseudo)

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

        # 加载点云 (PLY → N×3)
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
    # PCA 6D 位姿计算
    # ----------------------------------------------------------

    def compute_6d_pose(self, bbox, point_cloud, depth_data=None):
        """
        计算水瓶的 6D 位姿 (PCA 方法)

        online 模式: point_cloud 为 H×W×3 点云图,直接 bbox 区域采样
        offline 模式: point_cloud 为 N×3 PLY 点云,投影回图像筛选 bbox 内点

        参数:
            bbox        : [x1, y1, x2, y2, conf, cls_id]
            point_cloud : online: (H×W×3), offline: (N×3)
            depth_data  : 深度图 (online 不需要,offline 用于锚点深度)

        返回:
            position: (x, y, z) 重心坐标 (m)
            rotation: 3x3 旋转矩阵
            points: Nx3 点云
        """
        if point_cloud is None:
            return None, None, None

        # ---- 从点云中提取目标区域点 ----
        if len(point_cloud.shape) == 3:
            # online 模式: H×W×3 → bbox 区域直接采样
            H, W = point_cloud.shape[:2]
            x1 = max(0, min(int(bbox[0]), W - 1))
            y1 = max(0, min(int(bbox[1]), H - 1))
            x2 = max(0, min(int(bbox[2]), W - 1))
            y2 = max(0, min(int(bbox[3]), H - 1))

            # 锚点深度
            cx_bbox = int((x1 + x2) / 2)
            cy_bbox = int((y1 + y2) / 2)
            half = 3
            patch = point_cloud[
                max(0, cy_bbox - half):min(H, cy_bbox + half + 1),
                max(0, cx_bbox - half):min(W, cx_bbox + half + 1),
                :
            ]
            z_vals = patch[:, :, 2].flatten()
            valid_z = z_vals[(z_vals > DEPTH_MIN) & (z_vals < DEPTH_MAX) & np.isfinite(z_vals)]
            if len(valid_z) < 3:
                return None, None, None
            center_depth = float(np.median(valid_z))

            # 采样 bbox 区域
            region = point_cloud[y1:y2 + 1:2, x1:x2 + 1:2, :]
            points_flat = region.reshape(-1, 3)
            valid_mask = (
                np.isfinite(points_flat[:, 2]) &
                (points_flat[:, 2] > DEPTH_MIN) &
                (points_flat[:, 2] < DEPTH_MAX) &
                (np.abs(points_flat[:, 2] - center_depth) < DEPTH_Z_THRESHOLD)
            )
            points = points_flat[valid_mask]

            if len(points) < 10:
                valid_mask_wide = (
                    np.isfinite(points_flat[:, 2]) &
                    (points_flat[:, 2] > DEPTH_MIN) &
                    (points_flat[:, 2] < DEPTH_MAX)
                )
                points = points_flat[valid_mask_wide]
                if len(points) < 10:
                    return None, None, None
                print(f"[INFO] anchor depth: {center_depth:.3f}m, no-filter points: {len(points)}")
            else:
                print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)}")

        else:
            # offline 模式: PLY 点云 N×3 → 投影回图像筛选 bbox 区域
            if self.camera_matrix is not None:
                K = self.camera_matrix.reshape(3, 3)
            else:
                fx, fy = self.fx, self.fy
                cx, cy = self.cx, self.cy
                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)

            # 投影筛选 bbox 内点
            points = _filter_ply_by_bbox(point_cloud, bbox, K, (1200, 1920))

            if len(points) < 10:
                print(f"[WARN] PLY 投影到 bbox 后点数不足: {len(points)}")
                return None, None, None

            print(f"[INFO] PLY projected points in bbox: {len(points)}")

        # ---- Step: PCA 位姿 ----
        center = np.mean(points, axis=0)

        cov = np.cov(points, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主方向 (最大特征值)-> Y轴 (水瓶纵向)
        target_y_main = np.real(eig_vec[:, 0])
        # 最小方差方向 -> Z轴 (指向相机)
        target_z = np.real(eig_vec[:, 2])

        if target_z[2] > 0:
            target_z = -target_z

        # X轴 = Y x Z,确保右手系
        target_x = np.cross(target_y_main, target_z)
        target_x = target_x / np.linalg.norm(target_x)
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)

        return center, rot, points

    # ----------------------------------------------------------
    # 工具: 旋转矩阵 → 欧拉角
    # ----------------------------------------------------------

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """
        从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw)

        返回:
            tuple: (A, B, C) 角度值 (deg)
        """
        if abs(abs(R[2, 0]) - 1.0) < 1e-6:
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = 0.0
                yaw = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = 0.0
                yaw = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = -np.arcsin(R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    # ----------------------------------------------------------
    # 工具: 可视化 6D 坐标轴
    # ----------------------------------------------------------

    def visualize_pose(self, image, center_3d, rot, K, scale=0.08):
        """
        在图像上绘制 6D 位姿坐标轴

        红=X轴, 绿=Y轴 (主方向,水瓶纵向), 蓝=Z轴
        """
        center_2d = K @ np.array(center_3d)
        center_2d = center_2d / center_2d[2]
        cx_2d = int(center_2d[0])
        cy_2d = int(center_2d[1])

        axes = [
            (rot[:, 0], (0, 0, 255)),   # X -> 红色
            (rot[:, 1], (0, 255, 0)),   # Y -> 绿色
            (rot[:, 2], (255, 0, 0)),   # Z -> 蓝色
        ]

        cv2.circle(image, (cx_2d, cy_2d), 5, (124, 0, 255), 3)

        for axis, color in axes:
            end_3d = np.array(center_3d) + scale * axis
            end_2d = K @ end_3d
            end_2d = end_2d / end_2d[2]
            ex_2d = int(end_2d[0])
            ey_2d = int(end_2d[1])
            cv2.arrowedLine(image, (cx_2d, cy_2d), (ex_2d, ey_2d), color, 3, tipLength=0.2)

    # ----------------------------------------------------------
    # 核心处理: 检测 + 位姿估计 + 可视化
    # ----------------------------------------------------------

    def process_and_display(self, color_image, point_data, depth_data, win_name, display_label=""):
        """
        执行 YOLO 检测 → PCA 6D 位姿 → 绘制结果 → 显示

        参数:
            color_image  : 3通道图像
            point_data   : online=point_map (H×W×3), offline=point_cloud (N×3)
            depth_data   : 深度图 (online 不需要,offline 用于锚点深度)
            win_name     : OpenCV 窗口名
            display_label: 显示标签 (如时间戳)

        返回:
            display_img  : 显示图像
            best_pose    : (position, euler) 或 None
        """
        # YOLO 检测
        detections = self.detect_targets(color_image)
        print(f"  YOLO 检测到 {len(detections)} 个目标")

        display_img = color_image.copy()
        best_pose = None

        for det in detections:
            bbox = det[:4]
            conf = det[4]
            x1, y1, x2, y2 = map(int, bbox)

            # 始终绘制检测框 (不依赖位姿计算结果)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            position, rotation, points = self.compute_6d_pose(bbox, point_data, depth_data)

            if position is not None and rotation is not None:
                A, B, C = self.rotation_matrix_to_euler_angles(rotation)
                best_pose = (position, (A, B, C))

                print(f"  6D位姿: (X={position[0]:.3f}, Y={position[1]:.3f}, "
                      f"Z={position[2]:.3f}, A={A:.1f}, B={B:.1f}, C={C:.1f})")
                assert points is not None
                print(f"  点云点数: {points.shape[0]}")

                # 可视化坐标轴
                if self.camera_matrix is not None:
                    K = self.camera_matrix.reshape(3, 3)
                else:
                    fx, fy = self.fx, self.fy
                    cx, cy = self.cx, self.cy
                    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
                self.visualize_pose(display_img, position, rotation, K, scale=0.08)

                # 显示位姿文本 (仅在物体框上方)
                text_pos = f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                cv2.putText(display_img, text_pos, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                text_euler = f"Euler: (A={A:.1f}, B={B:.1f}, C={C:.1f})"
                cv2.putText(display_img, text_euler, (x1, y1 - 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                print("  [WARN] 点云不足,无法计算位姿")
                # 即使无位姿也显示 conf
                text_conf = f"conf: {conf:.2f}"
                cv2.putText(display_img, text_conf, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 缩放显示
        h, w = display_img.shape[:2]
        display_w = 960
        display_h = int(h * display_w / w)
        display_img = cv2.resize(display_img, (display_w, display_h))

        # 显示标签
        if display_label:
            cv2.putText(display_img, display_label, (10, display_img.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        cv2.imshow(win_name, display_img)
        return display_img, best_pose

    # ----------------------------------------------------------
    # Online 主循环
    # ----------------------------------------------------------

    def run_online(self):
        """online 模式主循环"""
        WIN_NAME = "EpicEye YOLO 6D Pose (Online)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 720)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 6D 水瓶位姿估计 [Online]")
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

                    color_image, point_map, depth_data = self.get_frames_online()

                    if color_image is None:
                        print("[失败]")
                        continue
                    print("完成")

                    # 检测 + 位姿
                    display_img, best_pose = self.process_and_display(
                        color_image, point_map, depth_data, WIN_NAME,
                        display_label="[SPACE/p] Capture  [q] Quit"
                    )

                    # 保存到 captures/
                    self._save_capture_online(color_image, point_map, depth_data)

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

        # 指定或加载最新
        if LOAD_TIMESTAMP:
            if LOAD_TIMESTAMP in timestamps:
                current_idx = timestamps.index(LOAD_TIMESTAMP)
            else:
                print(f"[WARN] 指定时间戳 {LOAD_TIMESTAMP} 未找到,加载最新")
                current_idx = len(timestamps) - 1
        else:
            current_idx = len(timestamps) - 1

        WIN_NAME = "EpicEye YOLO 6D Pose (Offline)"
        cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WIN_NAME, 960, 720)

        print("\n" + "=" * 60)
        print("  EpicEye YOLO 6D 水瓶位姿估计 [Offline]")
        print("  " + "-" * 56)
        print("  [n] 下一组  [p] 上一组  [q] 退出")
        print("=" * 60)

        def load_and_show(idx):
            """加载并显示指定索引的数据"""
            ts = timestamps[idx]
            print(f"\n[{ts}] 加载数据 ({idx + 1}/{len(timestamps)})...")
            color_image, point_cloud, depth_data = self._load_capture_offline(ts)
            if color_image is None:
                print("[ERROR] 加载失败")
                return

            display_img, best_pose = self.process_and_display(
                color_image, point_cloud, depth_data, WIN_NAME,
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
    detector = EpicEyeYOLO6DPoseDetector()
    try:
        detector.run()
    except Exception as e:
        print(f"[FATAL] 运行异常: {e}")
        detector.close()