# -*- coding: utf-8 -*-
"""
RealSense 水壶把手检测 共享工具函数
======================================
提供各文件共用的函数与基类:
  - load_camera_params     : 从 YAML 加载相机参数
  - color_to_depth_pixel   : 彩色像素 → 深度像素映射
  - depth_median_around    : 深度图局部中值
  - scanline_find_handle   : 水平扫描线分离把手
  - detect_red_tape        : HSV 红色胶带检测
  - PositionSmoother       : 3D 位置 EMA 平滑器
  - PoseSmoother           : 6D 位姿 EMA 平滑器
  - rotation_matrix_to_euler_angles : 旋转矩阵 → ZYX 欧拉角
  - visualize_pose         : 在图像上绘制 6D 坐标轴
  - BaseRealSenseDetector  : 检测器基类 (封装 RealSense/YOLO/SAM 通用基础设施)
"""

import os
import datetime
import yaml
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO, SAM, FastSAM


# ============================================================
#  0. 项目根目录 (模块级, 用于模型/日志路径)
# ============================================================

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ============================================================
#  1. 相机参数加载
# ============================================================

def load_camera_params(yaml_path):
    """
    从标准 YAML 文件加载相机参数 (深度+彩色+外参).

    参数:
        yaml_path : str, YAML 配置文件路径

    返回:
        dict:
          {
              'depth_intr': rs.intrinsics,
              'color_intr': rs.intrinsics,
              'd2c_extr': rs.extrinsics,
              'c2d_extr': rs.extrinsics,
          }
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    def _intr(src):
        i = rs.intrinsics()
        i.width, i.height = data['camera']['resolution']
        i.ppx, i.ppy = src['ppx'], src['ppy']
        i.fx, i.fy = src['fx'], src['fy']
        i.model = getattr(rs.distortion, src['model'])
        i.coeffs = src['distortion']
        return i

    def _extr(src):
        e = rs.extrinsics()
        e.rotation = src['rotation']
        e.translation = src['translation']
        return e

    return {
        'depth_intr': _intr(data['depth']['intrinsics']),
        'color_intr': _intr(data['color']['intrinsics']),
        'd2c_extr': _extr(data['extrinsics']['depth_to_color']),
        'c2d_extr': _extr(data['extrinsics']['color_to_depth']),
    }


# ============================================================
#  2. 深度映射工具
# ============================================================

def color_to_depth_pixel(cx, cy, depth_frame, depth_scale,
                         depth_intr, color_intr, c2d_extr, d2c_extr):
    """
    将彩色图上的像素 (cx, cy) 映射到深度图坐标系.

    参数:
        cx, cy        : int, 彩色图像像素坐标
        depth_frame   : rs.depth_frame
        depth_scale   : float, 深度缩放系数
        depth_intr    : rs.intrinsics, 深度内参
        color_intr    : rs.intrinsics, 彩色内参
        c2d_extr      : rs.extrinsics, 彩色→深度外参
        d2c_extr      : rs.extrinsics, 深度→彩色外参

    返回:
        (dx, dy) 或 None (映射失败时)
    """
    try:
        depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
            depth_frame.get_data(),
            depth_scale,
            0.1, 6.0,
            depth_intr,
            color_intr,
            c2d_extr,
            d2c_extr,
            [cx, cy]
        )
        if depth_pixel and np.isfinite(depth_pixel).all():
            dx, dy = int(depth_pixel[0]), int(depth_pixel[1])
            w = depth_frame.get_width()
            h = depth_frame.get_height()
            if 0 <= dx < w and 0 <= dy < h:
                return dx, dy
    except Exception:
        pass
    return None


def depth_median_around(dx, dy, depth_frame, half_size=3, depth_min=0.15, depth_max=1.5):
    """
    在深度图坐标 (dx, dy) 周围 half_size 窗口取深度中值.

    参数:
        dx, dy      : int, 深度图坐标
        depth_frame : rs.depth_frame
        half_size   : int, 窗口半宽 (默认 3, 即 7x7 窗口)
        depth_min   : float, 有效深度最小值 (米)
        depth_max   : float, 有效深度最大值 (米)

    返回:
        float 或 None (有效点不足时)
    """
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    x_s = max(0, dx - half_size)
    y_s = max(0, dy - half_size)
    x_e = min(w, dx + half_size + 1)
    y_e = min(h, dy + half_size + 1)

    depths = []
    for vy in range(y_s, y_e):
        for vx in range(x_s, x_e):
            d = depth_frame.get_distance(vx, vy)
            if depth_min < d < depth_max:
                depths.append(d)
    if len(depths) < 3:
        return None
    depths.sort()
    return depths[len(depths) // 2]


# ============================================================
#  3. 水平扫描线法分离把手
# ============================================================

def scanline_find_handle(full_mask, bbox,
                         handle_min_ratio=0.05, handle_max_ratio=0.55,
                         handle_side="right",
                         handle_center_deviation_ratio=0.3):
    """
    水平扫描线法分离壶身与把手.

    原理:
       逐行扫描, 对每行 y 在搜索范围内找出所有 mask 段.
       取把手侧最外的一个段作为把手候选 (不跨空洞合并).
       right 侧取最右段, left 侧取最左段.
       然后校验面积比例和质心位置.

    参数:
        full_mask                     : np.ndarray (H, W), bool, SAM 生成的完整水壶 mask
        bbox                          : [x1, y1, x2, y2], YOLO 检测框
        handle_min_ratio              : float, 把手面积/壶身面积 最小比例
        handle_max_ratio              : float, 把手面积/壶身面积 最大比例
        handle_side                   : "right" 或 "left", 把手预期侧
        handle_center_deviation_ratio : float, 把手 x 中心偏差容差(相对把手宽度比例),
                                        从底部+顶部双向剔除偏离过大的行

    返回:
        (body_mask, handle_mask) 或 (None, None)
    """
    h, w = full_mask.shape
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # 限制搜索范围到 bbox 内
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))

    if y2 - y1 < 5 or x2 - x1 < 5:
        return None, None

    # ---- 逐行扫描: 有 gap 模式的行, 小段直接标记为把手 ----
    handle_mask = np.zeros((h, w), dtype=bool)
    pattern_rows = 0

    for y in range(y1, y2 + 1):
        row = full_mask[y, x1:x2 + 1]
        if not np.any(row):
            continue

        diffs = np.diff(row.astype(np.int8))
        rising = np.where(diffs == 1)[0] + 1
        falling = np.where(diffs == -1)[0] + 1

        if row[0]:
            rising = np.concatenate([[0], rising])
        if row[-1]:
            falling = np.concatenate([falling, [len(row) - 1]])

        segments = []
        for r, f in zip(rising, falling):
            segments.append((int(r), int(f)))

        if len(segments) >= 2:
            # 取把手侧最外段:
            #   handle_side="right" -> 取 x 最大的段 (最右侧)
            #   handle_side="left"  -> 取 x 最小的段 (最左侧)
            if handle_side == "right":
                candidate = segments[-1]  # 最后一个段 = 最右侧
            else:
                candidate = segments[0]  # 第一个段 = 最左侧

            cand_start, cand_end = candidate
            cand_len = cand_end - cand_start
            if cand_len >= 3:
                pattern_rows += 1
                handle_mask[y, x1 + cand_start:x1 + cand_end + 1] = True

    if pattern_rows < 3:
        return None, None

    if np.sum(handle_mask) < 20:
        return None, None

    # ---- 从 full_mask 中减去 handle_mask 得到 body_mask ----
    body_mask = full_mask.copy()
    body_mask[handle_mask] = False

    # ---- 校验 ----
    handle_area = float(np.sum(handle_mask))
    body_area = float(np.sum(body_mask))
    if body_area < 1:
        return None, None

    ratio = handle_area / body_area
    if not (handle_min_ratio <= ratio <= handle_max_ratio):
        return None, None

    # 把手质心在指定侧
    ys, xs = np.where(handle_mask)
    cx = float(np.mean(xs))
    bbox_cx = (bbox[0] + bbox[2]) / 2.0

    if handle_side == "right" and cx < bbox_cx:
        return None, None
    if handle_side == "left" and cx > bbox_cx:
        return None, None

    # ---- 形态学闭运算填充把手内部空洞 ----
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    handle_mask = cv2.morphologyEx(
        handle_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel_close
    ).astype(bool)

    # ---- 只保留最大连通分量 (排除噪声) ----
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        handle_mask.astype(np.uint8), connectivity=8
    )
    if num_labels > 1:
        areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
        largest_label = max(areas, key=lambda x: x[1])[0]
        handle_mask = (labels == largest_label)

    # ---- X 中心一致性: 从底部向上 + 顶部向下 双向剔除偏离行 ----
    row_centers = []  # (y, center_x, width)
    for y in range(y1, y2 + 1):
        xs_row = np.where(handle_mask[y])[0]
        if len(xs_row) > 5:
            row_centers.append((y, float(np.mean(xs_row)), float(xs_row[-1] - xs_row[0])))
    if len(row_centers) >= 5:
        cx_median = np.median([c for _, c, _ in row_centers])
        avg_width = np.mean([w for _, _, w in row_centers])
        tol = avg_width * handle_center_deviation_ratio
        # 从底部向上剔除
        for y, cx, _ in sorted(row_centers, key=lambda x: -x[0]):
            if abs(cx - cx_median) > tol:
                handle_mask[y:, :] = False
            else:
                break
        # 从顶部向下剔除
        for y, cx, _ in sorted(row_centers, key=lambda x: x[0]):
            if abs(cx - cx_median) > tol:
                handle_mask[:y + 1, :] = False
            else:
                break

    # 重新计算 body_mask
    body_mask = full_mask.copy()
    body_mask[handle_mask] = False

    return body_mask, handle_mask


# ============================================================
#  4. 红色胶带检测工具
# ============================================================

def detect_red_tape(bgr_img, hsv_low1=None, hsv_up1=None,
                    hsv_low2=None, hsv_up2=None, min_area=20):
    """
    在 BGR 图像中检测红色区域.
    红色在 HSV 空间中分布在 0 和 180 两端, 需双阈值.

    参数:
        bgr_img    : np.ndarray, BGR 图像
        hsv_low1   : np.ndarray 或 None, 低阈值下限, 默认 [0, 50, 50]
        hsv_up1    : np.ndarray 或 None, 低阈值上限, 默认 [10, 255, 255]
        hsv_low2   : np.ndarray 或 None, 高阈值下限, 默认 [170, 50, 50]
        hsv_up2    : np.ndarray 或 None, 高阈值上限, 默认 [180, 255, 255]
        min_area   : int, 最小有效面积

    返回:
        red_mask  : np.ndarray (H, W), bool, 红色区域 mask
        center    : (cx, cy) 或 None, 红色区域中心 (像素坐标)
        area      : int, 红色区域像素面积
    """
    if hsv_low1 is None:
        hsv_low1 = np.array([0, 50, 50])
    if hsv_up1 is None:
        hsv_up1 = np.array([10, 255, 255])
    if hsv_low2 is None:
        hsv_low2 = np.array([170, 50, 50])
    if hsv_up2 is None:
        hsv_up2 = np.array([180, 255, 255])

    hsv = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv, hsv_low1, hsv_up1)
    mask2 = cv2.inRange(hsv, hsv_low2, hsv_up2)
    red_mask = np.bitwise_or(mask1, mask2).astype(bool)

    # 形态学开运算去噪 (去除孤立噪点)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    red_mask = cv2.morphologyEx(red_mask.astype(np.uint8),
                                cv2.MORPH_OPEN, kernel).astype(bool)

    # 先检查有效像素数
    total_px = np.sum(red_mask)
    if total_px < min_area:
        return red_mask, None, 0

    # 保留最大连通分量 (排除小噪声)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        red_mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return red_mask, None, 0

    areas = [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)]
    largest_idx, largest_area = max(areas, key=lambda x: x[1])

    if largest_area < min_area:
        return red_mask, None, 0

    red_mask = (labels == largest_idx)

    # 计算中心 (使用中位数, 对边缘离群点更鲁棒)
    ys, xs = np.where(red_mask)
    center = (int(np.median(xs)), int(np.median(ys)))

    return red_mask, center, largest_area


# ============================================================
#  5. 平滑工具类
# ============================================================

class PositionSmoother:
    """3D 位置 EMA 平滑器."""
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


class PoseSmoother:
    """6D 位姿平滑器: 位置 EMA + 旋转矩阵指数移动平均 + SVD 正交化."""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_pos = None
        self.filtered_rot = None

    def update(self, pos, rot):
        if pos is None or rot is None:
            return None, None
        if self.filtered_pos is None or self.filtered_rot is None:
            self.filtered_pos = list(pos)
            self.filtered_rot = rot.copy()
            return tuple(self.filtered_pos), self.filtered_rot
        for i in range(3):
            self.filtered_pos[i] = (1 - self.alpha) * self.filtered_pos[i] + self.alpha * pos[i]
        self.filtered_rot = (1 - self.alpha) * self.filtered_rot + self.alpha * rot
        # SVD 正交化, 确保是有效的旋转矩阵
        U, _, Vt = np.linalg.svd(self.filtered_rot)
        self.filtered_rot = U @ Vt
        return tuple(self.filtered_pos), self.filtered_rot

    def reset(self):
        self.filtered_pos = None
        self.filtered_rot = None


# ============================================================
#  6. 6D 位姿工具
# ============================================================

def rotation_matrix_to_euler_angles(R):
    """
    从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw).
    返回角度值 (deg).
    """
    if abs(abs(R[2, 0]) - 1.0) < 1e-6:
        # 奇异情况 (万向锁)
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


def visualize_pose(image, center_3d, rot, K, scale=0.08):
    """
    在图像上绘制 6D 位姿坐标轴.
    红=X轴, 绿=Y轴, 蓝=Z轴
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


# ============================================================
#  7. 检测器基类 BaseRealSenseDetector
# ============================================================

class BaseRealSenseDetector:
    """
    RealSense 检测器基类, 封装通用基础设施.

    子类可通过文件级类属性配置行为:
        USE_YOLO_WORLD : bool  - 使用 YOLO-World (True) 或标准 YOLOv8 (False)
        USE_SAM        : bool  - 是否加载 SAM 模型
        LOG_NAME       : str   - 日志文件名 (不含路径, 写入 logs/RealSense_R/)
        TARGET_CLASS_ID: int   - 标准 YOLO 目标类别 ID (默认 39 = bottle)
    """
    USE_YOLO_WORLD = False
    USE_FINETUNED_KETTLE = False
    USE_SAM = False
    LOG_NAME = "realsense_detect.log"
    LOG_MAX_SIZE = 5 * 1024 * 1024  # 5MB, 超过则自动清空
    TARGET_CLASS_ID = 39

    # ---- 分辨率默认值 ----
    FRAME_W = 640
    FRAME_H = 480
    FRAME_FPS = 30

    # ---- 深度默认值 ----
    DEPTH_MIN = 0.15
    DEPTH_MAX = 1.5
    DEPTH_Z_THRESHOLD = 0.05

    # ---- 平滑默认值 ----
    SMOOTH_ALPHA = 0.4
    SMOOTH_POSE_ALPHA = 0.3

    # ---- 标准 YOLO 默认值 ----
    MODEL_NAME = ""
    CONF_THRESHOLD = 0.5

    # ---- YOLO-World 默认值 ----
    YW_MODEL_PATH = "./model/yolov8m-worldv2.pt"
    YW_CONF_THRESHOLD = 0.25
    YW_CLASSES = ["kettle", "electric kettle", "water jug", "teapot"]

    # ---- 微调水壶模型默认值 ----
    KETTLE_MODEL_PATH = "./model/kettle_yolov8n_finetune.pt"
    KETTLE_CONF_THRESHOLD = 0.7
    KETTLE_DEVICE = 'cpu'

    # ---- SAM 默认值 ----
    SAM_MODEL_NAME = "mobile_sam.pt"
    SAM_CONF = 0.3

    # ---- 把手默认值 ----
    HANDLE_SIDE = "right"
    HANDLE_MIN_RATIO = 0.05
    HANDLE_MAX_RATIO = 0.55
    ROI_EXPAND_RATIO = 0.3
    SKIP_FRAMES = 10
    HANDLE_CENTER_DEVIATION_RATIO = 0.3

    # ---- 红色胶带 ----
    USE_RED_TAPE = False
    RED_TAPE_MIN_AREA = 20

    # ---- 设备默认值 ----
    DEVICE = 'cpu'
    YW_DEVICE = 'cpu'
    SAM_DEVICE = 'cpu'

    def __init__(self, config_dict=None, start_pipeline=True):
        # 先执行 sys.path 插入 (确保能导入 pyrealsense2)
        self._ensure_sys_path()

        # 如果外部传入配置字典, 覆盖类属性默认值
        if config_dict is not None:
            self._apply_config(config_dict)

        # 注入帧缓存 (用于复用 pipeline 时外部注入帧数据)
        self._injected_frames = None

        # 类型标注: 下述字段在 start_pipeline=True 时由 _init_realsense 赋值,
        # False 时保持 None, 调用处需用 assert 守卫
        self.pipeline: rs.pipeline = None  # type: ignore[assignment]
        self.profile: rs.pipeline_profile = None  # type: ignore[assignment]
        self.depth_scale: float = None  # type: ignore[assignment]
        self.depth_intrinsics: rs.intrinsics = None  # type: ignore[assignment]
        self.color_intrinsics: rs.intrinsics = None  # type: ignore[assignment]
        self.color_to_depth_extrinsics: rs.extrinsics = None  # type: ignore[assignment]
        self.depth_to_color_extrinsics: rs.extrinsics = None  # type: ignore[assignment]
        self.colorizer: rs.colorizer = None  # type: ignore[assignment]

        # 初始化 RealSense 相机
        if start_pipeline:
            self._init_realsense()
        else:
            print("[INFO] pipeline 跳过初始化 (start_pipeline=False)")

        # 初始化 YOLO/SAM 模型
        self._init_models()

        # 初始化日志
        self._init_logging()

        # 耗时统计
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # 跳帧缓存 (从类属性 SKIP_FRAMES 读取)
        self._frame_counter = 0
        self._skip_frames = self.SKIP_FRAMES
        self._cached_result = None

    # ------------------------------------------------------------
    @staticmethod
    def _ensure_sys_path():
        """确保项目根目录在 sys.path 中."""
        import sys
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if root not in sys.path:
            sys.path.insert(0, root)

    # ------------------------------------------------------------
    def _apply_config(self, cfg):
        """
        用配置字典覆盖类属性默认值.
        调用时机: 子类 __init__ 中 super().__init__(config_dict=...) 时.
        """
        # ---- 分辨率 ----
        if "resolution" in cfg:
            self.FRAME_W = cfg["resolution"]["width"]
            self.FRAME_H = cfg["resolution"]["height"]
            self.FRAME_FPS = cfg["resolution"]["fps"]

        # ---- 深度 ----
        _depth = cfg.get("depth", {})
        if "min" in _depth:
            self.DEPTH_MIN = _depth["min"]
        if "max" in _depth:
            self.DEPTH_MAX = _depth["max"]
        if "z_threshold" in _depth:
            self.DEPTH_Z_THRESHOLD = _depth["z_threshold"]

        # ---- 平滑 ----
        _sm = cfg.get("smoothing", {})
        if "alpha" in _sm:
            self.SMOOTH_ALPHA = _sm["alpha"]
        if "pose_alpha" in _sm:
            self.SMOOTH_POSE_ALPHA = _sm["pose_alpha"]

        # ---- 标准 YOLO (仅当 USE_YOLO_WORLD=False 时读取) ----
        if not self.USE_YOLO_WORLD:
            _model = cfg.get("model", {})
            if "name" in _model:
                self.MODEL_NAME = os.path.join(_project_root, _model["name"])
            if "conf_threshold" in _model:
                self.CONF_THRESHOLD = _model["conf_threshold"]
            if "target_class_id" in _model:
                self.TARGET_CLASS_ID = _model["target_class_id"]
            if "device" in _model:
                pref = _model["device"]
                self.DEVICE = 'cuda:0' if (pref == "gpu" and torch.cuda.is_available()) else 'cpu'

        # ---- YOLO-World (仅当 USE_YOLO_WORLD=True 时读取) ----
        if self.USE_YOLO_WORLD:
            _yw = cfg.get("yolo_world", {})
            if "model" in _yw:
                self.YW_MODEL_PATH = os.path.join(_project_root, _yw["model"])
            if "conf_threshold" in _yw:
                self.YW_CONF_THRESHOLD = _yw["conf_threshold"]
            if "classes" in _yw:
                self.YW_CLASSES = _yw["classes"]
            if "device" in _yw:
                yw_pref = _yw["device"]
                self.YW_DEVICE = 'cuda:0' if (yw_pref == "gpu" and torch.cuda.is_available()) else 'cpu'

        # ---- 微调水壶模型 (仅当 USE_FINETUNED_KETTLE=True 时读取) ----
        if self.USE_FINETUNED_KETTLE:
            _kettle = cfg.get("kettle_model", {})
            if "model" in _kettle:
                self.KETTLE_MODEL_PATH = os.path.join(_project_root, _kettle["model"])
            if "conf_threshold" in _kettle:
                self.KETTLE_CONF_THRESHOLD = _kettle["conf_threshold"]
            if "device" in _kettle:
                k_pref = _kettle["device"]
                self.KETTLE_DEVICE = 'cuda:0' if (k_pref == "gpu" and torch.cuda.is_available()) else 'cpu'

        # ---- SAM (仅当 USE_SAM=True 时读取) ----
        if self.USE_SAM:
            _sam = cfg.get("sam", {})
            if "model" in _sam:
                self.SAM_MODEL_NAME = _sam["model"]
            if "conf" in _sam:
                self.SAM_CONF = _sam["conf"]
            if "device" in _sam:
                sam_pref = _sam["device"]
                self.SAM_DEVICE = 'cuda:0' if (sam_pref == "gpu" and torch.cuda.is_available()) else 'cpu'

        # ---- 把手 (仅当 USE_SAM=True 时读取, 因为把手分离依赖 SAM) ----
        if self.USE_SAM:
            _handle = cfg.get("handle", {})
            if "handle_side" in _handle:
                self.HANDLE_SIDE = _handle["handle_side"]
            if "handle_min_ratio" in _handle:
                self.HANDLE_MIN_RATIO = _handle["handle_min_ratio"]
            if "handle_max_ratio" in _handle:
                self.HANDLE_MAX_RATIO = _handle["handle_max_ratio"]
            if "roi_expand_ratio" in _handle:
                self.ROI_EXPAND_RATIO = _handle["roi_expand_ratio"]
            if "skip_frames" in _handle:
                self.SKIP_FRAMES = _handle["skip_frames"]
            if "handle_center_deviation_ratio" in _handle:
                self.HANDLE_CENTER_DEVIATION_RATIO = _handle["handle_center_deviation_ratio"]

        # ---- 红色胶带 (始终读取, 不管 USE_SAM) ----
        _rt = cfg.get("red_tape", {})
        if "enabled" in _rt:
            self.USE_RED_TAPE = _rt["enabled"]
        if "min_area" in _rt:
            self.RED_TAPE_MIN_AREA = _rt["min_area"]

        # ---- 打印设备信息 ----
        if self.USE_FINETUNED_KETTLE:
            print("[INFO] 微调水壶模型运行设备: %s" % self.KETTLE_DEVICE)
        elif self.USE_YOLO_WORLD:
            print("[INFO] YOLO-World 运行设备: %s" % self.YW_DEVICE)
        else:
            print("[INFO] 标准 YOLO 运行设备: %s" % self.DEVICE)
        if self.USE_SAM:
            print("[INFO] SAM 运行设备: %s" % self.SAM_DEVICE)

    # ------------------------------------------------------------
    def _init_realsense(self):
        """初始化 RealSense 相机管道与参数."""
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, self.FRAME_W, self.FRAME_H, rs.format.bgr8, self.FRAME_FPS)
        config.enable_stream(rs.stream.depth, self.FRAME_W, self.FRAME_H, rs.format.z16, self.FRAME_FPS)

        self.profile = self.pipeline.start(config)

        # 深度缩放系数
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("[INFO] depth_scale = %f" % self.depth_scale)

        # 激光功率
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
            print("[INFO] laser_power set to 150")
        except Exception as e:
            print("[WARN] laser_power set failed: %s" % str(e))

        # 相机参数 (从 YAML 加载)
        _cam_params = load_camera_params(
            os.path.join(os.path.dirname(__file__), 'D435i_info.yaml')
        )
        self.depth_intrinsics = _cam_params['depth_intr']
        self.color_intrinsics = _cam_params['color_intr']
        self.color_to_depth_extrinsics = _cam_params['c2d_extr']
        self.depth_to_color_extrinsics = _cam_params['d2c_extr']

        print("[INFO] 深度内参 (YAML): fx=%.2f, fy=%.2f, ppx=%.2f, ppy=%.2f" % (
            self.depth_intrinsics.fx, self.depth_intrinsics.fy,
            self.depth_intrinsics.ppx, self.depth_intrinsics.ppy))
        print("[INFO] 彩色内参 (YAML): fx=%.2f, fy=%.2f, ppx=%.2f, ppy=%.2f" % (
            self.color_intrinsics.fx, self.color_intrinsics.fy,
            self.color_intrinsics.ppx, self.color_intrinsics.ppy))

        self.colorizer = rs.colorizer()

    # ------------------------------------------------------------
    def _init_models(self):
        """初始化 YOLO (标准/YOLO-World) 与 SAM 模型."""
        if self.USE_YOLO_WORLD:
            # ---- YOLO-World ----
            model_path = self.YW_MODEL_PATH
            print("[INFO] 加载 YOLO-World 模型: %s" % model_path)
            if not os.path.exists(model_path):
                raise FileNotFoundError("YOLO-World 模型文件不存在: %s" % model_path)
            _model = YOLO(model_path)  # type: ignore[arg-type]
            _model.to(self.YW_DEVICE)  # type: ignore[union-attr]
            _model.set_classes(self.YW_CLASSES)  # type: ignore[union-attr]
            self.model = _model
            print("[INFO] YOLO-World 词汇表: %s" % str(self.YW_CLASSES))
        else:
            # ---- 标准 YOLOv8 ----
            model_path = self.MODEL_NAME
            model_path_pt = "%s.pt" % model_path
            if os.path.exists(model_path_pt):
                model_path = model_path_pt
            self.model = YOLO(model_path).to(self.DEVICE)

        # ---- SAM (可选) ----
        self.sam_model = None
        if self.USE_SAM:
            sam_path = os.path.join(_project_root, "model", self.SAM_MODEL_NAME)
            if not os.path.exists(sam_path):
                print("[WARN] SAM 模型未找到: %s" % sam_path)
                print("[WARN] 将跳过 SAM 推理")
            else:
                is_fast_sam = self.SAM_MODEL_NAME.lower().startswith("fastsam")
                if is_fast_sam:
                    self.sam_model = FastSAM(sam_path)
                else:
                    self.sam_model = SAM(sam_path)
                self.sam_model.to(self.SAM_DEVICE)
                print("[INFO] SAM 模型已加载: %s" % sam_path)

        # ---- 微调水壶模型 (可选) ----
        self.kettle_model = None
        if self.USE_FINETUNED_KETTLE:
            kettle_path = self.KETTLE_MODEL_PATH
            print("[INFO] 加载微调水壶模型: %s" % kettle_path)
            if not os.path.exists(kettle_path):
                raise FileNotFoundError("微调水壶模型不存在: %s" % kettle_path)
            self.kettle_model = YOLO(kettle_path)
            self.kettle_model.to(self.KETTLE_DEVICE)

    # ------------------------------------------------------------
    def inject_frames(self, color_image, depth_image, depth_colormap, depth_frame):
        """
        注入外部帧数据 (复用 pipeline 时使用).
        之后调用 get_frames() 将返回注入的帧而非从 pipeline 获取.
        """
        self._injected_frames = (color_image, depth_image, depth_colormap, depth_frame)

    def get_frames(self):
        """
        获取原始彩色帧和原始深度帧.
        不经过 rs.align
        返回 (color_image, depth_image, depth_colormap, depth_frame)

        若存在注入帧 (_injected_frames), 优先返回注入帧.
        """
        if self._injected_frames is not None:
            return self._injected_frames

        if self.pipeline is None:
            return None, None, None, None

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        # 此时 pipeline 已启动且 colorizer 已初始化
        assert self.colorizer is not None  # type guard for Pylance
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image, depth_colormap, depth_frame

    # ------------------------------------------------------------
    def _color_to_depth_pixel(self, cx, cy, depth_frame):
        """
        将彩色图上的像素 (cx, cy) 映射到深度图坐标系.
        返回 (dx, dy) 或 None
        """
        return color_to_depth_pixel(
            cx, cy, depth_frame, self.depth_scale,
            self.depth_intrinsics, self.color_intrinsics,
            self.color_to_depth_extrinsics, self.depth_to_color_extrinsics
        )

    # ------------------------------------------------------------
    def _depth_median_around(self, dx, dy, depth_frame, half_size=3):
        """在深度图坐标 (dx, dy) 周围 half_size 窗口取深度中值."""
        return depth_median_around(
            dx, dy, depth_frame, half_size=half_size,
            depth_min=self.DEPTH_MIN, depth_max=self.DEPTH_MAX
        )

    # ------------------------------------------------------------
    def _color_pixel_to_3d(self, cx, cy, depth_frame):
        """
        将彩色像素 (cx, cy) 反投影到彩色坐标系下的 3D 点.

        步骤:
          1. 彩色像素 → 映射到深度图坐标
          2. 读取深度值 d
          3. 用彩色内参反投影: X=(cx-ppx)*d/fx, Y=(cy-ppy)*d/fy, Z=d

        返回 (x, y, z) 或 None (映射失败/深度无效时)
        """
        depth_pixel = self._color_to_depth_pixel(cx, cy, depth_frame)
        if depth_pixel is None:
            return None
        dx, dy = depth_pixel
        d = depth_frame.get_distance(dx, dy)
        if not (self.DEPTH_MIN < d < self.DEPTH_MAX):
            return None
        # 彩色内参反投影 → 彩色坐标系
        # 此函数仅当 pipeline 启动后调用, color_intrinsics 已由 _init_realsense 赋值
        assert self.color_intrinsics is not None  # type guard for Pylance
        x = (cx - self.color_intrinsics.ppx) * d / self.color_intrinsics.fx
        y = (cy - self.color_intrinsics.ppy) * d / self.color_intrinsics.fy
        return x, y, d

    # ------------------------------------------------------------
    def _get_anchor_depth(self, bbox, depth_frame):
        """
        获取 bbox 中心锚点深度 (单点).
        步骤:
         1. bbox 中心彩色坐标 → 映射到深度图
         2. 深度图局部 3x3 中值
         返回 (cx_color, cy_color, center_depth) 或 None
        """
        cx_color = (int(bbox[0]) + int(bbox[2])) // 2
        cy_color = (int(bbox[1]) + int(bbox[3])) // 2

        depth_pixel = self._color_to_depth_pixel(cx_color, cy_color, depth_frame)
        if depth_pixel is None:
            return None

        dx, dy = depth_pixel
        depth_val = self._depth_median_around(dx, dy, depth_frame, half_size=3)
        if depth_val is None:
            return None

        return cx_color, cy_color, depth_val

    # ------------------------------------------------------------
    def _init_logging(self):
        """根据 LOG_NAME 初始化日志文件 (追加模式, 超限自动轮转)."""
        if not self.LOG_NAME:
            return
        log_dir = os.path.join(_project_root, "logs", "RealSense_R")
        os.makedirs(log_dir, exist_ok=True)
        self._log_path = os.path.join(log_dir, self.LOG_NAME)
        if os.path.exists(self._log_path) and os.path.getsize(self._log_path) > self.LOG_MAX_SIZE:
            os.remove(self._log_path)
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 60 + "\n")
            f.write("  [%s] %s 启动\n" % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                self.__class__.__name__))
            f.write("=" * 60 + "\n")

    # ------------------------------------------------------------
    def _log_write(self, tag, text):
        """追加一行日志到 LOG_NAME 文件."""
        if not hasattr(self, '_log_path'):
            return
        with open(self._log_path, "a", encoding="utf-8") as f:
            f.write("[%s] %s %s\n" % (
                datetime.datetime.now().strftime("%H:%M:%S"), tag, text))

    # ------------------------------------------------------------
    def detect_targets(self, color_image):
        """
        标准 YOLO 检测.
        仅当 USE_YOLO_WORLD=False 时使用.
        返回 [(x1, y1, x2, y2, conf, class_id), ...]
        """
        if self.USE_YOLO_WORLD:
            raise NotImplementedError("USE_YOLO_WORLD=True 时请使用 _detect_kettle_bbox")
        results = self.model(
            color_image, conf=self.CONF_THRESHOLD,
            classes=[self.TARGET_CLASS_ID], verbose=False
        )
        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            for box, conf, cls_id in zip(boxes, confs, class_ids):
                detections.append([*box, float(conf), int(cls_id)])
        return detections

    # ------------------------------------------------------------
    def _detect_kettle_bbox(self, color_image):
        """
        YOLO-World 检测水壶 bbox.
        仅当 USE_YOLO_WORLD=True 时使用.
        返回 {"bbox": [x1, y1, x2, y2], "conf": float} 或 None
        """
        if not self.USE_YOLO_WORLD:
            raise NotImplementedError("USE_YOLO_WORLD=False 时请使用 detect_targets")
        results = self.model(color_image, conf=self.YW_CONF_THRESHOLD, verbose=False)
        best = None
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if best is None or conf > best["conf"]:
                    best = {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "conf": float(conf),
                    }
        return best

    # ------------------------------------------------------------
    def _detect_kettle_finetuned(self, color_image):
        """
        微调 YOLOv8n 检测水壶 bbox.
        仅当 USE_FINETUNED_KETTLE=True 时使用.
        返回 {"bbox": [x1, y1, x2, y2], "conf": float} 或 None
        """
        if self.kettle_model is None:
            return None
        results = self.kettle_model(
            color_image, conf=self.KETTLE_CONF_THRESHOLD, verbose=False
        )
        best = None
        for result in results:
            if result.boxes is None:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if best is None or conf > best["conf"]:
                    best = {
                        "bbox": [int(box[0]), int(box[1]), int(box[2]), int(box[3])],
                        "conf": float(conf),
                    }
        return best

    # ------------------------------------------------------------
    def _segment_full_mask(self, rgb_image, bbox):
        """
        在 bbox 区域内用 SAM 生成完整水壶 mask (壶身+把手).
        仅当 USE_SAM=True 且 sam_model 可用时使用.
        返回 np.ndarray (H, W), bool 或 None
        """
        if self.sam_model is None:
            return None

        bbox_w = bbox[2] - bbox[0]
        expand_lr = max(int(bbox_w * self.ROI_EXPAND_RATIO), 10)
        expand_tb = 10
        bbox_exp = [
            max(0, bbox[0] - expand_lr),
            max(0, bbox[1] - expand_tb),
            min(self.FRAME_W, bbox[2] + expand_lr),
            min(self.FRAME_H, bbox[3] + expand_tb),
        ]

        sam_results = self.sam_model(
            rgb_image,
            bboxes=[bbox_exp],
            conf=self.SAM_CONF,
            device=self.SAM_DEVICE,
            verbose=False
        )

        if len(sam_results) == 0 or sam_results[0].masks is None:
            return None

        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask = mask > 0.5
        return mask

    # ------------------------------------------------------------
    def stop(self):
        """停止管道并销毁窗口."""
        if self.pipeline is not None:
            self.pipeline.stop()
        cv2.destroyAllWindows()
