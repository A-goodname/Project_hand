# -*- coding: utf-8 -*-
"""
Orchestrator Camera A Handle+Cup 检测器
=============================================
独立于 RealSense_R 体系, 不继承 BaseRealSenseDetector.

Handle 检测 (yolo+sam): YOLO kettle bbox → SAM bbox 提示分割 full_mask
             → 形态学后处理 (腐蚀/开运算/闭运算/最大连通域)
             → 水平扫描线分离把手 (v2: mask 外接矩形范围, 保守剔除)
             → PCA 6D 位姿 + 正对参考对比旋转角 θ → 完整动态 euler
Cup 检测:    YOLO cup bbox → bbox 采样 → 深度反投影 → 3D 位置 (原样保留)

euler_cam 语义: 检测成功 → 相机系动态 ZYX 欧拉角 (deg, 完整检测姿态);
                检测失败 → 回落 config 固定值 (永不为 None).

yolo+sam 检测链拷贝自 RealSense_R/test_handle_6D.py + realsense_shared.py
(上游文件冻结, 此处私有化改造, 改动独立演化).

参考: RealSense_R/test_handle_6D.py, test_SAM_handle.py
"""

import os
import sys
import time
import math
from typing import Optional

# 项目根目录必须在 import pyrealsense2 之前加入 sys.path,
# 因为 pyrealsense2 是项目根目录下的 .pyd 文件 (非 pip 安装).
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO, SAM

from RealSense_R.realsense_shared import (
    load_camera_params,
    color_to_depth_pixel,
    depth_median_around,
    rotation_matrix_to_euler_angles,
)


# ============================================================
#  固定 ABC (检测失败时的回落值)
# ============================================================
HANDLE_EULER = (-90.0, 89.0, -90.0)
CUP_EULER_BACK = (-90.0, 89.0, -110.0)
CUP_EULER_LEFT = (-90.0, 89.0, -90.0)


# ============================================================
#  1. SAM mask 后处理 (去噪点/腐蚀/填洞/最大连通域)
# ============================================================

def _postprocess_full_mask(full_mask, cfg, handle_side="right"):
    """SAM 原始 mask → 质量 mask.

    链: OPEN(去孤立 speckle, 不缩大区域) → ERODE(消边界渗漏/单像素噪点)
        → CLOSE(填洞) → 连通域过滤.
    (OPEN 先于 ERODE: 两者都能去噪点, 但 ERODE 会缩细把手条,
     OPEN 无损 — 先 OPEN 后 ERODE 总损失最小)

    连通域过滤: 保留最大连通域 (壶身) + 位于壶身把手侧的次要连通域
    (SAM 偶发把把手切成孤岛时不能丢) ; 其余 (孤立噪点) 丢弃.

    参数:
        full_mask   : np.ndarray (H, W), bool, SAM 生成的完整水壶 mask
        cfg         : dict, config 的 mask_postprocess 段
        handle_side : "right" | "left", 把手预期侧 (用于保留把手孤岛)
    返回:
        np.ndarray (H, W), bool
    """
    m = full_mask.astype(np.uint8) * 255

    k_erode = int(cfg.get("erode_kernel", 3))
    k_open = int(cfg.get("open_kernel", 3))
    k_close = int(cfg.get("close_kernel", 5))
    min_area = float(cfg.get("min_area", 500.0))

    # 1) OPEN: 去孤立 speckle (无损, 不缩大区域)
    if k_open > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k_open, k_open), np.uint8))

    # 2) ERODE: 消边界渗漏/单像素噪点 (把手环臂 8-15px 厚, 核不宜 >5)
    if k_erode > 0:
        m = cv2.erode(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_erode, k_erode)))

    # 3) CLOSE: 填洞 (与 scanline 内部 close 对齐, 保证幂等)
    if k_close > 0:
        m = cv2.morphologyEx(
            m, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)),
        )

    # 4) 连通域过滤: 最大连通域 + 把手侧次要连通域 (防把手孤岛被丢)
    if cfg.get("largest_cc_only", True):
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if num_labels > 1:
            largest_idx = max(
                ((i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, num_labels)),
                key=lambda x: x[1],
            )[0]
            lx1 = int(stats[largest_idx, cv2.CC_STAT_LEFT])
            lx2 = lx1 + int(stats[largest_idx, cv2.CC_STAT_WIDTH])
            keep = {largest_idx}
            for i in range(1, num_labels):
                if i in keep:
                    continue
                if stats[i, cv2.CC_STAT_AREA] < min_area:
                    continue  # 孤立噪点
                # 把手侧次要连通域 (与最大连通域把手侧相邻)
                c_left = int(stats[i, cv2.CC_STAT_LEFT])
                c_right = c_left + int(stats[i, cv2.CC_STAT_WIDTH])
                if handle_side == "right" and c_left >= lx2 - 10:
                    keep.add(i)
                elif handle_side == "left" and c_right <= lx1 + 10:
                    keep.add(i)
            keep_mask = np.isin(labels, list(keep))
            m = keep_mask.astype(np.uint8) * 255

    return m.astype(bool)


# ============================================================
#  2. 水平扫描线分离把手 (拷贝自 realsense_shared.scanline_find_handle, 私有化改造)
# ============================================================

def _scanline_find_handle_v2(full_mask, bbox,
                             handle_min_ratio=0.05, handle_max_ratio=0.55,
                             handle_side="right",
                             handle_center_deviation_ratio=0.5,
                             trim_top=False, trim_bottom=True,
                             use_mask_extent=True):
    """水平扫描线法分离壶身与把手 (v2).

    相对上游 scanline_find_handle 的改造 (缺陷修复):
      1. 搜索范围 = mask 非零外接矩形 ∪ bbox (原: 钳到 YOLO bbox) —
         把手外弧超出 bbox 的部分也能扫到, 不再截断 (只向把手侧扩展 x,
         壶身侧仍限 bbox 防吸背景; y 用并集).
      2. X 中心一致性剔除保守化 (原: 双向整段清零会切掉弧形把手):
         默认只从底部向上剔 (trim_top=False 保把手弧), 连续 ≥2 行超差
         才剔, 且只删对应行 (原: 从超差行起整段 y: / :y+1 清零).

    参数:
        full_mask                     : np.ndarray (H, W), bool, SAM 生成的完整水壶 mask
        bbox                          : [x1, y1, x2, y2], YOLO 检测框
        handle_min_ratio / max_ratio  : 把手面积/壶身面积 最小/最大比例
        handle_side                   : "right" 或 "left", 把手预期侧
        handle_center_deviation_ratio : X 中心偏差容差(相对把手宽度比例)
        trim_top / trim_bottom        : 顶部/底部 X 中心一致性剔除开关
        use_mask_extent               : True=搜索范围扩展到 mask 外接矩形

    返回:
        (body_mask, handle_mask) 或 (None, None)
    """
    h, w = full_mask.shape
    bx1, by1, bx2, by2 = [int(v) for v in bbox]

    # ---- 搜索范围: bbox ∪ mask 外接矩形 ----
    if use_mask_extent and full_mask.any():
        ys, xs = np.where(full_mask)
        mx1, my1, mx2, my2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
        # 只向把手侧扩展 x; 壶身侧仍限 bbox (防吸背景); y 用并集
        if handle_side == "right":
            x1, x2 = bx1, max(bx2, mx2)
        else:
            x1, x2 = min(bx1, mx1), bx2
        y1, y2 = min(by1, my1), max(by2, my2)
    else:
        x1, y1, x2, y2 = bx1, by1, bx2, by2

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

        segments = [(int(r), int(f)) for r, f in zip(rising, falling)]

        if len(segments) >= 2:
            # 取把手侧最外段:
            #   handle_side="right" -> 取 x 最大的段 (最右侧)
            #   handle_side="left"  -> 取 x 最小的段 (最左侧)
            if handle_side == "right":
                cand_start, cand_end = segments[-1]
            else:
                cand_start, cand_end = segments[0]

            if cand_end - cand_start >= 3:
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
    bbox_cx = (bx1 + bx2) / 2.0

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

    # ---- X 中心一致性 (保守版): 连续 ≥2 行超差才剔除, 只删对应行 ----
    row_centers = []  # (y, center_x, width)
    for y in range(y1, y2 + 1):
        xs_row = np.where(handle_mask[y])[0]
        if len(xs_row) > 5:
            row_centers.append((y, float(np.mean(xs_row)), float(xs_row[-1] - xs_row[0])))
    if len(row_centers) >= 5:
        cx_median = np.median([c for _, c, _ in row_centers])
        avg_width = np.mean([wd for _, _, wd in row_centers])
        tol = avg_width * handle_center_deviation_ratio

        def _trim(rows_descending):
            """从一端向另一端扫描, 连续 ≥2 行超差才剔除该行 (只删对应行)."""
            streak = 0
            for y, cxc, _ in rows_descending:
                if abs(cxc - cx_median) > tol:
                    streak += 1
                    if streak >= 2:
                        handle_mask[y, :] = False
                else:
                    streak = 0

        if trim_bottom:
            _trim(sorted(row_centers, key=lambda x: -x[0]))  # 底部向上
        if trim_top:
            _trim(sorted(row_centers, key=lambda x: x[0]))   # 顶部向下

    # 重新计算 body_mask
    body_mask = full_mask.copy()
    body_mask[handle_mask] = False

    return body_mask, handle_mask


# ============================================================
#  3. 把手 mask → 6D 位姿 (PCA, 拷贝自 test_handle_6D._compute_handle_6d_pose)
# ============================================================

def _compute_handle_6d_pose(handle_mask, depth_frame, anchor_depth,
                             depth_scale, depth_intr, color_intr,
                             c2d_extr, d2c_extr,
                             step=2, z_threshold=0.05,
                             depth_min=0.15, depth_max=1.5):
    """对 handle_mask 区域计算 6D 位姿 (PCA, 彩色坐标系).

    流程:
      1. handle_mask 像素逐点: 彩色像素 → 深度映射 → 读深度
      2. 彩色内参反投影 → 彩色坐标系 3D 点
      3. Z 离群过滤 |d - anchor_depth| < z_threshold
      4. 点不足时放宽 Z 过滤重试 (z_threshold × 2)
      5. PCA → 主方向 → 右手系旋转矩阵
      6. 返回 位置 + 旋转矩阵 + 点云

    返回:
        (position, rotation, point_cloud) 或 (None, None, None)
          position : (x, y, z) 点云重心 (m) — 彩色坐标系
          rotation : 3x3 旋转矩阵 (Y=把手主方向, Z=指向相机)
          point_cloud: Nx3 点云
    """
    ys, xs = np.where(handle_mask)
    if len(ys) < 10:
        return None, None, None

    # ---- 构建点云: handle_mask 区域逐像素采样, 彩色内参反投影 ----
    def _build_cloud(z_thr):
        pts = []
        indices = np.arange(0, len(ys), step)
        for idx in indices:
            cy = ys[idx]
            cx = xs[idx]
            dp = color_to_depth_pixel(cx, cy, depth_frame, depth_scale,
                                      depth_intr, color_intr, c2d_extr, d2c_extr)
            if dp is None:
                continue
            dx, dy = dp
            d = depth_frame.get_distance(dx, dy)
            if not (depth_min < d < depth_max):
                continue
            if z_thr is not None and anchor_depth is not None:
                if abs(d - anchor_depth) > z_thr:
                    continue
            # 彩色内参反投影 → 彩色坐标系
            x = (cx - color_intr.ppx) * d / color_intr.fx
            y = (cy - color_intr.ppy) * d / color_intr.fy
            pts.append([x, y, d])
        return pts

    # 先尝试带 Z 过滤
    points = _build_cloud(z_threshold)

    # 点不足时放宽 Z 过滤
    if len(points) < 10:
        points = _build_cloud(z_threshold * 2)

    if len(points) < 10:
        return None, None, None

    point_cloud = np.array(points)

    # ---- PCA 主成分分析 ----
    center = np.mean(point_cloud, axis=0)

    cov = np.cov(point_cloud, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)

    idx_sort = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, idx_sort]

    # 主方向 (最大特征值) -> Y 轴
    target_y = np.real(eig_vec[:, 0])
    # 最小方差方向 -> Z 轴 (指向相机)
    target_z = np.real(eig_vec[:, 2])

    if target_z[2] > 0:
        target_z = -target_z

    # X 轴 = Y × Z, 确保右手系
    target_x = np.cross(target_y, target_z)
    norm_x = np.linalg.norm(target_x)
    if norm_x < 1e-10:
        return None, None, None
    target_x = target_x / norm_x
    # 重新正交化 Y = Z × X
    target_y = np.cross(target_z, target_x)
    norm_y = np.linalg.norm(target_y)
    if norm_y < 1e-10:
        return None, None, None
    target_y = target_y / norm_y

    rot = np.stack((target_x, target_y, target_z), axis=1)

    return center, rot, point_cloud


# ============================================================
#  4. 把手 2D 特征 + 正对参考旋转角 θ
# ============================================================

def _extract_handle_features(body_mask, handle_mask, handle_side="right"):
    """从壶身/把手 mask 提取旋转角相关特征.

    核心特征: dx = 把手 x 中心 − 壶身 x 中心 (像素, 按 handle_side 归一化右侧为正).
    正对相机时 |dx| 最大 (把手弧半径投影); 水壶绕竖直轴旋转 θ 后 |dx| 变小.

    返回:
        dict {"dx", "width", "ratio", "body_w"} 或 None (点数不足)
    """
    xs = np.where(handle_mask)[1]
    bxs = np.where(body_mask)[1]
    if len(xs) < 20 or len(bxs) < 50:
        return None

    dx = float(np.median(xs) - np.median(bxs))
    if handle_side == "left":
        dx = -dx  # 归一化: 右侧为正

    return {
        "dx": dx,
        "width": float(xs.max() - xs.min()),
        "ratio": len(xs) / max(len(bxs), 1),
        "body_w": float(bxs.max() - bxs.min()),
    }


def _rotate_around_axis(axis, angle_deg):
    """绕任意轴 axis 旋转 angle_deg 的旋转矩阵 (Rodrigues).

    返回 3x3 旋转矩阵 (作用于列向量, 即 R @ v).
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n < 1e-10:
        return np.eye(3)
    axis = axis / n
    rvec = axis * math.radians(angle_deg)
    return cv2.Rodrigues(rvec)[0]


# ============================================================
#  5. HandleCupDetector
# ============================================================

class HandleCupDetector:
    """Camera A Handle+Cup 检测器 (Handle: yolo+sam; Cup: 原样保留)."""

    def __init__(self, orch_config: dict, serial: str = "", device: str = "cuda"):
        """
        Args:
            orch_config: Orchestrator/config.yaml 加载后的字典 (全部配置).
            serial:      Camera A 序列号 (D435i_info.yaml 中的 serial).
            device:      "cuda" | "cpu".
        """
        _c = orch_config
        self._serial = serial
        self._device = device

        # ---- 检测参数 ----
        self._depth_min = float(_c.get("depth_min", 0.15))
        self._depth_max = float(_c.get("depth_max", 1.5))
        self._depth_z_thr = float(_c.get("depth_z_threshold", 0.05))
        self._cup_handle_side = _c.get("cup_handle_side", "back")
        self._cup_rim_radius = float(_c.get("cup_rim_radius", 0.035))
        self._kettle_conf = float(_c.get("kettle_conf", 0.7))
        self._cup_conf = float(_c.get("cup_conf", 0.5))
        self._handle_side = _c.get("handle_side", "right")

        # ---- 模型路径 ----
        self._kettle_model_path = os.path.join(
            _project_root, _c.get("kettle_model", "model/kettle_yolov8n_finetune.pt"))
        _cup_m = _c.get("cup_model", "model/yolov8m")
        self._cup_model_path = os.path.join(_project_root, _cup_m)
        if not self._cup_model_path.endswith(".pt"):
            self._cup_model_path += ".pt"
        self._sam_model_path = os.path.join(
            _project_root, _c.get("sam_model", "model/sam2_b.pt"))
        self._sam_conf = float(_c.get("sam_conf", 0.3))
        self._sam_roi_expand_ratio = float(_c.get("sam_roi_expand_ratio", 0.3))

        # ---- mask 后处理 / 扫描线 / 正对参考 配置段 ----
        self._mask_pp_cfg = _c.get("mask_postprocess", {})
        self._scan_cfg = _c.get("scanline", {})
        self._ref_cfg = _c.get("handle_ref", {})

        # ---- 点云采样步长 (性能预算: 反投影为 Python 循环, step↑ 耗时↓) ----
        self._pca_step_handle = int(_c.get("pca_step_handle", 2))
        self._pca_step_body = int(_c.get("pca_step_body", 4))

        # ---- 固定 ABC (检测失败时的回落值) ----
        _h = _c.get("handle_euler")
        self._handle_euler = tuple(_h) if _h else HANDLE_EULER
        _cb = _c.get("cup_euler_back")
        _cl = _c.get("cup_euler_left")
        self._cup_euler_back = tuple(_cb) if _cb else CUP_EULER_BACK
        self._cup_euler_left = tuple(_cl) if _cl else CUP_EULER_LEFT
        self._cup_euler = self._cup_euler_left if self._cup_handle_side == "left" else self._cup_euler_back

        # ---- RealSense pipeline ----
        self._pipeline = None
        self._colorizer = None
        self._depth_scale = 0.0
        self._color_intr = None
        self._depth_intr = None
        self._c2d_extr = None
        self._d2c_extr = None
        self._init_pipeline()

        # ---- YOLO + SAM 模型 ----
        self._kettle_model: Optional[YOLO] = None
        self._cup_model: Optional[YOLO] = None
        self._sam_model: Optional[SAM] = None
        self._init_models()

        # ---- 状态缓存 ----
        self._last_kettle_bbox = None
        self._last_kettle_time = 0.0
        self._kettle_hold_sec = 1.0  # YOLO 漏检缓存时间
        self._last_cup_bbox = None
        self._last_handle_pos = None   # (x, y, z) or None
        self._last_cup_pos = None      # (x, y, z) or None
        # 可视化缓存 (检测成功后更新)
        self._last_handle_mask = None  # bool mask
        self._last_body_mask = None    # bool mask
        self._last_handle_rot = None   # 3x3 旋转矩阵
        self._last_handle_euler = None # 动态 euler (相机系 ZYX deg)

        # ---- 正对参考状态 (Step 4: 旋转角 θ 估计) ----
        self._ref_dx = None       # 自标定参考: 把手 x 中心偏移 (px)
        self._R_ref = None        # 正对参考旋转矩阵 (θ 小时缓存的 PCA 结果)
        self._theta_ema = 0.0     # θ 平滑值 (deg)

        # ---- 供 state_machine 写入 (类型声明防 Pylance 报错) ----
        self.last_known_pose: Optional[tuple] = None      # (pos_ee, euler_ee) or None
        self.last_known_pose_ts: float = 0.0             # last_known_pose 更新时间戳 (P2 复核新鲜度)
        self.last_known_pose_observation_id: int = 0
        self.last_known_pose_at_monotonic_ns: int = 0
        self.last_known_cup_pose: Optional[tuple] = None  # (pos_cam, euler_cam) or None

    # ------------------------------------------------------------

    def _init_pipeline(self):
        """打开 Camera A pipeline (color + depth)."""
        yaml_path = os.path.join(
            _project_root, "RealSense_R", "D435i_info.yaml"
        )
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self._serial:
            config.enable_device(self._serial)
            print(f"[HandleCup] 绑定 serial: {self._serial}")
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self._pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
        except Exception:
            pass

        params = load_camera_params(yaml_path)
        self._color_intr = params["color_intr"]
        self._depth_intr = params["depth_intr"]
        self._c2d_extr = params["c2d_extr"]
        self._d2c_extr = params["d2c_extr"]
        self._colorizer = rs.colorizer()

        print(f"[HandleCup] depth_scale={self._depth_scale:.6f}")
        print(f"[HandleCup] color_intr: fx={self._color_intr.fx:.1f} fy={self._color_intr.fy:.1f}")

    # ------------------------------------------------------------

    def _init_models(self):
        """加载 YOLO kettle + YOLO cup + SAM (无降级: SAM 缺失即启动失败)."""
        print(f"[HandleCup] Kettle: {self._kettle_model_path}")
        self._kettle_model = YOLO(self._kettle_model_path)
        self._kettle_model.to(self._device)

        print(f"[HandleCup] Cup: {self._cup_model_path}")
        self._cup_model = YOLO(self._cup_model_path)
        self._cup_model.to(self._device)

        print(f"[HandleCup] SAM: {self._sam_model_path}")
        if not os.path.exists(self._sam_model_path):
            raise FileNotFoundError(
                f"SAM 模型未找到: {self._sam_model_path} (handle 检测依赖 SAM, 无降级)")
        self._sam_model = SAM(self._sam_model_path)
        self._sam_model.to(self._device)

    # ------------------------------------------------------------
    #  帧获取
    # ------------------------------------------------------------

    def get_frames(self):
        """返回 (color, depth_image, depth_colormap, depth_frame)."""
        if self._pipeline is None:
            return None, None, None, None
        assert self._colorizer is not None  # type guard for Pylance
        frames = self._pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None, None, None
        color = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())
        depth_cmap = np.asanyarray(self._colorizer.colorize(depth_frame).get_data())
        return color, depth_img, depth_cmap, depth_frame

    # ------------------------------------------------------------
    #  Kettle YOLO
    # ------------------------------------------------------------

    def _detect_kettle(self, color_image):
        """YOLO 检测水壶, 带缓存."""
        assert self._kettle_model is not None  # type guard for Pylance
        results = self._kettle_model(
            color_image, conf=self._kettle_conf, verbose=False
        )
        best = None
        for r in results:
            if r.boxes is None:
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                if best is None or conf > best["conf"]:
                    best = {
                        "bbox": [int(b) for b in box],
                        "conf": float(conf),
                    }
        if best is not None:
            self._last_kettle_bbox = best["bbox"]
            self._last_kettle_time = time.time()
            return best["bbox"]
        # 缓存回退
        if self._last_kettle_bbox is not None:
            if time.time() - self._last_kettle_time < self._kettle_hold_sec:
                return self._last_kettle_bbox
        return None

    # ------------------------------------------------------------
    #  SAM 分割 (拷贝自 realsense_shared._segment_full_mask)
    # ------------------------------------------------------------

    def _sam_segment(self, color_image, bbox):
        """SAM bbox 提示分割 → 完整水壶 mask (壶身+把手).

        返回 np.ndarray (H, W), bool 或 None.
        """
        assert self._sam_model is not None  # type guard for Pylance
        bbox_w = bbox[2] - bbox[0]
        expand_lr = max(int(bbox_w * self._sam_roi_expand_ratio), 10)
        expand_tb = 10
        bbox_exp = [
            max(0, bbox[0] - expand_lr),
            max(0, bbox[1] - expand_tb),
            min(color_image.shape[1], bbox[2] + expand_lr),
            min(color_image.shape[0], bbox[3] + expand_tb),
        ]

        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        sam_results = self._sam_model(
            rgb_image,
            bboxes=[bbox_exp],
            conf=self._sam_conf,
            device=self._device,
            verbose=False,
        )

        if len(sam_results) == 0 or sam_results[0].masks is None:
            return None

        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask = mask > 0.5
        return mask

    # ------------------------------------------------------------
    #  锚点深度 (拷贝自 test_handle_6D._get_anchor_depth)
    # ------------------------------------------------------------

    def _get_anchor_depth(self, bbox, depth_frame):
        """bbox 内均匀采样 50 个点, 取有效深度中位数."""
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_roi = y2 - y1
        w_roi = x2 - x1

        depths = []
        for _ in range(50):
            cy = y1 + np.random.randint(0, max(h_roi, 1))
            cx = x1 + np.random.randint(0, max(w_roi, 1))
            dp = color_to_depth_pixel(
                cx, cy, depth_frame, self._depth_scale,
                self._depth_intr, self._color_intr,
                self._c2d_extr, self._d2c_extr,
            )
            if dp is None:
                continue
            dx, dy = dp
            d = depth_frame.get_distance(dx, dy)
            if self._depth_min < d < self._depth_max:
                depths.append(d)

        if len(depths) < 5:
            return None
        depths.sort()
        return depths[len(depths) // 2]

    # ------------------------------------------------------------
    #  6D 位姿 (PCA, 拷贝自 test_handle_6D._compute_handle_6d_pose)
    # ------------------------------------------------------------

    def _handle_6d_pose(self, handle_mask, depth_frame, anchor_depth, step=2):
        """PCA 6D 位姿 (彩色坐标系). 返回 (pos, rot, point_cloud) 或 (None,None,None)."""
        assert self._color_intr is not None  # type guard for Pylance
        return _compute_handle_6d_pose(
            handle_mask, depth_frame, anchor_depth,
            self._depth_scale, self._depth_intr, self._color_intr,
            self._c2d_extr, self._d2c_extr,
            step=step, z_threshold=self._depth_z_thr,
            depth_min=self._depth_min, depth_max=self._depth_max,
        )

    # ------------------------------------------------------------
    #  旋转角 θ 估计 (正对参考对比)
    # ------------------------------------------------------------

    def _ref_quality_ok(self, feat):
        """自标定质量门控: dx 同侧 + 面积比在界内."""
        ratio_min = float(self._scan_cfg.get("min_ratio", 0.05))
        ratio_max = float(self._scan_cfg.get("max_ratio", 0.55))
        if feat["dx"] <= 0:  # 右侧把手 → dx 应为正 (归一化后)
            return False
        if not (ratio_min <= feat["ratio"] <= ratio_max):
            return False
        return True

    def _update_reference(self, feat, rot_pca, theta_deg):
        """更新正对参考 (自标定):

        - dx_ref: 运行期 max(|dx|) 迟滞更新 (正对时 |dx| 最大, 只会向正对收敛)
        - R_ref : θ 小 (接近正对) 时 PCA 结果可信, 缓存为参考姿态
        """
        cfg = self._ref_cfg
        dx = abs(feat["dx"])

        # dx_ref 自标定 (仅质量好帧)
        if cfg.get("auto_calib", True) and self._ref_quality_ok(feat):
            if self._ref_dx is None or dx > self._ref_dx:
                self._ref_dx = dx

        # R_ref 缓存: 接近正对时 PCA 可信
        near_front = theta_deg < 15.0  # |dx|/dx_ref > cos15° ≈ 0.966
        if near_front and rot_pca is not None:
            rot_aligned = rot_pca
            if self._R_ref is not None:
                # PCA 主轴符号不定: 与现有参考的 Y 轴对齐 (防 180° 跳变)
                if rot_pca[:, 1] @ self._R_ref[:, 1] < 0:
                    rot_aligned = np.diag([-1.0, -1.0, 1.0]) @ rot_pca
            if self._R_ref is None:
                self._R_ref = rot_aligned.copy()
            else:
                # EMA 更新 + SVD 正交化
                alpha = 0.5
                self._R_ref = (1 - alpha) * self._R_ref + alpha * rot_aligned
                U, _, Vt = np.linalg.svd(self._R_ref)
                self._R_ref = U @ Vt

    def _estimate_yaw_signed(self, feat):
        """由 dx 特征估计绕壶身轴旋转角 θ (deg) + 方向符号.

        θ = arccos(|dx| / dx_ref), dx_ref 优先级:
          运行期自标定 max > config dx_px > 几何兜底 body_w × dx_frac_body.

        符号: PCA 相对 R_ref 的绕轴角符号 (非退化时可信), 否则 config sign.
        """
        cfg = self._ref_cfg
        if not cfg.get("enabled", True):
            return 0.0, 1.0

        dx = abs(feat["dx"])

        # 确定 dx_ref
        if self._ref_dx is not None:
            dx_ref = self._ref_dx
        elif cfg.get("dx_px"):
            dx_ref = float(cfg["dx_px"])
        else:
            dx_ref = feat["body_w"] * float(cfg.get("dx_frac_body", 0.45))

        if dx_ref <= 1.0:
            return 0.0, 1.0

        r = min(1.0, dx / dx_ref)
        theta = math.degrees(math.acos(r))
        theta = min(theta, float(cfg.get("theta_max_deg", 50)))

        # 方向符号融合在 _synthesize_handle_pose 内完成 (需 rot_pca 与 body_axis);
        # 此处仅返回 config 默认符号
        sign = float(cfg.get("sign", 1))
        return theta, sign

    # ------------------------------------------------------------
    #  姿态合成: PCA + θ → 完整动态姿态
    # ------------------------------------------------------------

    def _synthesize_handle_pose(self, rot_pca, body_axis, theta_deg):
        """合成完整姿态旋转矩阵.

        R_final = Rot(壶身轴, θ) @ R_ref (正对参考缓存).
        正对参考 R_ref 在 θ≈0 时由 PCA 缓存 → θ 增大后姿态 = 参考 + θ,
        PCA 的透视/遮挡表观误差被消除; R_ref 未建立时退回 PCA 原样.
        """
        sign = float(self._ref_cfg.get("sign", 1))

        # PCA 主轴符号不定: 与 R_ref 的 Y 轴对齐 (防 180° 跳变)
        if self._R_ref is not None and rot_pca[:, 1] @ self._R_ref[:, 1] < 0:
            rot_pca = np.diag([-1.0, -1.0, 1.0]) @ rot_pca

        # PCA 相对 R_ref 的绕壶身轴符号 (非退化时可信)
        if self._R_ref is not None and body_axis is not None:
            y_now = rot_pca[:, 1]
            y_ref = self._R_ref[:, 1]
            a = body_axis
            # 投影到垂直壶身轴的平面
            y_ref_p = y_ref - (y_ref @ a) * a
            y_p = y_now - (y_now @ a) * a
            n1 = np.linalg.norm(y_ref_p)
            n2 = np.linalg.norm(y_p)
            if n1 > 1e-9 and n2 > 1e-9:
                y_ref_p = y_ref_p / n1
                y_p = y_p / n2
                cross_ang = np.cross(y_ref_p, y_p)
                ang_s = float(np.dot(cross_ang, a))
                if abs(ang_s) > 1e-6:
                    sign = 1.0 if ang_s > 0 else -1.0

        theta_signed = sign * theta_deg

        # θ EMA 平滑 (防帧间跳变)
        alpha = float(self._ref_cfg.get("ema_alpha", 0.4))
        self._theta_ema = (1 - alpha) * self._theta_ema + alpha * theta_signed

        # 合成: 绕壶身轴旋转参考姿态
        if self._R_ref is not None and body_axis is not None and abs(self._theta_ema) > 0.5:
            R_theta = _rotate_around_axis(body_axis, self._theta_ema)
            R_final = R_theta @ self._R_ref
            # 正交化
            U, _, Vt = np.linalg.svd(R_final)
            R_final = U @ Vt
            return R_final

        return rot_pca

    # ------------------------------------------------------------
    #  Cup YOLO
    # ------------------------------------------------------------

    def _detect_cup_bbox(self, color_image):
        """YOLO 检测水杯 (class=41)."""
        assert self._cup_model is not None  # type guard for Pylance
        results = self._cup_model(
            color_image, classes=[41], conf=self._cup_conf, verbose=False
        )
        if results and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            confs = boxes.conf.cpu().numpy()
            best_idx = confs.argmax()
            if float(confs[best_idx]) >= self._cup_conf:
                xyxy = boxes.xyxy.cpu().numpy()[best_idx]
                return [int(x) for x in xyxy]
        return None

    # ------------------------------------------------------------
    #  Cup 3D 位置
    # ------------------------------------------------------------

    def _compute_cup_3d(self, color_image, depth_frame, cup_bbox):
        """bbox 内采样 → 3D 点云 → 中值位置.

        移植自 test_cup_6D.py compute_cup_6d_pose, 含 left/right 模式 bbox 裁剪.
        """
        assert self._color_intr is not None  # type guard for Pylance
        x1, y1, x2, y2 = cup_bbox

        # 锚点深度
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        dp = color_to_depth_pixel(
            cx, cy, depth_frame, self._depth_scale,
            self._depth_intr, self._color_intr,
            self._c2d_extr, self._d2c_extr,
        )
        if dp is None:
            return None
        anchor_d = depth_median_around(
            dp[0], dp[1], depth_frame,
            depth_min=self._depth_min, depth_max=self._depth_max,
        )
        if anchor_d is None:
            return None

        # ---- 根据 handle_side 裁剪 bbox (排除把手侧的背景点) ----
        # 移植自 test_cup_6D.py:159-171
        _x1_orig, _x2_orig = x1, x2
        if self._cup_handle_side in ("left", "right"):
            rim_px = max(1, int(self._cup_rim_radius * self._color_intr.fx / anchor_d))
            if self._cup_handle_side == "left":
                # 把手在左侧, 从右边界向左取 2 倍杯口半径
                x1 = max(int(x2) - 2 * rim_px, int(x1))
            elif self._cup_handle_side == "right":
                # 把手在右侧, 从左边界向右取 2 倍杯口半径
                x2 = min(int(x1) + 2 * rim_px, int(x2))
            if x2 - x1 < 4:
                x1, x2 = _x1_orig, _x2_orig  # 回退到原始 bbox
            # 更新锚点深度为裁剪后 bbox 中心
            _cx2 = (x1 + x2) // 2
            _dp2 = color_to_depth_pixel(
                _cx2, cy, depth_frame, self._depth_scale,
                self._depth_intr, self._color_intr,
                self._c2d_extr, self._d2c_extr,
            )
            if _dp2 is not None:
                _ad2 = depth_median_around(
                    _dp2[0], _dp2[1], depth_frame,
                    depth_min=self._depth_min, depth_max=self._depth_max,
                )
                if _ad2 is not None:
                    anchor_d = _ad2
            # 存储裁剪后 bbox 供可视化
            self._last_cup_bbox = [x1, y1, x2, y2]

        # 步长采样
        z_thr = self._depth_z_thr
        points = []
        for cy_i in range(y1, y2 + 1, 2):
            for cx_i in range(x1, x2 + 1, 2):
                dp2 = color_to_depth_pixel(
                    cx_i, cy_i, depth_frame, self._depth_scale,
                    self._depth_intr, self._color_intr,
                    self._c2d_extr, self._d2c_extr,
                )
                if dp2 is None:
                    continue
                dx2, dy2 = dp2
                d = depth_frame.get_distance(dx2, dy2)
                if not (self._depth_min < d < self._depth_max):
                    continue
                if abs(d - anchor_d) > z_thr:
                    continue
                px = (cx_i - self._color_intr.ppx) * d / self._color_intr.fx
                py = (cy_i - self._color_intr.ppy) * d / self._color_intr.fy
                points.append([px, py, d])

        if len(points) < 10:
            # 放宽 Z 过滤
            points2 = []
            for cy_i in range(y1, y2 + 1, 2):
                for cx_i in range(x1, x2 + 1, 2):
                    dp3 = color_to_depth_pixel(
                        cx_i, cy_i, depth_frame, self._depth_scale,
                        self._depth_intr, self._color_intr,
                        self._c2d_extr, self._d2c_extr,
                    )
                    if dp3 is None:
                        continue
                    dx3, dy3 = dp3
                    d = depth_frame.get_distance(dx3, dy3)
                    if not (self._depth_min < d < self._depth_max):
                        continue
                    px = (cx_i - self._color_intr.ppx) * d / self._color_intr.fx
                    py = (cy_i - self._color_intr.ppy) * d / self._color_intr.fy
                    points2.append([px, py, d])
            if len(points2) < 10:
                return None
            points = points2

        pts = np.array(points)
        return tuple(np.median(pts, axis=0))

    # ------------------------------------------------------------
    #  主检测接口
    # ------------------------------------------------------------

    def detect(self, color_image, depth_frame):
        """同步检测 Handle + Cup.

        Returns:
            dict with keys "handle", "cup". 每个子 dict:
              {"pos_cam": (x,y,z) or None,
               "euler_cam": (A,B,C),          # 检测成功=动态相机系 ZYX; 失败=config 固定值
               "kettle_bbox": [...] or None,   # handle only
               "red_center": None,             # 兼容键 (红胶带已移除)
               "white_roi": None,              # 兼容键 (白把手流程已移除)
               "red_contour": None,            # 兼容键
               }
        """
        handle_result = {
            "pos_cam": None, "euler_cam": self._handle_euler,
            "kettle_bbox": None, "red_center": None,
            "white_roi": None, "red_contour": None,
        }
        cup_result = {
            "pos_cam": None, "euler_cam": self._cup_euler,
            "cup_bbox": None,
        }

        # === Handle 检测 (yolo+sam) ===
        kettle_bbox = self._detect_kettle(color_image)
        handle_result["kettle_bbox"] = kettle_bbox

        if kettle_bbox is not None:
            self._last_handle_mask = None
            self._last_body_mask = None
            self._last_handle_rot = None

            # ---- 1. SAM 分割完整 mask ----
            full_mask = self._sam_segment(color_image, kettle_bbox)
            if full_mask is not None:
                # ---- 2. mask 后处理 (腐蚀/去噪点/填洞/连通域过滤) ----
                pp_mask = _postprocess_full_mask(full_mask, self._mask_pp_cfg, self._handle_side)

                # ---- 3. 扫描线分离把手 ----
                body_mask, handle_mask = self._scanline_split(pp_mask, kettle_bbox)

                # 后处理失败 → 原始 mask 重试一次 (SAM 偶发把把手切成孤岛)
                if handle_mask is None and self._mask_pp_cfg.get("retry_raw", True):
                    body_mask, handle_mask = self._scanline_split(full_mask, kettle_bbox)

                if handle_mask is not None and body_mask is not None:
                    self._last_handle_mask = handle_mask
                    self._last_body_mask = body_mask

                    # ---- 4. 锚点深度 + PCA 6D ----
                    anchor_depth = self._get_anchor_depth(kettle_bbox, depth_frame)
                    if anchor_depth is not None:
                        pos, rot, _cloud = self._handle_6d_pose(
                            handle_mask, depth_frame, anchor_depth,
                            step=self._pca_step_handle)
                        if pos is not None and rot is not None:
                            handle_result["pos_cam"] = pos
                            self._last_handle_pos = pos

                            # ---- 5. 壶身轴 (body PCA 主方向) ----
                            # 仅 R_ref 建立后才需要 (合成姿态用); θ=0 期间省一次点云
                            body_axis = None
                            if self._R_ref is not None:
                                _bpos, _brot, _ = self._handle_6d_pose(
                                    body_mask, depth_frame, anchor_depth,
                                    step=self._pca_step_body)
                                if _brot is not None:
                                    body_axis = _brot[:, 1]  # 壶身长轴

                            # ---- 6. 旋转角 θ + 完整动态姿态 ----
                            feat = _extract_handle_features(
                                body_mask, handle_mask, self._handle_side)
                            if feat is not None:
                                theta, _sign = self._estimate_yaw_signed(feat)
                                self._update_reference(feat, rot, theta)
                                R_final = self._synthesize_handle_pose(
                                    rot, body_axis, theta)
                            else:
                                R_final = rot

                            euler_dyn = rotation_matrix_to_euler_angles(R_final)
                            handle_result["euler_cam"] = tuple(euler_dyn)
                            self._last_handle_rot = R_final
                            self._last_handle_euler = tuple(euler_dyn)

                            # 调试日志 (θ / dx / 参考)
                            if feat is not None:
                                print(
                                    f"[Handle] dx={feat['dx']:.1f}px ratio={feat['ratio']:.2f} "
                                    f"theta={theta:.1f}° ema={self._theta_ema:.1f}° "
                                    f"ref_dx={self._ref_dx if self._ref_dx is not None else '-'} "
                                    f"euler=({euler_dyn[0]:.1f},{euler_dyn[1]:.1f},{euler_dyn[2]:.1f})")
                        else:
                            self._last_handle_pos = None
                # handle_mask 分离失败: 保持 pos=None, euler 回落固定值
            # SAM 失败: 保持 pos=None

        # === Cup 检测 ===
        cup_bbox = self._detect_cup_bbox(color_image)
        cup_result["cup_bbox"] = cup_bbox
        self._last_cup_bbox = cup_bbox

        if cup_bbox is not None:
            cup_pos = self._compute_cup_3d(color_image, depth_frame, cup_bbox)
            cup_result["pos_cam"] = cup_pos
            self._last_cup_pos = cup_pos

        return {"handle": handle_result, "cup": cup_result}

    # ------------------------------------------------------------

    def _scanline_split(self, full_mask, kettle_bbox):
        """扫描线分离把手 (v2), 参数来自 config 的 scanline 段."""
        cfg = self._scan_cfg
        return _scanline_find_handle_v2(
            full_mask, kettle_bbox,
            handle_min_ratio=float(cfg.get("min_ratio", 0.05)),
            handle_max_ratio=float(cfg.get("max_ratio", 0.55)),
            handle_side=self._handle_side,
            handle_center_deviation_ratio=float(cfg.get("center_deviation_ratio", 0.5)),
            trim_top=bool(cfg.get("trim_top", False)),
            trim_bottom=bool(cfg.get("trim_bottom", True)),
            use_mask_extent=bool(cfg.get("use_mask_extent", True)),
        )

    # ------------------------------------------------------------
    #  可视化
    # ------------------------------------------------------------

    def visualize(self, color_image):
        """绘制检测结果, 返回带标注的 BGR 图像."""
        display = color_image.copy()
        h, w = display.shape[:2]

        # ---- 状态行 ----
        h_status = "H:OK" if self._last_handle_pos is not None else "H:..."
        c_status = "C:OK" if self._last_cup_pos is not None else "C:..."
        cv2.putText(display, f"{h_status} {c_status}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # ---- 3D 位姿文字 ----
        _y_off = 55
        if self._last_handle_pos is not None:
            p = self._last_handle_pos
            e = self._last_handle_euler if self._last_handle_euler is not None else self._handle_euler
            cv2.putText(display, f"H: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) A={e[0]:.0f} B={e[1]:.0f} C={e[2]:.0f}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            _y_off += 20
            cv2.putText(display, f"   theta={self._theta_ema:.1f}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
            _y_off += 20
        if self._last_cup_pos is not None:
            p = self._last_cup_pos
            cv2.putText(display, f"C: ({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) A={self._cup_euler[0]:.0f} B={self._cup_euler[1]:.0f} C={self._cup_euler[2]:.0f}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

        # ---- Kettle bbox (橙色) ----
        if self._last_kettle_bbox is not None:
            x1, y1, x2, y2 = self._last_kettle_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 165, 255), 2)
            cv2.putText(display, "kettle", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        # ---- 壶身 mask (橙色半透明) ----
        if self._last_body_mask is not None:
            overlay = np.zeros_like(display)
            overlay[self._last_body_mask] = (0, 100, 255)
            display = cv2.addWeighted(display, 1.0, overlay, 0.35, 0)
            cv2.putText(display, "body", (10, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 255), 1)

        # ---- 把手 mask (绿色半透明) ----
        if self._last_handle_mask is not None:
            overlay = np.zeros_like(display)
            overlay[self._last_handle_mask] = (0, 255, 0)
            display = cv2.addWeighted(display, 1.0, overlay, 0.45, 0)

        # ---- 6D 坐标轴 (彩色坐标系投影) ----
        if self._last_handle_pos is not None and self._last_handle_rot is not None:
            if self._color_intr is not None:
                K = np.array([
                    [self._color_intr.fx, 0, self._color_intr.ppx],
                    [0, self._color_intr.fy, self._color_intr.ppy],
                    [0, 0, 1],
                ])
                center = np.array(self._last_handle_pos)
                for axis_col, color in [
                    (0, (0, 0, 255)),    # X → 红
                    (1, (0, 255, 0)),    # Y → 绿
                    (2, (255, 0, 0)),    # Z → 蓝
                ]:
                    end_3d = center + 0.08 * self._last_handle_rot[:, axis_col]
                    c2d = K @ center
                    c2d = c2d / c2d[2]
                    e2d = K @ end_3d
                    e2d = e2d / e2d[2]
                    cv2.arrowedLine(display,
                                    (int(c2d[0]), int(c2d[1])),
                                    (int(e2d[0]), int(e2d[1])),
                                    color, 2, tipLength=0.2)

        # ---- Cup bbox (品红) ----
        if self._last_cup_bbox is not None:
            x1, y1, x2, y2 = self._last_cup_bbox
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display, "cup", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

        return display

    # ------------------------------------------------------------
    #  清理
    # ------------------------------------------------------------

    def close(self):
        """停止 pipeline. (不调 destroyAllWindows — 由 main 线程统一清理)"""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("[HandleCup] pipeline 已停止")
            except Exception:
                pass
            self._pipeline = None

    # ------------------------------------------------------------
    #  独立测试
    # ------------------------------------------------------------

    def run(self):
        """交互式测试主循环."""
        print("=" * 55)
        print("  HandleCupDetector 独立测试")
        print("  Handle: YOLO kettle + SAM + 扫描线 + PCA 6D + 动态 euler")
        print("  Cup:   YOLO cup + 深度反投影")
        print("  按 'q' 退出, 's' 保存当前帧")
        print("=" * 55)

        frame_counter = 0
        test_skip = 5  # 独立测试固定间隔
        print(f"[INFO] 检测间隔: 每 {test_skip} 帧")

        _save_dir = os.path.join(_project_root, "logs")
        os.makedirs(_save_dir, exist_ok=True)

        _t_detect = 0.0
        try:
            while True:
                color, depth_img, depth_cmap, depth_frame = self.get_frames()
                if color is None or depth_frame is None:
                    time.sleep(0.03)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        break
                    continue

                frame_counter += 1
                display = color.copy()

                if frame_counter % test_skip == 1:
                    _t0 = time.perf_counter()
                    result = self.detect(color, depth_frame)
                    _t_detect = (time.perf_counter() - _t0) * 1000
                    h_info = result["handle"]
                    c_info = result["cup"]

                    # 终端输出 (相机系 3D 位置 + 动态 euler)
                    if h_info["pos_cam"] is not None:
                        p = h_info["pos_cam"]
                        e = h_info["euler_cam"]
                        print(f"[Handle] cam=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                              f"euler=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f}) "
                              f"[{_t_detect:.0f}ms]")
                    else:
                        kettle = "Y" if h_info["kettle_bbox"] is not None else "N"
                        print(f"[Handle] no detection (kettle={kettle}) [{_t_detect:.0f}ms]")

                    if c_info["pos_cam"] is not None:
                        p = c_info["pos_cam"]
                        print(f"[Cup]    cam=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
                    else:
                        print("[Cup]    no detection")

                display = self.visualize(color)
                if depth_cmap is not None:
                    h_vis = display.shape[0]
                    d_small = cv2.resize(depth_cmap, (h_vis, h_vis))
                    vis_panel = np.hstack((display, d_small))
                else:
                    vis_panel = display
                cv2.imshow("HandleCupDetector Test", vis_panel)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(_save_dir, f"handlecup_{ts}.png")
                    cv2.imwrite(path, vis_panel)
                    print(f"[SAVE] {path}")

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            self.close()
            print("[INFO] 已退出")


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    import yaml

    _orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    with open(_orch_path, "r", encoding="utf-8") as f:
        _orch_cfg = yaml.safe_load(f)

    _serial = ""
    _info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(_info_path, "r", encoding="utf-8") as f:
            _info = yaml.safe_load(f)
        _serial = str(_info["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")

    import torch
    _dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    detector = HandleCupDetector(orch_config=_orch_cfg, serial=_serial, device=_dev)
    detector.run()
