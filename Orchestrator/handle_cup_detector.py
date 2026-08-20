# -*- coding: utf-8 -*-
"""
Orchestrator Camera A Handle+Cup 检测器
=============================================
独立于 RealSense_R 体系, 不继承 BaseRealSenseDetector.

Handle 检测 (yolo+HSV, 2026-08-17 简化): YOLO kettle bbox → bbox 右 30%
             (把手侧) HSV 青色贴纸分割 → 最大域
             质心 = 中心 (3D, 深度: 彩色→深度坐标映射 + 质心邻域中位数)
             长短轴 = 6D 姿态 (X=短轴, Y=长轴, Z=X×Y 面法线, z<0 朝相机)
             → 帧间锚定 + 矩阵 EMA → euler_cam (完整动态 euler)
             (SAM 链已移除加载, 方法定义保留未调用)
Cup 检测:    YOLO cup bbox → bbox 采样 → 深度反投影 → 3D 位置 (原样保留)

euler_cam 语义: 检测成功 → 相机系动态 ZYX 欧拉角 (deg, 完整检测姿态);
                检测失败 → 回落 config 固定值 (永不为 None).

参考: Orchestrator/tests/test_tag_points.py (HSV 链实机验证), 旧 yolo+sam 链
      (RealSense_R/test_handle_6D.py, 方法定义保留)
"""

import os
import sys
import time
import math
from typing import Optional

# 项目根目录加入 sys.path (pyrealsense2 为 pip 包, 此注入供 import RealSense_R 等使用)
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO, SAM

# 姿态数学统一在 rot_math (协作线 B2: 动态抓取 + 测地连续性), 本模块只留别名.
from Orchestrator.rot_math import (
    MatrixSmoother as _MatrixSmoother,
    align_axes as _align_axes,
    rotate_around_axis as _rotate_around_axis,
    rotation_matrix_to_quaternion,
    unwrap_euler as _unwrap_euler,
)

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
#  HSV 青色贴纸检测 (2026-08-17 简化: 替代 SAM 链)
#  把手正面贴青色贴纸; YOLO bbox 右 30% 内 HSV 分割 → 最大域
#  质心 = 中心 (3D), 长短轴 = 6D 姿态 (X=短轴, Y=长轴, Z=X×Y).
#  逻辑移植自 Orchestrator/tests/test_tag_points.py (实机验证:
#  中心 ±0.2px, euler 稳定, 深度 0.63m 正常).
# ============================================================
HSV_HUE_LOW, HSV_HUE_HIGH = 40, 110   # 贴纸实测 H 36-105 (青绿渐变)
HSV_SAT_MIN, HSV_VAL_MIN = 50, 50     # 白色 S 低 / 黑色 V 低 排除
# 青色贴纸搜索区域 — bbox 宽度的百分比区间 (2026-08-20 用户定):
#   右侧把手 [70%, 110%] (向右扩 10%, 防贴纸被 bbox 右缘裁掉)
#   左侧把手 [-10%, 30%] (向左扩 10%)
HANDLE_ROI_X_RIGHT = (0.70, 1.10)
HANDLE_ROI_X_LEFT = (-0.10, 0.30)
# 2026-08-19: TAG_ROI_VPAD 纵向扩展已回退 — 贴纸被 YOLO 框裁是 +45 的稳定特征
# (ratio 虚高到 0.34~0.36 才能与 +15 的 0.30 区分), 保留原 bbox 高度检索
TAG_MIN_AREA, TAG_MAX_AREA = 8, 5000  # 青色域面积过滤
DEPTH_MIN = 0.1                       # 有效深度下限 (m)


def _detect_tag_region(color_image, bbox, handle_side="right"):
    """bbox 把手侧百分比区域内 HSV 青色贴纸检测 (最大域).

    区域 = bbox 宽度的百分比区间 (2026-08-20):
      右侧把手 [70%, 110%] (向右扩 10%, 防 bbox 右缘裁贴纸)
      左侧把手 [-10%, 30%] (向左扩 10%)
    区间越出图像边界时 clamp; 空区域返回 None.

    返回 (u, v, area, angle, axis1, axis2, region_mask) 或 None.
    region_mask 为全图系 bool; angle/axis1/axis2 来自 fitEllipse
    (angle 对应 width 边, 长轴按 axis1/axis2 大小取).
    """
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    x_lo, x_hi = (HANDLE_ROI_X_LEFT if handle_side == "left"
                  else HANDLE_ROI_X_RIGHT)
    rx0 = max(0, int(round(x0 + w * x_lo)))
    rx1 = min(color_image.shape[1], int(round(x0 + w * x_hi)))
    if rx1 <= rx0:
        return None
    roi = color_image[y0:y1, rx0:rx1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([HSV_HUE_LOW, HSV_SAT_MIN, HSV_VAL_MIN], dtype=np.uint8)
    upper = np.array([HSV_HUE_HIGH, 255, 255], dtype=np.uint8)
    bw = cv2.inRange(hsv, lower, upper)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(bw, 8)
    best_i, best_area = None, 0
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if TAG_MIN_AREA <= area <= TAG_MAX_AREA and area > best_area:
            best_i, best_area = i, area
    if best_i is None:
        return None
    u = rx0 + float(cents[best_i][0])
    v = y0 + float(cents[best_i][1])
    angle = None
    axis1 = axis2 = 0.0
    ys, xs = np.where(labels == best_i)
    if len(xs) >= 5:
        (_, _), (axis1, axis2), angle = cv2.fitEllipse(
            np.column_stack((xs, ys)).astype(np.float32))
    mask_full = np.zeros(color_image.shape[:2], dtype=bool)
    mask_full[y0:y1, rx0:rx1] = (labels == best_i)
    return (u, v, best_area, angle, axis1, axis2, mask_full)


def _depth_at_center(depth_frame, du, dv, depth_scale, radius=5):
    """深度坐标邻域有效深度中位数 (中心深度).

    排坑: color/depth FOV 不同, 彩色坐标必须经 color_to_depth_pixel
    映射后再取 (直接取会错位到背景); 深色贴纸区域空洞由中位数抗噪.
    """
    depth_img = np.asanyarray(depth_frame.get_data())
    h, w = depth_img.shape[:2]
    du, dv = int(round(du)), int(round(dv))
    vals = []
    for vv in range(-radius, radius + 1):
        for uu in range(-radius, radius + 1):
            x, y = du + uu, dv + vv
            if 0 <= x < w and 0 <= y < h:
                d = depth_img[y, x] * depth_scale
                if d > DEPTH_MIN:
                    vals.append(d)
    if not vals:
        return None
    return float(np.median(vals))


def _depth_in_region(depth_frame, region_mask, depth_scale):
    """贴纸区域 mask 内有效深度中位数 (区域内, 不取区域外背景)."""
    depth_img = np.asanyarray(depth_frame.get_data())
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return None
    vals = depth_img[ys, xs] * depth_scale
    vals = vals[vals > DEPTH_MIN]
    if len(vals) == 0:
        return None
    return float(np.median(vals))


def _pixel_to_3d(u, v, depth_m, intr):
    """图像像素 → 相机系 3D (pyrealsense2 intrinsics: 主点 ppx/ppy)."""
    return np.array([(u - intr.ppx) / intr.fx * depth_m,
                     (v - intr.ppy) / intr.fy * depth_m,
                     depth_m])


def _pose_from_tag_axes(u, v, angle, axis1, axis2, depth, intr):
    """贴纸长短轴 → 把手坐标系 R.

    Y = 长轴 (把手主轴, 原把手 PCA 主轴), X = 短轴 (替代原壶身 PCA),
    Z = X×Y, 约束 z<0 朝相机 (RealSense 约定).
    fitEllipse: angle 对应 width 边 (axis1), 长轴取较大边.
    返回 3x3 旋转矩阵或 None (退化).
    """
    if axis1 >= axis2:
        long_ang, L, S = angle, axis1 / 2.0, axis2 / 2.0
    else:
        long_ang, L, S = angle + 90.0, axis2 / 2.0, axis1 / 2.0
    rad = np.deg2rad(long_ang)
    rad_s = np.deg2rad(long_ang + 90.0)
    p1 = _pixel_to_3d(u + L * np.cos(rad), v + L * np.sin(rad), depth, intr)
    p2 = _pixel_to_3d(u - L * np.cos(rad), v - L * np.sin(rad), depth, intr)
    p3 = _pixel_to_3d(u + S * np.cos(rad_s), v + S * np.sin(rad_s), depth, intr)
    p4 = _pixel_to_3d(u - S * np.cos(rad_s), v - S * np.sin(rad_s), depth, intr)
    d_long = p1 - p2
    d_short = p3 - p4
    n_long = np.linalg.norm(d_long)
    n_short = np.linalg.norm(d_short)
    if n_long < 1e-6 or n_short < 1e-6:
        return None
    y_axis = d_long / n_long
    x_raw = d_short / n_short
    z_axis = np.cross(x_raw, y_axis)
    zn = np.linalg.norm(z_axis)
    if zn < 1e-6:
        return None
    z_axis = z_axis / zn
    if z_axis[2] > 0:      # z 朝相机 (RealSense 约定)
        z_axis = -z_axis
    x_axis = np.cross(y_axis, z_axis)  # 重正交, 右手系
    return np.column_stack([x_axis, y_axis, z_axis])


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
                             handle_center_deviation_ratio=0.25,
                             trim_top=True, trim_bottom=True,
                             use_mask_extent=True):
    """水平扫描线法分离壶身与把手 (v2).

    相对上游 scanline_find_handle 的改造 (缺陷修复):
      1. 搜索范围 = mask 非零外接矩形 ∪ bbox (原: 钳到 YOLO bbox) —
         把手外弧超出 bbox 的部分也能扫到, 不再截断 (只向把手侧扩展 x,
         壶身侧仍限 bbox 防吸背景; y 用并集).
      2. X 中心一致性剔除**严格化** (为 Y 主轴 PCA 稳定):
         双向剔 (trim_top=True), 单行超差即剔 (原: 连续 ≥2 行),
         且只删对应行 (原: 从超差行起整段 y: / :y+1 清零).
         (剔掉的把手弧两端由 _extend_handle_mask 扩展补回)

    参数:
        full_mask                     : np.ndarray (H, W), bool, SAM 生成的完整水壶 mask
        bbox                          : [x1, y1, x2, y2], YOLO 检测框
        handle_min_ratio / max_ratio  : 把手面积/壶身面积 最小/最大比例
        handle_side                   : "right" 或 "left", 把手预期侧
        handle_center_deviation_ratio : X 中心偏差容差(相对把手宽度比例, 严格=小值)
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
            """从一端向另一端扫描, 单行超差即剔除该行 (严格版, 只删对应行)."""
            for y, cxc, _ in rows_descending:
                if abs(cxc - cx_median) > tol:
                    handle_mask[y, :] = False

        if trim_bottom:
            _trim(sorted(row_centers, key=lambda x: -x[0]))  # 底部向上
        if trim_top:
            _trim(sorted(row_centers, key=lambda x: x[0]))   # 顶部向下

    # 重新计算 body_mask
    body_mask = full_mask.copy()
    body_mask[handle_mask] = False

    return body_mask, handle_mask


def _extend_handle_mask(handle_mask, full_mask, kettle_bbox, extend_px, handle_side):
    """以把手核心 mask 为基准, 向上/向下扩展补回把手弧两端.

    严格 X 中心一致性剔除会把弧形把手的上/下端切掉; 扩展把 full_mask 中
    位于把手侧 (远离壶身侧) 的像素补回 → Y 主轴 (长轴 PCA) 稳定且 mask 完整.

    参数:
        handle_mask : np.ndarray (H, W), bool, 严格剔除后的把手核心 mask
        full_mask   : np.ndarray (H, W), bool, SAM 完整 mask (补像素来源)
        kettle_bbox : [x1, y1, x2, y2], YOLO 检测框 (把手侧边界基准)
        extend_px   : int, 上下扩展行数
        handle_side : "right" | "left"
    返回:
        np.ndarray (H, W), bool, 扩展后的把手 mask
    """
    if extend_px <= 0:
        return handle_mask
    h, w = full_mask.shape
    ys, xs = np.where(handle_mask)
    if len(ys) == 0:
        return handle_mask
    y_top, y_bot = int(ys.min()), int(ys.max())

    # 把手侧判定基准: 核心段把手条的 x 范围 (保持长方形/平行四边形形状,
    # 不依赖 bbox — bbox 含把手时 bx2 偏大, 且扩展行把手条与壶身相连时
    # 用 bbox 只取到右侧段会破坏形状)
    core_xs = xs
    x_lo = float(np.percentile(core_xs, 5))
    x_hi = float(np.percentile(core_xs, 95))

    def _row_handle_side(y):
        """该行 full_mask 中落在核心段把手条 x 范围内的像素 (整条加入, 形状保持)."""
        idx = np.where(full_mask[y, :])[0]
        if len(idx) == 0:
            return None
        ext_idx = idx[(idx >= x_lo) & (idx <= x_hi)]
        return ext_idx

    ext = handle_mask.copy()
    gap_limit = 5  # 连续无把手像素行数 → 停止 (弧端连接壶身处自然停止)
    max_rows = extend_px * 4  # 扩展上限 (防失控)

    # 向上贪心扩展
    gap = 0
    for y in range(y_top - 1, max(-1, y_top - 1 - max_rows), -1):
        if y < 0:
            break
        ext_idx = _row_handle_side(y)
        if ext_idx is not None and len(ext_idx) > 0:
            ext[y, ext_idx] = True
            gap = 0
        else:
            gap += 1
            if gap >= gap_limit:
                break

    # 向下贪心扩展
    gap = 0
    for y in range(y_bot + 1, min(h, y_bot + 1 + max_rows)):
        ext_idx = _row_handle_side(y)
        if ext_idx is not None and len(ext_idx) > 0:
            ext[y, ext_idx] = True
            gap = 0
        else:
            gap += 1
            if gap >= gap_limit:
                break

    return ext


# ============================================================
#  3. mask 区域 PCA 主成分方向 (拷贝自 test_handle_6D._compute_handle_6d_pose, 简化)
# ============================================================

def _compute_pca_axis(mask, depth_frame, anchor_depth,
                       depth_scale, depth_intr, color_intr,
                       c2d_extr, d2c_extr,
                       step=2, z_threshold=0.05,
                       depth_min=0.15, depth_max=1.5):
    """对 mask 区域计算 PCA 主成分方向 (3D 点云, 彩色坐标系).

    流程:
      1. mask 像素逐点: 彩色像素 → 深度映射 → 读深度
      2. 彩色内参反投影 → 彩色坐标系 3D 点
      3. Z 离群过滤 |d - anchor_depth| < z_threshold
      4. 点不足时放宽 Z 过滤重试 (z_threshold × 2)
      5. PCA → 最大主成分方向 (归一化) + 点云重心

    ⚠️ 只返回主成分方向, 不构造完整旋转矩阵 —
    最终姿态坐标系由 _compute_handle_pose_geometry 构建
    (X=壶身轴, Y=壶身轴×环法线, Z=X×Y).

    返回:
        (position, axis_max, axis_min, point_cloud) 或 (None, None, None, None)
          position : (x, y, z) 点云重心 (m) — 彩色坐标系
          axis_max : 最大主成分方向 (长轴, 单位向量, 符号不定)
          axis_min : 最小方差方向 (环法线/厚度方向, 单位向量, 符号不定)
          point_cloud: Nx3 点云
    """
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return None, None, None

    # ---- 构建点云: mask 区域逐像素采样, 彩色内参反投影 ----
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

    # ---- PCA: 最大主成分方向 = 点云长轴 (把手长轴 / 壶身长轴) ----
    center = np.mean(point_cloud, axis=0)

    cov = np.cov(point_cloud, rowvar=False)
    eig_val, eig_vec = np.linalg.eig(cov)

    idx_sort = np.argsort(eig_val)[::-1]
    eig_vec = eig_vec[:, idx_sort]

    y_axis = np.real(eig_vec[:, 0])
    norm_y = np.linalg.norm(y_axis)
    if norm_y < 1e-10:
        return None, None, None
    y_axis = y_axis / norm_y

    # 最小方差方向 (环法线/厚度方向): 物理固定, 比最大主轴稳定得多 —
    # 几何法 Y 轴用"壶身轴 × 环法线"替代 PCA 主轴 (弧面主轴对点云变化敏感)
    min_axis = np.real(eig_vec[:, 2])
    norm_m = np.linalg.norm(min_axis)
    if norm_m < 1e-10:
        return None, None, None
    min_axis = min_axis / norm_m

    return center, y_axis, min_axis, point_cloud


# ============================================================
#  4. 把手 2D 特征 + 正对参考旋转角 θ
# ============================================================

# 姿态数学 (rotate_around_axis / align_axes / MatrixSmoother / unwrap_euler)
# 已迁移至 Orchestrator/rot_math.py, 本模块以别名导入 (见文件头), 行为与
# 7c7a6fd 完全一致; 新增输出键 rot_cam/quat_cam/source 供动态抓取消费.


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

        # ---- mask 后处理 / 扫描线 配置段 ----
        self._mask_pp_cfg = _c.get("mask_postprocess", {})
        self._scan_cfg = _c.get("scanline", {})
        self._grasp_direction = _c.get("grasp_direction", "right")  # 姿态 Y 符号规则

        # ---- 旋转角度判定 (方案A宽高比法, 2026-08-18) ----
        _rot = _c.get("handle_rotation", {})
        self._rot_enabled = bool(_rot.get("enabled", False))
        _gm = _rot.get("global_model", {})
        self._rot_c0 = float(_gm.get("c0", 0.1147))
        self._rot_c1 = float(_gm.get("c1_cos", 0.1642))
        self._rot_c2 = float(_gm.get("c2_sin", 0.1027))
        self._rot_c3 = float(_gm.get("c3_dd", -0.0725))
        self._rot_dref = float(_gm.get("d_ref", 0.63))
        # 纯 θ 区间判档边界 (2026-08-19 简化 5 档)
        #   band0: 0° 半宽 (θ∈[-band0,band0]→0); band15_hi: (band0, band15_hi]→15, >→30
        self._rot_band0 = float(_rot.get("band0_halfwidth", 8.0))
        self._rot_band15_hi = float(_rot.get("band15_hi", 22.0))
        # B2 2026-08-20: 正侧 ratio 上界 (标定物理域之外 = 非物理/裁切虚高) → REJECT, 不归 +30.
        #   数据依据 (标定 json 2026-08-18, 39 点): 合法档位 ratio 最大 0.3177 (+30, 含 d 修正
        #   抬升仍 <0.33), 0.34 留 ~0.02 余量, 对合法档位 0 误杀; >0.34 只有被 YOLO 框裁抬高
        #   的 +45 高尾. 物理干净 +45 与 +30 重合无法区分 → 靠物理摆位限定 [-35,+35] 排除.
        self._rot_ratio_max = float(_rot.get("ratio_max", 0.34))
        # 2026-08-20: 正侧反解 θ 校正 (仅 θ>band0 应用). 反解在正侧受 φ≈28.8° 峰值压缩
        #   (饱和, +30 档被压到 ~20°); 负侧/0° 保持原始反解 (C 补偿精度).
        #  pos_side (首选): 双阈值 rd 判档 + 分段线性连续角度, rd>rd30 用 30 档仿射 —
        #    中间角度 (+20~+25) 连续解算, 不二值化.
        #  g_correction (后备): 三次多项式 (已弃用).
        _ps = _rot.get("pos_side", {})
        if _ps.get("enabled", False):
            self._rot_rd_a = float(_ps.get("rd_a", 0.314))
            self._rot_rd_b = float(_ps.get("rd_b", 0.320))
            self._rot_rd30 = float(_ps.get("rd30", 0.3229))
            self._rot_k = float(_ps.get("k_slope", 50.0))
            self._rot_nodes = [(float(r), float(a))
                               for r, a in _ps.get("nodes", [])]
        else:
            self._rot_rd_a = self._rot_rd_b = self._rot_rd30 = None
            self._rot_k = 50.0
            self._rot_nodes = []
        _gc = _rot.get("g_correction", {})
        if (_gc.get("enabled", False)
                and all(k in _gc for k in ("a3", "a2", "a1", "a0"))):
            self._rot_g = (float(_gc["a3"]), float(_gc["a2"]),
                           float(_gc["a1"]), float(_gc["a0"]))
        else:
            self._rot_g = None

        # ---- 把手中心模式: mask_center (点云重心) | edge_offset (边界偏移法) ----
        self._pos_mode = _c.get("handle_pos_mode", "edge_offset")
        _eo = _c.get("edge_offset", {})
        self._edge_in_m = float(_eo.get("in_cm", 1.0)) / 100.0       # 把手侧边界向内 (cm→m)

        # ---- 诊断: 保存 mask/点云样本 (PCA 主方向偏斜排查用) ----
        _dbg = _c.get("debug", {})
        self._debug_save = bool(_dbg.get("save_samples", False))
        self._debug_dir = os.path.join(_project_root, _dbg.get("save_dir", "logs/detector_debug"))
        self._debug_interval = int(_dbg.get("save_interval", 30))
        self._debug_counter = 0

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
        self._last_handle_ratio = None # 方案A: 贴纸短/长轴比
        self._last_handle_theta = None # 方案A: 反解连续角度 θ (deg)
        self._last_handle_band = None  # 方案A: 旋转档位 (45/30/15/0/-15/-30/None)

        # ---- 姿态稳定状态 (180° 翻转修复) ----
        self._prev_rot = None     # 上一帧旋转矩阵 (Y 翻转锚定)
        self._matrix_smoother = _MatrixSmoother(  # 旋转矩阵 EMA (避万向锁跳变)
            alpha=float(_c.get("rot_smooth_alpha", 0.4)))

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

        # 2026-08-17 简化: handle 检测改用 HSV 青色贴纸, 不再加载 SAM
        # (SAM 链方法保留定义未调用; _sam_model 恒 None)

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
        """bbox 内均匀采样 50 个点, 取有效深度中位数.

        ⚠️ 固定 seed: 随机采样无 seed 时每帧采样点不同 → anchor 抖动 →
        Z 离群过滤边界变化 → 点云集合变化 → PCA 主轴漂移 (mask 不变但
        姿态波动的一大来源). 固定 seed 保证帧间采样点一致 (test_handle_6D
        原版有 seed(0), 重构时曾丢失).
        """
        rng = np.random.default_rng(0)  # 固定 seed → 帧间确定性采样
        x1, y1, x2, y2 = [int(v) for v in bbox]
        h_roi = y2 - y1
        w_roi = x2 - x1

        depths = []
        for _ in range(50):
            cy = y1 + rng.integers(0, max(h_roi, 1))
            cx = x1 + rng.integers(0, max(w_roi, 1))
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
    #  PCA 主成分方向 (拷贝自 test_handle_6D._compute_handle_6d_pose, 简化)
    # ------------------------------------------------------------

    def _pca_axis(self, mask, depth_frame, anchor_depth, step=2):
        """mask 区域 PCA 主成分方向 (彩色坐标系).

        返回 (pos, axis_max, axis_min, point_cloud):
          axis_max = 最大主成分方向 (长轴, 符号不定)
          axis_min = 最小方差方向 (环法线/厚度方向, 符号不定)
        几何法: Y = 壶身轴 × 环法线 (axis_min) — 比 PCA 长轴稳定
        (弧面长轴 λ1/λ2 接近, 方向对点云变化敏感).
        """
        assert self._color_intr is not None  # type guard for Pylance
        return _compute_pca_axis(
            mask, depth_frame, anchor_depth,
            self._depth_scale, self._depth_intr, self._color_intr,
            self._c2d_extr, self._d2c_extr,
            step=step, z_threshold=self._depth_z_thr,
            depth_min=self._depth_min, depth_max=self._depth_max,
        )

    # ------------------------------------------------------------
    #  旋转角 θ 估计 (正对参考对比)
    # ------------------------------------------------------------
    #  姿态构建: 3D 几何法 (X=壶身长轴, Y=把手长轴, Z=X×Y 面法线)
    # ------------------------------------------------------------

    def _compute_handle_pose_geometry(self, handle_axis, body_axis):
        """3D 几何法构建完整姿态旋转矩阵.

        坐标约定 (相机系, 符号规则按 grasp_direction + 图像边界):
          Y = 把手长轴 (handle PCA 主成分方向, 符号不定)
              符号: right → Y 指向图像下边界 (Y·(0,1,0)>0, 相机系 y 向下);
                    left  → Y 指向图像上边界 (Y·(0,-1,0)>0)
          X = 壶身长轴 (body PCA 主成分方向, 符号不定)
              符号: 右手法则 + Z 朝向相机约束 — Z=X×Y 需满足 z>0,
                    否则翻转 X (X=-X 时 Z 同步变号)
          Z = X×Y (把手面法线, TCP 贴合面; 不假设把手面正对相机,
              "Z>0 朝向相机一侧, 但不完全指向")

        ⚠️ 曾试用 Y = 壶身轴×环法线 (min_axis) — 实机最小方差方向是
        环厚方向而非环平面法线, B 角大幅倾斜 (-29°), 已回退 PCA 主轴.
        """
        if body_axis is None:
            return None

        Y = np.asarray(handle_axis, dtype=float)
        if self._grasp_direction == "right":
            img_dir = np.array([0.0, 1.0, 0.0])    # 图像下边界 (相机系 +y)
        else:
            img_dir = np.array([0.0, -1.0, 0.0])   # 图像上边界 (相机系 -y)
        if Y @ img_dir < 0:
            Y = -Y

        X = np.asarray(body_axis, dtype=float)
        Z = np.cross(X, Y)
        nz = np.linalg.norm(Z)
        if nz < 1e-9:
            return None
        Z = Z / nz
        if Z[2] > 0:  # Z 朝向相机一侧: RealSense/OpenCV 相机系 z 正=远离相机,
            #             故朝向相机 = z<0 (与 test_handle_6D PCA 的 Z 约定一致)
            X = -X
            Z = -Z

        nx = np.linalg.norm(X)
        ny = np.linalg.norm(Y)
        if nx < 1e-9 or ny < 1e-9:
            return None
        X = X / nx
        Y = Y / ny

        R = np.column_stack((X, Y, Z))
        # SVD 正交化 (X 壶身轴主导, 保持右手系)
        U, _, Vt = np.linalg.svd(R)
        return U @ Vt

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
    #  把手中心: 边界偏移法 (mask 覆盖不全时重心不可靠)
    # ------------------------------------------------------------

    def _compute_handle_center_edge(self, handle_mask, depth_frame, anchor_depth,
                                    handle_cloud=None):
        """用 mask 可靠边界估计把手中心.

        mask 覆盖不全时点云重心会偏移; mask 上下边界与把手侧边界通常可靠 →
        中心 y = 上下边界中点 (自适应 mask 高度, 不依赖固定偏移 — 固定 4cm
        在 mask 被严格剔除切短时会超出到下边界); 中心 x = 把手侧边界向内
        in_m (沿相机系图像方向).

        参数:
            handle_mask  : np.ndarray (H, W), bool
            depth_frame  : rs.depth_frame
            anchor_depth : float, 深度锚点 (Z 过滤)
        返回:
            (x, y, z) 或 None (边界像素不足时)
        """
        assert self._color_intr is not None  # type guard for Pylance
        ys, xs = np.where(handle_mask)
        if len(ys) < 10:
            return None
        y_min = int(ys.min())
        y_max = int(ys.max())
        if self._handle_side == "right":
            x_edge = int(xs.max())   # 右边界 (right 侧)
        else:
            x_edge = int(xs.min())   # 左边界

        def _edge_points(pixels):
            """像素列表 → 3D 点 (中值)."""
            pts = []
            for px, py in pixels:
                dp = color_to_depth_pixel(
                    px, py, depth_frame, self._depth_scale,
                    self._depth_intr, self._color_intr,
                    self._c2d_extr, self._d2c_extr,
                )
                if dp is None:
                    continue
                dx, dy = dp
                d = depth_frame.get_distance(dx, dy)
                if not (self._depth_min < d < self._depth_max):
                    continue
                if anchor_depth is not None and abs(d - anchor_depth) > self._depth_z_thr * 2:
                    continue
                pts.append([
                    (px - self._color_intr.ppx) * d / self._color_intr.fx,
                    (py - self._color_intr.ppy) * d / self._color_intr.fy,
                    d,
                ])
            if len(pts) < 3:
                return None
            return np.median(np.array(pts), axis=0)

        # 上/下边界行 (y_min/y_max) 与 把手侧边界列 的 3D 中值点
        top_row = [(int(cx), y_min) for cx in np.where(handle_mask[y_min, :])[0]]
        bot_row = [(int(cx), y_max) for cx in np.where(handle_mask[y_max, :])[0]]
        edge_col = [(x_edge, int(cy)) for cy in np.where(handle_mask[:, x_edge])[0]]
        P_top = _edge_points(top_row)
        P_bot = _edge_points(bot_row)
        P_edge = _edge_points(edge_col)
        if P_top is None or P_bot is None or P_edge is None:
            return None

        inward = float(self._edge_in_m)   # 把手侧边界向内 (m, 沿图像水平方向)

        # z: 优先用 mask 点云中位数 (所有把手像素 3D z 的中位, 对边缘噪声鲁棒);
        # 边界点平均对弧顶/弧侧深度差敏感且样本少 → z 波动大
        if handle_cloud is not None and len(handle_cloud) >= 10:
            z_center = float(np.median(handle_cloud[:, 2]))
        else:
            z_center = float(np.median([P_top[2], P_bot[2], P_edge[2]]))

        if self._handle_side == "right":
            center = np.array([P_edge[0] - inward,
                               (P_top[1] + P_bot[1]) / 2.0,
                               z_center])
        else:
            center = np.array([P_edge[0] + inward,
                               (P_top[1] + P_bot[1]) / 2.0,
                               z_center])
        return tuple(center)

    # ------------------------------------------------------------
    #  诊断: 保存 mask/点云样本 (PCA 主方向偏斜排查)
    # ------------------------------------------------------------

    def _save_debug_sample(self, color_image, full_mask, body_mask, handle_mask,
                           handle_cloud, body_cloud, kettle_bbox,
                           euler, handle_axis, body_axis):
        """保存 mask 叠加图 (壶身橙/把手绿) + 点云/参数 npz, 供离线分析.

        排查场景: PCA 有时把对角线当主方向 (Y 角 6~10° 而非 -2~2°) —
        需看 mask 覆盖完整性 (缺弧端? 形状偏斜?) 与点云分布.
        """
        os.makedirs(self._debug_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        tag = f"{ts}_{self._debug_counter:05d}"

        vis = color_image.copy()
        overlay = np.zeros_like(vis)
        if body_mask is not None:
            overlay[body_mask] = (0, 100, 255)   # 壶身 → 橙
        if handle_mask is not None:
            overlay[handle_mask] = (0, 255, 0)   # 把手 → 绿
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.5, 0)
        if kettle_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in kettle_bbox]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 165, 255), 1)
        cv2.imwrite(os.path.join(self._debug_dir, f"{tag}_mask.png"), vis)

        np.savez(os.path.join(self._debug_dir, f"{tag}_cloud.npz"),
                 handle_cloud=handle_cloud if handle_cloud is not None else np.zeros((0, 3)),
                 body_cloud=body_cloud if body_cloud is not None else np.zeros((0, 3)),
                 bbox=np.array(kettle_bbox, dtype=float),
                 euler=np.array(euler, dtype=float),
                 handle_axis=handle_axis if handle_axis is not None else np.zeros(3),
                 body_axis=body_axis if body_axis is not None else np.zeros(3))
        print(f"[DEBUG] 样本已保存: {self._debug_dir}/{tag}_*")

    # ------------------------------------------------------------
    #  旋转角度判定 (方案A宽高比法)
    # ------------------------------------------------------------

    def _solve_theta(self, ratio, d):
        """全局模型反解连续角度 θ (deg), 正侧经 rd 判档+仿射 或 g 校正.

        ratio = c0 + c1·cosθ + c2·sinθ + c3·(d-d_ref)
        θ = φ - arccos((ratio - c0 - c3·(d-d_ref)) / A),  φ=atan2(c2,c1), A=√(c1²+c2²).
        工作范围 [-30,+30] 内用负分支 (θ<φ). 超出 clip 到 ±32° (45° 档不靠反解).
        2026-08-20: θ > band0 (正侧, +15/+30 域) 时反解受 φ≈28.8° 峰值饱和压缩,
        不再直接用反解 — 改为:
          pos_side (首选): d 修正比 rd=ratio−c3(d−dref) 判档 (+15/+30), 再仿射校正到档中心
            (θ=center+k·(rd−centroid)), 绕开饱和区, 档内 spread<0.8°.
          g_correction (后备): 三次多项式 (已弃用).
        负侧/0° 保持原始反解 (C 补偿精度).
        """
        phi = math.atan2(self._rot_c2, self._rot_c1)
        amp = math.hypot(self._rot_c1, self._rot_c2)
        if amp < 1e-9:
            return 0.0
        x = (ratio - self._rot_c0 - self._rot_c3 * (d - self._rot_dref)) / amp
        x = max(-1.0, min(1.0, x))
        theta = math.degrees(phi - math.acos(x))
        if theta > self._rot_band0:
            if getattr(self, "_rot_nodes", None):
                # pos_side v2: 分段线性连续 (rd≤rd30), rd>rd30 用 30 档仿射
                rd = ratio - self._rot_c3 * (d - self._rot_dref)
                if rd > self._rot_rd30:
                    theta = 30.0 + self._rot_k * (rd - self._rot_rd30)
                else:
                    theta = self._interp_pos(rd)
            elif getattr(self, "_rot_g", None) is not None:
                a3, a2, a1, a0 = self._rot_g
                theta = a3 * theta**3 + a2 * theta**2 + a1 * theta + a0
        return theta

    def _interp_pos(self, rd):
        """正侧 rd → 连续角度 (分段线性过标定节点).

        节点: (rd, angle) = 0°/15°/20°/30° 各档 rd 中位 (config pos_side.nodes).
        使中间角度 (+15~+30 之间的实际摆位) 连续解算, 不二值化到档位.
        """
        nodes = self._rot_nodes
        if not nodes:
            return rd
        if rd <= nodes[0][0]:
            return nodes[0][1]
        for i in range(len(nodes) - 1):
            r0, a0 = nodes[i]
            r1, a1 = nodes[i + 1]
            if rd <= r1:
                return a0 + (rd - r0) * (a1 - a0) / (r1 - r0)
        return nodes[-1][1]

    def _classify_rotation(self, ratio, u, d):
        """旋转档位判定 (2026-08-19 简化: 纯 θ 区间, 5 档).

        固定抓取 5 档: +30/+15/0/-15/-30 (无 ±45; +45 物理摆位统一落 +30).
        边界 (用户定案 2026-08-19): [-8,8]=0, (8,22]=15, >22=30; 负侧对称
              [-22,-8)=-15, <-22→-30, <-40 不可达.
        纯 θ 反解区间, 无 per-band / u 门限 / 45 识别.

        Returns:
            (band, theta_deg): band ∈ {30,15,0,-15,-30,None}; None=不可达.
        """
        if not self._rot_enabled or ratio is None or d is None:
            return None, None
        # B2 2026-08-20: 反解前识别非物理/未标定 ratio (正侧越界) → REJECT, 不以 +30 代替.
        #   检测器不输出 θ (theta_deg=None) → data_sm 稳定窗口不加帧 → B3 缺样本 REJECT 链,
        #   Phase B 持续无效则超时 → 不发计划 → 受控停止.
        if ratio > self._rot_ratio_max:
            print(f"[ROT-REJECT] ratio={ratio:.4f} > {self._rot_ratio_max} 越界 "
                  f"(非物理/裁切虚高), 拒绝判档", flush=True)
            return None, None
        theta = self._solve_theta(ratio, d)
        b0 = self._rot_band0
        b15 = self._rot_band15_hi

        # 正侧 (θ>band0): pos_side 双阈值 rd 判档 (15 上限 / 30 下限; 中间区按连续 θ 取最近档)
        if theta > b0:
            if getattr(self, "_rot_nodes", None):
                rd = ratio - self._rot_c3 * (d - self._rot_dref)
                if rd >= self._rot_rd_b:
                    return 30, theta
                if rd <= self._rot_rd_a:
                    return 15, theta
                return (15 if theta < 22.5 else 30), theta
            # 无 pos_side: 回退 θ 区间判档
            if theta > b15:
                return 30, theta
            return 15, theta
        if theta < -40.0:
            return None, theta       # 不可达
        if theta < -b15:
            return -30, theta
        if theta < -b0:
            return -15, theta
        return 0, theta

    # ------------------------------------------------------------
    #  主检测接口
    # ------------------------------------------------------------

    def detect(self, color_image, depth_frame):
        """同步检测 Handle + Cup.

        Returns:
            dict with keys "handle", "cup". 每个子 dict:
              {"pos_cam": (x,y,z) or None,
               "euler_cam": (A,B,C),          # 检测成功=动态相机系 ZYX; 失败=config 固定值
               "rot_cam": 3x3 ndarray or None,  # 动态链平滑后旋转矩阵 (B2 动态抓取)
               "quat_cam": (w,x,y,z) or None,   # 同一旋转的四元数
               "source": "dynamic" | "fallback",# 本帧 euler_cam 来源 (动态链/配置回退)
               "kettle_bbox": [...] or None,   # handle only
               "red_center": None,             # 兼容键 (红胶带已移除)
               "white_roi": None,              # 兼容键 (白把手流程已移除)
               "red_contour": None,            # 兼容键
               }
        """
        handle_result = {
            "pos_cam": None, "euler_cam": self._handle_euler,
            "rot_cam": None, "quat_cam": None, "source": "fallback",
            "kettle_bbox": None, "red_center": None,
            "white_roi": None, "red_contour": None,
            # 方案A旋转判定 (2026-08-18): ratio/theta/band, 检测失败=均 None
            "ratio": None, "theta_deg": None, "band": None,
            # 18 回合证据集 (对方 08-20 addendum §3): 每有效样本记录字段
            "rd": None, "tag_u": None, "tag_v": None,
            "handle_side": self._handle_side,
        }
        cup_result = {
            "pos_cam": None, "euler_cam": self._cup_euler,
            "cup_bbox": None,
        }

        # === Handle 检测 (YOLO bbox + HSV 青色贴纸; 2026-08-17 简化, 替代 SAM 链) ===
        kettle_bbox = self._detect_kettle(color_image)
        handle_result["kettle_bbox"] = kettle_bbox

        if kettle_bbox is not None:
            self._last_handle_mask = None
            self._last_body_mask = None
            self._last_handle_rot = None

            # ---- HSV 青色贴纸检测 (把手侧百分比区域, 2026-08-20 扩 10%) ----
            tag = _detect_tag_region(color_image, kettle_bbox, self._handle_side)
            if tag is not None:
                (u, v, _area, angle, axis1, axis2, region_mask) = tag
                # 深度: 彩色→深度坐标映射 (FOV 不同) → 质心邻域 → 区域内兜底
                dpix = color_to_depth_pixel(
                    int(round(u)), int(round(v)), depth_frame,
                    self._depth_scale, self._depth_intr, self._color_intr,
                    self._c2d_extr, self._d2c_extr)
                d = None
                if dpix is not None:
                    d = _depth_at_center(depth_frame, dpix[0], dpix[1],
                                         self._depth_scale)
                if d is None:
                    d = _depth_in_region(depth_frame, region_mask,
                                         self._depth_scale)
                if d is not None:
                    pos = _pixel_to_3d(u, v, d, self._color_intr)
                    handle_result["pos_cam"] = pos
                    self._last_handle_pos = pos

                    # ---- 方案A旋转判定: ratio → θ 反解 → 档位 (2026-08-18) ----
                    if angle is not None and axis1 >= 2 and axis2 >= 2:
                        ratio = min(axis1, axis2) / max(axis1, axis2)
                        handle_result["ratio"] = ratio
                        band, theta = self._classify_rotation(ratio, u, d)
                        handle_result["theta_deg"] = theta
                        handle_result["band"] = band
                        # 18 回合证据集 (对方 08-20 addendum §3): rd 深度校正比 + 贴纸中心像素
                        handle_result["rd"] = ratio - self._rot_c3 * (d - self._rot_dref)
                        handle_result["tag_u"] = u
                        handle_result["tag_v"] = v
                        handle_result["handle_side"] = self._handle_side
                        self._last_handle_ratio = ratio
                        self._last_handle_theta = theta
                        self._last_handle_band = band

                    # ---- 6D 姿态: 贴纸长短轴 → 把手坐标系 → 锚定/EMA → euler ----
                    euler_dyn = None
                    if angle is not None and axis1 >= 2 and axis2 >= 2:
                        R_final = _pose_from_tag_axes(u, v, angle, axis1, axis2,
                                                      d, self._color_intr)
                        if R_final is not None:
                            R_aligned = _align_axes(R_final, self._prev_rot)
                            R_smooth = self._matrix_smoother.update(R_aligned)
                            self._prev_rot = R_smooth
                            euler_dyn = rotation_matrix_to_euler_angles(R_smooth)
                            if self._last_handle_euler is not None:
                                euler_dyn = _unwrap_euler(
                                    tuple(euler_dyn), self._last_handle_euler)
                            else:
                                euler_dyn = tuple(euler_dyn)
                            handle_result["rot_cam"] = R_smooth
                            handle_result["quat_cam"] = rotation_matrix_to_quaternion(R_smooth)
                            self._last_handle_rot = R_smooth
                    if euler_dyn is not None:
                        handle_result["euler_cam"] = euler_dyn
                        handle_result["source"] = "dynamic"
                        self._last_handle_euler = euler_dyn
                        lock_dist = abs(abs(euler_dyn[1]) - 90.0)
                        print(
                            f"[Handle] HSV pos=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                            f"euler=({euler_dyn[0]:.1f},{euler_dyn[1]:.1f},"
                            f"{euler_dyn[2]:.1f}) lock={lock_dist:.1f} "
                            f"grasp={self._grasp_direction}", flush=True)
                    else:
                        # 姿态退化 (长短轴无效): 位置有效但姿态回退
                        handle_result["pos_cam"] = None
                        self._last_handle_pos = None
                        self._prev_rot = None
                        self._matrix_smoother.reset()
                else:
                    # 深度空洞: 姿态状态重置
                    self._last_handle_pos = None
                    self._prev_rot = None
                    self._matrix_smoother.reset()
                    self._last_handle_ratio = None
                    self._last_handle_theta = None
                    self._last_handle_band = None
            else:
                # HSV 失败 (无青色): 姿态状态重置
                self._last_handle_pos = None
                self._prev_rot = None
                self._matrix_smoother.reset()
                self._last_handle_ratio = None
                self._last_handle_theta = None
                self._last_handle_band = None

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
        """扫描线分离把手 (v2, 严格模式), 参数来自 config 的 scanline 段."""
        cfg = self._scan_cfg
        return _scanline_find_handle_v2(
            full_mask, kettle_bbox,
            handle_min_ratio=float(cfg.get("min_ratio", 0.05)),
            handle_max_ratio=float(cfg.get("max_ratio", 0.55)),
            handle_side=self._handle_side,
            handle_center_deviation_ratio=float(cfg.get("center_deviation_ratio", 0.25)),
            trim_top=bool(cfg.get("trim_top", True)),
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
        # ---- 方案A旋转判定 (ratio/θ/band) ----
        if self._last_handle_ratio is not None:
            band_str = str(self._last_handle_band) if self._last_handle_band is not None else "N/A"
            th_str = f"{self._last_handle_theta:.1f}" if getattr(self, "_last_handle_theta", None) is not None else "N/A"
            cv2.putText(display, f"ROT: ratio={self._last_handle_ratio:.4f} th={th_str} band={band_str}",
                        (10, _y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
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

                    # 终端输出 (相机系 3D 位置 + 动态 euler + 方案A旋转判定)
                    if h_info["pos_cam"] is not None:
                        p = h_info["pos_cam"]
                        e = h_info["euler_cam"]
                        rot_str = ""
                        if h_info["ratio"] is not None:
                            b = h_info["band"]
                            th = h_info["theta_deg"]
                            b_str = str(b) if b is not None else "N/A"
                            th_str = f"{th:.1f}" if th is not None else "N/A"
                            rot_str = f" ratio={h_info['ratio']:.4f} th={th_str} band={b_str}"
                        print(f"[Handle] cam=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                              f"euler=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f})"
                              f"{rot_str} [{_t_detect:.0f}ms]")
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
