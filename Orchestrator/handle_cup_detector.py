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

# 项目根目录加入 sys.path (pyrealsense2 为 pip 包, 此注入供 import RealSense_R 等使用)
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
        kettle_bbox : [x1, y1, x2, y2], YOLO 检测框 (把手侧边界基准:
                      用 bbox 而非 body mask — body 可能含未剔除行的把手像素)
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

    bx1, _by1, bx2, _by2 = [int(v) for v in kettle_bbox]

    def _row_handle_side(y):
        """该行中位于把手侧 (远离壶身) 的 full_mask 像素."""
        idx = np.where(full_mask[y, :])[0]
        if len(idx) == 0:
            return None
        if handle_side == "right":
            ext_idx = idx[idx >= bx2 - 10]  # 壶身右侧 (把手侧, 放宽防漏)
        else:
            ext_idx = idx[idx <= bx1 + 10]  # 壶身左侧
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


def _align_axes(rot, prev_rot):
    """帧间姿态锚定: 在 4 个候选 (恒等 + 绕 X/Y/Z 各 180°) 中选与上一帧最接近者.

    治 PCA 主轴符号逐帧任意导致的帧间 180° 翻转 (根因 1).
    覆盖全部翻转模式:
      - Y 翻转 (绕壶身轴 X 180°)
      - Z 翻转 (绕把手轴 Y 180°, Z[2]≈0 时 z 符号约束失效的兜底)
      - X/Y 组合翻转 (绕相机 Z 180°)
    锚点: 上一帧旋转矩阵 (帧间锚定, 不依赖正对参考).
    """
    if prev_rot is None:
        return rot
    cands = [rot]
    for axis_i in range(3):
        cands.append(_rotate_around_axis(rot[:, axis_i], 180.0) @ rot)
    return max(cands, key=lambda R: float(np.trace(R.T @ prev_rot)))


class _MatrixSmoother:
    """旋转矩阵 EMA 平滑 + SVD 正交化 (私有化自 realsense_shared.PoseSmoother).

    在矩阵空间平滑, 避开 euler 万向锁区域的帧间跳变 (根因 2 放大因子):
    接近 B≈±90° 时 euler 提取对微小矩阵噪声敏感 ~100 倍, 矩阵 EMA 使
    输出平滑摆动而非跳变.
    """

    def __init__(self, alpha=0.4):
        self.alpha = alpha
        self.filtered = None

    def update(self, rot):
        if self.filtered is None:
            self.filtered = rot.copy()
            return self.filtered
        self.filtered = (1 - self.alpha) * self.filtered + self.alpha * rot
        U, _, Vt = np.linalg.svd(self.filtered)
        self.filtered = U @ Vt
        return self.filtered

    def reset(self):
        self.filtered = None


def _unwrap_euler(euler_new, euler_prev):
    """A/C 取与上一帧最接近的 ±360k 等价表示 (治 ±180 边界表观跳变).

    B 由矩阵 EMA 保证平滑, 不需要 wrap.
    """
    out = list(euler_new)
    for i in (0, 2):
        while out[i] - euler_prev[i] > 180:
            out[i] -= 360
        while euler_prev[i] - out[i] > 180:
            out[i] += 360
    return tuple(out)


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
    #  姿态构建: 3D 几何法 (X=壶身长轴, Y=把手长轴, Z=X×Y 面法线)
    # ------------------------------------------------------------

    def _compute_handle_pose_geometry(self, handle_rot, body_axis):
        """3D 几何法构建完整姿态旋转矩阵 (替代 dx→θ→R_ref 自标定链).

        坐标约定 (相机系, 符号规则按 grasp_direction + 图像边界):
          Y = 把手长轴 (handle PCA 主成分)
              符号: right → Y 指向图像下边界 (Y·(0,1,0)>0, 相机系 y 向下);
                    left  → Y 指向图像上边界 (Y·(0,-1,0)>0)
          X = 壶身长轴 (body PCA 主成分)
              符号: 右手法则 + Z 朝向相机约束 — Z=X×Y 需满足 z>0,
                    否则翻转 X (X=-X 时 Z 同步变号)
          Z = X×Y (把手面法线, TCP 贴合面; 不假设把手面正对相机,
              "Z>0 朝向相机一侧, 但不完全指向")

        优点 (解决 dx/θ 链问题):
          - 姿态由 3D 几何自足, 不需要"正对参考"自标定 — 水壶不动也能
            确定姿态 (解决原 dx_ref 依赖"最大 mask"的问题)
          - Z 是面法线而非"PCA 最小方差+指向相机" — 水壶旋转后把手面
            朝向 TCP 贴合面而非相机 (解决长方体把手抓取贴面问题)
        """
        if body_axis is None:
            return None

        Y = handle_rot[:, 1]
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
                src_mask = pp_mask
                if handle_mask is None and self._mask_pp_cfg.get("retry_raw", True):
                    body_mask, handle_mask = self._scanline_split(full_mask, kettle_bbox)
                    src_mask = full_mask

                if handle_mask is not None and body_mask is not None:
                    # ---- 3.5 把手 mask 精细化: 严格剔除后上下扩展补回把手弧两端 ----
                    handle_mask = _extend_handle_mask(
                        handle_mask, src_mask, kettle_bbox,
                        int(self._scan_cfg.get("extend_px", 15)),
                        self._handle_side)

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

                            # ---- 5. 壶身轴 (body PCA 主成分 = 壶身长轴) ----
                            body_axis = None
                            _bpos, _brot, _ = self._handle_6d_pose(
                                body_mask, depth_frame, anchor_depth,
                                step=self._pca_step_body)
                            if _brot is not None:
                                body_axis = _brot[:, 1]  # 壶身长轴

                            # ---- 6. 3D 几何法姿态: X=壶身长轴, Y=把手长轴,
                            #        Z=X×Y (面法线, z>0 朝向相机一侧) ----
                            R_final = self._compute_handle_pose_geometry(rot, body_axis)
                            if R_final is None:
                                R_final = rot  # 壶身轴退化时退回把手 PCA

                            # ---- 姿态稳定链 (180° 翻转修复):
                            #   帧间锚定 (4 候选最近, 覆盖 Y/Z/X 翻转) → 矩阵 EMA → euler unwrap ----
                            R_aligned = _align_axes(R_final, self._prev_rot)
                            R_smooth = self._matrix_smoother.update(R_aligned)
                            self._prev_rot = R_smooth

                            euler_dyn = rotation_matrix_to_euler_angles(R_smooth)
                            if self._last_handle_euler is not None:
                                euler_dyn = _unwrap_euler(
                                    tuple(euler_dyn), self._last_handle_euler)
                            else:
                                euler_dyn = tuple(euler_dyn)
                            handle_result["euler_cam"] = euler_dyn
                            self._last_handle_rot = R_smooth
                            self._last_handle_euler = euler_dyn

                            # 调试日志 (姿态 / B 角万向锁距离)
                            lock_dist = abs(abs(euler_dyn[1]) - 90.0)
                            print(
                                f"[Handle] euler=({euler_dyn[0]:.1f},{euler_dyn[1]:.1f},"
                                f"{euler_dyn[2]:.1f}) lock={lock_dist:.1f} "
                                f"grasp={self._grasp_direction}")
                        else:
                            # PCA 失败: 姿态状态重置 (防陈旧锚点/平滑缓存)
                            self._last_handle_pos = None
                            self._prev_rot = None
                            self._matrix_smoother.reset()
                else:
                    # 分离失败: 姿态状态重置
                    self._last_handle_pos = None
                    self._prev_rot = None
                    self._matrix_smoother.reset()
            # SAM 失败: 保持 pos=None (姿态状态保留, 短时漏检锚点仍有效)

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
