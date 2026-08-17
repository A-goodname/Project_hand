# -*- coding: utf-8 -*-
"""
把手青色贴纸检测试验
====================
方案 (2026-08-17): 把手正面贴青色贴纸 (标准长方形, 斜视投影为平行四边形),
YOLO bbox 内 HSV 色相分割青色 (bbox 内唯一青色):
  - 中心 = 最大青色域质心 (落在青色像素上, 白字空洞不偏置)
  - C = 最大域 fitEllipse 主轴 → ±L 两点 3D 水平投影
  - 深度 = 区域外扩环形带中位数 (深色贴纸吸红外 → 区域内空洞)

背景:
  - yolo+sam 链中 SAM mask 不稳定 (覆盖区域变化 → 中心/姿态漂移)
  - 黑点方案弃用: 2cm 把手面上两点仅 2-4px 太小; 灰色阈值受光照影响
  - 青色贴纸: HSV 色相分割, 白色 (S 低) / 黑色盖子 (V 低) 均被排除;
    bbox 内青色唯一 → 不依赖 SAM mask (mask 常只覆盖侧边)
  - minAreaRect 版已回退 (2026-08-17): 矩形被 bbox 内贴纸外的青色拉偏,
    中心偏离把手 (rect_c 摆动 ±16px); 回第一版最大域

输出: 每帧 cyan: 最大域质心/面积/主轴角 + 3D 中点 + C (deg);
      每 30 帧 [STAT] 中点/方向抖动.

轻量直连 (2026-08-17): D435i pipeline + YOLO kettle bbox (无 SAM 加载) +
HSV 青色贴纸检测 (bbox 右侧 30%). 拖拽圈定 (左键) = HSV 统计 + 保存原图.

排坑: D435i 主动红外, 深色贴纸深度空洞 → 深度取区域外扩环形带中位数.

用法:
  conda run -n Proj python Orchestrator/test_tag_points.py
按键: q 退出 | s 保存原图+青色二值到 logs/
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np
from collections import deque

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from RealSense_R.realsense_shared import (
    rotation_matrix_to_euler_angles,
    color_to_depth_pixel,
)


# =============================================================
# 参数 (实机调参集中地)
# =============================================================
# ---- 青色贴纸方案 (2026-08-17): 把手正面贴青色贴纸, HSV 色相检测 ----
# 黑点方案弃用: 2cm 把手面上两点仅 2-4px 太小; 青色在壶身 bbox 内唯一,
# 不依赖 SAM mask (mask 常只覆盖侧边), 盖子黑色/把手白色均被 HSV 排除
HUE_LOW, HUE_HIGH = 40, 110   # OpenCV H: 0-180; 贴纸实测 H 36-105 (青绿渐变, 2026-08-17 ROI 分析)
SAT_MIN = 50                  # 白色 S 低 → 排除 (渐变段 S 52-104 可纳入)
VAL_MIN = 50                  # 黑色 V 低 → 排除

HANDLE_SIDE_RATIO = 0.3   # 把手在 bbox 侧边的占比: 只检测 bbox 右侧 30% (排除壶身干扰)
MIN_AREA = 8              # 最小区域面积 (px)
MAX_AREA = 5000           # 最大区域面积 (px) — 防误检大片 (bbox 内青色唯一, 放宽)
MASK_DILATE = 3           # mask 膨胀核 (px, 备用: 若用 mask 约束时补容差)
OPEN_KERNEL = 1           # 形态学开核 (1=关; 小区域禁开运算)
DEPTH_RING_DILATE = 8     # 深度取样: 贴纸区域外扩环形带宽度 (px, 落在白色把手上)
DEPTH_MIN = 0.1           # 有效深度下限 (m)
STAT_N = 30               # 抖动统计窗口 (帧)


def detect_tag_region(color_img, bbox):
    """在 YOLO bbox 内检测青色贴纸 (第一版: 最大青色域 → 质心 + fitEllipse 主轴).

    返回 [(u, v, area, bbox, angle_deg, region_mask), ...] 按面积降序.
    angle_deg: 域主轴与图像 x 轴夹角 (fitEllipse, 面积 <5px 时为 None).
    说明: bbox 内青色唯一 (把手白/盖子黑), HSV 色相直接分割, 不依赖 SAM mask;
          白字/图标在青色上抠洞 → 青色碎片, 取最大域 (质心在青色像素上,
          不含白字空洞; minAreaRect 版矩形被贴纸外青色拉偏, 已回退 2026-08-17).
    """
    x0, y0, x1, y1 = bbox
    # 把手在壶身右侧 (grasp=right): 检测区域收窄到 bbox 右侧 30%, 排除壶身/背景青色干扰
    x0 = x1 - int((x1 - x0) * HANDLE_SIDE_RATIO)
    roi = color_img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([HUE_LOW, SAT_MIN, VAL_MIN], dtype=np.uint8)
    upper = np.array([HUE_HIGH, 255, 255], dtype=np.uint8)
    bw = cv2.inRange(hsv, lower, upper)
    # 闭运算填小洞 + 开运算去噪点
    kernel = np.ones((3, 3), np.uint8)
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    if OPEN_KERNEL > 1:
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN,
                              np.ones((OPEN_KERNEL, OPEN_KERNEL), np.uint8))
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, 8)
    pts = []
    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < MIN_AREA or area > MAX_AREA:
            continue
        bbx = (x0 + int(stats[i, cv2.CC_STAT_LEFT]), y0 + int(stats[i, cv2.CC_STAT_TOP]),
               int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        u = x0 + float(centroids[i][0])
        v = y0 + float(centroids[i][1])
        angle = None
        axis1 = axis2 = 0.0
        if area >= 5:
            ys_, xs_ = np.where(labels == i)
            if len(xs_) >= 5:
                (_, _), (axis1, axis2), angle = cv2.fitEllipse(
                    np.column_stack((xs_, ys_)).astype(np.float32))
        # 区域 mask 偏移回全图坐标 (供 depth_img 取样)
        mask_full = np.zeros((color_img.shape[0], color_img.shape[1]), dtype=bool)
        mask_full[y0:y1, x0:x1] = (labels == i)
        pts.append((u, v, area, bbx, angle, axis1, axis2, mask_full))
    pts.sort(key=lambda p: -p[2])
    return pts


def depth_at_center(depth_img, u, v, depth_scale, radius=5):
    """质心邻域方形窗口有效深度中位数 (中心深度).

    2026-08-17 用户指示: 外扩环形带取到的是贴纸周围背景 (贴纸盖满
    把手面时环形带落在背景上); 中心深度应取质心邻域 — 青色贴纸
    红外吸收有限, 邻域内有有效深度. 中位数抗少量空洞.
    """
    h, w = depth_img.shape[:2]
    u, v = int(round(u)), int(round(v))
    vals = []
    for dv in range(-radius, radius + 1):
        for du in range(-radius, radius + 1):
            x, y = u + du, v + dv
            if 0 <= x < w and 0 <= y < h:
                d = depth_img[y, x] * depth_scale
                if d > DEPTH_MIN:
                    vals.append(d)
    if not vals:
        return None
    return float(np.median(vals))


def depth_in_region(depth_img, region_mask, depth_scale):
    """贴纸区域 mask 内有效深度中位数 (区域内, 不取区域外背景)."""
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return None
    vals = depth_img[ys, xs] * depth_scale
    vals = vals[vals > DEPTH_MIN]
    if len(vals) == 0:
        return None
    return float(np.median(vals))


def pixel_to_3d(u, v, depth_m, intr):
    # pyrealsense2 intrinsics: 主点为 ppx/ppy (不是 cx/cy)
    return np.array([(u - intr.ppx) / intr.fx * depth_m,
                     (v - intr.ppy) / intr.fy * depth_m,
                     depth_m])


def analyze_roi(color_img, roi, save_dir):
    """圈定区域 HSV 统计 + 保存原图 (排 HSV 参数用, 2026-08-17).

    roi: (x0, y0, x1, y1) 全图坐标. 输出青色像素占比 + H 直方图 +
    青色区 H/S/V 范围 (现参数), 保存原图裁剪与标注图.
    """
    x0, y0, x1, y1 = roi
    crop = color_img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(int)
    s = hsv[..., 1].astype(int)
    v = hsv[..., 2].astype(int)
    cyan = (h >= HUE_LOW) & (h <= HUE_HIGH) & (s >= SAT_MIN) & (v >= VAL_MIN)
    n_total = h.size
    n_cyan = int(cyan.sum())
    print(f"[MARK] ROI=({x0},{y0})-({x1},{y1}) {x1-x0}x{y1-y0}px")
    print(f"[MARK] 现参数青色像素: {n_cyan}/{n_total} ({n_cyan/n_total*100:.1f}%)")
    hist_h, edges = np.histogram(h, bins=10, range=(0, 180))
    print("[MARK] H 直方图(0-180,每桶18°): " + " ".join(
        f"{int(e)}-{int(e2)}:{c}" for e, e2, c in zip(edges[:-1], edges[1:], hist_h)))
    if n_cyan:
        print(f"[MARK] 青色区 H: {h[cyan].min()}-{h[cyan].max()} "
              f"(median {np.median(h[cyan]):.0f})")
        print(f"[MARK] 青色区 S: {s[cyan].min()}-{s[cyan].max()} "
              f"(median {np.median(s[cyan]):.0f})")
        print(f"[MARK] 青色区 V: {v[cyan].min()}-{v[cyan].max()} "
              f"(median {np.median(v[cyan]):.0f})")
    else:
        cand = (h >= HUE_LOW) & (h <= HUE_HIGH)
        if cand.any():
            print(f"[MARK] 无青色像素; H 在 [{HUE_LOW},{HUE_HIGH}] 的像素 {int(cand.sum())} 个"
                  f" (S median {np.median(s[cand]):.0f}, V median {np.median(v[cand]):.0f})")
        else:
            print("[MARK] H 无落在 [70,110] 的像素 — 贴纸色相不在参数范围")
    ts = time.strftime("%Y%m%d_%H%M%S")
    cv2.imwrite(os.path.join(save_dir, f"roi_orig_{ts}.png"), crop)
    annot = color_img.copy()
    cv2.rectangle(annot, (x0, y0), (x1, y1), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(save_dir, f"roi_annot_{ts}.png"), annot)
    print(f"[MARK] 已保存: roi_orig_{ts}.png (原图裁剪) + roi_annot_{ts}.png (全图标框)")


def main():
    # ---- 轻量直连: D435i pipeline + YOLO kettle (无 SAM, 2026-08-17) ----
    orch_path = os.path.join(_project_root, "Orchestrator", "config.yaml")
    with open(orch_path, "r", encoding="utf-8") as f:
        orch_cfg = yaml.safe_load(f)
    serial = ""
    info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            serial = str(yaml.safe_load(f)["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")

    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    import pyrealsense2 as rs
    pipeline = rs.pipeline()
    rs_cfg = rs.config()
    if serial:
        rs_cfg.enable_device(serial)
    rs_cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    rs_cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(rs_cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    from RealSense_R.realsense_shared import load_camera_params
    params = load_camera_params(info_path)
    color_intr = params["color_intr"]
    depth_intr = params["depth_intr"]
    c2d_extr = params["c2d_extr"]
    d2c_extr = params["d2c_extr"]
    print(f"[INFO] pipeline OK, depth_scale={depth_scale:.6f}, "
          f"fx={color_intr.fx:.1f} fy={color_intr.fy:.1f}")

    # ---- YOLO kettle (bbox 源, 稳定) ----
    from ultralytics import YOLO
    kettle_path = os.path.join(
        _project_root, orch_cfg.get("kettle_model", "model/kettle_yolov8n_finetune.pt"))
    kettle_conf = float(orch_cfg.get("kettle_conf", 0.25))
    kettle_model = YOLO(kettle_path)
    kettle_model.to(dev)
    print(f"[INFO] Kettle YOLO: {kettle_path} (conf={kettle_conf}, {dev})")

    save_dir = os.path.join(_project_root, "logs")
    os.makedirs(save_dir, exist_ok=True)

    stats_mid = deque(maxlen=STAT_N)   # 中点 3D (相机系)
    stats_ang = deque(maxlen=STAT_N)   # 水平投影方向角 (deg)
    frame_counter = 0
    ratio_ref = None                   # 宽高比基准 r0 (正对时按 c 记录, 方案 A)
    current_ratio = None               # 最近一帧宽高比 (供 c 键记录)

    # ---- 鼠标圈定: 左键拖拽框选把手区域 → HSV 统计 + 保存原图 ----
    # (不依赖按键 — Linux Qt 后端 waitKey 需窗口聚焦才收按键, 鼠标事件无此限制)
    mark_state = {"p0": None, "p1": None, "done": False}

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            mark_state["p0"] = (x, y)
            mark_state["p1"] = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and flags & cv2.EVENT_FLAG_LBUTTON:
            mark_state["p1"] = (x, y)
        elif event == cv2.EVENT_LBUTTONUP:
            mark_state["p1"] = (x, y)
            mark_state["done"] = True

    cv2.namedWindow("Tag Points Test", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Tag Points Test", on_mouse)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color = np.asanyarray(color_frame.get_data())
            depth_img = np.asanyarray(depth_frame.get_data())
            frame_counter += 1

            # YOLO kettle bbox (无 SAM 链)
            bbox = None
            results = kettle_model(color, conf=kettle_conf, verbose=False)
            for r in results:
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                best = boxes[int(np.argmax(confs))]
                bbox = [int(b) for b in best]
                break
            display = color.copy()
            if bbox is not None:
                cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                              (255, 0, 0), 1)

            ok = False
            if bbox is not None:
                pts = detect_tag_region(color, bbox)
                if pts:
                    # ---- 第一版: 最大青色域 → 质心 = 中心, fitEllipse 主轴 = C ----
                    ok = True
                    if len(pts) > 1 and frame_counter % 10 == 1:
                        # 碎片数变化是白字/反光切碎所致, 不逐帧刷屏, 每 10 帧提示一次
                        print(f"[{frame_counter}] {len(pts)} 域, 取最大 (面积 "
                              f"{pts[0][2]}, 其他 {[p[2] for p in pts[1:]]})",
                              flush=True)
                    (u, v, area, _, angle, axis1, axis2, region_mask) = pts[0]
                    # 中心深度: 彩色坐标 → 深度坐标 (D435i color/depth FOV 不同,
                    # 直接取会错位到背景 — z=1.6m 根因, 2026-08-17 实机确认);
                    # 质心邻域优先, 空洞 → 区域 mask 内中位数 (不取背景)
                    dpix = color_to_depth_pixel(int(round(u)), int(round(v)),
                                                depth_frame, depth_scale,
                                                depth_intr, color_intr,
                                                c2d_extr, d2c_extr)
                    d = None
                    if dpix is not None:
                        d = depth_at_center(depth_img, dpix[0], dpix[1], depth_scale)
                    if d is None:
                        d = depth_in_region(depth_img, region_mask, depth_scale)
                    if d is not None:
                        intr = color_intr
                        mid = pixel_to_3d(u, v, d, intr)
                        ang = None
                        euler_cam = None
                        R_pose = None
                        L = S = 0.0
                        rad = 0.0
                        if angle is not None and axis1 >= 2 and axis2 >= 2:
                            # 完整 6D 姿态: 贴纸长方形长短轴 = 把手坐标系
                            #   Y = 长轴 (把手主轴, 原把手 PCA 主轴)
                            #   X = 短轴 (替代原壶身 PCA 主方向)
                            #   Z = X×Y, 约束 z<0 朝相机 (RealSense 约定)
                            # 长短轴 3D: 中心 ± 半轴两点反投影 (深度用环形带近似,
                            # 同处把手平面; 两端深度一致 → 方向由图像几何主导)
                            d_ring = d  # 同把手平面近似 (方向由图像几何主导)
                            # fitEllipse: angle 对应 width 边 (axis1);
                            # 长轴 = 较大边 (axis1/axis2 谁大谁长)
                            if axis1 >= axis2:
                                long_ang = angle
                                L = axis1 / 2.0
                                S = axis2 / 2.0
                            else:
                                long_ang = angle + 90.0
                                L = axis2 / 2.0
                                S = axis1 / 2.0
                            rad = np.deg2rad(long_ang)
                            rad_s = np.deg2rad(long_ang + 90.0)
                            p1 = pixel_to_3d(u + L * np.cos(rad), v + L * np.sin(rad),
                                             d_ring, intr)
                            p2 = pixel_to_3d(u - L * np.cos(rad), v - L * np.sin(rad),
                                             d_ring, intr)
                            d_long = p1 - p2
                            p3 = pixel_to_3d(u + S * np.cos(rad_s), v + S * np.sin(rad_s),
                                             d_ring, intr)
                            p4 = pixel_to_3d(u - S * np.cos(rad_s), v - S * np.sin(rad_s),
                                             d_ring, intr)
                            d_short = p3 - p4
                            y_axis = d_long / np.linalg.norm(d_long)
                            x_raw = d_short / np.linalg.norm(d_short)
                            z_axis = np.cross(x_raw, y_axis)
                            zn = np.linalg.norm(z_axis)
                            if zn > 1e-6:
                                z_axis = z_axis / zn
                                if z_axis[2] > 0:      # z 朝相机 (RealSense 约定)
                                    z_axis = -z_axis
                                x_axis = np.cross(y_axis, z_axis)  # 重正交右手系
                                R_pose = np.column_stack([x_axis, y_axis, z_axis])
                                euler_cam = rotation_matrix_to_euler_angles(R_pose)
                                # ang 保留作参考 (长轴水平投影方向)
                                d_horiz = np.array([d_long[0], d_long[1], 0.0])
                                ang = np.degrees(np.arctan2(d_horiz[1], d_horiz[0]))
                        stats_mid.append(mid)
                        if ang is not None:
                            stats_ang.append(ang)
                        # 方案 A (2026-08-17): 宽高比 → 旋转角
                        # r = 短轴/长轴 = r0·cosθ (正对 r0 标定, 按 c 记录当前帧)
                        ratio = None
                        theta = None
                        if angle is not None and axis1 >= 2 and axis2 >= 2:
                            ratio = min(axis1, axis2) / max(axis1, axis2)
                            current_ratio = ratio
                            if ratio_ref is not None:
                                _c = max(-1.0, min(1.0, ratio / ratio_ref))
                                theta = float(np.degrees(np.arccos(_c)))
                        euler_str = (f"{euler_cam[0]:.1f},{euler_cam[1]:.1f},"
                                     f"{euler_cam[2]:.1f}" if euler_cam is not None
                                     else "None")
                        print(f"[{frame_counter}] cyan: img=({u:.1f},{v:.1f}) "
                              f"area={area} "
                              f"angle_img={angle if angle is None else f'{angle:.1f}'} "
                              f"mid=({mid[0]:.3f},{mid[1]:.3f},{mid[2]:.3f}) "
                              f"ang={ang if ang is None else f'{ang:.1f}'}deg "
                              f"ratio={ratio if ratio is None else f'{ratio:.3f}'} "
                              f"theta={theta if theta is None else f'{theta:.1f}'}deg "
                              f"euler_cam=({euler_str})", flush=True)
                        # 可视化: 青色区域 mask 轮廓 + 长轴方向线 + 中心
                        cnts, _ = cv2.findContours(
                            region_mask.astype(np.uint8),
                            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, cnts, -1, (255, 255, 0), 1)
                        if angle is not None and L > 0:
                            cv2.line(display,
                                     (int(u - L * np.cos(rad)), int(v - L * np.sin(rad))),
                                     (int(u + L * np.cos(rad)), int(v + L * np.sin(rad))),
                                     (0, 0, 255), 1)
                        cv2.circle(display, (int(u), int(v)), 4, (0, 255, 255), -1)
                        # 3D 姿态坐标轴投影 (红=X 绿=Y 蓝=Z, 5cm) + 位置/姿态文字
                        if R_pose is not None:
                            for axis, col in zip(R_pose.T,
                                                 [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
                                p3d = mid + axis * 0.05
                                if p3d[2] > 0.1:
                                    pu = int(color_intr.ppx + p3d[0] / p3d[2] * color_intr.fx)
                                    pv = int(color_intr.ppy + p3d[1] / p3d[2] * color_intr.fy)
                                    cv2.line(display, (int(u), int(v)), (pu, pv), col, 2)
                            cv2.putText(display,
                                        f"mid=({mid[0]:.2f},{mid[1]:.2f},{mid[2]:.2f})",
                                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 255), 1)
                            cv2.putText(display,
                                        f"euler=({euler_cam[0]:.0f},{euler_cam[1]:.0f},"
                                        f"{euler_cam[2]:.0f})",
                                        (10, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 255), 1)
                    else:
                        print(f"[{frame_counter}] 深度空洞, 调整 DEPTH_RING_DILATE", flush=True)
                else:
                    print(f"[{frame_counter}] 无青色 (bbox内, 调 HUE/SAT/VAL)", flush=True)

                if len(stats_ang) == STAT_N:
                    a = np.array(stats_ang)
                    m = np.array(stats_mid)
                    print(f"[STAT] {STAT_N}帧 中点std=({m[:,0].std()*1000:.1f},"
                          f"{m[:,1].std()*1000:.1f},{m[:,2].std()*1000:.1f})mm "
                          f"ang_std={a.std():.1f}deg", flush=True)
                status = f"cyan={len(pts)}"
            else:
                status = "no handle"

            # ---- 圈定完成处理 (HSV 统计 + 原图保存) ----
            if mark_state["done"]:
                mark_state["done"] = False
                p0, p1 = mark_state["p0"], mark_state["p1"]
                mark_state["p0"] = mark_state["p1"] = None
                if p0 and p1:
                    rx0, ry0 = min(p0[0], p1[0]), min(p0[1], p1[1])
                    rx1, ry1 = max(p0[0], p1[0]), max(p0[1], p1[1])
                    if rx1 - rx0 >= 10 and ry1 - ry0 >= 10:
                        analyze_roi(color, (rx0, ry0, rx1, ry1), save_dir)
                    else:
                        print("[MARK] 区域太小, 忽略")

            cv2.putText(display, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255) if ok else (0, 0, 255), 2)
            if mark_state["p0"] and mark_state["p1"]:
                cv2.rectangle(display, mark_state["p0"], mark_state["p1"],
                              (0, 255, 0), 1)
            cv2.imshow("Tag Points Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("c"):
                # 记录正对基准宽高比 r0 (方案 A 标定)
                if current_ratio is not None:
                    ratio_ref = current_ratio
                    print(f"[CALIB] 正对基准 r0={ratio_ref:.3f} 已记录 "
                          f"(旋转后 theta=arccos(r/r0))")
                else:
                    print("[CALIB] 尚无检测帧, 无法记录")
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(save_dir, f"tagpoints_{ts}.png")
                cv2.imwrite(path, display)
                print(f"[SAVE] {path}")
                # 调试图: 原图 | 青色二值 (bbox 内) (排坑用)
                if bbox is not None:
                    x0, y0, x1, y1 = bbox
                    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
                    bw_save = cv2.inRange(
                        hsv,
                        np.array([HUE_LOW, SAT_MIN, VAL_MIN], dtype=np.uint8),
                        np.array([HUE_HIGH, 255, 255], dtype=np.uint8))
                    bw_save[:y0, :] = 0
                    bw_save[y1:, :] = 0
                    bw_save[:, :x0] = 0
                    bw_save[:, x1:] = 0
                    dbg_panel = np.hstack((color,
                                           cv2.cvtColor(bw_save, cv2.COLOR_GRAY2BGR)))
                    dbg_path = os.path.join(save_dir, f"tagpoints_debug_{ts}.png")
                    cv2.imwrite(dbg_path, dbg_panel)
                    print(f"[SAVE] {dbg_path} (orig|cyan)")

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        pipeline.stop()
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
