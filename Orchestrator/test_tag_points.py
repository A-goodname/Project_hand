# -*- coding: utf-8 -*-
"""
贴纸宽高比基准 r0 建立工具 (方案 A)
=====================================
正对水壶时持续输出每帧 ratio = 短轴/长轴，自动累积均值/方差作为 r0 候选值。
旋转时 r = r0·cosθ（±45° 内有效，60° 以上因遮挡失效）。

轻量直连 (2026-08-17): D435i pipeline + YOLO kettle bbox (无 SAM) +
HSV 青色贴纸检测 (bbox 右侧 30%).

用法:
  conda run -n Proj python Orchestrator/test_tag_points.py
按键: q 退出 | r 重置累积（换位置/角度后重新采样）
"""

import os
import sys
import time
import yaml
import cv2
import numpy as np

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from RealSense_R.realsense_shared import color_to_depth_pixel


# =============================================================
# 参数 (实机调参集中地)
# =============================================================
HUE_LOW, HUE_HIGH = 40, 110   # OpenCV H: 0-180; 贴纸实测 H 36-105 (青绿渐变)
SAT_MIN = 50                  # 白色 S 低 → 排除
VAL_MIN = 50                  # 黑色 V 低 → 排除

# 青色贴纸搜索区域 — bbox 宽度的百分比区间 (2026-08-20 用户定, 与 handle_cup_detector 一致):
#   右侧把手 [70%, 110%] (向右扩 10%, 防 bbox 右缘裁贴纸)
HANDLE_ROI_X_RIGHT = (0.70, 1.10)
# 2026-08-19: TAG_ROI_VPAD 纵向扩展已回退 — 贴纸被 YOLO 框裁是 +45 的稳定特征
# (ratio 虚高 0.34~0.36 才能与 +15 区分), 与 handle_cup_detector 保持一致
MIN_AREA = 8
MAX_AREA = 5000
OPEN_KERNEL = 1           # 形态学开核 (1=关)
DEPTH_MIN = 0.1           # 有效深度下限 (m)


def detect_tag_region(color_img, bbox):
    """在 YOLO bbox 内检测青色贴纸，返回最大域信息。

    返回 [(u, v, area, bbox, angle_deg, axis1, axis2, region_mask), ...] 按面积降序。
    """
    bx0, y0, bx1, y1 = bbox
    w = bx1 - bx0
    rx0 = max(0, int(round(bx0 + w * HANDLE_ROI_X_RIGHT[0])))
    rx1 = min(color_img.shape[1], int(round(bx0 + w * HANDLE_ROI_X_RIGHT[1])))
    if rx1 <= rx0:
        return []
    roi = color_img[y0:y1, rx0:rx1]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([HUE_LOW, SAT_MIN, VAL_MIN], dtype=np.uint8)
    upper = np.array([HUE_HIGH, 255, 255], dtype=np.uint8)
    bw = cv2.inRange(hsv, lower, upper)
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
        bbx = (rx0 + int(stats[i, cv2.CC_STAT_LEFT]), y0 + int(stats[i, cv2.CC_STAT_TOP]),
               int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]))
        u = rx0 + float(centroids[i][0])
        v = y0 + float(centroids[i][1])
        angle = None
        axis1 = axis2 = 0.0
        if area >= 5:
            ys_, xs_ = np.where(labels == i)
            if len(xs_) >= 5:
                (_, _), (axis1, axis2), angle = cv2.fitEllipse(
                    np.column_stack((xs_, ys_)).astype(np.float32))
        mask_full = np.zeros((color_img.shape[0], color_img.shape[1]), dtype=bool)
        mask_full[y0:y1, rx0:rx1] = (labels == i)
        pts.append((u, v, area, bbx, angle, axis1, axis2, mask_full))
    pts.sort(key=lambda p: -p[2])
    return pts


def depth_at_center(depth_img, u, v, depth_scale, radius=5):
    """质心邻域方形窗口有效深度中位数。"""
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
    """贴纸区域 mask 内有效深度中位数。"""
    ys, xs = np.where(region_mask)
    if len(xs) == 0:
        return None
    vals = depth_img[ys, xs] * depth_scale
    vals = vals[vals > DEPTH_MIN]
    if len(vals) == 0:
        return None
    return float(np.median(vals))


def pixel_to_3d(u, v, depth_m, intr):
    return np.array([(u - intr.ppx) / intr.fx * depth_m,
                     (v - intr.ppy) / intr.fy * depth_m,
                     depth_m])


def main():
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
    print(f"[INFO] pipeline ok, depth_scale={depth_scale:.6f}, "
          f"fx={color_intr.fx:.1f} fy={color_intr.fy:.1f}")

    from ultralytics import YOLO
    kettle_path = os.path.join(
        _project_root, orch_cfg.get("kettle_model", "model/kettle_yolov8n_finetune.pt"))
    kettle_conf = float(orch_cfg.get("kettle_conf", 0.25))
    kettle_model = YOLO(kettle_path)
    kettle_model.to(dev)
    print(f"[INFO] Kettle YOLO: {kettle_path} (conf={kettle_conf}, {dev})")

    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"ratio_r0_{ts}.log")
    log_f = open(log_path, "w", encoding="utf-8")
    print(f"[INFO] log -> {log_path}")

    def log(msg):
        """写 log 文件 + 终端。"""
        print(msg, flush=True)
        log_f.write(msg + "\n")
        log_f.flush()

    def log_file(msg):
        """只写 log 文件（逐帧检测不刷屏）。"""
        log_f.write(msg + "\n")
        log_f.flush()

    ratio_samples = []        # 累积 ratio 值 (每帧追加)
    frame_counter = 0

    cv2.namedWindow("Tag Points Test", cv2.WINDOW_NORMAL)

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

            # YOLO kettle bbox
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
                    ok = True
                    (u, v, area, _, angle, axis1, axis2, region_mask) = pts[0]
                    # 深度
                    dpix = color_to_depth_pixel(int(round(u)), int(round(v)),
                                                depth_frame, depth_scale,
                                                depth_intr, color_intr,
                                                c2d_extr, d2c_extr)
                    d = None
                    if dpix is not None:
                        d = depth_at_center(depth_img, dpix[0], dpix[1], depth_scale)
                    if d is None:
                        d = depth_in_region(depth_img, region_mask, depth_scale)
                    mid = None
                    if d is not None:
                        mid = pixel_to_3d(u, v, d, color_intr)
                    # ratio = 短轴/长轴
                    ratio = None
                    if angle is not None and axis1 >= 2 and axis2 >= 2:
                        ratio = min(axis1, axis2) / max(axis1, axis2)
                        ratio_samples.append(ratio)
                    # 累积统计
                    n = len(ratio_samples)
                    if n > 0:
                        arr = np.array(ratio_samples)
                        r_mean = arr.mean()
                        r_std = arr.std()
                    else:
                        r_mean = r_std = 0.0
                    # 输出
                    d_str = f"{d:.3f}" if d is not None else "None"
                    mid_str = (f"({mid[0]:.3f},{mid[1]:.3f},{mid[2]:.3f})"
                               if mid is not None else "None")
                    ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
                    log_file(f"[{frame_counter}] pos=({u:.1f},{v:.1f}) "
                             f"area={area} axis=({axis1:.1f},{axis2:.1f}) "
                             f"d={d_str}m mid={mid_str} "
                             f"ratio={ratio_str} | "
                             f"N={n} mean={r_mean:.4f} std={r_std:.4f}")
                    # 可视化: 轮廓 + 中心
                    cnts, _ = cv2.findContours(
                        region_mask.astype(np.uint8),
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(display, cnts, -1, (255, 255, 0), 1)
                    cv2.circle(display, (int(u), int(v)), 4, (0, 255, 255), -1)
                    # 画面上显示 ratio + 均值
                    cv2.putText(display, f"ratio={ratio_str}  N={n}  mean={r_mean:.4f}",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
                    cv2.putText(display, f"std={r_std:.4f}",
                                (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1)
                else:
                    log_file(f"[{frame_counter}] no cyan (bbox)")
            else:
                log_file(f"[{frame_counter}] no kettle")

            cv2.imshow("Tag Points Test", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                n_old = len(ratio_samples)
                if n_old > 0:
                    arr = np.array(ratio_samples)
                    log(f"[RESET] 清除 {n_old} 帧 "
                        f"(mean={arr.mean():.4f} std={arr.std():.4f} "
                        f"min={arr.min():.4f} max={arr.max():.4f})")
                ratio_samples.clear()

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        # 退出汇总
        n = len(ratio_samples)
        if n > 0:
            arr = np.array(ratio_samples)
            log(f"[SUMMARY] {n} 帧  mean={arr.mean():.4f}  std={arr.std():.4f}  "
                f"min={arr.min():.4f}  max={arr.max():.4f}")
        else:
            log("[SUMMARY] 无有效帧")
        log_f.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        print(f"[INFO] log 已保存: {log_path}")
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
