# -*- coding: utf-8 -*-
"""
RealSense YOLO-World + SAM + 水平扫描线 水壶把手定位
=====================================================
使用 YOLO-World(GPU) 检测水壶本体 + SAM(CPU) 分割 + 水平扫描线分离把手.

策略:
  - YOLO-World → 零样本检测 kettle bbox
  - SAM → 在 bbox 内生成壶身+把手完整 mask
  - 水平扫描线 mask|gap|mask 模式分离把手
  - 校验: 把手面积占比 + 质心位置 + 最大连通分量

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/test_SAM_handle.py
  3. 按 'q' 键退出
"""

import sys
import os
import yaml

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO, SAM, FastSAM


# ============================================================
#  0. 参数加载工具函数
# ============================================================

def _load_camera_params(yaml_path):
    """
    从标准 YAML 文件加载相机参数 (深度+彩色+外参).
    返回 dict:
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


# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

FRAME_W = cfg["resolution"]["width"]
FRAME_H = cfg["resolution"]["height"]
FRAME_FPS = cfg["resolution"]["fps"]

# ---- YOLO-World 配置 ----
_yw_cfg = cfg.get("yolo_world", {})
YW_MODEL_PATH = os.path.join(_project_root, _yw_cfg.get("model", "./model/yolov8m-worldv2.pt"))
YW_CONF_THRESHOLD = _yw_cfg.get("conf_threshold", 0.25)
YW_CLASSES = _yw_cfg.get("classes", ["kettle", "electric kettle", "water jug", "teapot"])
YW_DEVICE = 'cuda:0' if (_yw_cfg.get("device", "gpu") == "gpu" and torch.cuda.is_available()) else 'cpu'

# ---- SAM 配置 ----
_sam_cfg = cfg.get("sam", {})
SAM_MODEL_NAME = _sam_cfg.get("model", "mobile_sam.pt")
SAM_CONF = _sam_cfg.get("conf", 0.3)
_sam_device_pref = _sam_cfg.get("device", "gpu")
SAM_DEVICE = 'cuda:0' if (_sam_device_pref == "gpu" and torch.cuda.is_available()) else 'cpu'
print("[INFO] SAM 运行设备: %s" % SAM_DEVICE)

SAM_MODEL_PATH = os.path.join(_project_root, "model", SAM_MODEL_NAME)
IS_FAST_SAM = SAM_MODEL_NAME.lower().startswith("fastsam")

# ---- 把手检测配置 ----
_handle_cfg = cfg.get("handle", {})
HANDLE_SIDE = _handle_cfg.get("handle_side", "right")
HANDLE_MIN_RATIO = _handle_cfg.get("handle_min_ratio", 0.05)
HANDLE_MAX_RATIO = _handle_cfg.get("handle_max_ratio", 0.55)

_cam_params = _load_camera_params(
    os.path.join(os.path.dirname(__file__), 'D435i_info.yaml')
)


# ============================================================
#  1. 把手分离: 水平扫描线法
# ============================================================

def _scanline_find_handle(full_mask, bbox, side="right"):
    """
    用水平扫描线在 mask 上找 mask|gap|mask 模式, 分离壶身与把手.

    原理:
      对每一行 y, 从左到右扫描像素:
        连续 mask 段 = non-mask 段 = mask 段 → 穿过壶身与把手的横切面
        两个 mask 段中面积较小者直接标记为该行的把手区域.
      所有行把手区域的并集 = handle_mask.

    参数:
        full_mask : np.ndarray (H, W), bool, SAM 生成的完整水壶 mask
        bbox      : [x1, y1, x2, y2], YOLO 检测框
        side      : "right" 或 "left" (仅用于最终校验)

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
            pattern_rows += 1
            # 取面积最大的两个段
            segments.sort(key=lambda s: s[1] - s[0], reverse=True)
            seg1 = segments[0]
            seg2 = segments[1]

            # seg 左, seg2 右
            if seg1[0] > seg2[0]:
                seg1, seg2 = seg2, seg1

            area1 = seg1[1] - seg1[0]
            area2 = seg2[1] - seg2[0]

            # 小段 = 把手 (直接标记, 不投票)
            if area1 <= area2:
                handle_start, handle_end = seg1
            else:
                handle_start, handle_end = seg2

            if handle_end - handle_start >= 3:
                handle_mask[y, x1 + handle_start:x1 + handle_end + 1] = True

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
    if not (HANDLE_MIN_RATIO <= ratio <= HANDLE_MAX_RATIO):
        return None, None

    # 把手质心在指定侧
    ys, xs = np.where(handle_mask)
    cx = float(np.mean(xs))
    bbox_cx = (bbox[0] + bbox[2]) / 2.0

    if side == "right" and cx < bbox_cx:
        return None, None
    if side == "left" and cx > bbox_cx:
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

    # 重新计算 body_mask
    body_mask = full_mask.copy()
    body_mask[handle_mask] = False

    return body_mask, handle_mask


# ============================================================
#  2. 主检测器
# ============================================================

class RealSenseHandleDetector:
    """
    YOLO-World + SAM + 水平扫描线分离 水壶把手检测器.

    流程:
      1. YOLO-World 检测水壶本体 bbox
      2. SAM 在 bbox 内生成完整 mask (壶身+把手)
      3. 水平扫描线 mask|gap|mask 分离壶身和把手
    """

    def __init__(self):
        # --- RealSense 初始化 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FRAME_FPS)

        self.profile = self.pipeline.start(config)

        # --- 深度缩放系数 ---
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print("[INFO] depth_scale = %f" % self.depth_scale)

        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
            print("[INFO] laser_power set to 150")
        except Exception as e:
            print("[WARN] laser_power set failed: %s" % str(e))

        # --- 相机参数 ---
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

        # --- YOLO-World 模型 ---
        print("[INFO] 加载 YOLO-World 模型: %s" % YW_MODEL_PATH)
        if not os.path.exists(YW_MODEL_PATH):
            raise FileNotFoundError("YOLO-World 模型文件不存在: %s" % YW_MODEL_PATH)
        self.yolo_world_model = YOLO(YW_MODEL_PATH)
        self.yolo_world_model.to(YW_DEVICE)
        self.yolo_world_model.set_classes(YW_CLASSES)
        print("[INFO] YOLO-World 词汇表: %s" % str(YW_CLASSES))

        # --- SAM ---
        if not os.path.exists(SAM_MODEL_PATH):
            print("[WARN] SAM 模型未找到: %s" % SAM_MODEL_PATH)
            print("[WARN] 将跳过 SAM 推理")
            self.sam_model = None
        elif IS_FAST_SAM:
            self.sam_model = FastSAM(SAM_MODEL_PATH)
            self.sam_model.to(SAM_DEVICE)
            print("[INFO] FastSAM 模型已加载: %s" % SAM_MODEL_PATH)
        else:
            self.sam_model = SAM(SAM_MODEL_PATH)
            self.sam_model.to(SAM_DEVICE)
            print("[INFO] SAM 模型已加载: %s" % SAM_MODEL_PATH)

        self.colorizer = rs.colorizer()

        # --- 耗时统计 ---
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # ---- 跳帧缓存配置 ----
        self._frame_counter = 0
        self._skip_frames = 10
        self._cached_result = None

    # ------------------------------------------------------------
    def get_frames(self):
        """获取原始彩色帧和原始深度帧."""
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image, depth_colormap, depth_frame

    # ------------------------------------------------------------
    def _segment_full_mask(self, rgb_image, bbox):
        """在 bbox 区域内用 SAM 生成完整水壶 mask (壶身+把手)."""
        if self.sam_model is None:
            return None

        expand_px = 10
        bbox_exp = [
            max(0, bbox[0] - expand_px),
            max(0, bbox[1] - expand_px),
            min(FRAME_W, bbox[2] + expand_px),
            min(FRAME_H, bbox[3] + expand_px),
        ]

        sam_results = self.sam_model(
            rgb_image,
            bboxes=[bbox_exp],
            conf=SAM_CONF,
            device=SAM_DEVICE,
            verbose=False
        )

        if len(sam_results) == 0 or sam_results[0].masks is None:
            return None

        mask = sam_results[0].masks.data[0].cpu().numpy()
        mask = mask > 0.5
        return mask

    # ------------------------------------------------------------
    def _detect_kettle_bbox(self, color_image):
        """用 YOLO-World 检测水壶 bbox, 返回最高置信度结果."""
        results = self.yolo_world_model(color_image, conf=YW_CONF_THRESHOLD, verbose=False)
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
    def detect_handle(self, color_image, depth_frame=None):
        """
        主检测函数: YOLO-World → SAM → 水平扫描线分离 → 输出.

        返回:
            handle_mask    : np.ndarray (H, W), bool, 把手 mask 或 None
            handle_bbox    : [x1, y1, x2, y2], 把手 bbox 或 None
            body_mask      : np.ndarray (H, W), bool, 壶身 mask 或 None
            full_mask      : np.ndarray (H, W), bool, 完整水壶 mask 或 None
            stage          : str, "success" / "none"
        """
        # === Step 1: YOLO-World 检测水壶本体 ===
        _t_yolo_start = time.perf_counter()
        best = self._detect_kettle_bbox(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t_yolo_start) * 1000

        if best is None:
            return None, None, None, None, "none"

        bbox = best["bbox"]

        # === Step 2: SAM 生成完整水壶 mask ===
        _t_sam_start = time.perf_counter()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        full_mask = self._segment_full_mask(rgb_image, bbox)
        self._sam_time_ms = (time.perf_counter() - _t_sam_start) * 1000

        if full_mask is None:
            return None, None, None, None, "none"

        # === Step 3: 水平扫描线分离把手 ===
        body_mask, handle_mask = _scanline_find_handle(
            full_mask, bbox, side=HANDLE_SIDE
        )

        if handle_mask is not None:
            # ---- 使用 minAreaRect 的 axis-aligned 包围盒 (比 np.where 更紧) ----
            mask_uint8 = handle_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(largest_contour)
                box_pts = cv2.boxPoints(rect)
                xs_pts = [int(p[0]) for p in box_pts]
                ys_pts = [int(p[1]) for p in box_pts]
                handle_bbox = [
                    float(min(xs_pts)), float(min(ys_pts)),
                    float(max(xs_pts)), float(max(ys_pts)),
                ]
            else:
                ys, xs = np.where(handle_mask)
                handle_bbox = [
                    float(np.min(xs)), float(np.min(ys)),
                    float(np.max(xs)), float(np.max(ys)),
                ]

            handle_area = float(np.sum(handle_mask))
            body_area = float(np.sum(body_mask))
            print("[INFO] Handle detected: area_ratio=%.1f%%" % (
                handle_area / body_area * 100
            ))
            return handle_mask, handle_bbox, body_mask, full_mask, "success"

        print("[WARN] No handle found (scanline detection failed)")
        return None, None, full_mask, full_mask, "none"

    # ------------------------------------------------------------
    def visualize(self, display_img, body_mask, handle_mask,
                  handle_bbox, stage):
        """可视化检测结果."""
        overlay = display_img.copy()

        if body_mask is not None:
            body_only = body_mask.copy()
            if handle_mask is not None:
                body_only = body_only & (~handle_mask)
            overlay[body_only] = (255, 0, 0)
            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)

        if handle_mask is not None:
            overlay2 = display_img.copy()
            overlay2[handle_mask] = (0, 255, 0)
            cv2.addWeighted(overlay2, 0.4, display_img, 0.6, 0, display_img)

            mask_uint8 = handle_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)

        if body_mask is not None:
            mask_uint8 = body_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(display_img, contours, -1, (255, 0, 0), 1)

        if handle_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in handle_bbox]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 额外绘制旋转矩形轮廓 (更贴合把手形状)
            if handle_mask is not None:
                mask_uint8 = handle_mask.astype(np.uint8) * 255
                ctrs, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if ctrs:
                    big_ctr = max(ctrs, key=cv2.contourArea)
                    rect = cv2.minAreaRect(big_ctr)
                    box_pts = cv2.boxPoints(rect)
                    # np.int0 is an alias for np.int64, which cv2.drawContours accepts
                    box_pts = box_pts.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(display_img, [box_pts], 0, (0, 255, 255), 2)

        if stage == "success":
            color = (0, 255, 0)
            text = "Handle OK"
        else:
            color = (0, 0, 255)
            text = "No handle"

        cv2.putText(display_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_img, "YOLO: %dms  SAM: %dms" % (
            int(self._yolo_time_ms), int(self._sam_time_ms)),
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # ------------------------------------------------------------
    def run(self):
        """
        主循环: 每 N 帧执行一次完整检测, 中间帧复用缓存.
        按 'q' 退出, 按 's' 保存当前帧 (原图+mask) 至 images/ 目录.
        """
        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        print("按 'q' 键退出程序...")
        print("按 's' 键保存当前帧 (原图+mask) 至 %s" % _save_dir)
        print("水壶检测: YOLO-World (%s)" % str(YW_CLASSES))
        print("把手侧: %s, 把手面积比: [%.0f%%, %.0f%%]" % (
            HANDLE_SIDE, HANDLE_MIN_RATIO * 100, HANDLE_MAX_RATIO * 100))
        print("跳帧策略: 每 %d 帧执行一次完整检测" % self._skip_frames)

        _last_color = None
        _last_full_mask = None

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                self._frame_counter += 1
                display_img = color_image.copy()

                # --- 每 N 帧执行一次完整检测 ---
                if self._frame_counter % self._skip_frames == 1:
                    handle_mask, handle_bbox, body_mask, full_mask, stage = \
                        self.detect_handle(color_image, depth_frame)
                    self._cached_result = (
                        handle_mask, handle_bbox, body_mask, stage,
                    )
                    _last_color = color_image.copy()
                    _last_full_mask = full_mask.copy() if full_mask is not None else None
                else:
                    if self._cached_result is not None:
                        handle_mask, handle_bbox, body_mask, stage = self._cached_result
                    else:
                        handle_mask = None
                        handle_bbox = None
                        body_mask = None
                        stage = "none"

                self.visualize(display_img, body_mask, handle_mask,
                               handle_bbox, stage)

                cv2.imshow('YOLO-World+SAM Handle Detection', display_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if _last_color is not None:
                        ts = time.strftime("%Y%m%d_%H%M%S")
                        color_path = os.path.join(_save_dir, "frame_%s_color.png" % ts)
                        cv2.imwrite(color_path, _last_color)
                        print("[SAVE] %s" % color_path)
                        if _last_full_mask is not None:
                            mask_path = os.path.join(_save_dir, "frame_%s_mask.png" % ts)
                            cv2.imwrite(mask_path, (_last_full_mask.astype(np.uint8) * 255))
                            print("[SAVE] %s" % mask_path)
                    else:
                        print("[WARN] No frame saved yet (wait for detection)")

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("[INFO] 程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseHandleDetector()
    detector.run()