# -*- coding: utf-8 -*-
"""
RealSense 微调YOLO + SAM + 水平扫描线 水壶把手定位
==================================================
继承 BaseRealSenseDetector 基类.
使用微调 YOLO (GPU/CPU) 检测水壶本体 + SAM (GPU) 分割 + 水平扫描线分离把手.

相比旧版主要改进:
  1. 微调 YOLO bbox 更精准, 能完整框住水壶(含把手), 为 SAM 提供更优 ROI
  2. SAM 分割质量更高
  3. 手动触发模式: 默认实时显示, 按 's' 才执行单次推理, 避免持续 GPU 负载
  4. 按 'v' 保存水壶 mask 到 images/ 目录 (不再保存 .txt)
  5. full_mask 后处理: 连通分量分析, 剔除外围孤立分量

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/test_SAM_handle.py
  3. 按 's' 键触发单次推理 (YOLO -> SAM -> 扫描线分离)
  4. 按 'v' 键保存水壶 mask (RESULT 模式下)
  5. 按 'q' 键退出
"""

import sys
import os

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import os
import time
import yaml
import cv2
import numpy as np

from RealSense_R.realsense_shared import (
    BaseRealSenseDetector,
    scanline_find_handle,
)


# ============================================================
#  1. 模块级配置加载
# ============================================================

_cfg_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  2. 主检测器
# ============================================================

class RealSenseHandleDetector(BaseRealSenseDetector):
    """
    微调 YOLO + SAM + 水平扫描线分离 水壶把手检测器.

    流程:
      1. 微调 YOLO 检测水壶本体 bbox (能完整框住壶身+把手)
      2. SAM 在 bbox ROI 内生成完整 mask (壶身+把手)
      3. full_mask 后处理: 连通分量分析, 剔除孤立小分量
      4. 水平扫描线 mask|gap|mask 分离壶身和把手

    运行模式:
      - 默认实时显示相机画面 (不推理, 30fps)
      - 按 's' 键触发单次完整推理, 冻结当前帧显示结果
      - 按 'v' 键保存水壶 mask
      - 按 'q' 键退出
    """
    USE_YOLO_WORLD = False
    USE_FINETUNED_KETTLE = True
    USE_SAM = True
    LOG_NAME = "test_SAM_handle.log"

    # ---- 显示模式 ----
    MODE_LIVE = "live"        # 纯实时画面, 不推理
    MODE_INFERRING = "inferring"  # 推理中
    MODE_RESULT = "result"    # 显示推理结果

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

        # --- 耗时统计 ---
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # --- 显示状态 ---
        self._display_mode = self.MODE_LIVE
        self._result_cache = None  # 推理结果缓存 (供 MODE_RESULT 使用)
        self._infer_start_time = 0.0

    # ------------------------------------------------------------
    def _calc_handle_bbox(self, handle_mask):
        """从 handle_mask 计算 axis-aligned 包围盒."""
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
        return handle_bbox

    # ------------------------------------------------------------
    def detect_handle(self, color_image, depth_frame=None):
        """
        主检测函数: 微调 YOLO -> SAM -> 后处理 -> 水平扫描线分离 -> 输出.

        参数:
            color_image : np.ndarray (H, W, 3), BGR 彩色图
            depth_frame : 可选, 暂未使用 (保留接口兼容性)

        返回:
            handle_mask    : np.ndarray (H, W), bool, 把手 mask 或 None
            handle_bbox    : [x1, y1, x2, y2], 把手 bbox 或 None
            body_mask      : np.ndarray (H, W), bool, 壶身 mask 或 None
            full_mask      : np.ndarray (H, W), bool, 完整水壶 mask 或 None
            stage          : str, "success" / "none"
            yolo_bbox      : [x1, y1, x2, y2], 微调 YOLO 原始检测框, 或 None
        """
        # === Step 1: 微调 YOLO 检测水壶本体 ===
        _t_yolo_start = time.perf_counter()
        best = self._detect_kettle_finetuned(color_image)
        self._yolo_time_ms = (time.perf_counter() - _t_yolo_start) * 1000

        if best is None:
            return None, None, None, None, "none", None

        bbox = best["bbox"]

        # === Step 2: SAM 生成完整水壶 mask ===
        _t_sam_start = time.perf_counter()
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
        full_mask = self._segment_full_mask(rgb_image, bbox)
        self._sam_time_ms = (time.perf_counter() - _t_sam_start) * 1000

        if full_mask is None:
            return None, None, None, None, "none", bbox

        # === Step 3: 水平扫描线分离把手 ===
        body_mask, handle_mask = scanline_find_handle(
            full_mask, bbox,
            handle_min_ratio=self.HANDLE_MIN_RATIO,
            handle_max_ratio=self.HANDLE_MAX_RATIO,
            handle_side=self.HANDLE_SIDE,
            handle_center_deviation_ratio=self.HANDLE_CENTER_DEVIATION_RATIO
        )

        if handle_mask is not None and body_mask is not None:
            handle_bbox = self._calc_handle_bbox(handle_mask)

            handle_area = float(np.sum(handle_mask))
            body_area = float(np.sum(body_mask))
            print("[INFO] Handle detected: area_ratio=%.1f%%" % (
                handle_area / body_area * 100))
            return handle_mask, handle_bbox, body_mask, full_mask, "success", bbox

        print("[WARN] No handle found (scanline detection failed)")
        return None, None, full_mask, full_mask, "none", bbox

    # ------------------------------------------------------------
    def _run_single_inference(self, color_image, depth_frame):
        """
        执行单次推理, 返回结果字典供显示使用.

        返回:
            dict: {
                "body_mask", "handle_mask", "handle_bbox",
                "full_mask", "stage", "yolo_bbox",
                "yolo_time_ms", "sam_time_ms"
            }
        """
        handle_mask, handle_bbox, body_mask, full_mask, stage, yolo_bbox = \
            self.detect_handle(color_image, depth_frame)

        return {
            "body_mask": body_mask,
            "handle_mask": handle_mask,
            "handle_bbox": handle_bbox,
            "full_mask": full_mask,
            "stage": stage,
            "yolo_bbox": yolo_bbox,
            "yolo_time_ms": self._yolo_time_ms,
            "sam_time_ms": self._sam_time_ms,
        }

    # ------------------------------------------------------------
    def _draw_live_info(self, display_img):
        """在实时模式下绘制提示信息."""
        cv2.putText(display_img, "[LIVE] Press 's' to run inference",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(display_img, "Press 'q' to quit",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ------------------------------------------------------------
    def _draw_inferring_info(self, display_img):
        """在推理模式下显示进度."""
        elapsed = time.perf_counter() - self._infer_start_time
        cv2.putText(display_img, "[INFERRING] %.1fs" % elapsed,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(display_img, "YOLO -> SAM -> scanline...",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # ------------------------------------------------------------
    def _draw_result(self, display_img, result):
        """绘制推理结果 (mask 叠加 + bbox + 统计)."""
        body_mask = result["body_mask"]
        handle_mask = result["handle_mask"]
        handle_bbox = result["handle_bbox"]
        stage = result["stage"]
        yolo_bbox = result["yolo_bbox"]
        yolo_time_ms = result["yolo_time_ms"]
        sam_time_ms = result["sam_time_ms"]

        # ---- 先绘制微调 YOLO 原始检测框 (紫色) ----
        if yolo_bbox is not None:
            x1, y1, x2, y2 = [int(v) for v in yolo_bbox]
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.putText(display_img, "YOLO bbox", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

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

            # 额外绘制旋转矩形轮廓
            if handle_mask is not None:
                mask_uint8 = handle_mask.astype(np.uint8) * 255
                ctrs, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
                if ctrs:
                    big_ctr = max(ctrs, key=cv2.contourArea)
                    rect = cv2.minAreaRect(big_ctr)
                    box_pts = cv2.boxPoints(rect)
                    box_pts = box_pts.reshape((-1, 1, 2)).astype(np.int32)
                    cv2.drawContours(display_img, [box_pts], 0, (0, 255, 255), 2)

        # ---- 底部提示信息 ----
        if stage == "success":
            color = (0, 255, 0)
            text = "Handle OK"
        else:
            color = (0, 0, 255)
            text = "No handle"

        cv2.putText(display_img, text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(display_img, "YOLO: %dms  SAM: %dms" % (
            int(yolo_time_ms), int(sam_time_ms)),
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(display_img, "[RESULT] 'v'-save mask | 's'-rerun | 'c'-clear | 'q'-quit",
                    (10, display_img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # ------------------------------------------------------------
    def _save_result(self, display_img, result):
        """
        保存推理结果到 images/ 目录:
          水壶完整 mask: frame_YYYYMMDD_HHMMSS_mask.png (二值图, uint8)
          不再保存同名 .txt 文件.

        参数:
            display_img : np.ndarray, 当前显示图像 (仅用于日志, 不保存显示图)
            result      : dict, 推理结果 (_run_single_inference 返回的字典)
        """
        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        # 生成唯一序号 (避免同一秒内覆盖)
        _seq = 0
        while True:
            seq_str = "_%03d" % _seq
            filename = "frame_%s%s_mask.png" % (timestamp, seq_str)
            save_path = os.path.join(_save_dir, filename)
            if not os.path.exists(save_path):
                break
            _seq += 1

        # ---- 保存水壶 mask (full_mask) 为 8-bit PNG ----
        full_mask = result.get("full_mask")
        if full_mask is not None:
            mask_uint8 = full_mask.astype(np.uint8) * 255
            cv2.imwrite(save_path, mask_uint8)
            print("[INFO] Saved kettle mask: %s" % save_path)
        else:
            print("[WARN] No full_mask to save")

    # ------------------------------------------------------------
    def run(self):
        """
        主循环: 默认实时显示, 按 's' 触发单次推理, 显示结果.

        显示模式转换:
          MODE_LIVE  --按's'-->  MODE_INFERRING --推理完成-->  MODE_RESULT
             ^                                                     |
             +------------------按's'再跑一次-----------------------+

        按键:
          's' : 触发推理 (从 LIVE -> INFERRING, 或从 RESULT -> INFERRING 重新跑)
          'v' : 保存当前水壶 mask 到 images/ 目录 (仅 RESULT 模式有效)
          'c' : 清除结果, 回到 LIVE 模式
          'q' : 退出
        """
        _save_dir = os.path.join(os.path.dirname(__file__), "images")
        os.makedirs(_save_dir, exist_ok=True)

        print("=" * 50)
        print("RealSense SAM Handle Detection (手动触发模式)")
        print("=" * 50)
        print("检测模型: 微调 YOLO + SAM (GPU)")
        print("把手侧: %s, 把手面积比: [%.0f%%, %.0f%%]" % (
            self.HANDLE_SIDE, self.HANDLE_MIN_RATIO * 100, self.HANDLE_MAX_RATIO * 100))
        print("ROI 左右拓展: %.0f%%" % (self.ROI_EXPAND_RATIO * 100))
        print("")
        print("操作说明:")
        print("  [LIVE模式]   按 's' -> 执行 YOLO+SAM+扫描线 推理")
        print("  [RESULT模式] 按 'v' -> 保存水壶 mask (到 images/ 目录)")
        print("  [RESULT模式] 按 's' -> 重新推理 (冻结新帧)")
        print("  任意模式按    'q' -> 退出")
        print("  按 'c' -> 清除结果, 回到纯实时显示")
        print("")

        # ---- 缓存: 冻结帧 (推理时锁定画面, 不随相机更新) ----
        _frozen_color = None
        _frozen_depth_frame = None

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- 根据当前模式决定显示内容 ----
                if self._display_mode == self.MODE_LIVE:
                    # 实时显示: 不推理, 只显示提示
                    display_img = color_image.copy()
                    self._draw_live_info(display_img)

                elif self._display_mode == self.MODE_INFERRING:
                    # 推理中: 显示冻结帧 + "推理中" 提示
                    if _frozen_color is not None:
                        display_img = _frozen_color.copy()
                    else:
                        display_img = color_image.copy()
                    self._draw_inferring_info(display_img)

                    # ---- 执行单次推理 ----
                    if _frozen_color is not None and _frozen_depth_frame is not None:
                        result = self._run_single_inference(
                            _frozen_color, _frozen_depth_frame
                        )
                        self._result_cache = result

                        # 有结果后立即切换到 RESULT 模式
                        self._display_mode = self.MODE_RESULT
                        # 重绘当前帧 (用推理结果覆盖)
                        display_img = _frozen_color.copy()
                        self._draw_result(display_img, result)

                elif self._display_mode == self.MODE_RESULT:
                    # 显示结果: 冻结帧 + 推理结果覆盖
                    if _frozen_color is not None:
                        display_img = _frozen_color.copy()
                    else:
                        display_img = color_image.copy()

                    if self._result_cache is not None:
                        self._draw_result(display_img, self._result_cache)

                # ---- 按键处理 ----
                cv2.imshow('SAM Handle Detection (Finetuned YOLO + SAM)', display_img)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('s'):
                    # 触发推理: 冻结当前帧, 切换到推理模式
                    _frozen_color = color_image.copy()
                    _frozen_depth_frame = depth_frame
                    self._infer_start_time = time.perf_counter()
                    self._display_mode = self.MODE_INFERRING

                elif key == ord('v') and self._display_mode == self.MODE_RESULT:
                    # 保存水壶 mask
                    if self._result_cache is not None:
                        self._save_result(display_img, self._result_cache)

                elif key == ord('c'):
                    # 清除结果, 回到实时模式
                    self._display_mode = self.MODE_LIVE
                    self._result_cache = None
                    _frozen_color = None
                    _frozen_depth_frame = None
                    print("[INFO] Cleared result, back to live mode")

        finally:
            self.stop()
            print("[INFO] Program exited")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseHandleDetector(config_dict=_MODULE_CFG)
    detector.run()