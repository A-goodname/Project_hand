# -*- coding: utf-8 -*-
"""
EyeToHand 外置相机标定工具 — 1D最小二乘线性回归
===================================================
将红色胶带贴在机械臂末端, 机械臂末端保持 YZ 和 ABC 不变,
仅沿基座 X 方向移动到 N 个位置 (N ≥ 2),
终端输入示教器读取的 X_base 值, 相机检测胶带中心 X_cam,
最小二乘拟合 scale+bias.

前置要求:
  - 递送位姿的 YZ 和 ABC 由示教获得 (写入 config.yaml delivery 段)
  - 标定时末端 YZ/ABC 使用示教值, 全程保持不变
  - 仅 X_base 在校准过程中变化

为什么只需要 X:
  递送位姿 = (X_person, Y_TEACH, Z_TEACH, A_TEACH, B_TEACH, C_TEACH)
            ↑                      ↑
        外置相机人体检测获得      示教固定值
  YZ/ABC 是预先示教的安全递送姿态, 标定只需解决相机 X → 基座 X 的映射.

原理:
  外置相机固定安装, 其 X 轴与基座 X 轴通常不平行.
  标定计算相机 X_cam → 基座 X_base 的 1D 线性映射:
    X_base = scale × X_cam + bias

采集:
  N 次: 机械臂沿 X 移到任意位置 → 按 'c' → 终端输入当前 X_base → 记录 (X_base, X_cam)

拟合 (最小二乘):
  对 N 个点 (x_base_i, x_cam_i) 做线性回归:
    scale = Σ(xc_i - x̄c)(xb_i - x̄b) / Σ(xc_i - x̄c)²
    bias  = x̄b - scale × x̄c

输出:
  EyeToHand/calibration/result.yaml

使用方法:
  python EyeToHand/calibration/calibrate.py
  'c' 采集  'f' 拟合  's' 保存  'q' 退出
"""

import sys
import os

_calib_dir = os.path.dirname(os.path.abspath(__file__))
_eye_to_hand_dir = os.path.dirname(_calib_dir)
_project_root = os.path.dirname(os.path.dirname(_eye_to_hand_dir))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml
import cv2
import numpy as np

from RealSense.realsense_shared import BaseRealSenseDetector, detect_red_tape

# ---- 加载 EyeToHand 配置 ----
_cfg_path = os.path.join(_eye_to_hand_dir, "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  CalibrationDetector
# ============================================================

class CalibrationDetector(BaseRealSenseDetector):
    """
    标定用检测器: 检测红色胶带中心 3D 位置 (相机系).

    继承 BaseRealSenseDetector, 与 test_3D.py 模式一致.
    不依赖 YOLO 检测, 仅使用红色胶带 HSV 检测.
    """

    USE_YOLO_WORLD = False
    USE_SAM = False
    LOG_NAME = "calibrate.log"
    TARGET_CLASS_ID = 39           # 占位, 不使用 YOLO

    def __init__(self, config_dict=None):
        super().__init__(config_dict=config_dict)

    # ------------------------------------------------------------
    def detect_tape_3d(self, color_image, depth_frame):
        """
        检测红色胶带中心, 返回相机系 3D 坐标.

        返回:
            (x_cam, y_cam, z_cam) 或 None
        """
        _, center, area = detect_red_tape(color_image)
        if center is None or area < 10:
            return None
        cx, cy = center
        return self._color_pixel_to_3d(cx, cy, depth_frame)

    # ------------------------------------------------------------
    def run(self):
        """
        交互式标定主循环.

        前置: 机械臂末端 YZ/ABC 保持不变, 仅 X 方向移动.

        键盘:
          'c' → 终端输入当前 X_base, 采集 (X_base, X_cam)
          'f' → 最小二乘拟合 scale + bias (需 ≥2 点)
          's' → 保存到 EyeToHand/calibration/result.yaml
          'q' → 退出
        """
        points = []                 # [(X_base, X_cam), ...]
        scale = None
        bias = None

        print("=" * 60)
        print("  EyeToHand 相机标定 — 1D 最小二乘线性回归")
        print("  外置相机 X_cam → 基座 X_base")
        print()
        print("  递送位姿 = (X_person, Y_TEACH, Z_TEACH, A_TEACH, B_TEACH, C_TEACH)")
        print("             ↑ 人体检测得到    ↑ 示教固定 (config.yaml delivery)")
        print()
        print("  因此标定只需解决 1D: X_cam → X_base")
        print()
        print("  !! 末端 YZ/ABC = 示教递送值, 全程保持不变 !!")
        print("  !! 仅沿基座 X 方向移动                    !!")
        print()
        print("  操作:")
        print("    1. 末端贴红色胶带, 使用示教递送 YZ/ABC, 沿 X 平移")
        print("    2. 按 'c' → 终端输入示教器显示的 X_base")
        print("    3. 重复 2+ 次 (建议 3-5 点, 范围尽量大)")
        print("    4. 按 'f' 最小二乘拟合  按 's' 保存  按 'q' 退出")
        print("=" * 60)

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                # ---- 检测红色胶带 ----
                tape_3d = self.detect_tape_3d(color_image, depth_frame)

                # ---- 可视化 ----
                display_img = color_image.copy()

                # 左上角状态
                cv2.putText(display_img, "points: %d  (min 2 for fit)" % len(points),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if tape_3d is not None:
                    _, center, _ = detect_red_tape(color_image)
                    if center is not None:
                        cv2.drawMarker(display_img, center, (0, 255, 0),
                                       cv2.MARKER_CROSS, 20, 2)
                    cv2.putText(display_img, "tape X_cam=%.4f m" % tape_3d[0],
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                else:
                    cv2.putText(display_img, "tape: not found",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # 已采集点列表
                for i, (xb, xc) in enumerate(points):
                    cv2.putText(display_img, "[%d] X_base=%.4f  X_cam=%.4f" % (i + 1, xb, xc),
                                (10, 80 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)

                # 拟合结果
                y_offset = 80 + len(points) * 20 + 10
                if scale is not None and bias is not None:
                    cv2.putText(display_img, "fit: scale=%.4f  bias=%.4f" % (scale, bias),
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 并排显示
                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img
                cv2.imshow("EyeToHand Calibration", images)

                key = cv2.waitKey(25) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    if tape_3d is None:
                        print("[WARN] 未检测到红色胶带, 请确认胶带在视野内")
                    else:
                        try:
                            val_str = input("  输入当前 X_base (米): ").strip()
                            x_base = float(val_str)
                            x_cam = tape_3d[0]
                            points.append((x_base, x_cam))
                            print("  [CAPTURE] #%d: X_base=%.4f  X_cam=%.4f" % (
                                len(points), x_base, x_cam))
                        except ValueError:
                            print("  [WARN] 输入格式错误, 请输入数字 (如 0.35)")

                elif key == ord("f"):
                    if len(points) < 2:
                        print("[WARN] 至少需要 2 个采集点 (当前 %d 个)" % len(points))
                    else:
                        # ---- 最小二乘线性回归 ----
                        xb_arr = np.array([p[0] for p in points])
                        xc_arr = np.array([p[1] for p in points])

                        xc_mean = xc_arr.mean()
                        xb_mean = xb_arr.mean()
                        scale = np.sum((xc_arr - xc_mean) * (xb_arr - xb_mean)) / np.sum((xc_arr - xc_mean) ** 2)
                        bias = xb_mean - scale * xc_mean

                        # 残差统计
                        preds = scale * xc_arr + bias
                        errors = np.abs(preds - xb_arr)

                        print()
                        print("[FIT]  最小二乘 (N=%d 点)" % len(points))
                        print("[FIT]  scale = %.6f" % scale)
                        print("[FIT]  bias  = %.6f" % bias)
                        print("[FIT]  公式: X_base = %.6f * X_cam + (%.6f)" % (scale, bias))
                        print("[FIT]  逐点误差:")
                        for i in range(len(points)):
                            print("[FIT]    点%d: pred=%.4f  actual=%.4f  err=%.1fmm" % (
                                i + 1, preds[i], xb_arr[i], errors[i] * 1000))
                        print("[FIT]  max_err=%.1fmm  mean_err=%.1fmm" % (
                            errors.max() * 1000, errors.mean() * 1000))

                elif key == ord("s"):
                    if scale is None or bias is None:
                        print("[WARN] 请先拟合 ('f' 键) 再保存")
                    else:
                        self._save_result(scale, bias)

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断 (Ctrl+C)")
        except Exception as e:
            print("\n[ERROR] 运行异常: %s" % str(e))
        finally:
            self.stop()
            cv2.destroyAllWindows()
            print("[INFO] 程序已退出")

    # ------------------------------------------------------------
    @staticmethod
    def _save_result(scale, bias):
        """保存标定结果到 YAML 文件."""
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.yaml")

        data = {
            "calibration": {
                "scale": float(scale),
                "bias": float(bias),
            }
        }

        with open(result_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        print("[SAVE] 标定结果已写入: %s" % result_path)


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    detector = CalibrationDetector(config_dict=_MODULE_CFG)
    detector.run()
