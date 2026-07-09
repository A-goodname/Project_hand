# -*- coding: utf-8 -*-
"""
EyeToHand 外置相机标定工具 — 3D→1D 人体标定
===============================================
人站在桌边不同位置 (左/中/右), Camera B 检测 person_3d_cam,
机械臂手动示教 Cup P3 递送 X_base, 拟合 3D→1D 线性映射.

原理:
  递送位姿 = (X_person, Y_TEACH, Z_TEACH, A_TEACH, B_TEACH, C_TEACH)
              ↑ 相机检测拟合      ↑ 示教固定 (config.yaml delivery)
  YZ/ABC 是固定值, 只需标定 person_3d_cam → X_base.

拟合模型 (3D 线性):
  X_base = a·x_cam + b·y_cam + c·z_cam + d

  其中 (x_cam, y_cam, z_cam) 为 PersonDetector 检测的人体 bbox 中心 3D 坐标.

采集:
  1. 人站在位置 A (如桌子左边)
  2. 机械臂示教 Cup P3 递送位姿, 记录 X_base
  3. 按 'c' → 记录当前 person_3d_cam
  4. 终端输入 X_base → 自动配对
  5. 换位置 B, C... 重复 (建议 ≥3 点)
  6. 按 'f' 拟合  按 's' 保存  按 'q' 退出

输出:
  EyeToHand/calibration/result.yaml  →  {a, b, c, d}

使用方法:
  python EyeToHand/calibration/calibrate.py
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

from EyeToHand.person_detector import PersonDetector

# ---- 加载 EyeToHand 配置 ----
_cfg_path = os.path.join(_eye_to_hand_dir, "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as _f:
    _MODULE_CFG = yaml.safe_load(_f)


# ============================================================
#  PersonCalibrationDetector
# ============================================================

class PersonCalibrationDetector(PersonDetector):
    """
    标定用检测器: 继承 PersonDetector, 复用人体检测 + 稳定器.

    在 PersonDetector 基础上增加:
      - 记录 person_3d_cam → X_base 采集
      - 3D 线性最小二乘拟合
      - 保存标定结果
    """

    def run(self):
        """
        交互式标定主循环.

        操作:
          'c' → 记录当前 person_3d_cam + 终端输入 X_base
          'f' → 3D 线性拟合 (需 ≥3 点, 4 参数)
          's' → 保存到 result.yaml
          'q' → 退出
        """
        points = []          # [(X_base, x_cam, y_cam, z_cam), ...]
        coeffs = None        # (a, b, c, d)

        print("=" * 60)
        print("  EyeToHand 相机标定 — 3D→1D 人体标定")
        print("  person_3d_cam → X_base (Cup P3 递送)")
        print()
        print("  递送位姿 = (X_person, Y_TEACH, Z_TEACH, A_TEACH, B_TEACH, C_TEACH)")
        print("              ↑ 相机拟合          ↑ 示教固定")
        print()
        print("  拟合模型: X_base = a·x + b·y + c·z + d")
        print()
        print("  操作:")
        print("    1. 人站在桌边位置 (左/中/右)")
        print("    2. 机械臂示教 Cup P3 递送, 记录 X_base")
        print("    3. 按 'c' → 终端输入 X_base")
        print("    4. 换位置重复 ≥3 次 (建议 3-5 点, 范围尽量大)")
        print("    5. 按 'f' 拟合  按 's' 保存  按 'q' 退出")
        print("=" * 60)

        try:
            while True:
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None:
                    continue

                # ---- 人体检测 (复用 PersonDetector API) ----
                stable_result = self.detect_person_stable(color_image, depth_frame)
                raw_result = self.detect_person(color_image, depth_frame)

                # ---- 可视化 ----
                display_img = color_image.copy()

                # 人体 bbox + 稳定位置
                if raw_result is not None:
                    bbox = raw_result["bbox"]
                    x1, y1, x2, y2 = [int(v) for v in bbox]
                    bbox_color = (0, 255, 0) if stable_result else (0, 255, 255)
                    cv2.rectangle(display_img, (x1, y1), (x2, y2), bbox_color, 2)
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    cv2.drawMarker(display_img, (cx, cy), bbox_color,
                                   cv2.MARKER_CROSS, 15, 1)

                # 当前稳定位置 (用于采集)
                person_3d = stable_result["pos_3d_cam"] if stable_result else None

                # 左上角状态
                stability = "STABLE" if stable_result else "unstable"
                cv2.putText(display_img, "points: %d (min 3)  [%s]" % (len(points), stability),
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                if person_3d is not None:
                    cv2.putText(display_img,
                                "person: x=%.3f y=%.3f z=%.3f" % person_3d,
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                elif raw_result is not None:
                    pos = raw_result["pos_3d_cam"]
                    if pos is not None:
                        cv2.putText(display_img,
                                    "person(raw): x=%.3f y=%.3f z=%.3f" % pos,
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                    else:
                        cv2.putText(display_img, "person: detected (no 3D)",
                                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 2)
                else:
                    cv2.putText(display_img, "person: not found",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                # 已采集点列表
                for i, (xb, xc, yc, zc) in enumerate(points):
                    cv2.putText(display_img,
                                "[%d] X_base=%.4f  person=(%.3f,%.3f,%.3f)" % (
                                    i+1, xb, xc, yc, zc),
                                (10, 85 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

                # 拟合结果
                y_offset = 85 + len(points) * 20 + 15
                if coeffs is not None:
                    a, b, c, d = coeffs
                    cv2.putText(display_img,
                                "fit: a=%.4f b=%.4f c=%.4f d=%.4f" % (a, b, c, d),
                                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

                # 显示
                if depth_colormap is not None and display_img.shape == depth_colormap.shape:
                    images = np.hstack((display_img, depth_colormap))
                else:
                    images = display_img
                cv2.imshow("EyeToHand Person Calibration", images)

                key = cv2.waitKey(25) & 0xFF

                if key == ord("q"):
                    break

                elif key == ord("c"):
                    if stable_result is None:
                        print("[WARN] 人体位置未稳定, 请等人站稳后再采集 (绿色=稳定)")
                    else:
                        person_3d = stable_result["pos_3d_cam"]
                        try:
                            val_str = input("  输入 Cup P3 X_base (米): ").strip()
                            x_base = float(val_str)
                            points.append((x_base,
                                           float(person_3d[0]),
                                           float(person_3d[1]),
                                           float(person_3d[2])))
                            print("  [CAPTURE] #%d: X_base=%.4f  person=(%.3f,%.3f,%.3f)" % (
                                len(points), x_base,
                                person_3d[0], person_3d[1], person_3d[2]))
                        except ValueError:
                            print("  [WARN] 输入格式错误, 请输入数字 (如 0.45)")

                elif key == ord("f"):
                    if len(points) < 3:
                        print("[WARN] 至少需要 3 个采集点 (当前 %d 个)" % len(points))
                    else:
                        # ---- 3D 线性最小二乘 ----
                        # X_base = a·x + b·y + c·z + d
                        # 构造: A @ [a,b,c,d]^T = X_base_arr
                        n = len(points)
                        A = np.zeros((n, 4))
                        for i, (xb, xc, yc, zc) in enumerate(points):
                            A[i, 0] = xc   # x_cam
                            A[i, 1] = yc   # y_cam
                            A[i, 2] = zc   # z_cam
                            A[i, 3] = 1.0  # bias d
                        xb_arr = np.array([p[0] for p in points])

                        # SVD 求解: min ||A·coeff - xb||
                        coeff_vec, residuals, rank, sv = np.linalg.lstsq(A, xb_arr, rcond=None)
                        a, b, c, d = float(coeff_vec[0]), float(coeff_vec[1]), float(coeff_vec[2]), float(coeff_vec[3])
                        coeffs = (a, b, c, d)

                        # 残差统计
                        preds = A @ coeff_vec
                        errors = np.abs(preds - xb_arr)

                        print()
                        print("[FIT]  3D 线性最小二乘 (N=%d 点)" % n)
                        print("[FIT]  a = %.6f  (x_cam 系数)" % a)
                        print("[FIT]  b = %.6f  (y_cam 系数)" % b)
                        print("[FIT]  c = %.6f  (z_cam 系数)" % c)
                        print("[FIT]  d = %.6f  (偏置)" % d)
                        print("[FIT]  公式: X_base = %.4f·x + %.4f·y + %.4f·z + (%.4f)" % (a, b, c, d))
                        print("[FIT]  逐点误差:")
                        for i in range(n):
                            print("[FIT]    点%d: pred=%.4f  actual=%.4f  err=%.1fmm" % (
                                i+1, preds[i], xb_arr[i], errors[i]*1000))
                        print("[FIT]  max_err=%.1fmm  mean_err=%.1fmm" % (
                            errors.max()*1000, errors.mean()*1000))

                elif key == ord("s"):
                    if coeffs is None:
                        print("[WARN] 请先拟合 ('f' 键) 再保存")
                    else:
                        self._save_result(coeffs)

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
    def _save_result(coeffs):
        """保存标定结果到 YAML 文件."""
        a, b, c, d = coeffs
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.yaml")

        data = {
            "calibration": {
                "a": float(a),
                "b": float(b),
                "c": float(c),
                "d": float(d),
                "formula": "X_base = a*x_cam + b*y_cam + c*z_cam + d",
            }
        }

        with open(result_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

        print("[SAVE] 标定结果已写入: %s" % result_path)
        print("[SAVE] 公式: X_base = %.6f*x + %.6f*y + %.6f*z + (%.6f)" % (a, b, c, d))


# ============================================================
#  主入口
# ============================================================

if __name__ == "__main__":
    detector = PersonCalibrationDetector(config_dict=_MODULE_CFG)
    detector.run()
