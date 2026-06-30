# -*- coding: utf-8 -*-
"""
EyeToHand 共享工具函数
=======================
提供标定数据加载、相机系→基系坐标转换等工具.
不包含检测器类 — 检测器见 person_detector.py (直接继承 BaseRealSenseDetector).
"""

import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml


# ============================================================
#  标定工具
# ============================================================

def load_calibration(calib_path=None):
    """
    加载 2 点 1D 线性映射标定结果.

    参数:
        calib_path : str 或 None
            标定文件路径, 为 None 时使用 EyeToHand/calibration/result.yaml

    返回:
        dict: {"scale": float, "bias": float}
    """
    if calib_path is None:
        calib_path = os.path.join(
            os.path.dirname(__file__), "calibration", "result.yaml"
        )

    if not os.path.exists(calib_path):
        print("[WARN] 标定文件不存在: %s, 使用默认 scale=1.0, bias=0.0" % calib_path)
        return {"scale": 1.0, "bias": 0.0}

    with open(calib_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return {
        "scale": float(data["calibration"]["scale"]),
        "bias": float(data["calibration"]["bias"]),
    }


def camera_x_to_base_x(x_cam, calib=None):
    """
    相机系 X 坐标 → 基系 X 坐标 (1D 线性映射).

    参数:
        x_cam : float, 相机系 X 坐标 (米)
        calib : dict 或 None, {"scale", "bias"}

    返回:
        float, 基系 X 坐标 (米)
    """
    if calib is None:
        calib = load_calibration()
    return calib["scale"] * x_cam + calib["bias"]
