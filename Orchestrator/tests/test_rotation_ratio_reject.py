# -*- coding: utf-8 -*-
"""旋转判档 ratio 上界 REJECT (B2 2026-08-20).

检测器在反解前识别非物理/未标定 ratio (正侧越界) → REJECT (band=None,
theta=None). 检测器不输出 θ → data_sm 稳定窗口不加帧 → 走 B3 缺样本
REJECT 链, 不以 +30 代替. 阈值 0.34 基于标定 json (2026-08-18, 39 点):
合法档位 ratio 最大 0.3177 (+30), 对合法档位 0 误杀; >0.34 只有被 YOLO
框裁抬高的 +45 高尾. u 不作门限 (u 是摆位量, 与 +30/+45 区间重叠, 不可分).

不触相机: HandleCupDetector.__new__ 跳过 __init__.
"""
import unittest

from Orchestrator.handle_cup_detector import HandleCupDetector


def make_rot_detector(ratio_max=0.34):
    det = HandleCupDetector.__new__(HandleCupDetector)
    det._rot_enabled = True
    # 全局模型系数 (config.yaml handle_rotation.global_model)
    det._rot_c0, det._rot_c1, det._rot_c2, det._rot_c3 = 0.1071, 0.1739, 0.0988, -0.0768
    det._rot_dref = 0.63
    det._rot_band0 = 8.0
    det._rot_band15_hi = 22.0
    det._rot_ratio_max = float(ratio_max)
    return det


class RotationRatioRejectTest(unittest.TestCase):
    # 用 d = d_ref (Δd=0), 隔离 ratio 上界检查本身
    D = 0.63

    def test_ratio_above_max_rejected(self):
        det = make_rot_detector()
        band, theta = det._classify_rotation(0.35, 600, self.D)
        self.assertIsNone(band)
        self.assertIsNone(theta)

    def test_cropped_45_tail_rejected(self):
        # 标定 json: +45 被 YOLO 框裁抬高的高尾 0.3415/0.3447/0.3556/0.3598 → REJECT
        det = make_rot_detector()
        for r in (0.3415, 0.3447, 0.3556, 0.3598):
            band, theta = det._classify_rotation(r, 600, self.D)
            self.assertIsNone(band, f"裁切虚高 ratio={r} 应 REJECT")
            self.assertIsNone(theta)

    def test_legal_bands_not_rejected(self):
        # 标定实测各合法档位 ratio 均远低于 0.34, 必须不误杀:
        #   0.2040(-30) 0.2584(-15) 0.2782(0) 0.2997(+15) 0.3101/+0.3177(+30)
        det = make_rot_detector()
        for r in (0.2040, 0.2584, 0.2782, 0.2997, 0.3101, 0.3177):
            band, _ = det._classify_rotation(r, 600, self.D)
            self.assertIsNotNone(band, f"合法档位 ratio={r} 不应 REJECT")

    def test_ratio_at_max_boundary_allowed(self):
        # 0.34 = 边界值本身不 REJECT (留余量); 反解 x>1 钳位 → θ=φ → +30
        det = make_rot_detector()
        band, theta = det._classify_rotation(0.34, 600, self.D)
        self.assertEqual(band, 30)
        self.assertIsNotNone(theta)

    def test_ratio_max_configurable(self):
        # 阈值可配置: 更紧的 0.32 也工作 (仍 > +30 最大 0.3177)
        det = make_rot_detector(ratio_max=0.32)
        band, _ = det._classify_rotation(0.35, 600, self.D)
        self.assertIsNone(band)
        band, _ = det._classify_rotation(0.3177, 600, self.D)
        self.assertIsNotNone(band)

    def test_low_ratio_unreachable_failclosed(self):
        # 负侧极低 ratio → 反解 <-40° 不可达 → band=None (既有 fail-closed 保持)
        det = make_rot_detector()
        band, _ = det._classify_rotation(0.10, 600, self.D)
        self.assertIsNone(band)

    def test_rotation_disabled_no_band(self):
        # 旋转禁用时本就无 band (与 B2 无关, 确认不因新检查改变)
        det = make_rot_detector()
        det._rot_enabled = False
        band, _ = det._classify_rotation(0.50, 600, self.D)
        self.assertIsNone(band)


if __name__ == "__main__":
    unittest.main()
