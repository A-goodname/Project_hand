# -*- coding: utf-8 -*-
"""旋转判档 ratio 上界 REJECT + pos_side 运行分支 (B2 2026-08-20 + 对方 08-20 addendum).

- 2.1 ratio_max=0.36 真实语义: 严格 `ratio > 0.36` 才 REJECT; 旧裁切高尾 0.34~0.36
  与边界值 0.3600 均通过落 +30, 0.3601 才 REJECT — 物理限位是 +45 主约束,
  软件不挡 0.34~0.36 (不得把 0.3598 误记为软件 REJECT).
- 2.2 pos_side 运行分支实际覆盖: 从 config.yaml 加载完整 handle_rotation 配置
  (含 pos_side `_rot_nodes`/`rd_a`/`rd_b`), 验证 rd 判档表 (d=d_ref → rd=ratio).

不触相机: HandleCupDetector.__new__ 跳过 __init__, 配置从 config.yaml 显式加载.
"""
import unittest
from pathlib import Path

import yaml

from Orchestrator.handle_cup_detector import HandleCupDetector

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.yaml"


def load_rot_config():
    cfg = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    return cfg.get("handle_rotation", {})


def make_rot_detector(ratio_max=None):
    """按 config.yaml handle_rotation 完整初始化 (含 pos_side 运行分支字段)."""
    rot = load_rot_config()
    det = HandleCupDetector.__new__(HandleCupDetector)
    det._rot_enabled = bool(rot.get("enabled", True))
    gm = rot.get("global_model", {})
    det._rot_c0 = float(gm.get("c0", 0.0903))
    det._rot_c1 = float(gm.get("c1_cos", 0.2052))
    det._rot_c2 = float(gm.get("c2_sin", 0.1127))
    det._rot_c3 = float(gm.get("c3_dd", -0.0200))
    det._rot_dref = float(gm.get("d_ref", 0.63))
    det._rot_band0 = float(rot.get("band0_halfwidth", 8.0))
    det._rot_band15_hi = float(rot.get("band15_hi", 22.0))
    det._rot_ratio_max = float(rot.get("ratio_max", 0.36)
                               if ratio_max is None else ratio_max)
    # pos_side 运行分支 (2.2): 必须实际初始化, 否则走旧回退分支无法证明运行配置
    ps = rot.get("pos_side", {})
    if ps.get("enabled", False):
        det._rot_rd_a = float(ps["rd_a"])
        det._rot_rd_b = float(ps["rd_b"])
        det._rot_rd30 = float(ps["rd30"])
        det._rot_k = float(ps["k_slope"])
        det._rot_nodes = [(float(r), float(a)) for r, a in ps.get("nodes", [])]
    else:
        det._rot_rd_a = det._rot_rd_b = det._rot_rd30 = None
        det._rot_k = 50.0
        det._rot_nodes = []
    det._rot_g = None   # g_correction 已停用
    return det


class RatioMaxSemanticsTest(unittest.TestCase):
    """2.1 ratio_max=0.36 真实语义 (严格 >0.36 才 REJECT)."""
    D = 0.63   # d = d_ref → rd = ratio

    def test_cropped_45_tail_below_036_passes(self):
        # 旧裁切高尾 0.3388/0.3415/0.3447/0.3556/0.3598 均 <0.36 → 软件不 REJECT,
        # 落 +30 (物理限位是 +45 主约束). 对方 addendum 2.1 明确要求记录此语义.
        det = make_rot_detector()
        for r in (0.3388, 0.3415, 0.3447, 0.3556, 0.3598):
            band, _ = det._classify_rotation(r, 600, self.D)
            self.assertEqual(band, 30, f"ratio={r} <0.36 应落 +30, 不 REJECT")

    def test_boundary_03600_passes(self):
        # 边界值 0.3600: 严格 `>0.36` 才 REJECT, 0.3600 不 REJECT → 落 +30
        det = make_rot_detector()
        band, theta = det._classify_rotation(0.3600, 600, self.D)
        self.assertEqual(band, 30)
        self.assertIsNotNone(theta)

    def test_03601_rejected(self):
        det = make_rot_detector()
        band, theta = det._classify_rotation(0.3601, 600, self.D)
        self.assertIsNone(band)
        self.assertIsNone(theta)

    def test_legal_bands_not_rejected(self):
        # 标定实测合法档位 ratio 均 <0.34 (0.2040(-30) 0.2584(-15) 0.2782(0)
        # 0.2997(+15) 0.3101/0.3177(+30)) → 必须不误杀
        det = make_rot_detector()
        for r in (0.2040, 0.2584, 0.2782, 0.2997, 0.3101, 0.3177):
            band, _ = det._classify_rotation(r, 600, self.D)
            self.assertIsNotNone(band, f"合法档位 ratio={r} 不应 REJECT")

    def test_threshold_configurable(self):
        det = make_rot_detector(ratio_max=0.32)
        band, _ = det._classify_rotation(0.35, 600, self.D)
        self.assertIsNone(band)
        band, _ = det._classify_rotation(0.3177, 600, self.D)
        self.assertIsNotNone(band)


class PosSideBranchTest(unittest.TestCase):
    """2.2 pos_side 运行分支实际覆盖 (加载完整 config, d=d_ref → rd=ratio)."""
    D = 0.63

    def test_rd_table_bands(self):
        # 对方 addendum 2.2 给定 rd 表 (d=d_ref): 0.3090→0, 0.3094→+15,
        # 0.3177→+15, 0.3180→+30, 0.3601→None. 数值由配置推导 (非复制独立常量).
        det = make_rot_detector()
        cases = [
            (0.3090, 0,    "theta_raw ≤8 前 (0 档)"),
            (0.3094, 15,   "theta_raw >8 后 (rd<rd_a → +15)"),
            (0.3177, 15,   "最近档切换前 (rd_a<rd<rd_b, 连续θ<22.5 → +15)"),
            (0.3180, 30,   "最近档切换后 (rd_a<rd<rd_b, 连续θ≥22.5 → +30)"),
            (0.3601, None, "运行阈值 REJECT"),
        ]
        for rd, expect_band, why in cases:
            band, _ = det._classify_rotation(rd, 600, self.D)
            self.assertEqual(band, expect_band, f"rd={rd}: {why}")

    def test_pos_side_fields_initialized(self):
        # 确保测试确实走 pos_side 分支 (而非旧回退): _rot_nodes/rd_a/rd_b 已按 config 初始化
        det = make_rot_detector()
        self.assertTrue(det._rot_nodes)
        self.assertIsNotNone(det._rot_rd_a)
        self.assertIsNotNone(det._rot_rd_b)


class RotationRatioFailClosedTest(unittest.TestCase):
    """既有 fail-closed 语义保持."""
    D = 0.63

    def test_low_ratio_unreachable(self):
        det = make_rot_detector()
        band, _ = det._classify_rotation(0.10, 600, self.D)
        self.assertIsNone(band)   # 负侧反解 <-40° 不可达

    def test_rotation_disabled_no_band(self):
        det = make_rot_detector()
        det._rot_enabled = False
        band, _ = det._classify_rotation(0.50, 600, self.D)
        self.assertIsNone(band)


if __name__ == "__main__":
    unittest.main()
