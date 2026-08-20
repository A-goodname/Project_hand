# -*- coding: utf-8 -*-
"""旋转 θ 并入 PoseStabilizer 稳定窗口 (2026-08-19).

覆盖:
  1. θ 模式: 同一窗口内 pos/euler/θ 一致 → 稳定, get_mean 返回窗口 θ 中位数
  2. θ 极差超阈值 (分歧) → 不稳定 (B3 REJECT 语义)
  3. 窗口不满 N 帧 → 不稳定 (B3 恰满窗语义)
  4. 向后兼容: 无 θ (cup 用途) → 仅 pos/angle 稳定, get_mean 返回 (pos, euler, None)
  5. 代际失效: on_disconnect 后作废 (B3 同代累积)
  6. 滑动窗口: 超窗 pop 最旧, θ 中位数只基于最新窗口帧
"""
import unittest

import numpy as np

from Orchestrator.estun_handle_data_sm import PoseStabilizer, ROT_THETA_SPREAD_MAX


class PoseStabilizerThetaTest(unittest.TestCase):
    def test_theta_stable_consistent(self):
        """同一窗口 pos/euler/θ 全一致 → 稳定; 中位数 = 输入值."""
        st = PoseStabilizer(window=5, pos_thresh=0.01, angle_thresh=5.0)
        for _ in range(5):
            st.add((0.1, 0.2, 0.3), (0, 0, 0), 16.2)
        self.assertTrue(st.is_stable())
        mean_pos, mean_euler, th_med = st.get_mean()
        np.testing.assert_allclose(mean_pos, (0.1, 0.2, 0.3), atol=1e-6)
        np.testing.assert_allclose(mean_euler, (0, 0, 0), atol=1e-6)
        self.assertAlmostEqual(th_med, 16.2)

    def test_theta_median_of_window(self):
        """θ 中位数 = 窗口内中位数 (非均值)."""
        st = PoseStabilizer(window=5, pos_thresh=0.01, angle_thresh=5.0)
        for th in (16.0, 16.2, 16.4, 16.1, 16.3):
            st.add((0.1, 0.2, 0.3), (0, 0, 0), th)
        self.assertTrue(st.is_stable())
        _, _, th_med = st.get_mean()
        self.assertAlmostEqual(th_med, 16.2)  # sorted: 16.0,16.1,16.2,16.3,16.4

    def test_theta_spread_rejects(self):
        """窗口内 θ 极差超阈值 → 不稳定 (B3 分歧 REJECT)."""
        st = PoseStabilizer(window=5, pos_thresh=0.01, angle_thresh=5.0)
        for th in (10.0, 10.2, 10.1, 17.0, 10.3):  # 极差 7.0 > 5.0
            st.add((0.1, 0.2, 0.3), (0, 0, 0), th)
        self.assertFalse(st.is_stable())

    def test_window_insufficient(self):
        """窗口不满 5 帧 → 不稳定 (B3 恰满窗语义)."""
        st = PoseStabilizer(window=5, pos_thresh=0.01, angle_thresh=5.0)
        for _ in range(4):
            st.add((0.1, 0.2, 0.3), (0, 0, 0), 0.0)
        self.assertFalse(st.is_stable())

    def test_legacy_no_theta(self):
        """无 θ (cup 用途) → 仅 pos/angle 稳定; get_mean 返回 (pos, euler, None)."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        for _ in range(3):
            st.add((0.1, 0.2, 0.3), (0, 0, 0))
        self.assertTrue(st.is_stable())
        mean_pos, mean_euler, th_med = st.get_mean()
        np.testing.assert_allclose(mean_pos, (0.1, 0.2, 0.3), atol=1e-6)
        np.testing.assert_allclose(mean_euler, (0, 0, 0), atol=1e-6)
        self.assertIsNone(th_med)

    def test_legacy_pos_spread_rejects(self):
        """无 θ: 位置极差超阈 → 不稳定."""
        st = PoseStabilizer(window=3, pos_thresh=0.005, angle_thresh=5.0)
        for p in ((0.1, 0.2, 0.3), (0.1, 0.2, 0.3), (0.12, 0.2, 0.3)):  # 极差 0.02
            st.add(p, (0, 0, 0))
        self.assertFalse(st.is_stable())

    def test_generation_invalid(self):
        """会话重连 (on_disconnect) → 代际失效."""
        st = PoseStabilizer(window=3)
        st.capture_generation()
        self.assertTrue(st.is_generation_valid())
        PoseStabilizer.on_disconnect()
        self.assertFalse(st.is_generation_valid())

    def test_sliding_window(self):
        """超窗 pop 最旧: θ 中位数只基于最新窗口帧."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        # 填满 3 帧, 再 push 1 帧挤出最旧 (θ=0)
        for th in (0.0, 16.0, 16.2):
            st.add((0.1, 0.2, 0.3), (0, 0, 0), th)
        st.add((0.1, 0.2, 0.3), (0, 0, 0), 16.4)
        self.assertTrue(st.is_stable())
        _, _, th_med = st.get_mean()
        self.assertAlmostEqual(th_med, 16.2)  # [16.0, 16.2, 16.4] 中位数

    def test_theta_thresh_override(self):
        """is_stable(theta_thresh=...) 可覆盖实例阈值."""
        st = PoseStabilizer(window=5, pos_thresh=0.01, angle_thresh=5.0,
                            theta_thresh=ROT_THETA_SPREAD_MAX)
        for th in (10.0, 10.2, 10.1, 10.3, 10.4):  # 极差 0.4
            st.add((0.1, 0.2, 0.3), (0, 0, 0), th)
        self.assertTrue(st.is_stable())
        self.assertTrue(st.is_stable(theta_thresh=1.0))
        self.assertFalse(st.is_stable(theta_thresh=0.2))  # 0.4 > 0.2


if __name__ == "__main__":
    unittest.main()
