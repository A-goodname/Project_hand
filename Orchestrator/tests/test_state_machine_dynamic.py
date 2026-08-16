# -*- coding: utf-8 -*-
"""#9 B2 动态抓取 (2026-08-16): PoseStabilizer 测地扩展 + compute_grasp_rotation dynamic.

覆盖:
  1. 四元数一致 → 稳定; 含 180° 翻转 → 不稳定 (PCA 符号歧义场景)
  2. get_mean_attitude: Markley 均值旋转正确
  3. continuity_report: 帧数/位置范围/测地/判定
  4. 旧调用 (无四元数) 行为不变
  5. compute_grasp_rotation("dynamic"): 跟随检测姿态 + Y 轴消歧
"""
import unittest

import numpy as np

from Orchestrator.state_machine import PoseStabilizer
from Orchestrator.rot_math import (
    quaternion_to_rotation_matrix,
    rotation_geodesic_deg,
    rotation_matrix_to_quaternion,
)
from Estun.utils_shared import compute_grasp_rotation, euler_to_rotation_matrix


class PoseStabilizerDynamicTest(unittest.TestCase):
    def test_quat_stability_consistent(self):
        """四元数一致 → 稳定."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        for _ in range(3):
            st.add((0.1, 0.2, 0.3), (0, 0, 0), (1.0, 0.0, 0.0, 0.0))
        self.assertTrue(st.is_stable())

    def test_quat_stability_flip_rejected(self):
        """窗口含 180° 翻转四元数 → 测地距离超阈 → 不稳定 (PCA 翻转场景)."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        st.add((0.1, 0.2, 0.3), (0, 0, 0), (1.0, 0.0, 0.0, 0.0))
        st.add((0.1, 0.2, 0.3), (0, 0, 0), (1.0, 0.0, 0.0, 0.0))
        st.add((0.1, 0.2, 0.3), (0, 0, 0), (0.0, 1.0, 0.0, 0.0))  # 绕 x 轴 180°
        self.assertFalse(st.is_stable())
        rep = st.continuity_report()
        self.assertGreater(rep["max_geodesic_deg"], 100.0)

    def test_get_mean_attitude(self):
        """相同四元数窗口 → 均值旋转与输入一致 (测地 ≈ 0)."""
        st = PoseStabilizer(window=2)
        q = rotation_matrix_to_quaternion(
            euler_to_rotation_matrix(10, 20, 30, order="zyx"))
        for _ in range(2):
            st.add((0, 0, 0), (10, 20, 30), q)
        R_mean, q_mean = st.get_mean_attitude()
        self.assertIsNotNone(R_mean)
        self.assertIsNotNone(q_mean)
        geo = rotation_geodesic_deg(R_mean, quaternion_to_rotation_matrix(q))
        self.assertLess(geo, 1e-6)

    def test_continuity_report(self):
        """报告字段完整, 稳定窗口判定 stable."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        for i in range(3):
            st.add((0.1 + i * 0.001, 0.2, 0.3), (0, 0, 0), (1.0, 0.0, 0.0, 0.0))
        rep = st.continuity_report()
        self.assertEqual(rep["frames"], 3)
        self.assertLess(rep["pos_range_m"], 0.01)
        self.assertLess(rep["max_geodesic_deg"], 0.1)
        self.assertEqual(rep["verdict"], "stable")

    def test_backward_compat_no_quat(self):
        """旧调用 (无四元数) 行为不变."""
        st = PoseStabilizer(window=3, pos_thresh=0.01, angle_thresh=5.0)
        for _ in range(3):
            st.add((0.1, 0.2, 0.3), (0, 0, 0))
        self.assertTrue(st.is_stable())
        self.assertIsNone(st.get_mean_attitude()[0])
        rep = st.continuity_report()
        self.assertIsNone(rep["max_geodesic_deg"])


class ComputeGraspRotationDynamicTest(unittest.TestCase):
    def test_dynamic_follows_object(self):
        """dynamic: R_grasp_ee = R_obj_ee(消歧后) @ R_offset.T (偏移为 0 时恒等)."""
        # 纯绕基系 Z 旋转: Y 轴严格在 XY 平面 (z=0) → 不触发消歧翻转
        c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
        R_obj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        r_offset = np.array([0, 0, 0])
        Rg = compute_grasp_rotation(R_obj, "dynamic", r_offset,
                                    R_ee_base=np.eye(3),
                                    grasp_direction="right")
        np.testing.assert_allclose(Rg, R_obj, atol=1e-9)

    def test_dynamic_y_disambiguation(self):
        """Y 轴消歧: grasp_direction="right" → Y 朝下 (基系 z 分量为负)."""
        # 右手系 R_obj: Y = [0,0,1] (朝上) → "right" 应翻转为朝下
        R_obj = np.array([[1.0, 0.0, 0.0],
                          [0.0, 0.0, -1.0],
                          [0.0, 1.0, 0.0]])
        np.testing.assert_allclose(np.linalg.det(R_obj), 1.0)
        Rg = compute_grasp_rotation(R_obj, "dynamic", np.zeros(3),
                                    R_ee_base=np.eye(3),
                                    grasp_direction="right")
        self.assertLess(Rg[2, 1], 0.0, "right 抓取 Y 应朝下")

    def test_dynamic_with_offset(self):
        """叠加安装偏移: R_grasp_ee = R @ R_offset.T."""
        c, s = np.cos(np.radians(10)), np.sin(np.radians(10))
        R_obj = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        r_offset = np.array([0, 0, 45.0])
        Rg = compute_grasp_rotation(R_obj, "dynamic", r_offset,
                                    R_ee_base=np.eye(3),
                                    grasp_direction="right")
        R_offset = euler_to_rotation_matrix(*r_offset, order="zyx")
        np.testing.assert_allclose(Rg, R_obj @ R_offset.T, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
