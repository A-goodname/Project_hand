"""handle_cup_detector 3D 几何法姿态符号规则单测 (审计 B2 修复 3).

覆盖 _compute_handle_pose_geometry 的符号约束:
  - Y = 把手长轴, 符号按 grasp_direction: right → Y·(0,1,0)>0 (图像下边界),
    left → Y·(0,-1,0)>0 (图像上边界)
  - Z = X×Y 面法线, Z[2] < 0 朝向相机 (RealSense/OpenCV z 正=远离相机)
  - 右手系 (det=+1); 零范数/平行输入返回 None (fail-closed)

不触相机: HandleCupDetector.__new__ 跳过 __init__ (不开 pipeline).
"""

import unittest

import numpy as np

from Orchestrator.handle_cup_detector import HandleCupDetector


def make_detector(grasp_direction="right"):
    det = HandleCupDetector.__new__(HandleCupDetector)
    det._grasp_direction = grasp_direction
    return det


class TestGeometrySymbolRules(unittest.TestCase):
    def test_y_sign_by_grasp_direction(self):
        # body 沿 +x, handle 沿 +y: right 应得 Y=(0,1,0), left 应得 Y=(0,-1,0)
        body_axis = np.array([1.0, 0.0, 0.0])
        handle_axis = np.array([0.0, 1.0, 0.0])

        R = make_detector("right")._compute_handle_pose_geometry(handle_axis, body_axis)
        self.assertLess(np.max(np.abs(R[:, 1] - np.array([0.0, 1.0, 0.0]))), 1e-9)
        # Z = X×Y 朝向相机 (z<0): X=(1,0,0), Y=(0,1,0) → Z=(0,0,1) → 需翻 → (0,0,-1)
        self.assertLess(np.max(np.abs(R[:, 2] - np.array([0.0, 0.0, -1.0]))), 1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

        R = make_detector("left")._compute_handle_pose_geometry(handle_axis, body_axis)
        self.assertLess(np.max(np.abs(R[:, 1] - np.array([0.0, -1.0, 0.0]))), 1e-9)
        self.assertLess(np.max(np.abs(R[:, 2] - np.array([0.0, 0.0, -1.0]))), 1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_y_flip_when_pointing_away(self):
        # handle 沿 -y: right 规则要求 Y·(0,1,0)>0 → 翻转
        R = make_detector("right")._compute_handle_pose_geometry(
            np.array([0.0, -1.0, 0.0]), np.array([1.0, 0.0, 0.0]))
        self.assertGreater(R[1, 1], 0.99)

    def test_z_always_toward_camera(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            body_axis = rng.standard_normal(3)
            handle_axis = rng.standard_normal(3)
            R = make_detector("right")._compute_handle_pose_geometry(handle_axis, body_axis)
            if R is None:
                continue
            self.assertLess(R[2, 2], 1e-6)      # Z[2] < 0 朝向相机
            self.assertAlmostEqual(np.linalg.det(R), 1.0, places=6)  # 右手系
            for col in range(3):                # 单位列
                self.assertAlmostEqual(np.linalg.norm(R[:, col]), 1.0, places=9)

    def test_degenerate_returns_none(self):
        det = make_detector("right")
        self.assertIsNone(det._compute_handle_pose_geometry(
            np.array([0.0, 1.0, 0.0]), None))            # body_axis 缺失
        self.assertIsNone(det._compute_handle_pose_geometry(
            np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])))  # X ∥ Y → Z≈0
        self.assertIsNone(det._compute_handle_pose_geometry(
            np.zeros(3), np.array([1.0, 0.0, 0.0])))     # 零范数 handle_axis


if __name__ == "__main__":
    unittest.main(verbosity=2)
