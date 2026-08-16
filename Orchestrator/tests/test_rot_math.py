"""rot_math 姿态数学单测 (审计 B2 修复 3: Euler ±180° 等价 / PCA 轴符号翻转).

覆盖:
  - Euler ±180° 等价表示不会被误判为不同姿态 (测地距离 vs 欧拉逐分量差)
  - _align_axes 4 候选帧间锚定 (绕壶身轴/把手轴/相机 Z 翻转全部对齐)
  - MatrixSmoother 交替翻转帧平滑、SVD 正交
  - unwrap_euler 跨 ±180 边界连续
  - 测地距离/四元数均值 (Markley, double cover) 性质
  - Rodrigues numpy 实现与 cv2.Rodrigues 一致
  - sanity_check_pose 单位纪律 (m/mm)
"""

import math
import unittest

import numpy as np

from Orchestrator import rot_math
from Orchestrator.rot_math import (
    MatrixSmoother,
    align_axes,
    quaternion_geodesic_deg,
    quaternion_mean,
    quaternion_to_rotation_matrix,
    rotate_around_axis,
    rotation_geodesic_deg,
    rotation_matrix_to_quaternion,
    sanity_check_pose,
    unwrap_euler,
)


def euler_to_R(A, B, C):
    """ZYX 约定旋转矩阵 (与 Estun/utils_shared.py 一致): R = Rz(C)@Ry(B)@Rx(A)."""
    ra = math.radians(A)
    rb = math.radians(B)
    rc = math.radians(C)
    Rx = np.array([[1, 0, 0], [0, math.cos(ra), -math.sin(ra)], [0, math.sin(ra), math.cos(ra)]])
    Ry = np.array([[math.cos(rb), 0, math.sin(rb)], [0, 1, 0], [-math.sin(rb), 0, math.cos(rb)]])
    Rz = np.array([[math.cos(rc), -math.sin(rc), 0], [math.sin(rc), math.cos(rc), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def random_rotation(rng):
    """随机正交旋转矩阵 (SVD 正交化, 固定 rng 保证确定性)."""
    U, _, Vt = np.linalg.svd(rng.standard_normal((3, 3)))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, 2] *= -1
        R = U @ Vt
    return R


class TestEulerPm180Equivalence(unittest.TestCase):
    """审计 B2: 同一旋转的 Euler ±180° 等价表示不得被误判为不同姿态."""

    def test_pm180_representations_geodesically_identical(self):
        R1 = euler_to_R(-90, 89, -90)
        R2 = euler_to_R(90, 91, 90)  # (A+180, 180−B, C+180) 等价表示
        self.assertLess(rotation_geodesic_deg(R1, R2), 1e-4)

    def test_pm180_not_mistaken_for_different_attitude(self):
        """窗口 [R1, R2, R1] 最大两两测地距离 ≈0 < 阈值; 对照 naive 欧拉差会误判."""
        R1 = euler_to_R(-90, 89, -90)
        R2 = euler_to_R(90, 91, 90)
        geodesic = max(rotation_geodesic_deg(R1, R2),
                       rotation_geodesic_deg(R1, R1),
                       rotation_geodesic_deg(R2, R1))
        self.assertLess(geodesic, 3.0)  # 稳定器测地阈值
        # 对照: 欧拉角逐分量差会误判为 ~180° 大跳变
        naive = max(abs(a - b) for a, b in zip((-90, 89, -90), (90, 91, 90)))
        self.assertGreater(naive, 170.0)

    def test_unwrap_keeps_pm180_continuous(self):
        # 表观 ±180 边界跳变经 unwrap 后与上一帧逐分量接近
        out = unwrap_euler((-179.0, 89.0, -179.0), (179.0, 89.0, 179.0))
        self.assertLess(abs(out[0] - 179.0), 2.1)
        self.assertLess(abs(out[2] - 179.0), 2.1)
        out2 = unwrap_euler((179.0, 89.0, -179.0), (179.0, 89.0, 179.0))
        self.assertLess(abs(out2[2] - 179.0), 2.1)


class TestAlignAxes(unittest.TestCase):
    """审计 B2: PCA 主轴符号翻转样本经 _align_axes 对齐后与锚点一致."""

    def test_four_flip_candidates_all_align(self):
        """锚点=真实位姿: 任意列符号翻转样本经 align 后恢复真实位姿 (PCA 翻转样本)."""
        rng = np.random.default_rng(0)
        for _ in range(20):
            R = random_rotation(rng)
            for axis_i in range(3):
                flip = rotate_around_axis(R[:, axis_i], 180.0) @ R  # 列符号翻转
                aligned = align_axes(flip, R)
                self.assertLess(rotation_geodesic_deg(aligned, R), 1e-4)
            # 含恒等候选: 锚点即 R 本身时, 对齐结果应与 R 一致 (1e-4 容忍 acos 浮点噪声)
            self.assertLess(rotation_geodesic_deg(align_axes(R, R), R), 1e-4)

    def test_noisy_anchor_recovers_true_pose(self):
        """上一帧锚点带小噪声 (真实运行场景): 对齐后仍接近真实位姿."""
        rng = np.random.default_rng(1)
        for _ in range(10):
            R = random_rotation(rng)
            noise_axis = rng.standard_normal(3)
            anchor = rotate_around_axis(noise_axis, 1.0) @ R  # 锚点偏 1°
            for axis_i in range(3):
                flip = rotate_around_axis(R[:, axis_i], 180.0) @ R
                aligned = align_axes(flip, anchor)
                self.assertLess(rotation_geodesic_deg(aligned, anchor), 1.1)

    def test_prev_none_identity(self):
        rng = np.random.default_rng(2)
        R = random_rotation(rng)
        self.assertLess(rotation_geodesic_deg(align_axes(R, None), R), 1e-4)

    def test_alternating_flip_frames_smooth(self):
        """交替翻转帧 [R, flip, R, flip] 经 align+smoother 后相邻测地 < 1°."""
        rng = np.random.default_rng(2)
        R = random_rotation(rng)
        flip = rotate_around_axis(R[:, 0], 180.0) @ R
        sm = MatrixSmoother(alpha=0.4)
        prev_smooth = None
        prev_raw = None
        for raw in (R, flip, R, flip):
            aligned = align_axes(raw, prev_raw)
            prev_raw = aligned
            smooth = sm.update(aligned)
            if prev_smooth is not None:
                self.assertLess(rotation_geodesic_deg(smooth, prev_smooth), 1.0)
            prev_smooth = smooth

    def test_matrix_smoother_orthogonal(self):
        sm = MatrixSmoother(alpha=0.3)
        rng = np.random.default_rng(3)
        R = random_rotation(rng)
        sm.update(R)
        out = sm.update(random_rotation(rng))
        self.assertLess(np.max(np.abs(out.T @ out - np.eye(3))), 1e-9)


class TestUnwrapEuler(unittest.TestCase):
    def test_crosses_180_continuous(self):
        out = unwrap_euler((179.0, 89.0, -179.0), (179.0, 89.0, 179.0))
        self.assertLess(abs(out[2] - 179.0), 2.1)
        out2 = unwrap_euler((-179.0, 89.0, 179.0), (179.0, 89.0, 179.0))
        self.assertLess(abs(out2[0] - 179.0), 2.1)

    def test_wrap_roundtrip_idempotent(self):
        # 语义: 取与上一帧最接近的 ±360k 表示 (170 ≡ -190 同角)
        out = unwrap_euler((170.0, 0.0, 170.0), (-190.0, 0.0, -190.0))
        self.assertEqual(out, (-190.0, 0.0, -190.0))
        out2 = unwrap_euler(out, (170.0, 0.0, 170.0))
        self.assertEqual(out2, (170.0, 0.0, 170.0))
        # 幂等: prev 与输入同表示时输出不变
        self.assertEqual(unwrap_euler(out2, out2), out2)


class TestGeodesic(unittest.TestCase):
    def test_properties(self):
        rng = np.random.default_rng(4)
        R1 = random_rotation(rng)
        R2 = random_rotation(rng)
        self.assertLess(rotation_geodesic_deg(R1, R1), 1e-9)  # 自反
        self.assertAlmostEqual(rotation_geodesic_deg(R1, R2),
                               rotation_geodesic_deg(R2, R1), places=9)  # 对称
        Rz30 = rotate_around_axis(np.array([0.0, 0.0, 1.0]), 30.0)
        self.assertAlmostEqual(rotation_geodesic_deg(np.eye(3), Rz30), 30.0, places=9)
        # 四元数版与矩阵版一致
        q1, q2 = rotation_matrix_to_quaternion(R1), rotation_matrix_to_quaternion(R2)
        self.assertAlmostEqual(quaternion_geodesic_deg(q1, q2),
                               rotation_geodesic_deg(R1, R2), places=6)

    def test_quaternion_mean_double_cover(self):
        rng = np.random.default_rng(5)
        R = random_rotation(rng)
        q = rotation_matrix_to_quaternion(R)
        mean = quaternion_mean(np.array([q, -q, q]))  # q 与 -q 同一旋转
        self.assertLess(rotation_geodesic_deg(
            quaternion_to_rotation_matrix(mean), R), 1e-9)
        # 混合相近四元数均值应在几何中间
        qs = np.array([rotation_matrix_to_quaternion(rotate_around_axis(
            np.array([0.0, 0.0, 1.0]), a)) for a in (-10.0, 0.0, 10.0)])
        mean_q = quaternion_mean(qs)
        mean_R = quaternion_to_rotation_matrix(mean_q)
        self.assertLess(rotation_geodesic_deg(mean_R, np.eye(3)), 8.0)


class TestRodrigues(unittest.TestCase):
    def test_matches_cv2(self):
        import cv2
        rng = np.random.default_rng(6)
        for _ in range(10):
            axis = rng.standard_normal(3)
            angle = rng.uniform(-180.0, 180.0)
            R1 = rotate_around_axis(axis, angle)
            rvec = (axis / np.linalg.norm(axis)) * math.radians(angle)
            R2 = cv2.Rodrigues(rvec)[0]
            self.assertLess(np.max(np.abs(R1 - R2)), 1e-12)
        # 零模轴 → 单位阵
        self.assertLess(np.max(np.abs(rotate_around_axis(np.zeros(3), 45.0) - np.eye(3))), 1e-12)


class TestQuaternion(unittest.TestCase):
    def test_roundtrip(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            R = random_rotation(rng)
            q = rotation_matrix_to_quaternion(R)
            R2 = quaternion_to_rotation_matrix(q)
            self.assertLess(rotation_geodesic_deg(R, R2), 1e-9)
            self.assertGreaterEqual(q[0], 0.0)  # w>=0 约定


class TestSanityCheckPose(unittest.TestCase):
    def test_units(self):
        sanity_check_pose(np.array([0.477, 0.231, 0.624, -135, 4.5, -86]), "m")  # 合法
        sanity_check_pose(np.array([477.0, 231.0, 624.0, -135, 4.5, -86]), "mm")  # 合法
        with self.assertRaises(ValueError):
            sanity_check_pose(np.array([10.0, 0.0, 0.0, 0, 0, 0]), "m")  # xyz 越界
        with self.assertRaises(ValueError):
            sanity_check_pose(np.array([6000.0, 0.0, 0.0, 0, 0, 0]), "mm")
        with self.assertRaises(ValueError):
            sanity_check_pose(np.array([0.0, 0.0, 0.0, 800.0, 0, 0]), "m")  # 角度越界
        with self.assertRaises(ValueError):
            sanity_check_pose(np.array([0.0, 0.0, 0.0, 0, 0]), "m")  # 字段数错
        with self.assertRaises(ValueError):
            sanity_check_pose(np.array([0.0, 0.0, 0.0, 0, 0, 0]), "cm")  # 未知单位


if __name__ == "__main__":
    unittest.main(verbosity=2)
