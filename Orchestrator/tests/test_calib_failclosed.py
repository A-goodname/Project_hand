# -*- coding: utf-8 -*-
"""标定加载失败关闭 (B4 2026-08-20).

data_sm::load_cam2ee 对齐 state_machine 的 D0 门禁: 文件/schema/样本数/
重投影/方法/非有限 4x4/非刚体/单位变换任一失败抛 CalibrationError,
绝不返回 np.eye(4) 或默认 Tsai 回退. 全部离线, 不触碰硬件.
"""
import os
import tempfile
import unittest

import numpy as np
import yaml

from Orchestrator.estun_handle_data_sm import CalibrationError, load_cam2ee


def _good_matrix():
    """有效刚体 4x4: 绕 Z 旋转 30° + 平移 [0.1, 0.2, 0.3] m."""
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [0.1, 0.2, 0.3]
    return T.tolist()


def _good_payload():
    return {
        "calibration": {
            "num_samples": 26,
            "reprojection_error_px": 0.228,
            "handeye": {
                "use_method": "Daniilidis",
                "Daniilidis": {"T_cam_to_gripper": _good_matrix()},
            },
        }
    }


class LoadCam2eeFailClosedTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, payload):
        p = os.path.join(self._tmp.name, "calib.yaml")
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f)
        return p

    def _mutate(self, fn):
        """mutate 一份好 payload 并写盘, 断言 load_cam2ee 抛 CalibrationError."""
        payload = _good_payload()
        fn(payload)
        with self.assertRaises(CalibrationError):
            load_cam2ee(self._write(payload))

    # ---- B4 检查项: 全部失败关闭, 绝不返回单位矩阵 ----

    def test_missing_file(self):
        with self.assertRaises(CalibrationError):
            load_cam2ee(os.path.join(self._tmp.name, "nonexistent.yaml"))

    def test_invalid_yaml(self):
        p = os.path.join(self._tmp.name, "bad.yaml")
        with open(p, "w", encoding="utf-8") as f:
            f.write("::: 非法 yaml\n")
        with self.assertRaises(CalibrationError):
            load_cam2ee(p)

    def test_missing_calibration_section(self):
        self._mutate(lambda p: p.clear() or p.update({"foo": "bar"}))

    def test_sample_count_below_min(self):
        self._mutate(lambda p: p["calibration"].update(num_samples=3))

    def test_sample_count_non_numeric(self):
        self._mutate(lambda p: p["calibration"].update(num_samples="abc"))

    def test_reproj_above_max(self):
        self._mutate(lambda p: p["calibration"].update(reprojection_error_px=3.0))

    def test_missing_handeye(self):
        self._mutate(lambda p: p["calibration"].pop("handeye"))

    def test_missing_use_method(self):
        self._mutate(lambda p: p["calibration"]["handeye"].pop("use_method"))

    def test_missing_method_data(self):
        self._mutate(lambda p: p["calibration"]["handeye"].update(Daniilidis={}))

    def test_missing_T(self):
        self._mutate(lambda p: p["calibration"]["handeye"]["Daniilidis"].pop("T_cam_to_gripper"))

    def test_bad_shape(self):
        self._mutate(lambda p: p["calibration"]["handeye"]["Daniilidis"].update(
            T_cam_to_gripper=[[1, 0, 0], [0, 1, 0]]))

    def test_non_finite(self):
        def fn(p):
            T = _good_matrix()
            T[0][0] = float("nan")
            p["calibration"]["handeye"]["Daniilidis"]["T_cam_to_gripper"] = T
        self._mutate(fn)

    def test_identity_transform_rejected(self):
        self._mutate(lambda p: p["calibration"]["handeye"]["Daniilidis"].update(
            T_cam_to_gripper=np.eye(4).tolist()))

    def test_non_rigid(self):
        def fn(p):
            T = _good_matrix()
            T[1][0] = 0.9  # 破坏正交
            p["calibration"]["handeye"]["Daniilidis"]["T_cam_to_gripper"] = T
        self._mutate(fn)

    def test_homogeneous_row_bad(self):
        def fn(p):
            T = np.array(_good_matrix())
            T[3, :] = [0.0, 0.0, 0.0, 2.0]
            p["calibration"]["handeye"]["Daniilidis"]["T_cam_to_gripper"] = T.tolist()
        self._mutate(fn)

    # ---- 正常路径: 返回有效矩阵, 绝非单位变换 ----

    def test_good_matrix_returned(self):
        T = load_cam2ee(self._write(_good_payload()))
        self.assertEqual(T.shape, (4, 4))
        self.assertFalse(np.allclose(T, np.eye(4)))
        self.assertTrue(np.allclose(T[:3, 3], [0.1, 0.2, 0.3]))
        self.assertTrue(np.all(np.isfinite(T)))


if __name__ == "__main__":
    unittest.main()
