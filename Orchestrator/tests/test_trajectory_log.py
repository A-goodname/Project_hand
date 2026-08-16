# -*- coding: utf-8 -*-
"""轨迹日志测试 (audit 修复 4): schema 字段 / numpy 序列化 / 非有限值 / 线程安全."""
import json
import os
import tempfile
import threading
import time
import unittest

import numpy as np

from Orchestrator.trajectory_log import TrajectoryLogger, _jsonable


class TrajectoryLogTest(unittest.TestCase):
    def _tmp_path(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        os.unlink(path)
        return path

    def test_session_records(self):
        """session_start / obs / event / grasp / session_end 全记录结构."""
        path = self._tmp_path()
        lg = TrajectoryLogger(path, run_id="r1", lease_epoch=2,
                              calib_fingerprint="a" * 16,
                              t_cam2ee=np.eye(4), p0_pose=((0.1, 0.2, 0.3), (1, 2, 3)))
        lg.log_observation(
            monotonic_ns=1000, observation_id=7,
            pos_cam=(0.5, 0.6, 0.7), euler_cam_deg=(-90, 0, 0),
            rot_cam=np.eye(3), quat_cam=(1, 0, 0, 0),
            source="dynamic", confidence=0.88, depth_points=213,
            stabilizer={"verdict": "stable", "geodesic_deg": 1.1, "samples": 5},
        )
        lg.log_event("P2_ARRIVED", code="OK")
        lg.log_grasp(ticket_id="t-1", pose_base=[1, 2, 3], observation_id=7)
        lg.close()
        records = lg.read_back()
        self.assertEqual([r["type"] for r in records],
                         ["session_start", "obs", "event", "grasp", "session_end"])
        # session_start 元数据
        self.assertEqual(records[0]["run_id"], "r1")
        self.assertEqual(records[0]["lease_epoch"], 2)
        self.assertEqual(records[0]["t_cam2ee"], np.eye(4).tolist())
        # obs: audit 修复 4 全字段
        obs = records[1]
        self.assertEqual(obs["monotonic_ns"], 1000)
        self.assertEqual(obs["observation_id"], 7)
        self.assertEqual(obs["pos_cam"], [0.5, 0.6, 0.7])
        self.assertEqual(obs["rot_cam"], [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(obs["quat_cam"], [1, 0, 0, 0])
        self.assertEqual(obs["source"], "dynamic")
        self.assertEqual(obs["confidence"], 0.88)
        self.assertEqual(obs["depth_points"], 213)
        self.assertEqual(obs["stabilizer"], {"verdict": "stable",
                                             "geodesic_deg": 1.1, "samples": 5})
        # 事件/抓取/结束
        self.assertEqual(records[2]["event"], "P2_ARRIVED")
        self.assertEqual(records[3]["ticket_id"], "t-1")
        self.assertIn("session_end", records[4]["type"])

    def test_nan_and_inf_become_null(self):
        """非有限数值 → null (JSON 纪律, 与 NOT_FINITE 一致)."""
        path = self._tmp_path()
        lg = TrajectoryLogger(path, run_id="r", lease_epoch=1)
        lg.log_observation(monotonic_ns=1, observation_id=1,
                           pos_cam=(np.nan, np.inf, 0.5),
                           euler_cam_deg=(-np.inf, 0, 0))
        lg.close()
        obs = lg.read_back()[1]
        self.assertIsNone(obs["pos_cam"][0])
        self.assertIsNone(obs["pos_cam"][1])
        self.assertEqual(obs["pos_cam"][2], 0.5)
        self.assertIsNone(obs["euler_cam_deg"][0])

    def test_append_durability(self):
        """追加写 + 逐行 flush: close 前写入的行已落盘."""
        path = self._tmp_path()
        lg = TrajectoryLogger(path, run_id="r", lease_epoch=1)
        lg.log_observation(monotonic_ns=1, observation_id=1,
                           pos_cam=(0, 0, 0), euler_cam_deg=(0, 0, 0))
        # 不 close, 直接读文件 (模拟进程崩溃取证)
        with open(path, encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        self.assertEqual(len(lines), 2)   # session_start + obs
        lg.close()

    def test_thread_safety(self):
        """多线程并发写: 行不交错, 全部可解析."""
        path = self._tmp_path()
        lg = TrajectoryLogger(path, run_id="r", lease_epoch=1)
        errors = []

        def writer(i):
            try:
                for j in range(50):
                    lg.log_observation(
                        monotonic_ns=i * 1000 + j, observation_id=i * 100 + j,
                        pos_cam=(i, j, i + j), euler_cam_deg=(0, 0, 0))
            except Exception as e:  # noqa: BLE001
                errors.append(repr(e))

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        lg.close()
        self.assertFalse(errors, errors)
        records = lg.read_back()
        obs = [r for r in records if r["type"] == "obs"]
        self.assertEqual(len(obs), 200)
        ids = {r["observation_id"] for r in obs}
        self.assertEqual(len(ids), 200, "观测 ID 应无重复/丢失")

    def test_jsonable_scalars(self):
        """numpy 标量与 int 直接序列化."""
        self.assertEqual(_jsonable(np.float32(1.5)), 1.5)
        self.assertEqual(_jsonable(np.int64(3)), 3)
        self.assertEqual(_jsonable(np.eye(2)), [[1.0, 0.0], [0.0, 1.0]])
        self.assertIsNone(_jsonable(float("nan")))


if __name__ == "__main__":
    unittest.main()
