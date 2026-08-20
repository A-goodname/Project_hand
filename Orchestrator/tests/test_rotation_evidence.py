# -*- coding: utf-8 -*-
"""18 回合证据集记录器 (RotationEvidence) 单测 (对方 08-20 addendum §3).

覆盖:
  1. round/sample 两级记录: type 区分, round 自动带 target_angle/config_sha256/physical_limit
  2. sample 每有效检测帧: ratio/d/rd/handle_side/bbox/贴纸中心像素
  3. config SHA-256 计算正确 (与 sha256sum 一致)
  4. 关闭后不再写入 (socket 断开自动保存语义)
  5. numpy 标量序列化
"""
import hashlib
import json
import os
import tempfile
import unittest

import numpy as np

from Orchestrator.rotation_evidence import RotationEvidence

DUMMY_CONFIG = tempfile.NamedTemporaryFile(
    "w", suffix=".yaml", delete=False, encoding="utf-8")
DUMMY_CONFIG.write("handle_rotation:\n  enabled: true\n")
DUMMY_CONFIG.close()


def _expected_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class RotationEvidenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._out = os.path.join(self._tmp.name, "ev")

    def tearDown(self):
        self._tmp.cleanup()

    def _read_lines(self, path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def test_round_and_sample_records(self):
        ev = RotationEvidence(self._out, "R01", target_angle_deg=-30,
                              config_path=DUMMY_CONFIG.name)
        ev.write_round(status="session_start", ts=123.0)
        ev.write_sample(ratio=0.204, d=0.63, rd=0.204, handle_side="right",
                        kettle_bbox=[1, 2, 3, 4], tag_u=500.0, tag_v=260.0,
                        theta_deg=-29.5, band=-30)
        ev.close()
        lines = self._read_lines(ev.path)
        self.assertEqual(len(lines), 2)
        r, s = lines
        self.assertEqual(r["type"], "round")
        self.assertEqual(r["round_id"], "R01")
        self.assertEqual(r["target_angle_deg"], -30)
        self.assertEqual(r["config_sha256"], _expected_sha(DUMMY_CONFIG.name))
        self.assertTrue(r["physical_limit"])
        self.assertEqual(s["type"], "sample")
        self.assertEqual(s["round_id"], "R01")
        self.assertEqual(s["ratio"], 0.204)
        self.assertEqual(s["tag_u"], 500.0)
        self.assertNotIn("target_angle_deg", s)   # sample 不带回合级字段

    def test_numpy_values_serialized(self):
        ev = RotationEvidence(self._out, "R02", config_path=DUMMY_CONFIG.name)
        ev.write_round(theta_med=float(np.float32(-29.5)), p1=list(np.array([1.0, 2.0, 3.0])))
        ev.close()
        r = self._read_lines(ev.path)[0]
        self.assertEqual(r["theta_med"], -29.5)
        self.assertEqual(r["p1"], [1.0, 2.0, 3.0])

    def test_write_after_close_noop(self):
        ev = RotationEvidence(self._out, "R03", config_path=DUMMY_CONFIG.name)
        ev.write_round(status="session_start")
        ev.close()
        ev.write_round(status="after_close")   # 不抛错且不写入
        lines = self._read_lines(ev.path)
        self.assertEqual(len(lines), 1)

    def test_round_auto_fields(self):
        ev = RotationEvidence(self._out, "R04", target_angle_deg=22,
                              config_path=DUMMY_CONFIG.name, physical_limit=False)
        ev.write_round(status="done", grasp_result="SUCCESS")
        ev.close()
        r = self._read_lines(ev.path)[0]
        self.assertEqual(r["target_angle_deg"], 22)
        self.assertFalse(r["physical_limit"])
        self.assertEqual(r["grasp_result"], "SUCCESS")

    def test_path_under_out_dir(self):
        ev = RotationEvidence(self._out, "R05", config_path=DUMMY_CONFIG.name)
        self.assertEqual(os.path.basename(ev.path), "R05.jsonl")
        self.assertTrue(os.path.dirname(ev.path).endswith("ev"))
        ev.close()


if __name__ == "__main__":
    unittest.main()
