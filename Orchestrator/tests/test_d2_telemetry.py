# -*- coding: utf-8 -*-
"""D2 遥测服务端离线测试 (方案 §6.A, 2026-08-17).

覆盖: 正常完整序列 / 分片帧 / 粘连帧 / 一次 recv 多帧 / 乱序事件 /
      非数字 / NaN-Infinity / 单位错误 / 断链残帧 (PROTOCOL_TRUNCATED) /
      白名单外事件 / P2-P1 去分号后过既有 SessionBarrier 回归.
"""
import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from Orchestrator.estun_d2_telemetry_sm import (  # noqa: E402
    D2TelemetryServer, D2_EVENT_SEQUENCE, parse_event_frame,
)
from Orchestrator.session_barrier import (  # noqa: E402
    SessionBarrier, SessionState, WristObservation, calib_fingerprint,
)

P0 = ((0.477, 0.231, 0.624), (-135.0, 4.5, -86.0))
P2_T = ((0.40, 0.10, 0.60), (-90.0, 89.0, -90.0))
P1_T = ((0.41, 0.11, 0.61), (-90.0, 89.0, -90.0))
CALIB = calib_fingerprint(np.eye(4))


def _frame(event, xyz=(0.1, 0.9, 0.3), abc=(-115.2, 85.0, -21.1)):
    return f"{event}|{xyz[0]},{xyz[1]},{xyz[2]},{abc[0]},{abc[1]},{abc[2]};"


class ParseTest(unittest.TestCase):
    def test_valid(self):
        ev, xyz, abc = parse_event_frame("P1_ARRIVED|0.0255,0.9376,0.3594,"
                                         "-115.20,85.00,-21.10")
        self.assertEqual(ev, "P1_ARRIVED")
        self.assertEqual(xyz, [0.0255, 0.9376, 0.3594])

    def test_unknown_event(self):
        with self.assertRaises(ValueError):
            parse_event_frame("EVIL_CMD|0.1,0.2,0.3,1,2,3")

    def test_bad_count(self):
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|0.1,0.2,0.3,1,2")

    def test_non_numeric(self):
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|0.1,abc,0.3,1,2,3")

    def test_nan_inf(self):
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|nan,0.2,0.3,1,2,3")
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|0.1,inf,0.3,1,2,3")

    def test_unit_error(self):
        # 位置疑似 mm 误作 m (>3)
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|111.9,378.3,698.3,1,2,3")
        # 角度疑似非 deg (>360)
        with self.assertRaises(ValueError):
            parse_event_frame("P2_ARRIVED|0.1,0.2,0.3,720,2,3")


class _Harness:
    """socketpair 直连 handle_connection, 输出到临时目录."""

    def __init__(self, on_p2_p1=None):
        self.out = tempfile.mkdtemp(prefix="d2tel-")
        self.srv = D2TelemetryServer(out_dir=self.out,
                                     episode_id="ep-test-001",
                                     scene_batch_id="canonical-a",
                                     on_p2_p1=on_p2_p1)
        self.conn, self.cli = socket.socketpair()
        self.t = threading.Thread(
            target=lambda: self.srv.handle_connection(self.conn),
            daemon=True)
        self.t.start()

    def send(self, data: bytes):
        self.cli.sendall(data)

    def close(self):
        self.cli.close()
        self.t.join(timeout=5.0)

    def robot_state(self):
        with open(os.path.join(self.out, "ep-test-001", "signals",
                               "robot_state.jsonl"), encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]

    def ep_manifest(self):
        with open(os.path.join(self.out, "ep-test-001",
                               "episode_manifest.json"), encoding="utf-8") as f:
            return json.load(f)

    def art_manifest(self):
        with open(os.path.join(self.out, "ep-test-001",
                               "artifact_manifest.json"), encoding="utf-8") as f:
            return json.load(f)


class ServerTest(unittest.TestCase):
    @staticmethod
    def _free_port():
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def test_bind_accept_smoke(self):
        """真实 bind+accept 冒烟: 端口监听 → 连接 → 一帧落盘."""
        out = tempfile.mkdtemp(prefix="d2tel-bind-")
        port = self._free_port()
        srv = D2TelemetryServer(host="127.0.0.1", ports=(port,), out_dir=out,
                                episode_id="ep-bind-001")
        t = threading.Thread(target=srv.start, daemon=True)
        t.start()
        time.sleep(0.3)
        cli = socket.create_connection(("127.0.0.1", port), timeout=3.0)
        cli.sendall(_frame("P0_ACTUAL").encode("utf-8"))
        cli.close()
        time.sleep(0.5)
        srv.stop()
        t.join(timeout=3.0)
        rows_path = os.path.join(out, "ep-bind-001", "signals",
                                 "robot_state.jsonl")
        with open(rows_path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["event"] for r in rows], ["P0_ACTUAL"])

    def test_dual_port_listen(self):
        """双端口监听: 8082 与 8080 任一连接均收事件 (CRP 走 Socket0=8080 兜底)."""
        out = tempfile.mkdtemp(prefix="d2tel-dual-")
        p1, p2 = self._free_port(), self._free_port()
        srv = D2TelemetryServer(host="127.0.0.1", ports=(p1, p2),
                                out_dir=out, episode_id="ep-dual-001")
        t = threading.Thread(target=srv.start, daemon=True)
        t.start()
        time.sleep(0.3)
        # 连 p1 (8082 角色) 发 P0_ACTUAL
        cli1 = socket.create_connection(("127.0.0.1", p1), timeout=3.0)
        cli1.sendall(_frame("P0_ACTUAL").encode("utf-8"))
        cli1.close()
        time.sleep(0.3)
        # 连 p2 (8080 角色) 发 P2_ARRIVED
        cli2 = socket.create_connection(("127.0.0.1", p2), timeout=3.0)
        cli2.sendall(_frame("P2_ARRIVED").encode("utf-8"))
        cli2.close()
        time.sleep(0.5)
        srv.stop()
        t.join(timeout=3.0)
        rows_path = os.path.join(out, "ep-dual-001", "signals",
                                 "robot_state.jsonl")
        with open(rows_path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        self.assertEqual([r["event"] for r in rows],
                         ["P0_ACTUAL", "P2_ARRIVED"])

    def test_full_sequence_normal(self):
        """完整 9 事件正常帧 → 顺序完成, manifest 生成."""
        h = _Harness()
        for ev in D2_EVENT_SEQUENCE:
            h.send(_frame(ev).encode("utf-8"))
        h.close()
        rows = h.robot_state()
        self.assertEqual([r["event"] for r in rows], D2_EVENT_SEQUENCE)
        man = h.ep_manifest()
        self.assertTrue(man["sequence_complete"])
        self.assertEqual(man["outcome"], "SUCCESS")
        self.assertFalse(man["expert_valid"])       # 保守 false 待审计
        art = h.art_manifest()
        self.assertEqual({f["relpath"] for f in art["files"]},
                         {"signals/robot_state.jsonl",
                          "signals/state_transition.jsonl",
                          "signals/action_receipt.jsonl"})
        self.assertTrue(all(f["sha256"] for f in art["files"]))
        # schema 字段 (方案 §3.3)
        self.assertEqual(rows[0]["schema_version"], "estun-cpos-event-v1")
        self.assertEqual(rows[0]["state_kind"], "ESTUN_TCP_EVENT_SNAPSHOT")
        self.assertFalse(rows[0]["joint_state_available"])
        self.assertFalse(rows[0]["controller_timestamp_available"])
        self.assertEqual(rows[0]["clock_id"], "pc-monotonic-ingress-v1")

    def test_session_survives_long_idle_gap(self):
        """事件间长静默 (超过旧 2s 超时) 会话存活 — 修复: 超时≠断链."""
        h = _Harness()
        h.send(_frame("P0_ACTUAL").encode("utf-8"))
        time.sleep(3.5)                      # 真实 CRP 运动间隔 (秒级)
        h.send(_frame("P2_ARRIVED").encode("utf-8"))
        h.close()
        rows = h.robot_state()
        self.assertEqual([r["event"] for r in rows],
                         ["P0_ACTUAL", "P2_ARRIVED"])
        man = h.ep_manifest()
        self.assertFalse(any("IDLE_TIMEOUT" in r or "PROTOCOL_TRUNCATED" in r
                             for r in man["invalid_reasons"]))

    def test_fragmented_frame(self):
        """一帧逐字节分片 → 完整还原."""
        h = _Harness()
        frame = _frame("P0_ACTUAL").encode("utf-8")
        for b in frame:
            h.send(bytes([b]))
        h.close()
        self.assertEqual(len(h.robot_state()), 1)

    def test_coalesced_frames(self):
        """两帧一次 send (粘连) → 逐帧读出."""
        h = _Harness()
        blob = (_frame("P0_ACTUAL") + _frame("P2_ARRIVED")).encode("utf-8")
        h.send(blob)
        h.close()
        self.assertEqual([r["event"] for r in h.robot_state()],
                         ["P0_ACTUAL", "P2_ARRIVED"])

    def test_out_of_order(self):
        """乱序事件 → 顺序纪律标记无效 (但帧仍落盘)."""
        h = _Harness()
        h.send(_frame("P1_ARRIVED").encode("utf-8"))   # 应为 P0_ACTUAL
        h.close()
        man = h.ep_manifest()
        self.assertFalse(man["sequence_complete"])
        self.assertTrue(any("顺序异常" in r for r in man["invalid_reasons"]))

    def test_invalid_frame_kept_invalid(self):
        """非数字帧 → 记录 protocol_error, episode 无效."""
        h = _Harness()
        h.send(_frame("P0_ACTUAL").encode("utf-8"))
        h.send(b"P2_ARRIVED|0.1,abc,0.3,1,2,3;")
        h.close()
        man = h.ep_manifest()
        self.assertTrue(any("解析失败" in r for r in man["invalid_reasons"]))
        self.assertEqual(len(h.robot_state()), 1)     # 非法帧不落盘

    def test_truncated_disconnect(self):
        """断链残留半帧 → PROTOCOL_TRUNCATED, episode 无效."""
        h = _Harness()
        h.send(_frame("P0_ACTUAL").encode("utf-8"))
        h.send(b"P2_ARRIVED|0.1,0.2,0.3,1,2")         # 无结束符
        h.close()
        man = h.ep_manifest()
        self.assertTrue(man["protocol_truncated"])
        self.assertTrue(any("PROTOCOL_TRUNCATED" in r
                            for r in man["invalid_reasons"]))

    def test_barrier_regression(self):
        """P2/P1 去分号后过既有 SessionBarrier (方案 §5.4/§6.A.2)."""
        forwarded = []

        def on_p2_p1(event, xyz, abc):
            forwarded.append((event, xyz, abc))

        h = _Harness(on_p2_p1=on_p2_p1)
        for ev in D2_EVENT_SEQUENCE:
            h.send(_frame(ev).encode("utf-8"))
        h.close()
        self.assertEqual([f[0] for f in forwarded], ["P2_ARRIVED", "P1_ARRIVED"])

        # 帧体 (去 ";") 直接喂既有 barrier: 容差/票据/一次性消费不变
        barrier = SessionBarrier(run_id="d2-regress", lease_epoch=1)
        barrier.arm(p0_pose=P0, p2_target=P2_T, p1_target=P1_T,
                    planning_observation_id=1000,
                    calibration_fingerprint=CALIB)
        barrier.receive_p2(
            f"P2_ARRIVED|{P2_T[0][0]},{P2_T[0][1]},{P2_T[0][2]},"
            f"{P2_T[1][0]},{P2_T[1][1]},{P2_T[1][2]}",
            calibration_fingerprint=CALIB, now_ns=time.monotonic_ns())
        ack = barrier.authorize_grasp(
            WristObservation(2001, time.monotonic_ns()),
            calibration_fingerprint=CALIB, now_ns=time.monotonic_ns())
        self.assertTrue(ack.startswith("<0.477"))
        ack2 = barrier.receive_p1(
            f"P1_ARRIVED|{P1_T[0][0]},{P1_T[0][1]},{P1_T[0][2]},"
            f"{P1_T[1][0]},{P1_T[1][1]},{P1_T[1][2]}",
            calibration_fingerprint=CALIB, now_ns=time.monotonic_ns())
        self.assertTrue(ack2.startswith("[0.477"))
        barrier.receive_gripdone("GRIPDONE", now_ns=time.monotonic_ns())
        self.assertIs(barrier.state, SessionState.COMPLETE)


if __name__ == "__main__":
    unittest.main()
