# -*- coding: utf-8 -*-
"""session_barrier 单元测试 (纯逻辑, 无硬件/socket).

覆盖对方验收关注点:
  - 严格解析: 空/非ASCII/空白/前缀/个数/有限数
  - P2: 正确通过 / 越序 / 容差 / ticket 过期 / 旧 ticket / 新鲜观测
  - P1: 一次性消费 / 重放 / 越序
  - GRIPDONE: 精确匹配才闭合 / 重复拒绝
  - P0 回执: 非全零实际位姿
"""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from session_barrier import (  # noqa: E402
    SessionBarrier, BarrierReject, parse_arrived, pose_within_tolerance,
)

P0 = ((0.477, 0.231, 0.624), (-135.0, 4.5, -86.0))       # Phase A 实际位姿
P2_T = ((0.40, 0.10, 0.60), (-90.0, 89.0, -90.0))        # 下发 P2 目标
P1_T = ((0.41, 0.11, 0.61), (-90.0, 89.0, -90.0))        # 下发 P1 目标


def make_barrier(ttl=30.0):
    b = SessionBarrier(run_id="run-test-1", calib_fingerprint="0.1,0.2,0.3",
                       pos_tol=0.02, ang_tol=5.0, ticket_ttl=ttl)
    b.issue_pregrasp()
    return b


def p2_ok():
    return f"P2_ARRIVED|{P2_T[0][0]},{P2_T[0][1]},{P2_T[0][2]},{P2_T[1][0]},{P2_T[1][1]},{P2_T[1][2]}"


def p1_ok():
    return f"P1_ARRIVED|{P1_T[0][0]},{P1_T[0][1]},{P1_T[0][2]},{P1_T[1][0]},{P1_T[1][1]},{P1_T[1][2]}"


# ============================================================
# 解析
# ============================================================

class TestParseArrived(unittest.TestCase):
    def test_ok(self):
        xyz, abc = parse_arrived("P2_ARRIVED|0.400,0.100,0.600,-90.00,89.00,-90.00", "P2_ARRIVED|")
        self.assertEqual(xyz, (0.4, 0.1, 0.6))
        self.assertEqual(abc, (-90.0, 89.0, -90.0))

    def test_empty(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("", "P2_ARRIVED|")

    def test_non_ascii(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("P2_ARRIVED|0.4,0.1,0.6,-90,89,好", "P2_ARRIVED|")

    def test_whitespace(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("P2_ARRIVED|0.4, 0.1,0.6,-90,89,-90", "P2_ARRIVED|")

    def test_bad_prefix(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("GRASP_AUTH|0.4,0.1,0.6,-90,89,-90", "P2_ARRIVED|")

    def test_bad_count(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("P2_ARRIVED|0.4,0.1,0.6,-90,89", "P2_ARRIVED|")

    def test_non_finite(self):
        with self.assertRaises(BarrierReject):
            parse_arrived("P2_ARRIVED|0.4,nan,0.6,-90,89,-90", "P2_ARRIVED|")

    def test_leading_zeros_ok(self):
        # 前缀为空体也拒绝
        with self.assertRaises(BarrierReject):
            parse_arrived("P2_ARRIVED|", "P2_ARRIVED|")


# ============================================================
# 容差
# ============================================================

class TestTolerance(unittest.TestCase):
    def test_within(self):
        ok, _ = pose_within_tolerance(P2_T[0], P2_T[1], P2_T[0], P2_T[1], 0.02, 5.0)
        self.assertTrue(ok)

    def test_pos_over(self):
        bad = (P2_T[0][0] + 0.05, P2_T[0][1], P2_T[0][2])
        ok, why = pose_within_tolerance(bad, P2_T[1], P2_T[0], P2_T[1], 0.02, 5.0)
        self.assertFalse(ok)
        self.assertIn("位置", why)

    def test_angle_over(self):
        bad = (P2_T[1][0], P2_T[1][1] + 10.0, P2_T[1][2])
        ok, why = pose_within_tolerance(P2_T[0], bad, P2_T[0], P2_T[1], 0.02, 5.0)
        self.assertFalse(ok)
        self.assertIn("角度", why)


# ============================================================
# 状态机
# ============================================================

class TestSessionBarrier(unittest.TestCase):
    def test_full_pass(self):
        """正常全流程: P2 通过 -> P1 通过 -> GRIPDONE 闭合."""
        b = make_barrier()
        action, ack = b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0,
                             fresh_obs_ts=time.time() + 1.0)
        self.assertEqual(action, "ACK")
        self.assertTrue(ack.startswith("<0.477"))
        self.assertTrue(ack.endswith(">"))

        action, ack = b.step(p1_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        self.assertEqual(action, "ACK")
        self.assertTrue(ack.startswith("[0.477"))
        self.assertTrue(ack.endswith("]"))

        action, _ = b.step("GRIPDONE", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        self.assertEqual(action, "GRIPDONE")
        self.assertEqual(b.state, "PASSED")

    def test_p2_out_of_order(self):
        """WAIT_P2 收到非 P2_ARRIVED (如旧 GRIPDONE) -> 拒绝."""
        b = make_barrier()
        with self.assertRaises(BarrierReject) as cm:
            b.step("GRIPDONE", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        self.assertEqual(cm.exception.code, "SEQ")
        self.assertEqual(b.state, "REJECTED")

    def test_p2_bad_format(self):
        b = make_barrier()
        with self.assertRaises(BarrierReject):
            b.step("P2_ARRIVED|0.4,0.1", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)

    def test_p2_tolerance(self):
        """P2 回执超容差 -> 拒绝, 不签 grasp ticket."""
        b = make_barrier()
        bad = f"P2_ARRIVED|{P2_T[0][0] + 0.5},{P2_T[0][1]},{P2_T[0][2]},{P2_T[1][0]},{P2_T[1][1]},{P2_T[1][2]}"
        with self.assertRaises(BarrierReject) as cm:
            b.step(bad, p2_target=P2_T, p1_target=P1_T, p0_pose=P0,
                   fresh_obs_ts=time.time() + 1.0)
        self.assertEqual(cm.exception.code, "P2_TOL")

    def test_p2_stale_obs(self):
        """无新鲜观测 (fresh_obs_ts 缺失或 <= P2 回执时间) -> 拒绝."""
        b = make_barrier()
        with self.assertRaises(BarrierReject) as cm:
            b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0, fresh_obs_ts=None)
        self.assertEqual(cm.exception.code, "STALE_OBS")
        # fresh_obs_ts 在 P2 回执之前 (模拟旧观测) 也拒绝
        b = make_barrier()
        with self.assertRaises(BarrierReject):
            b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0,
                   fresh_obs_ts=time.time() - 100.0)

    def test_pregrasp_ticket_expired(self):
        """ticket 过期 -> 拒绝."""
        b = make_barrier(ttl=0.001)
        time.sleep(0.01)
        with self.assertRaises(BarrierReject) as cm:
            b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0,
                   fresh_obs_ts=time.time() + 1.0)
        self.assertEqual(cm.exception.code, "PG_TICKET_EXPIRED")

    def test_p1_replay(self):
        """grasp ticket 一次性: 重复 P1 回执 -> 拒绝."""
        b = make_barrier()
        b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0, fresh_obs_ts=time.time() + 1.0)
        b.step(p1_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        # 重复 P1_ARRIVED (grasp ticket 已消费) -> 拒绝
        with self.assertRaises(BarrierReject) as cm:
            b.step(p1_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        self.assertEqual(cm.exception.code, "GRASP_REPLAY")

    def test_p1_before_p2(self):
        """P1 在 P2 之前 -> WAIT_P2 状态越序拒绝."""
        b = make_barrier()
        with self.assertRaises(BarrierReject):
            b.step(p1_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0)

    def test_gripdone_repeat_after_pass(self):
        """PASSED 后重复 GRIPDONE -> 拒绝 (防二次闭合)."""
        b = make_barrier()
        b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0, fresh_obs_ts=time.time() + 1.0)
        b.step(p1_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        b.step("GRIPDONE", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        with self.assertRaises(BarrierReject):
            b.step("GRIPDONE", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)

    def test_pregrip_gesture_only(self):
        """PREGRIP 仅手势: 状态不变, 不绕过屏障."""
        b = make_barrier()
        action, _ = b.step("PREGRIP", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)
        self.assertEqual(action, "PREGRIP")
        self.assertEqual(b.state, "WAIT_P2_ARRIVED")
        # 之后仍需 P2 才能前进
        with self.assertRaises(BarrierReject):
            b.step("GRIPDONE", p2_target=P2_T, p1_target=P1_T, p0_pose=P0)

    def test_p0_ack_not_zero(self):
        """P0 回执必须来自 Phase A 实际位姿 (非全零)."""
        b = make_barrier()
        action, ack = b.step(p2_ok(), p2_target=P2_T, p1_target=P1_T, p0_pose=P0,
                             fresh_obs_ts=time.time() + 1.0)
        self.assertNotIn(",0.000,0.000,0.000,", ack)


if __name__ == "__main__":
    unittest.main(verbosity=2)
