# -*- coding: utf-8 -*-
"""纯内存 D0 会话门禁测试；不导入相机、socket 或机械臂库。"""
import unittest

from Orchestrator.session_barrier import (
    BarrierReject,
    SessionBarrier,
    SessionState,
    WristObservation,
    calib_fingerprint,
    parse_arrived,
)


P0 = ((0.477, 0.231, 0.624), (-135.0, 4.5, -86.0))
P2 = ((0.400, 0.100, 0.600), (-90.0, 89.0, -90.0))
P1 = ((0.410, 0.110, 0.610), (-90.0, 89.0, -90.0))
CALIBRATION = (
    (0.0, -1.0, 0.0, 0.101),
    (1.0, 0.0, 0.0, -0.052),
    (0.0, 0.0, 1.0, 0.204),
    (0.0, 0.0, 0.0, 1.0),
)


def p2_frame():
    return "P2_ARRIVED|0.4,0.1,0.6,-90,89,-90"


def p1_frame():
    return "P1_ARRIVED|0.41,0.11,0.61,-90,89,-90"


def make_barrier():
    fingerprint = calib_fingerprint(CALIBRATION)
    barrier = SessionBarrier("run-001", lease_epoch=7, ticket_ttl_s=30.0)
    barrier.arm(
        p0_pose=P0,
        p2_target=P2,
        p1_target=P1,
        planning_observation_id=41,
        calibration_fingerprint=fingerprint,
        now_ns=1_000,
    )
    return barrier, fingerprint


class TestTwoPhaseP2Barrier(unittest.TestCase):
    def test_p2_is_recorded_before_fresh_wrist_observation(self):
        barrier, fingerprint = make_barrier()

        barrier.receive_p2(p2_frame(), calibration_fingerprint=fingerprint, now_ns=2_000)
        self.assertEqual(barrier.state, SessionState.WAIT_FRESH_WRIST_CONFIRMATION)
        self.assertEqual(barrier.p2_arrived_at_ns, 2_000)

        ack = barrier.authorize_grasp(
            WristObservation(observation_id=42, observed_at_ns=2_001),
            calibration_fingerprint=fingerprint,
            now_ns=2_002,
        )
        self.assertEqual(ack, "<0.477,0.231,0.624,-135.00,4.50,-86.00>")
        self.assertEqual(barrier.state, SessionState.WAIT_P1_ARRIVED)

    def test_old_or_future_observation_cannot_authorize_p1(self):
        barrier, fingerprint = make_barrier()
        barrier.receive_p2(p2_frame(), calibration_fingerprint=fingerprint, now_ns=2_000)

        with self.assertRaisesRegex(BarrierReject, "fresh"):
            barrier.authorize_grasp(
                WristObservation(observation_id=41, observed_at_ns=2_001),
                calibration_fingerprint=fingerprint,
                now_ns=2_002,
            )

        barrier, fingerprint = make_barrier()
        barrier.receive_p2(p2_frame(), calibration_fingerprint=fingerprint, now_ns=2_000)
        with self.assertRaisesRegex(BarrierReject, "future"):
            barrier.authorize_grasp(
                WristObservation(observation_id=42, observed_at_ns=2_003),
                calibration_fingerprint=fingerprint,
                now_ns=2_002,
            )


class TestTicketScopeAndClosure(unittest.TestCase):
    def test_full_flow_consumes_ticket_and_closes_once(self):
        barrier, fingerprint = make_barrier()
        barrier.receive_p2(p2_frame(), calibration_fingerprint=fingerprint, now_ns=2_000)
        barrier.authorize_grasp(
            WristObservation(observation_id=42, observed_at_ns=2_001),
            calibration_fingerprint=fingerprint,
            now_ns=2_002,
        )

        self.assertEqual(
            barrier.receive_p1(p1_frame(), calibration_fingerprint=fingerprint, now_ns=2_003),
            "[0.477,0.231,0.624,-135.00,4.50,-86.00]",
        )
        barrier.receive_gripdone("GRIPDONE", now_ns=2_004)
        self.assertEqual(barrier.state, SessionState.COMPLETE)
        with self.assertRaises(BarrierReject):
            barrier.receive_gripdone("GRIPDONE", now_ns=2_005)

    def test_calibration_rotation_change_faults_before_ack(self):
        barrier, fingerprint = make_barrier()
        changed = list(CALIBRATION)
        changed[0] = (1.0, 0.0, 0.0, 0.101)
        with self.assertRaisesRegex(BarrierReject, "calibration"):
            barrier.receive_p2(
                p2_frame(),
                calibration_fingerprint=calib_fingerprint(tuple(changed)),
                now_ns=2_000,
            )
        self.assertEqual(barrier.state, SessionState.REJECTED)

    def test_gripdone_has_its_own_finite_deadline(self):
        barrier, fingerprint = make_barrier()
        barrier.receive_p2(p2_frame(), calibration_fingerprint=fingerprint, now_ns=2_000)
        barrier.authorize_grasp(
            WristObservation(observation_id=42, observed_at_ns=2_001),
            calibration_fingerprint=fingerprint,
            now_ns=2_002,
        )
        barrier.receive_p1(p1_frame(), calibration_fingerprint=fingerprint, now_ns=2_003)
        with self.assertRaisesRegex(BarrierReject, "timeout"):
            barrier.receive_gripdone("GRIPDONE", now_ns=12_000_000_004)


class TestParsing(unittest.TestCase):
    def test_truthy_non_string_is_a_fail_closed_parse_error(self):
        with self.assertRaises(BarrierReject):
            parse_arrived(object(), "P2_ARRIVED|")

    def test_full_matrix_fingerprint_changes_when_rotation_changes(self):
        changed = list(CALIBRATION)
        changed[0] = (1.0, 0.0, 0.0, 0.101)
        self.assertNotEqual(calib_fingerprint(CALIBRATION), calib_fingerprint(tuple(changed)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
