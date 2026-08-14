# -*- coding: utf-8 -*-
"""AST-free source checks for the D0 integration; never imports runtime hardware modules."""
import os
from pathlib import Path
import unittest


class TestStateMachineGateSource(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        orchestrator_dir = Path(__file__).resolve().parents[1]
        state_path = Path(os.environ.get("PROJECT_HAND_STATE_MACHINE", orchestrator_dir / "state_machine.py"))
        detector_path = Path(os.environ.get("PROJECT_HAND_DETECTOR", orchestrator_dir / "handle_cup_detector.py"))
        cls.source = state_path.read_text(encoding="utf-8")
        cls.detector = detector_path.read_text(encoding="utf-8")

    def test_receipt_is_committed_before_post_p2_refresh(self):
        receipt = self.source.index("barrier.receive_p2(")
        refresh = self.source.index("self._refresh_handle_observation(", receipt)
        authorize = self.source.index("barrier.authorize_grasp(", refresh)
        self.assertLess(receipt, refresh)
        self.assertLess(refresh, authorize)

    def test_refresh_clears_old_observation_and_forces_new_detection(self):
        start = self.source.index("def _refresh_handle_observation")
        end = self.source.index("def _release_camera_a_late", start)
        refresh = self.source[start:end]
        self.assertIn("det.last_known_pose = None", refresh)
        self.assertIn("last_known_pose_observation_id", refresh)
        self.assertIn("self._detect(ctx, force=True)", refresh)
        self.assertIn("observed_ns > after_ns", refresh)

    def test_gate_and_close_latch_are_connection_local(self):
        self.assertNotIn("self._barrier", self.source)
        self.assertNotIn("self._grip_done_once", self.source)
        self.assertIn("self._session_lock", self.source)
        self.assertIn("lease_epoch", self.source)

    def test_detector_has_monotonic_observation_identity(self):
        self.assertIn("last_known_pose_observation_id", self.detector)
        self.assertIn("last_known_pose_at_monotonic_ns", self.detector)


if __name__ == "__main__":
    unittest.main(verbosity=2)
