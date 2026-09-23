"""No-hardware protocol checks for Phase D keyboard authority."""
import unittest
import threading
import numpy as np
from unittest import mock

from Orchestrator.estun_handle_cup_sm import HandleCupSM, phase_d_signal


class FakeConnection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def settimeout(self, _seconds):
        pass

    def recv(self, _size):
        return next(self.messages, b"")

    def sendall(self, data):
        self.sent.append(data)


class PhaseDKeyboardGateTest(unittest.TestCase):
    def run_case(self, key, expected, release_count):
        sm = HandleCupSM()
        sm._hand_available = True
        sm._no_preview = True
        conn = FakeConnection([b"READY", b"RELEASE", b""])

        def begin():
            sm._handover_active.set()
            sm._handle_key(key)

        with mock.patch.object(sm, "_begin_handover", side_effect=begin), \
             mock.patch.object(sm, "_close_handover"), \
             mock.patch("Orchestrator.estun_handle_cup_sm.release_object") as release:
            sm._phase_c(conn, "<0,0,0,0,0,0>[0,0,0,0,0,0]{0.229,0.442,0.383,148.79,82.79,-113.57}",
                        (0.2, 0.4, 0.6), (-90, 80, -90))
        self.assertEqual(len(conn.sent), 1)
        self.assertEqual(conn.sent[0], expected)
        self.assertEqual(release.call_count, release_count)

    def test_continue_releases(self):
        self.run_case(ord("c"), b"<0.229,0.442,0.383,148.79,82.79,-113.57>", 1)

    def test_abort_retains_grip(self):
        self.run_case(ord("a"), b"[0.229,0.442,0.383,148.79,82.79,-113.57]", 0)

    def test_abort_branch_releases_at_c_p1_then_resets(self):
        sm = HandleCupSM()
        sm._hand_available = True
        sm._no_preview = True
        conn = FakeConnection([b"READY", b"RELEASE", b"RELEASE", b"RESET", b""])

        def begin():
            sm._handover_active.set()
            sm._handle_key(ord("a"))

        with mock.patch.object(sm, "_begin_handover", side_effect=begin), \
             mock.patch.object(sm, "_close_handover"), \
             mock.patch("Orchestrator.estun_handle_cup_sm.release_object") as release, \
             mock.patch("Orchestrator.estun_handle_cup_sm.hand_reset") as reset:
            sm._phase_c(conn, "<0,0,0,0,0,0>[0,0,0,0,0,0]{0.229,0.442,0.383,148.79,82.79,-113.57}",
                        (0.2, 0.4, 0.6), (-90, 80, -90))
        self.assertEqual(len(conn.sent), 1)
        self.assertTrue(conn.sent[0].startswith(b"["))
        release.assert_called_once()
        reset.assert_called_once()

    def test_missing_c_p3_rejected(self):
        with self.assertRaises(ValueError):
            phase_d_signal(None, "CONTINUE")

    def test_visual_never_decides(self):
        sm = HandleCupSM()
        sm._no_preview = True
        sm._handover_active.set()
        sm._handover_vision_failed = True
        sm._update_handover()
        self.assertFalse(sm._handover_confirmed.is_set())
        self.assertIsNone(sm._handover_choice)

    def test_fusion_reuses_phase0_hand_and_prewarmed_detector(self):
        sm = HandleCupSM()
        sm._defer_devices = True
        sm._detector = object()
        sm._phase0_handoff = threading.Event()
        sm._phase0_handoff.set()
        with mock.patch("Orchestrator.estun_handle_cup_sm.is_hand_available", return_value=True), \
             mock.patch.object(sm, "_init_hand") as init_hand, \
             mock.patch.object(sm, "_init_detector") as init_detector:
            sm._ensure_devices()
        init_hand.assert_not_called()
        init_detector.assert_not_called()
        self.assertTrue(sm._hand_available)

    def test_handover_model_preloads_without_inference(self):
        sm = HandleCupSM()
        sm._no_preview = True
        fake_camera = mock.Mock()
        fake_camera.read.return_value = (np.zeros((10, 10, 3), dtype=np.uint8),)
        fake_detector = mock.Mock()
        fake_detector.is_hand_on_cup = False
        with mock.patch("Orchestrator.estun_handle_cup_sm.UGreenCamera", return_value=fake_camera), \
             mock.patch("Orchestrator.estun_handle_cup_sm.HandInCupDetector", return_value=fake_detector):
            self.assertTrue(sm._prepare_handover_vision())
            fake_camera.read.assert_not_called()
            fake_detector.update.assert_not_called()
            sm._handover_active.set()
            sm._update_handover()
            fake_detector.update.assert_called_once()
        sm._close_handover()

    def test_camera_a_closes_after_phase_b(self):
        sm = HandleCupSM()
        detector = mock.Mock()
        sm._detector = detector
        sm._detect_active = True
        sm._close_camera_a()
        detector.close.assert_called_once()
        self.assertIsNone(sm._detector)
        self.assertFalse(sm._detect_active)


if __name__ == "__main__":
    unittest.main()
