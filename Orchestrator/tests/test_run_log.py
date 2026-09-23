"""No-device checks for fused-run logging and stability diagnostics."""
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Estun.phase0_camera_worker import _camera_process
from Orchestrator.estun_handle_cup_sm import PoseStabilizer
from Orchestrator.run_log import capture_run_log


class RunLogTest(unittest.TestCase):
    def test_camera_worker_logged_entry_without_hardware(self):
        class FakeCamera:
            camera_id = 4

        class FakeGreeting:
            def __init__(self, *args, **kwargs):
                self._camera = FakeCamera()

            def _open_camera(self):
                print("fake camera opened")

            def _get_frame(self):
                return object()

            def _person_detected(self, frame):
                return False

            def _wait_for_person(self, timeout):
                return True

            def _close_camera(self):
                print("fake camera closed")

        class FakeConnection:
            def __init__(self):
                self.sent = []

            def send(self, value):
                self.sent.append(value)

            def recv(self):
                return ("detect", 1.0)

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "child.log"
            conn = FakeConnection()
            with patch("Estun.estun_greeting_phase0.Phase0Greeting", FakeGreeting):
                _camera_process(conn, {}, "UGREEN Camera 1080P", "", -1, str(path))
            contents = path.read_text(encoding="utf-8")
        self.assertEqual(conn.sent, [("ready", 4), ("detected", True)])
        self.assertIn("fake camera opened", contents)
        self.assertIn("fake camera closed", contents)

    def test_camera_worker_reports_constructor_failure(self):
        class BrokenGreeting:
            def __init__(self, *args, **kwargs):
                raise RuntimeError("startup failed")

        class FakeConnection:
            def __init__(self):
                self.sent = []

            def send(self, value):
                self.sent.append(value)

            def close(self):
                pass

        conn = FakeConnection()
        with patch("Estun.estun_greeting_phase0.Phase0Greeting", BrokenGreeting):
            with self.assertRaisesRegex(RuntimeError, "startup failed"):
                _camera_process(conn, {}, "UGREEN Camera 1080P", "", -1, None)
        self.assertEqual(conn.sent, [("error", "RuntimeError: startup failed")])

    def test_stdout_stderr_are_mirrored_and_restored(self):
        out, err = io.StringIO(), io.StringIO()
        original_out, original_err = sys.stdout, sys.stderr
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "fusion.log"
            try:
                sys.stdout, sys.stderr = out, err
                with capture_run_log("phase0_a2d", path):
                    print("main-thread output")
                    print("error output", file=sys.stderr)
                self.assertIs(sys.stdout, out)
                self.assertIs(sys.stderr, err)
            finally:
                sys.stdout, sys.stderr = original_out, original_err
            contents = path.read_text(encoding="utf-8")
        self.assertIn("main-thread output", contents)
        self.assertIn("error output", contents)
        self.assertIn("main-thread output", out.getvalue())
        self.assertIn("error output", err.getvalue())

    def test_stability_spreads_report_raw_window(self):
        stabilizer = PoseStabilizer(window=3)
        self.assertEqual(stabilizer.spreads(), (None, None, None))
        stabilizer.add((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 10.0)
        stabilizer.add((0.003, 0.0, 0.0), (2.0, 0.0, 0.0), 13.0)
        pos, angle, theta = stabilizer.spreads()
        self.assertAlmostEqual(pos, 0.003)
        self.assertAlmostEqual(angle, 2.0)
        self.assertAlmostEqual(theta, 3.0)


if __name__ == "__main__":
    unittest.main()
