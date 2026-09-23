"""Phase0 four-step socket contract without robot, camera or hand movement."""
import unittest
from unittest import mock

from Estun.estun_greeting_phase0 import Phase0Greeting, _load_estun_config


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

    def close(self):
        pass


class Phase0ProtocolNoMotionTest(unittest.TestCase):
    def setUp(self):
        self.service = Phase0Greeting(_load_estun_config(), mock=True, preview=False)

    def test_four_steps(self):
        conn = FakeConnection([b"pose", b"ZHAOSHOU1|pose", b"ZHAOSHOU2|pose", b"WOSHOU|pose"])
        with mock.patch.object(self.service, "_wait_for_person", return_value=True), \
             mock.patch.object(self.service, "_execute_wave"), \
             mock.patch.object(self.service, "_execute_handshake", return_value=True):
            self.service._handle_session(conn)
        self.assertEqual(len(conn.sent), 4)
        self.assertEqual(conn.sent[-1], b"Home\n")

    def test_no_person_no_second_motion_reply(self):
        conn = FakeConnection([b"pose", b"ZHAOSHOU1|pose"])
        with mock.patch.object(self.service, "_wait_for_person", return_value=False):
            self.service._handle_session(conn)
        self.assertEqual(len(conn.sent), 1)

    def test_no_handshake_no_home_reply(self):
        conn = FakeConnection([b"pose", b"ZHAOSHOU1|pose", b"ZHAOSHOU2|pose", b"WOSHOU|pose"])
        with mock.patch.object(self.service, "_wait_for_person", return_value=True), \
             mock.patch.object(self.service, "_execute_wave"), \
             mock.patch.object(self.service, "_execute_handshake", return_value=False):
            self.service._handle_session(conn)
        self.assertEqual(len(conn.sent), 3)

    def test_fusion_camera_process_interface(self):
        worker = mock.Mock()
        worker.wait_for_person.return_value = True
        service = Phase0Greeting(_load_estun_config(), mock=True, preview=False,
                                 camera_worker=worker)
        service.start()
        worker.start.assert_called_once()
        self.assertTrue(service._wait_for_person(timeout=1.0))
        worker.wait_for_person.assert_called_once_with(1.0)
        service._close_camera()
        worker.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
