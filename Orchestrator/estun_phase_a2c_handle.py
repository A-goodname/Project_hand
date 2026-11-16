"""Run the A2D kettle and pouring sequence without the cup pickup or Phase D."""
from __future__ import annotations

import argparse
from pathlib import Path
import socket
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Omnihand.omnihand_gesture import Gesture, OmniHandGesture
from Orchestrator.estun_handle_cup_sm import HandleCupSM
from Orchestrator.run_log import capture_run_log


class KettlePour(HandleCupSM):
    def __init__(self):
        super().__init__()
        self._no_preview = True
        self._hand = None
        self.completed = False

    def _phase_c(self, conn, cup_pose_data, robot_xyz, robot_ABC):
        """Use the original one-way GRIPDONE and RESET commands."""
        conn.settimeout(2.0)
        reset_received = False
        while not self._stop_event.is_set():
            try:
                raw = conn.recv(1024)
            except socket.timeout:
                continue
            if not raw:
                self.completed = reset_received
                return
            command = raw.decode("ascii").strip()
            if command == "GRIPDONE":
                self._hand.do_gesture(Gesture.GRIPDONE, delay=0.5)
            elif command == "RESET":
                self._hand.do_gesture(Gesture.RESET, delay=0.3)
                reset_received = True
            else:
                print(f"[A2C] Ignoring {command!r}")
                continue
            print(f"[A2C] {command}: hand driver returned")

    def run_once(self):
        self._load_configs()
        self._init_stabilizer_and_calib()
        self._init_detector()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            listener.bind((self._host, self._port))
            self._hand = OmniHandGesture(port=self._hand_port)
            self._hand.open()
            if not self._hand.enable():
                raise RuntimeError("Hand refused enable")
            self._hand.do_gesture(Gesture.RESET, delay=0.3)
            listener.listen(1)
            print(f"[A2C] Listening {self._host}:{self._port}; start dedicated CRP at detection pose")
            conn, peer = listener.accept()

        with conn:
            worker = threading.Thread(target=self._handle_session, args=(conn, peer))
            worker.start()
            try:
                self.run()
            finally:
                if worker.is_alive():
                    try:
                        conn.shutdown(socket.SHUT_RDWR)
                    except OSError:
                        pass
                worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print startup sequence, no devices")
    args = parser.parse_args()
    if args.dry_run:
        print("Dry run: A pose -> B stable handle and cup -> C grasp kettle, pour, return and RESET")
        return 0
    with capture_run_log("phase_a2c_handle"):
        sm = KettlePour()
        try:
            sm.run_once()
        except (Exception, KeyboardInterrupt) as exc:
            print(f"[A2C] Incomplete: {type(exc).__name__}: {exc}")
            return 1
        finally:
            sm._close_camera_a()
            if sm._hand is not None:
                sm._hand.serial.close()
        if not sm.completed:
            print("[A2C] Closed before RESET completed")
            return 1
        print("[A2C] RESET completed and controller disconnected")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
