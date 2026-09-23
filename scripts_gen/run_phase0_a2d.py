"""Single-process Windows fusion for Phase0 + PhaseA2D; no voice or RPS.

The same Estun.utils_shared hand singleton remains open and enabled across both
phases. A2D listens before Phase0 starts; D435i/YOLO can be prewarmed early.
"""
from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Estun.estun_greeting_phase0 import Phase0Greeting, _load_estun_config
from Estun.phase0_camera_worker import WelcomeCameraWorker
from Estun.utils_shared import disable_hand
from Orchestrator.estun_handle_cup_sm import HandleCupSM


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase0 + PhaseA2D in one Python process")
    parser.add_argument("--no-preview", action="store_true", help="Use terminal c/a for PhaseD")
    parser.add_argument("--no-prewarm", action="store_true", help="Open D435i and models only on A2D connection")
    parser.add_argument("--dry-run", action="store_true", help="Print startup sequence, no devices")
    args = parser.parse_args()
    if args.dry_run:
        print("1. Bind A2D 8080 (deferred devices)")
        print("2. Preopen PhaseD UGREEN/model, optionally prewarm D435i/Handle+Cup; no PhaseD inference")
        print("3. Open OmniHand COM5 once; welcome camera/model in own process; bind 8600")
        print("4. Phase0 completes; A2D reuses the same OmniHand connection")
        print("5. PhaseD keyboard C/Enter=CONTINUE, A=ABORT; no voice/RPS")
        return 0

    sm = HandleCupSM()
    sm._defer_devices = True
    sm._no_preview = args.no_preview
    sm._phase0_handoff = threading.Event()
    estun_cfg = _load_estun_config()
    camera_worker = WelcomeCameraWorker(estun_cfg)
    phase0 = Phase0Greeting(
        estun_cfg,
        camera_path="",
        camera_name="UGREEN Camera 1080P",
        preview=False,  # Phase0 socket runs in a worker; OpenCV GUI stays on main thread.
        camera_worker=camera_worker,
    )
    phase0_result = {"completed": False, "error": None}
    worker = None
    try:
        sm.start()  # 8080 must listen before the controller completes Phase0.
        sm._prepare_handover_vision()  # PhaseD camera/model ready, no inference yet.
        if not args.no_prewarm:
            sm._init_detector()  # D435i + models, but Handle/Cup inference is paused.
            sm._detect_active = False
        phase0.start()  # opens shared COM5; camera+YOLO live only in child process

        def serve_phase0():
            try:
                phase0_result["completed"] = bool(phase0.serve())
            except Exception as exc:  # noqa: BLE001
                phase0_result["error"] = exc
                print(f"[FUSION] Phase0 failed: {exc}")
            finally:
                phase0.stop()  # RESET only; shared serial remains open for A2D.
                sm._phase0_handoff.set()
                if not phase0_result["completed"]:
                    sm._stop_event.set()

        print("[FUSION] Python 8080 prepared; Phase0 8600 starts next.")
        print("[FUSION] When both listeners are ready, run PhaseAll-NoRPS.crp on the controller.")
        worker = threading.Thread(target=serve_phase0, name="phase0-socket", daemon=True)
        worker.start()
        sm.run()  # Main thread owns all OpenCV windows and PhaseD visual updates.
        worker.join(timeout=2)
        if phase0_result["error"] is not None or not phase0_result["completed"]:
            return 2
        if sm._handover_choice != "CONTINUE":
            print("[FUSION] ABORT or unfinished handover: retain current grip; inspect hardware manually.")
            return 3
        disable_hand(skip_reset=True)
        return 0
    except KeyboardInterrupt:
        print("[FUSION] Interrupted; inspect robot and hand before restart.")
        return 130
    finally:
        sm._stop_event.set()
        phase0._stop_event.set()
        if worker is not None and worker.is_alive():
            worker.join(timeout=2)


if __name__ == "__main__":
    raise SystemExit(main())
