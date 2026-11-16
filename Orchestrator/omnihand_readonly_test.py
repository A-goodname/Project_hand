"""Record OmniHand position, enable state and error feedback without commanding motion."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Omnihand.omnihand_usb_serial import OmniHandSerial


def read_feedback(hand, record):
    failures = 0

    def sample(field, query, index=None):
        nonlocal failures
        started = time.monotonic_ns()
        value = None
        error = None
        try:
            value = query()
            if field == "positions" and len(value) != 10:
                raise ValueError("expected ten axis positions")
        except Exception as exc:
            value = None
            failures += 1
            error = f"{type(exc).__name__}: {exc}"
        record(event="sample", field=field, index=index, value=value, error=error,
               pc_request_start_monotonic_ns=started, device_sample_time=None)

    sample("model_version", hand.query_model_version)
    sample("product_serial", hand.read_serial_number)
    for index in range(30):
        start = time.monotonic()
        sample("positions", hand.read_all_positions, index)
        if index % 5 == 0:
            sample("enable", hand.query_enable, index)
            sample("error", hand.read_error, index)
        if index < 29:
            time.sleep(max(0.0, 0.1 - (time.monotonic() - start)))
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM4")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = args.output or ROOT / "logs" / f"omnihand_readonly_{stamp}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        def record(**fields):
            output.write(json.dumps({"pc_utc_ns": time.time_ns(),
                                     "pc_monotonic_ns": time.monotonic_ns(), **fields}) + "\n")
            output.flush()

        class RecordedHand(OmniHandSerial):
            def _log(self, message):
                record(event="serial", message=message)

        hand = RecordedHand(port=args.port, verbose=True)
        try:
            hand.open()
            failures = read_feedback(hand, record)
        finally:
            hand.close()
    print(f"[RESULT] failed_queries={failures} | {path.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
