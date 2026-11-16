"""Receive numbered ESTUN pose samples over a separate, read-only test socket."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import socket
import time

SAMPLE_COUNT = 10
PORT = 8810
FRAME = re.compile(rb"([PJ])\|([0-9]{4})\|\[([^\[\]]+)\]")
MAX_FRAME = 512


def receive(conn, output, with_joints=False):
    buffer = b""
    ids = {"P": [], "J": []}
    rejected = 0
    issue = None

    def record(**fields):
        output.write(json.dumps(
            {"pc_recv_utc_ns": utc_ns, "pc_recv_monotonic_ns": monotonic_ns, **fields},
            allow_nan=False,
        ) + "\n")
        output.flush()

    while True:
        try:
            chunk = conn.recv(4096)
        except socket.timeout:
            issue = "socket timeout"
            break
        except OSError as exc:
            issue = f"socket error: {exc}"
            break
        if not chunk:
            break
        utc_ns = time.time_ns()
        monotonic_ns = time.monotonic_ns()
        buffer += chunk
        while b";" in buffer:
            raw, buffer = buffer.split(b";", 1)
            match = FRAME.fullmatch(raw.strip()) if len(raw) <= MAX_FRAME else None
            try:
                if not match:
                    raise ValueError("invalid frame")
                kind = match[1].decode("ascii")
                if kind == "J" and not with_joints:
                    raise ValueError("joint frame requires --with-joints")
                sample_id = int(match[2])
                pose = [float(value) for value in match[3].split(b",")]
                if len(pose) != 6 or not all(math.isfinite(value) for value in pose):
                    raise ValueError("expected six finite pose values")
            except ValueError as exc:
                rejected += 1
                record(event="invalid_frame", raw_hex=(raw + b";").hex(), error=str(exc))
                continue
            ids[kind].append(sample_id)
            fields = ({"event": "pose", "tcp_xyz_m": pose[:3], "tcp_abc_deg": pose[3:]}
                      if kind == "P" else {"event": "joints", "joint_deg": pose})
            record(sample_id=sample_id, raw_hex=(raw + b";").hex(),
                   controller_sample_time=None, **fields)
            print(f"[{fields['event'].upper()}] {sample_id:04d}: {pose}")
        if len(buffer) > MAX_FRAME:
            issue = "unterminated frame exceeds size limit"
            break
    if buffer.strip():
        rejected += 1
        record(event="partial_frame", raw_hex=buffer.hex())
    expected = list(range(1, SAMPLE_COUNT + 1))
    channels = ("P", "J") if with_joints else ("P",)
    complete = issue is None and rejected == 0 and all(ids[kind] == expected for kind in channels)
    missing = {kind: sorted(set(expected) - set(ids[kind])) for kind in channels}
    print(f"[RESULT] complete={complete} ids={ids} missing={missing} rejected={rejected} issue={issue}")
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.101.96")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-joints", action="store_true",
                        help="Require a six-axis joint sample for each TCP sample")
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%fZ")
    path = args.output or Path("logs") / f"estun_pose_test_{stamp}.jsonl"
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            server.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        server.bind((args.host, args.port))
        server.listen(1)
        print(f"[LISTEN] {args.host}:{args.port}")
        conn, peer = server.accept()
        with conn:
            conn.settimeout(8.0)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8") as output:
                print(f"[CLIENT] {peer[0]} | {path.resolve()}")
                return 0 if receive(conn, output, with_joints=args.with_joints) else 1


if __name__ == "__main__":
    raise SystemExit(main())
