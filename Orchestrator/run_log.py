"""Mirror a robot run's stdout/stderr to per-process files without changing stdin."""
from __future__ import annotations

import sys
import threading
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class _Tee:
    def __init__(self, console, file, lock):
        self.console = console
        self.file = file
        self.lock = lock

    def write(self, data):
        with self.lock:
            self.console.write(data)
            self.file.write(data)
            self.file.flush()
        return len(data)

    def flush(self):
        with self.lock:
            self.console.flush()
            self.file.flush()

    def isatty(self):
        return self.console.isatty()

    def __getattr__(self, name):
        return getattr(self.console, name)


@contextmanager
def capture_run_log(name: str, path: Path | None = None):
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    if path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        path = log_dir / f"{name}_{stamp}.log"
    path = Path(path)
    with path.open("a", encoding="utf-8", buffering=1) as file:
        original_out, original_err = sys.stdout, sys.stderr
        lock = threading.RLock()
        sys.stdout = _Tee(original_out, file, lock)
        sys.stderr = _Tee(original_err, file, lock)
        try:
            print(f"[LOG] 本次运行日志: {path}", flush=True)
            yield path
        except BaseException:
            traceback.print_exc()
            raise
        finally:
            sys.stdout, sys.stderr = original_out, original_err
