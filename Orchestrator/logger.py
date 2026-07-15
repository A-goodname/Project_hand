# -*- coding: utf-8 -*-
"""
轻量日志 — 同时输出到终端和文件
===============================
使用:
  from Orchestrator.logger import session_log
  session_log("初始化完成")
"""

import os
import sys
import time
import threading

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- 全局状态 ----
_log_file = None
_log_lock = threading.Lock()


def init_log(name: str = "orchestrator"):
    """初始化日志文件.

    Args:
        name: 日志文件名前缀.
    """
    global _log_file
    _log_dir = os.path.join(_project_root, "logs")
    os.makedirs(_log_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_log_dir, f"{name}_{ts}.log")
    _log_file = open(path, "w", encoding="utf-8")
    _write(f"[LOG] 日志文件: {path}")
    return path


def session_log(msg: str):
    """写一条日志 (终端 + 文件)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    _write(line)


def _write(line: str):
    """终端 print + 文件 write (线程安全)."""
    print(line, flush=True)
    with _log_lock:
        if _log_file is not None:
            try:
                _log_file.write(line + "\n")
                _log_file.flush()
            except Exception:
                pass


def close_log():
    """关闭日志文件."""
    global _log_file
    if _log_file is not None:
        _log_file.close()
        _log_file = None
