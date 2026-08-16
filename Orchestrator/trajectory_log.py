# -*- coding: utf-8 -*-
"""轨迹日志 (audit 修复 4, 2026-08-16): 逐帧可审计的 JSONL 轨迹记录.

记录内容 (每观测一行):
  - monotonic_ns (逐帧单调时钟, 唯一可排序)
  - observation_id (观测 ID, 屏障新鲜度校验引用)
  - 位姿: pos_cam / euler_cam_deg (检测成功=动态相机系 ZYX; 失败=config 回退)
  - 旋转矩阵: rot_cam (3x3, B2 动态链平滑后)
  - quat_cam (w,x,y,z)
  - source: "dynamic" | "fallback"
  - confidence (yolo 置信度) / depth_points (深度有效点数)
  - stabilizer: 稳定器判定 (verdict stable/unstable + geodesic_deg + samples)

配套会话记录: session_start (run_id/lease_epoch/标定指纹/T_cam2ee/P0) 与
session_end; 事件记录 log_event (屏障 P2/P1/GRIPDONE/拒绝等).

纪律:
  - 每行独立 JSON, 追加写 + flush (进程崩溃不丢已记行, 取证用)
  - 非有限数值 → null (JSON 无 NaN; 与协议 NOT_FINITE 纪律一致)
  - 线程安全: 内部锁保护写行 (生产在检测线程/主线程并发调用)
  - 不含凭证/密钥/模型权重; 序列号由调用方脱敏
"""
import json
import math
import os
import threading
import time

import numpy as np


def _jsonable(value):
    """ndarray/tuple → list, 非有限 float → None (JSON 无 NaN/Inf)."""
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return f if math.isfinite(f) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


class TrajectoryLogger:
    """JSONL 轨迹日志: 会话开始 → 逐帧观测/事件 → 会话结束."""

    def __init__(self, path: str, *, run_id: str, lease_epoch: int,
                 calib_fingerprint: str | None = None,
                 t_cam2ee=None, p0_pose=None) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._path = path
        self._fh = open(path, "a", encoding="utf-8")
        self._lock = threading.RLock()   # 可重入: close 内调 _write
        self._closed = False
        self._write({
            "type": "session_start",
            "monotonic_ns": time.monotonic_ns(),
            "run_id": run_id,
            "lease_epoch": lease_epoch,
            "calib_fingerprint": calib_fingerprint,
            "t_cam2ee": _jsonable(t_cam2ee),
            "p0_pose": _jsonable(p0_pose),
        })

    # ---- 写入 ----

    def _write(self, record: dict) -> None:
        with self._lock:
            if self._closed:
                return
            line = json.dumps(record, ensure_ascii=False, allow_nan=False)
            self._fh.write(line + "\n")
            self._fh.flush()

    def log_observation(self, *, monotonic_ns: int, observation_id: int,
                        pos_cam, euler_cam_deg, rot_cam=None, quat_cam=None,
                        source: str = "fallback",
                        confidence: float | None = None,
                        depth_points: int | None = None,
                        stabilizer: dict | None = None) -> None:
        """逐帧观测记录 (audit 修复 4 字段全量)."""
        self._write({
            "type": "obs",
            "monotonic_ns": _jsonable(monotonic_ns),
            "observation_id": _jsonable(observation_id),
            "pos_cam": _jsonable(pos_cam),
            "euler_cam_deg": _jsonable(euler_cam_deg),
            "rot_cam": _jsonable(rot_cam),
            "quat_cam": _jsonable(quat_cam),
            "source": source,
            "confidence": _jsonable(confidence),
            "depth_points": _jsonable(depth_points),
            "stabilizer": _jsonable(stabilizer),
        })

    def log_event(self, event: str, **detail) -> None:
        """协议/屏障事件 (P2_ARRIVED / P1_ARRIVED / GRIPDONE / 拒绝码 / 状态迁移)."""
        self._write({
            "type": "event",
            "monotonic_ns": time.monotonic_ns(),
            "event": event,
            "detail": _jsonable(detail),
        })

    def log_grasp(self, *, ticket_id: str, pose_base, pose_ee=None,
                  observation_id: int | None = None) -> None:
        """抓取授权记录 (grasp ticket 一次性消费审计)."""
        self._write({
            "type": "grasp",
            "monotonic_ns": time.monotonic_ns(),
            "ticket_id": ticket_id,
            "observation_id": _jsonable(observation_id),
            "pose_base": _jsonable(pose_base),
            "pose_ee": _jsonable(pose_ee),
        })

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._write({"type": "session_end",
                         "monotonic_ns": time.monotonic_ns()})
            self._fh.close()
            self._closed = True

    # ---- 只读 ----

    @property
    def path(self) -> str:
        return self._path

    def read_back(self) -> list[dict]:
        """读回全部记录 (取证/测试用)."""
        with self._lock:
            records = []
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            return records
