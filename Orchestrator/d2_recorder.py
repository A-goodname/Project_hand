# -*- coding: utf-8 -*-
"""D2 episode 记录器 (2026-08-17) — data_sm 集成版.

对方 D2 采集要求 §3/§4/§5 的记录层实现, 与 estun_d2_telemetry_sm.py 共用
schema estun-cpos-event-v1 (时钟/单位/状态标记纪律一致).

位姿事件发送格式 (实机侧 2026-08-17 确定):
    <PREFIX>|<x_m>,<y_m>,<z_m>,<a_deg>,<b_deg>,<c_deg>;
前缀 P0/P1/P2/P3/P4/Home, 按流程出现次序映射 D2 九事件:
    P0 → P0_ACTUAL
    P2 → P2_ARRIVED, P2_FINAL_ARRIVED        (第 1/2 次)
    P1 → P1_ARRIVED, P1_RETURNED             (第 1/2 次)
    P4 → P4_AFTER_LIFT, P4_AFTER_POUR        (第 1/2 次)
    P3 → P3_ARRIVED
    Home → HOME_ARRIVED

记录: robot_state.jsonl + state_transition.jsonl + action_receipt.jsonl,
会话结束 fsync + 双 manifest (SHA-256, expert_valid=false 保守待审计).
"""
import datetime
import hashlib
import json
import math
import os
import time

from Orchestrator.d2_streams import (  # noqa: E402
    write_force_unavailable, write_hand_state,
)
from Orchestrator.estun_d2_telemetry_sm import (  # noqa: E402
    D2_EVENT_SEQUENCE, SCHEMA_VERSION, CLOCK_ID,
)

# 前缀 → D2 事件 (按出现次序)
PREFIX_EVENT_MAP = {
    "P0": ("P0_ACTUAL",),
    "P2": ("P2_ARRIVED", "P2_FINAL_ARRIVED"),
    "P1": ("P1_ARRIVED", "P1_RETURNED"),
    "P4": ("P4_AFTER_LIFT", "P4_AFTER_POUR"),
    "P3": ("P3_ARRIVED",),
    "Home": ("HOME_ARRIVED",),
}

_UNIT_POS_MAX_M = 3.0
_UNIT_ANG_MAX_DEG = 360.0


def parse_pose_prefix(line, occurrences):
    """解析位姿帧 → (d2_event, xyz_m, abc_deg).

    现行格式 (实机 CRP 2026-08-17 下载版, CPosToStr strstring="P2[" endstring="]"):
      <PREFIX>[x,y,z,a,b,c]

    occurrences: dict {prefix: 已出现次数}, 决定第 N 次映射到哪个 D2 事件.
    失败抛 ValueError(原因).
    """
    if "[" not in line:
        raise ValueError(f"帧缺少 '[' 前缀分隔: {line!r}")
    prefix, _, payload = line.partition("[")
    payload = payload.split("]")[0].strip()
    names = PREFIX_EVENT_MAP.get(prefix.strip())
    if names is None:
        raise ValueError(f"未知前缀 {prefix!r} (P0/P1/P2/P3/P4/Home)")
    parts = payload.split(",")
    if len(parts) != 6:
        raise ValueError(f"字段数 {len(parts)} != 6")
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"非数值字段: {e}") from e
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("含 NaN/Infinity")
    xyz, abc = vals[:3], vals[3:]
    if any(abs(v) > _UNIT_POS_MAX_M for v in xyz):
        raise ValueError(f"位置疑似非 m 单位: {xyz}")
    if any(abs(v) > _UNIT_ANG_MAX_DEG for v in abc):
        raise ValueError(f"角度疑似非 deg 单位: {abc}")
    n = occurrences.get(prefix, 0)
    if n >= len(names):
        raise ValueError(f"{prefix} 出现 {n + 1} 次超出预期 (最多 {len(names)})")
    occurrences[prefix] = n + 1
    return names[n], xyz, abc


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class EpisodeRecorder:
    """单 episode 记录器: 位姿/转换/回执 JSONL + 会话结束双 manifest."""

    def __init__(self, out_dir, episode_id, scene_batch_id="canonical-a",
                 crp_ip="192.168.101.100"):
        self.episode_id = episode_id
        self.scene_batch_id = scene_batch_id
        self._ep_dir = os.path.join(out_dir, episode_id)
        raw_dir = os.path.join(self._ep_dir, "raw")
        sig_dir = os.path.join(self._ep_dir, "signals")
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(sig_dir, exist_ok=True)
        self._rs = open(os.path.join(sig_dir, "robot_state.jsonl"), "a",
                        encoding="utf-8")
        self._st = open(os.path.join(sig_dir, "state_transition.jsonl"), "a",
                        encoding="utf-8")
        self._ar = open(os.path.join(sig_dir, "action_receipt.jsonl"), "a",
                        encoding="utf-8")
        self._seen = []
        self._invalid_reasons = []
        self._occurrences = {}
        self._closed = False
        self.record_transition("session_start")
        # FORCE_TORQUE: 无真实力传感器 → UNAVAILABLE 健康记录 (每 episode 一次)
        write_force_unavailable(self._ep_dir)

    @property
    def ep_dir(self):
        return self._ep_dir

    def record_hand(self, **fields):
        """灵巧手状态事件 → signals/hand_state.jsonl (追加)."""
        write_hand_state(self._ep_dir, **fields)

    # ---- 写入 ----

    def _write(self, fh, rec):
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()

    def _ts(self, rec_type, **fields):
        return {"type": rec_type, "episode_id": self.episode_id,
                "pc_monotonic_ns": time.monotonic_ns(),
                "pc_wall_time_utc": datetime.datetime.now(
                    datetime.timezone.utc).isoformat(),
                "clock_id": CLOCK_ID, **fields}

    def record_pose(self, d2_event, xyz, abc, raw_payload):
        """位姿事件 → robot_state.jsonl (estun-cpos-event-v1 schema)."""
        self._seen.append(d2_event)
        self._write(self._rs, {
            "schema_version": SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "event": d2_event,
            "pc_monotonic_ns": time.monotonic_ns(),
            "pc_wall_time_utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "clock_id": CLOCK_ID,
            "pose_source": "data-sm-ingress",
            "controller_timestamp_available": False,
            "state_kind": "ESTUN_TCP_EVENT_SNAPSHOT",
            "tcp_frame": "base",
            "xyz_m": list(xyz),
            "abc_deg": list(abc),
            "joint_state_available": False,
            "raw_payload": raw_payload,
        })
        self.record_transition("event_received", event=d2_event,
                               xyz_m=list(xyz), abc_deg=list(abc))

    def record_transition(self, rec_type, **fields):
        self._write(self._st, self._ts(rec_type, **fields))

    def record_receipt(self, rec_type, **fields):
        self._write(self._ar, self._ts(rec_type, **fields))

    def note_invalid(self, reason):
        self._invalid_reasons.append(reason)

    # ---- 会话结束 ----

    def close(self):
        if self._closed:
            return
        self._closed = True
        seq_complete = self._seen == D2_EVENT_SEQUENCE
        self.record_transition("session_end", events_seen=self._seen,
                               sequence_complete=seq_complete,
                               invalid_reasons=self._invalid_reasons)
        for fh in (self._rs, self._st, self._ar):
            fh.flush()
            os.fsync(fh.fileno())
            fh.close()
        self._finalize_manifests(seq_complete)

    def _finalize_manifests(self, seq_complete):
        files = []
        for root, _dirs, names in os.walk(self._ep_dir):
            for n in sorted(names):
                p = os.path.join(root, n)
                files.append({
                    "artifact": ("ROBOT_STATE" if n == "robot_state.jsonl"
                                 else "STATE_TRANSITION" if n == "state_transition.jsonl"
                                 else "ACTION_RECEIPT" if n == "action_receipt.jsonl"
                                 else "RAW_PLACEHOLDER"),
                    "relpath": os.path.relpath(p, self._ep_dir),
                    "media_type": ("application/jsonl" if n.endswith(".jsonl")
                                   else "inode/directory"),
                    "sha256": _sha256_file(p),
                    "bytes": os.stat(p).st_size,
                    "clock_id": CLOCK_ID,
                })
        with open(os.path.join(self._ep_dir, "artifact_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"episode_id": self.episode_id, "files": files},
                      f, ensure_ascii=False, indent=2)
        with open(os.path.join(self._ep_dir, "episode_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "episode_id": self.episode_id,
                "scene_batch_id": self.scene_batch_id,
                "python_authorization_enabled": False,
                # 最终裁决 (2026-08-17): ESTUN 控制器唯一执行/停止权威,
                # Python 不发送实时运动授权/ACK/ticket, 仅记录.
                "outcome": ("SUCCESS" if seq_complete
                            else "ABORTED"),
                "automatic_motion_started": seq_complete,
                "phase_flags": {e: (e in self._seen)
                                for e in D2_EVENT_SEQUENCE},
                "events_seen": self._seen,
                "sequence_complete": seq_complete,
                "invalid_reasons": self._invalid_reasons,
                "expert_valid": False,   # 保守: 待来源/标定/回执审计
                "expert_valid_reason": ("待来源/标定/回执审计"
                                        if seq_complete
                                        else ("; ".join(self._invalid_reasons)
                                              or "事件序列不完整")),
                "source_evidence": {},
            }, f, ensure_ascii=False, indent=2)
        print(f"[D2-RECORDER] 会话结束: {self.episode_id} "
              f"seq_complete={seq_complete} events={len(self._seen)}")
