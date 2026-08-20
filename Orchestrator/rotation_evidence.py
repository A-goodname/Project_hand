# -*- coding: utf-8 -*-
"""18 回合旋转判档抓取证据集 JSONL 记录 (对方 2026-08-20 addendum §3).

两级记录 (每行一条 JSON, type 区分):
  type="round"  : 每回合一条 — round_id / target_angle_deg / config_sha256 /
                  5 个 θ 原始样本 theta_samples / theta_med / band / P1/P2/C /
                  钳位前后 Z / 实际 CPOS / 动作回执 / 抓取结果或拒绝原因 / 物理限位
  type="sample" : 每个有效检测帧一条 — ratio / d / rd / handle_side /
                  kettle bbox / 贴纸中心像素 (tag_u/tag_v)

输出: <out_dir>/<round_id>.jsonl
"""
import hashlib
import json
import os
import threading


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o):
    if hasattr(o, "item"):      # numpy 标量
        return o.item()
    if isinstance(o, (tuple, list)):
        return list(o)
    return str(o)


class RotationEvidence:
    """回合证据集写入器 (线程安全, 逐行 flush)."""

    def __init__(self, out_dir, round_id, target_angle_deg=None,
                 config_path=None, physical_limit=True):
        self._lock = threading.Lock()
        self._round_id = round_id
        self._target_angle = (float(target_angle_deg)
                              if target_angle_deg is not None else None)
        self._config_sha = (_sha256_file(config_path)
                            if config_path and os.path.exists(config_path) else None)
        self._physical_limit = bool(physical_limit)
        os.makedirs(out_dir, exist_ok=True)
        self._path = os.path.join(out_dir, f"{round_id}.jsonl")
        self._fh = open(self._path, "w", encoding="utf-8")
        self._closed = False

    @property
    def path(self):
        return self._path

    def write(self, type_, **fields):
        """写一行记录. type_ ∈ {"round", "sample"}; 通用字段自动带上."""
        rec = {"type": type_, "round_id": self._round_id}
        if type_ == "round":
            rec["target_angle_deg"] = self._target_angle
            rec["config_sha256"] = self._config_sha
            rec["physical_limit"] = self._physical_limit
        rec.update(fields)
        with self._lock:
            if not self._closed:
                self._fh.write(json.dumps(rec, ensure_ascii=False,
                                          default=_json_default) + "\n")
                self._fh.flush()

    def write_sample(self, **fields):
        """每有效检测帧一条 (ratio/d/rd/handle_side/bbox/贴纸中心)."""
        self.write("sample", **fields)

    def write_round(self, **fields):
        """每回合一条汇总 (θ 样本/中位数/band/位姿/结果)."""
        self.write("round", **fields)

    def close(self):
        with self._lock:
            if not self._closed:
                self._fh.close()
                self._closed = True
