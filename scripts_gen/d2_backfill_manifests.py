# -*- coding: utf-8 -*-
"""为缺失 manifest 的 D2 episode 补生成双 manifest (2026-08-17).

背景: 主循环退出杀 daemon 会话线程 (修复前) 导致 recorder.close() 未执行,
      已产出的 episode 缺 episode_manifest.json / artifact_manifest.json.
      本脚本从现有 signals/ 与 raw/ 文件真实推导 (哈希实算, 不伪造).

用法:
    conda run -n Proj python scripts_gen/d2_backfill_manifests.py \
        logs/d2/tea-canonical-a-20260817T191919 [episode-dir ...]

输出: 每个 episode 目录下生成双 manifest; 已存在 manifest 的跳过 (不覆盖).
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Orchestrator.d2_recorder import (  # noqa: E402
    _artifact_type, _media_type, _sha256_file,
)
from Orchestrator.estun_d2_telemetry_sm import (  # noqa: E402
    D2_EVENT_SEQUENCE, CLOCK_ID,
)


def _load_events(ep_dir):
    """从 signals/robot_state.jsonl 读事件序列 (顺序/重复原样保留)."""
    path = os.path.join(ep_dir, "signals", "robot_state.jsonl")
    if not os.path.exists(path):
        return []
    events = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line).get("event"))
    return events


def _load_invalid_reasons(ep_dir):
    """从 state_transition.jsonl 的 protocol_error 行收集原因."""
    path = os.path.join(ep_dir, "signals", "state_transition.jsonl")
    reasons = []
    if not os.path.exists(path):
        return reasons
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("type") == "protocol_error" and row.get("reason"):
                reasons.append(row["reason"])
    return reasons


def _scan_files(ep_dir):
    files = []
    for root, _dirs, names in os.walk(ep_dir):
        for n in sorted(names):
            p = os.path.join(root, n)
            files.append({
                "artifact": _artifact_type(n),
                "relpath": os.path.relpath(p, ep_dir),
                "media_type": _media_type(n),
                "sha256": _sha256_file(p),
                "bytes": os.stat(p).st_size,
                "clock_id": CLOCK_ID,
            })
    return files


def backfill(ep_dir):
    ep_dir = os.path.abspath(ep_dir)
    ep_manifest = os.path.join(ep_dir, "episode_manifest.json")
    art_manifest = os.path.join(ep_dir, "artifact_manifest.json")
    if os.path.exists(ep_manifest) or os.path.exists(art_manifest):
        print(f"[SKIP] {os.path.basename(ep_dir)}: manifest 已存在")
        return

    events = _load_events(ep_dir)
    seen = [e for e in events if e in D2_EVENT_SEQUENCE]
    seq_complete = seen == list(D2_EVENT_SEQUENCE)
    invalid = _load_invalid_reasons(ep_dir)

    with open(ep_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "episode_id": os.path.basename(ep_dir),
            "scene_batch_id": ("canonical-a" if "canonical" in os.path.basename(ep_dir)
                               else "unknown"),
            "python_authorization_enabled": False,
            # 最终裁决 (2026-08-17): ESTUN 控制器唯一执行/停止权威,
            # Python 不发送实时运动授权/ACK/ticket, 仅记录.
            "outcome": ("SUCCESS" if seq_complete else "ABORTED"),
            "automatic_motion_started": seq_complete,
            "phase_flags": {e: (e in seen) for e in D2_EVENT_SEQUENCE},
            "events_seen": seen,
            "sequence_complete": seq_complete,
            "invalid_reasons": invalid,
            "expert_valid": False,   # 保守: 待来源/标定/回执审计
            "expert_valid_reason": ("待来源/标定/回执审计"
                                    if seq_complete
                                    else ("; ".join(invalid) or "事件序列不完整")),
            "source_evidence": {},
            "_note": "backfill 生成 (2026-08-17): 进程退出未走 recorder.close(), "
                     "由 d2_backfill_manifests.py 从现有文件真实推导",
        }, f, ensure_ascii=False, indent=2)

    # 先收集再写: 若在 open("w") 之后扫描会把刚截断的空文件自身扫进清单
    files = _scan_files(ep_dir)
    with open(art_manifest, "w", encoding="utf-8") as f:
        json.dump({
            "episode_id": os.path.basename(ep_dir),
            "files": files,
        }, f, ensure_ascii=False, indent=2)

    print(f"[OK] {os.path.basename(ep_dir)}: seq_complete={seq_complete} "
          f"events={len(seen)} files={len(_scan_files(ep_dir))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    for d in sys.argv[1:]:
        if os.path.isdir(d):
            backfill(d)
        else:
            print(f"[WARN] 不是目录, 跳过: {d}")
