# -*- coding: utf-8 -*-
"""零运动稳定性验证 (对方 2026-08-21 受限开工令 §1).

固定水壶/水杯/机械臂, 零运动; 仅验证新版 skip_frames=3 / stabilization.window=10 /
ROT_THETA_WIN=10 的检测输出. 物理摆位 0°/+30° 各 2 次独立检测窗口 (共 4 窗口).

稳定机制与 data_sm 一致: 复用 PoseStabilizer — 窗口内每帧 (pos, euler, theta),
稳定判定同时约束 pos/angle/θ 极差 (θ < ROT_THETA_SPREAD_MAX 5°), 恰满 10 帧才
输出窗口; 丢弃开头不稳定帧 (不满窗/不一致不产出). 每窗口 = 一次稳定 (10 个 θ
原始样本), 输出 θ 中位数 → band.

每有效样本记录: theta / band / ratio / d / rd / handle_side / kettle bbox /
贴纸中心像素 / config.yaml SHA-256 / 源码 git_rev.
通过条件 (4 窗口全满足): 0°×2 均 band=0, +30°×2 均 band=+30, 每窗口完整 10 样本.

用法 (conda Proj, 实机, 操作员摆位):
  python scripts_gen/rotation_verify_zero_motion.py
按键: 摆好位后按 's' 开始采集当前窗口 | 'q' 退出
"""
import hashlib
import json
import os
import subprocess
import sys
import time
import argparse
import collections

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import yaml

from Orchestrator.handle_cup_detector import HandleCupDetector
from Orchestrator.estun_handle_data_sm import PoseStabilizer, ROT_THETA_SPREAD_MAX

WINDOW_SAMPLES = 10            # 每窗口 θ 原始样本数 (对齐 ROT_THETA_WIN=10)
WINDOWS = [                    # (目标摆位角, 窗口 id)
    (0, "W01"), (0, "W02"),
    (30, "W03"), (30, "W04"),
]


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_rev():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_project_root,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def resolve_rot_band(th_deg, band0, band15_hi):
    """θ → 档位 (与 data_sm._resolve_rot_band 一致, 纯 θ 区间)."""
    if th_deg > band15_hi:
        return 30
    if th_deg > band0:
        return 15
    if th_deg < -40.0:
        return None
    if th_deg < -band15_hi:
        return -30
    if th_deg < -band0:
        return -15
    return 0


def main():
    ap = argparse.ArgumentParser(description="零运动稳定性验证 (4 窗口 × 10 θ 样本, data_sm 稳定机制)")
    ap.add_argument("--out", default=None, help="输出 JSONL (默认 logs/rotation_verify_zero_motion/...)")
    args = ap.parse_args()

    orch_path = os.path.join(_project_root, "Orchestrator", "config.yaml")
    orch_cfg = yaml.safe_load(open(orch_path, "r", encoding="utf-8"))
    cfg_sha = _sha256_file(orch_path)
    git_rev = _git_rev()

    rot_cfg = orch_cfg.get("handle_rotation", {})
    band0 = float(rot_cfg.get("band0_halfwidth", 8.0))
    band15_hi = float(rot_cfg.get("band15_hi", 22.0))
    stab_cfg = orch_cfg.get("stabilization", {})
    pos_thresh = float(stab_cfg.get("pos_threshold", 0.01))
    angle_thresh = float(stab_cfg.get("angle_threshold", 5.0))

    # 相机 serial (参考 theta_stability_log)
    try:
        info = yaml.safe_load(open(os.path.join(
            _project_root, "RealSense_R", "D435i_info.yaml"), "r", encoding="utf-8"))
        serial = str(info["camera"]["serial"])
    except Exception:
        serial = ""

    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("[INFO] 创建 HandleCupDetector ...")
    detector = HandleCupDetector(orch_config=orch_cfg, serial=serial, device=dev)

    # data_sm 稳定机制: window=10 (与 config stabilization.window 对齐), θ 极差 < ROT_THETA_SPREAD_MAX
    stab = PoseStabilizer(window=int(stab_cfg.get("window", 10)),
                          pos_thresh=pos_thresh, angle_thresh=angle_thresh,
                          theta_thresh=ROT_THETA_SPREAD_MAX)
    # 完整检测字段缓存 (与稳定器同步入窗; 稳定后逐样本输出全字段, 对方 §1 要求)
    sample_buf = collections.deque(maxlen=WINDOW_SAMPLES)

    out_dir = os.path.join(_project_root, "logs", "rotation_verify_zero_motion")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        out_dir, f"zero_motion_{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")

    print(f"[INFO] window={stab.window} theta_thresh={ROT_THETA_SPREAD_MAX}°")
    print(f"[INFO] config_sha256={cfg_sha}")
    print(f"[INFO] git_rev={git_rev}")
    print(f"[INFO] out -> {out_path}")
    print(f"[INFO] 操作: 摆好 0° 后按 s → 稳定 10 帧自动出窗口 → 同摆位再按 s → "
          f"换 +30° 按 s → ... 共 4 窗口; q 退出")

    results = []        # 每窗口: {window_id, target, theta_med, band, samples}
    win_i = 0           # 0..3
    phase = "idle"      # idle(等 s) / collecting(稳定累积)
    cur_angle = None
    frame_counter = 0
    try:
        while win_i < len(WINDOWS):
            color, _depth_img, depth_cmap, depth_frame = detector.get_frames()
            if color is None or depth_frame is None:
                time.sleep(0.03)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            frame_counter += 1

            result = detector.detect(color, depth_frame)
            h = result["handle"]

            # ---- data_sm 稳定机制: 有效帧 (pos+theta 齐全) 入窗, 恰满 10 帧且一致才出窗口 ----
            if phase == "collecting" and h["pos_cam"] is not None \
                    and h.get("theta_deg") is not None:
                # 完整检测字段缓存 (与稳定器同步; 稳定后逐样本输出, 对方 §1 每有效样本全字段)
                sample_buf.append({
                    "theta_deg": float(h["theta_deg"]), "band": h.get("band"),
                    "ratio": h.get("ratio"), "d": float(h["pos_cam"][2]),
                    "rd": h.get("rd"), "handle_side": h.get("handle_side"),
                    "kettle_bbox": h.get("kettle_bbox"),
                    "tag_u": h.get("tag_u"), "tag_v": h.get("tag_v"),
                })
                stab.add(tuple(h["pos_cam"]), tuple(h["euler_cam"]),
                         float(h["theta_deg"]))
                if stab.is_stable():
                    ths = stab.get_theta_samples()
                    th_med = float(np.median(ths))
                    band = resolve_rot_band(th_med, band0, band15_hi)
                    # ---- 逐样本记录 (每有效样本: theta/band/ratio/d/rd/handle_side/bbox/贴纸中心/config SHA/git rev) ----
                    for i, s in enumerate(sample_buf, 1):
                        rec = {
                            "type": "sample", "window_id": WINDOWS[win_i][1],
                            "idx": i, "target_angle_deg": WINDOWS[win_i][0],
                            "config_sha256": cfg_sha, "git_rev": git_rev,
                            "ts": time.time(),
                            **s,
                        }
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # ---- 窗口汇总 ----
                    win_rec = {
                        "type": "window", "window_id": WINDOWS[win_i][1],
                        "target_angle_deg": WINDOWS[win_i][0],
                        "theta_samples": [round(float(t), 4) for t in ths],
                        "theta_med": round(th_med, 4), "band": band,
                        "config_sha256": cfg_sha, "git_rev": git_rev,
                        "ts": time.time(),
                    }
                    results.append(win_rec)
                    out_f.write(json.dumps(win_rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                    ok = (band == 0 if WINDOWS[win_i][0] == 0 else band == 30)
                    print(f"[窗口 {WINDOWS[win_i][1]} 稳定] target={WINDOWS[win_i][0]}° "
                          f"θ_med={th_med:.2f} band={band} → {'PASS' if ok else 'FAIL'}", flush=True)
                    win_i += 1
                    stab.reset()
                    stab.capture_generation()
                    sample_buf.clear()
                    phase = "idle"
                    if win_i < len(WINDOWS):
                        nxt = WINDOWS[win_i][0]
                        if nxt != cur_angle:
                            print(f"\n>>> 请换摆位到 {nxt}° (窗口 {WINDOWS[win_i][1]}), 摆好按 s", flush=True)
                        else:
                            print(f"\n>>> 同一摆位 {nxt}° 再按 s (窗口 {WINDOWS[win_i][1]})", flush=True)

            # ---- 显示 ----
            display = detector.visualize(color)
            status = (f"采集窗口 {WINDOWS[win_i][1]}: 累积 {len(stab.get_theta_samples())}/10"
                      if phase == "collecting" else
                      f"等 s 开始 (窗口 {WINDOWS[win_i][1]}/4)")
            cv2.putText(display, status, (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "s=start q=quit",
                        (display.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Zero-Motion Verify", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and phase == "idle" and win_i < len(WINDOWS):
                cur_angle = WINDOWS[win_i][0]
                phase = "collecting"
                stab.reset()
                stab.capture_generation()
                print(f"\n>>> 采集窗口 {WINDOWS[win_i][1]}: {cur_angle}° 保持不动, 等稳定 10 帧...",
                      flush=True)
    finally:
        out_f.close()

    # ---- 汇总 + 通过判定 ----
    print("\n================ 零运动稳定性验证结果 ================")
    all_pass = True
    for r in results:
        exp = 0 if r["target_angle_deg"] == 0 else 30
        ok = r["band"] == exp and len(r["theta_samples"]) == WINDOW_SAMPLES
        all_pass &= ok
        print(f"  {r['window_id']} target={r['target_angle_deg']:>3}° "
              f"θ_med={r['theta_med']:7.2f} band={r['band']} "
              f"samples={len(r['theta_samples'])} → {'PASS' if ok else 'FAIL'}")
    if len(results) < len(WINDOWS):
        print(f"  [WARN] 仅完成 {len(results)}/{len(WINDOWS)} 窗口")
        all_pass = False
    print(f"判定: {'PASS (可继续 18 回合)' if all_pass else 'FAIL (暂停实机, 保留资料回传)'}")
    print(f"输出: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
