# -*- coding: utf-8 -*-
"""零运动稳定性验证 (对方 2026-08-21 受限开工令 §1).

固定水壶/水杯/机械臂, 零运动; 仅验证新版 skip_frames=3 / stabilization.window=10 /
ROT_THETA_WIN=10 的检测输出. 物理摆位 0°/+30° 各 2 次独立检测窗口 (共 4 窗口),
每窗口严格保留 10 个 theta 原始样本; 每个有效样本记录:
theta / band / ratio / d / rd / handle_side / kettle bbox / 贴纸中心像素 /
config.yaml SHA-256 / 源码 revision.

复用 HandleCupDetector (不改检测器本体) + theta_stability_log 的初始化模式.
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

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import yaml

from Orchestrator.handle_cup_detector import HandleCupDetector

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


def main():
    ap = argparse.ArgumentParser(description="零运动稳定性验证 (4 窗口 × 10 θ 样本)")
    ap.add_argument("--out", default=None, help="输出 JSONL (默认 logs/rotation_verify_zero_motion/...)")
    args = ap.parse_args()

    orch_path = os.path.join(_project_root, "Orchestrator", "config.yaml")
    orch_cfg = yaml.safe_load(open(orch_path, "r", encoding="utf-8"))
    cfg_sha = _sha256_file(orch_path)
    git_rev = _git_rev()

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

    out_dir = os.path.join(_project_root, "logs", "rotation_verify_zero_motion")
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(
        out_dir, f"zero_motion_{time.strftime('%Y%m%dT%H%M%S')}.jsonl")
    out_f = open(out_path, "w", encoding="utf-8")

    print(f"[INFO] config_sha256={cfg_sha}")
    print(f"[INFO] git_rev={git_rev}")
    print(f"[INFO] out -> {out_path}")
    print(f"[INFO] 操作: 摆好 0° 后按 s 采集窗口1 → 换 +30° 按 s → ... 共 4 窗口; q 退出")

    results = []   # 每窗口: {window_id, target, band, theta_samples, samples}
    frame_counter = 0
    window_i = 0
    collecting = False
    cur = None    # 当前窗口累积
    last_detect = None
    try:
        while window_i < len(WINDOWS):
            color, _depth_img, depth_cmap, depth_frame = detector.get_frames()
            if color is None or depth_frame is None:
                time.sleep(0.03)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            frame_counter += 1

            # 检测 (逐帧或按 skip; 零运动验证逐帧更稳)
            t0 = time.perf_counter()
            result = detector.detect(color, depth_frame)
            h = result["handle"]
            last_detect = h

            # 收集有效样本 (theta 非 None 且 ratio 非 None)
            if collecting:
                th = h.get("theta_deg")
                if th is not None and h.get("ratio") is not None:
                    p = h["pos_cam"]
                    d = float(p[2]) if p is not None else None
                    sample = {
                        "window_id": cur["id"], "target_angle_deg": cur["angle"],
                        "idx": cur["n"] + 1,
                        "theta_deg": float(th), "band": h.get("band"),
                        "ratio": h.get("ratio"), "d": d, "rd": h.get("rd"),
                        "handle_side": h.get("handle_side"),
                        "kettle_bbox": h.get("kettle_bbox"),
                        "tag_u": h.get("tag_u"), "tag_v": h.get("tag_v"),
                        "config_sha256": cfg_sha, "git_rev": git_rev,
                        "ts": time.time(),
                    }
                    out_f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    out_f.flush()
                    cur["samples"].append(sample)
                    cur["n"] += 1
                    print(f"  [{cur['id']}] #{cur['n']}/10 θ={float(th):.2f} band={h.get('band')} "
                          f"ratio={h.get('ratio'):.4f} rd={sample['rd']}", flush=True)
                    if cur["n"] >= WINDOW_SAMPLES:
                        # 窗口完成
                        ths = [s["theta_deg"] for s in cur["samples"]]
                        bands = [s["band"] for s in cur["samples"]]
                        med = float(np.median(ths))
                        cur["theta_med"] = med
                        cur["bands"] = bands
                        results.append(cur)
                        print(f"[窗口 {cur['id']} 完成] target={cur['angle']}° "
                              f"θ_med={med:.2f} bands={bands} 10/10",
                              flush=True)
                        collecting = False
                        cur = None
                        window_i += 1
                        if window_i < len(WINDOWS):
                            print(f"\n>>> 请摆位到 {WINDOWS[window_i][0]}° "
                                  f"(窗口 {WINDOWS[window_i][1]}/{len(WINDOWS)}), 摆好按 s", flush=True)

            # 显示
            display = detector.visualize(color)
            status = "采集窗口 1/4" if collecting else f"等 s 开始 (窗口 {window_i+1}/4)"
            cv2.putText(display, status, (10, display.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "s=start q=quit",
                        (display.shape[1] - 160, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            cv2.imshow("Zero-Motion Verify", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s") and not collecting:
                ang, wid = WINDOWS[window_i]
                collecting = True
                cur = {"id": wid, "angle": ang, "n": 0, "samples": []}
                print(f"\n>>> 采集窗口 {wid}: 摆位 {ang}° (保持不动), 收集 10 样本...", flush=True)
    finally:
        out_f.close()

    # ---- 汇总 + 通过判定 ----
    print("\n================ 零运动稳定性验证结果 ================")
    all_pass = True
    for r in results:
        ok = all(b == r["angle"] for b in r["bands"]) if r["angle"] == 0 else all(
            b == 30 for b in r["bands"])
        all_pass &= ok
        print(f"  {r['id']} target={r['angle']:>3}° θ_med={r['theta_med']:7.2f} "
              f"bands={r['bands']} → {'PASS' if ok else 'FAIL'}")
    if len(results) < len(WINDOWS):
        print(f"  [WARN] 仅完成 {len(results)}/{len(WINDOWS)} 窗口")
        all_pass = False
    print(f"判定: {'PASS (可继续 18 回合)' if all_pass else 'FAIL (暂停实机, 保留资料回传)'}")
    print(f"输出: {out_path}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
