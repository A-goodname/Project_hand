# -*- coding: utf-8 -*-
"""D1-A Task 4: 视觉稳定性测量 (只读相机 + 检测推理, 不启动任何服务).

对应: project-hand-d1-preparation-and-measurement-checklist-2026-08-15.md §4
- 把手/杯口检测成功率 + 失败分类
- 把手 3D 位姿帧间抖动 (observation_id + 单调时间戳)
- P2_ARRIVED 模拟时刻 -> 新稳定观测耗时 (须 <6s) + 观测 ID 必须变化
注意: 壶嘴无独立检测器 (P3 全示教固定, 已知限制, 如实记录)
"""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from Orchestrator.handle_cup_detector import HandleCupDetector  # noqa: E402
import yaml  # noqa: E402

OUT_CSV = os.path.abspath(os.path.join(_PROJECT_ROOT, "..", "VLA-BD", "docs",
                                       "d1-measurements-2026-08-15", "vision-stability.csv"))
SERIAL = "342222071364"
N_FRAMES = 60
P2_REOBSERVE_DEADLINE_S = 6.0
STABLE_WINDOW = 3          # 连续稳定帧数
STABLE_POS_TOL = 0.005     # 位置抖动阈值 (m)
STABLE_ANG_TOL = 3.0       # 角度抖动阈值 (deg)


def load_orch_cfg():
    cfg_path = os.path.join(_PROJECT_ROOT, "Orchestrator", "config.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_orch_cfg()
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    det = HandleCupDetector(orch_config=cfg, serial=SERIAL, device=dev)
    print(f"检测器就绪 (yolo+sam 新链), device={dev}")

    # ---------- 1. 连续采集统计 ----------
    handle_ok = handle_fail = cup_ok = cup_fail = 0
    fail_reasons = {}          # 把手失败分类计数
    pos_seq, ang_seq = [], []  # 把手 pos_cam/euler_cam 序列
    obs_id = 0
    p2_sim_done = False
    p2_latency = None
    p2_old_id = None
    p2_new_id = None
    last_obs_ts = 0

    for i in range(N_FRAMES):
        color, depth_img, _cmap, depth_frame = det.get_frames()
        if color is None or depth_frame is None:
            print(f"[{i}] 取帧失败")
            continue
        result = det.detect(color, depth_frame)
        h, c = result["handle"], result["cup"]

        # 把手
        if h["pos_cam"] is not None:
            handle_ok += 1
            pos_seq.append(np.array(h["pos_cam"]))
            ang_seq.append(np.array(h["euler_cam"]))
            # 新观测 ID (每次成功检测计一次观测; 稳定窗口内取最新)
            obs_id += 1
            last_obs_ts = time.monotonic_ns()
        else:
            handle_fail += 1
            if h["kettle_bbox"] is None:
                key = "壶未检出(YOLO)"
            else:
                key = "mask/scanline/6D 失败"
            fail_reasons[key] = fail_reasons.get(key, 0) + 1

        # 杯
        if c["pos_cam"] is not None:
            cup_ok += 1
        else:
            cup_fail += 1

        # ---------- P2 新鲜观测模拟 (帧 30 处触发一次) ----------
        if i == 30 and not p2_sim_done:
            p2_sim_done = True
            p2_ts = time.monotonic_ns()          # 模拟 P2_ARRIVED 时刻
            p2_old_id = obs_id                    # 初始观测 ID
            det.last_known_pose = None            # 清旧缓存
            det.last_known_pose_observation_id = 0
            # 清稳定缓冲: 用连续成功帧重新判定
            _buf = []
            deadline = time.time() + P2_REOBSERVE_DEADLINE_S
            while time.time() < deadline:
                color2, _d2, _c2, depth2 = det.get_frames()
                r2 = det.detect(color2, depth2)
                if r2["handle"]["pos_cam"] is not None:
                    _buf.append(np.array(r2["handle"]["pos_cam"]))
                    if len(_buf) > STABLE_WINDOW:
                        _buf.pop(0)
                    if len(_buf) >= STABLE_WINDOW:
                        arr = np.array(_buf)
                        if np.ptp(arr, axis=0).max() < STABLE_POS_TOL:
                            obs_id += 1
                            p2_new_id = obs_id
                            p2_latency = (time.monotonic_ns() - p2_ts) / 1e9
                            break
            print(f"\n[P2 模拟] 旧观测ID={p2_old_id} -> 新观测ID={p2_new_id} "
                  f"耗时={p2_latency:.3f}s (deadline {P2_REOBSERVE_DEADLINE_S}s)")

        print(f"\r[{i+1}/{N_FRAMES}] handle_ok={handle_ok} cup_ok={cup_ok}", end="", flush=True)

    print()
    det.close()

    # ---------- 2. 抖动统计 (只统计后半段, 排除动态姿态初始收敛段) ----------
    n = len(pos_seq)
    if n > 10:
        half = n // 2
        pos_jitter = float(np.std(pos_seq[half:], axis=0).max())
        ang_jitter = float(np.std(ang_seq[half:], axis=0).max())
    elif n > 2:
        pos_jitter = float(np.std(pos_seq, axis=0).max())
        ang_jitter = float(np.std(ang_seq, axis=0).max())
    else:
        pos_jitter = ang_jitter = float("nan")

    # ---------- 3. 写 CSV ----------
    lines = [
        "layout_id,target,detect_success,detect_fail,fail_reason_class,pos_jitter_m,angle_jitter_deg,obs_id_initial,obs_id_p2,reobserve_latency_s,within_6s",
        f"1,handle,{handle_ok},{handle_fail},{dict(fail_reasons) or 'none'},{pos_jitter:.4f},{ang_jitter:.2f},{p2_old_id},{p2_new_id},{p2_latency if p2_latency else 'NA'},{p2_latency is not None and p2_latency < 6.0}",
        f"1,cup_rim,{cup_ok},{cup_fail},{'none'},{'NA'},{'NA'},NA,NA,NA,NA",
        "1,spout,NA,NA,无独立检测器(P3全示教固定),NA,NA,NA,NA,NA,NA",
    ]
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"\n===== 结果 =====")
    print(f"把手: {handle_ok}/{N_FRAMES} 成功, 失败分类 {fail_reasons}")
    print(f"杯:   {cup_ok}/{N_FRAMES} 成功")
    print(f"把手位置抖动: {pos_jitter:.4f} m | 姿态抖动: {ang_jitter:.2f} deg")
    print(f"P2 新鲜观测: 耗时 {p2_latency if p2_latency else 'FAIL'}s | "
          f"ID {p2_old_id}->{p2_new_id} {'变化OK' if p2_old_id != p2_new_id else '未变化!!'} "
          f"| {'<6s OK' if p2_latency and p2_latency < 6.0 else '>6s FAIL'}")
    print(f"CSV 已写入: {OUT_CSV}")


if __name__ == "__main__":
    main()
