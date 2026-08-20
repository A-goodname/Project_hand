# -*- coding: utf-8 -*-
"""
θ 稳定性记录工具 (2026-08-20)
=============================
逐帧记录 handle_cup_detector 的旋转检测结果 (ratio/rd/θ/band/d/位置/姿态),
累积 θ 均值/方差, 用于观察 θ 波动规律.

参考 test_tag_points.py: log 文件 + log()/log_file() 双写 + 'r' 重置累积 + 退出 SUMMARY.
检测复用 HandleCupDetector (Orchestrator/handle_cup_detector.py), 不改动检测器本体.

用法:
  conda run -n Proj python Orchestrator/tests/theta_stability_log.py
  conda run -n Proj python Orchestrator/tests/theta_stability_log.py --skip 5
按键: q 退出 | r 重置累积 (换角度/位置后重新采样) | s 保存当前帧
"""
import os
import sys
import time
import argparse

_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import yaml

from Orchestrator.handle_cup_detector import HandleCupDetector


def load_orch_config():
    orch_path = os.path.join(_project_root, "Orchestrator", "config.yaml")
    with open(orch_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_serial():
    info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(info_path, "r", encoding="utf-8") as f:
            return str(yaml.safe_load(f)["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")
        return ""


def main():
    ap = argparse.ArgumentParser(
        description="θ 稳定性记录 (复用 HandleCupDetector, 参考 test_tag_points)")
    ap.add_argument("--skip", type=int, default=1,
                    help="每 N 帧检测一次 (1=逐帧; 用 余数==0 语义, 避开 N=1 时 余数==1 恒 False 的坑)")
    ap.add_argument("--log", default=None,
                    help="日志文件路径 (默认 logs/theta_<ts>.log)")
    ap.add_argument("--no-log", action="store_true", help="不写日志文件")
    args = ap.parse_args()

    skip = max(1, args.skip)

    orch_cfg = load_orch_config()
    serial = load_serial()

    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("[INFO] 创建 HandleCupDetector ...")
    detector = HandleCupDetector(orch_config=orch_cfg, serial=serial, device=dev)

    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    if args.log is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(log_dir, f"theta_{ts}.log")
    log_f = None if args.no_log else open(args.log, "w", encoding="utf-8")

    def log(msg):
        """写 log 文件 + 终端."""
        print(msg, flush=True)
        if log_f is not None:
            log_f.write(msg + "\n")
            log_f.flush()

    def log_file(msg):
        """只写 log 文件 (逐帧检测不刷屏)."""
        if log_f is not None:
            log_f.write(msg + "\n")
            log_f.flush()

    log(f"[INFO] skip={skip}")
    log(f"[INFO] handle_side={orch_cfg.get('handle_side')}")
    log(f"[INFO] log -> {args.log}")
    log("[INFO] 操作: 摆好角度后按 r 开始累积 → 保持姿势观察 mean/std → 换角度再按 r; q 退出")

    theta_samples = []                             # 累积 θ (按姿势, 'r' 重置)

    frame_counter = 0
    last_stat = None   # 最近一次检测的统计 (跳帧时画面复用)
    try:
        while True:
            color, _depth_img, depth_cmap, depth_frame = detector.get_frames()
            if color is None or depth_frame is None:
                time.sleep(0.03)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
                continue
            frame_counter += 1

            # ---- 跳帧检测 (==0 语义: skip=1 逐帧触发) ----
            if frame_counter % skip == 0:
                t0 = time.perf_counter()
                result = detector.detect(color, depth_frame)
                t_ms = (time.perf_counter() - t0) * 1000
                h = result["handle"]

                th = h.get("theta_deg")
                if th is not None:
                    theta_samples.append(float(th))
                n = len(theta_samples)
                th_mean = th_std = 0.0
                if n:
                    arr = np.array(theta_samples)
                    th_mean, th_std = float(arr.mean()), float(arr.std())

                p = h["pos_cam"]
                if p is not None:
                    e = h["euler_cam"]
                    ratio = h.get("ratio")
                    rd = h.get("rd")
                    ratio_str = f"{ratio:.4f}" if ratio is not None else "None"
                    rd_str = f"{rd:.4f}" if rd is not None else "None"
                    th_str = f"{th:.2f}" if th is not None else "None"
                    band_str = str(h.get("band")) if h.get("band") is not None else "None"
                    # 全量帧进 log 文件 (不刷屏)
                    log_file(
                        f"[{frame_counter}] pos=({p[0]:.3f},{p[1]:.3f},{p[2]:.3f}) "
                        f"euler=({e[0]:.1f},{e[1]:.1f},{e[2]:.1f}) "
                        f"ratio={ratio_str} rd={rd_str} th={th_str} band={band_str} "
                        f"d={p[2]:.3f} side={h.get('handle_side')} | "
                        f"N={n} mean={th_mean:.2f} std={th_std:.2f} "
                        f"[{t_ms:.0f}ms]")
                else:
                    kettle = "Y" if h["kettle_bbox"] is not None else "N"
                    log_file(f"[{frame_counter}] no handle (kettle={kettle})")
                last_stat = {
                    "p": p, "th": th, "band": h.get("band"),
                    "n": n, "th_mean": th_mean, "th_std": th_std,
                }

            # ---- 显示 (每帧刷新, 统计复用最近一次检测) ----
            display = detector.visualize(color)
            if last_stat is not None and last_stat["p"] is not None:
                # th/band 已由 visualize() 的 ROT 行显示 (y≈75); 这里只补累积统计,
                # 放 y=120 避开 visualize 左上文字区 (y=30/55/75/95, 底部 y=h-10)
                cv2.putText(display, f"acc: N={last_stat['n']} "
                                     f"mean={last_stat['th_mean']:.2f} "
                                     f"std={last_stat['th_std']:.2f}",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(display, "q=quit r=reset s=save",
                        (display.shape[1] - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            if depth_cmap is not None:
                h_vis = display.shape[0]
                d_small = cv2.resize(depth_cmap, (h_vis, h_vis))
                vis_panel = np.hstack((display, d_small))
            else:
                vis_panel = display
            cv2.imshow("Theta Stability Log | Camera A", vis_panel)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                n_old = len(theta_samples)
                if n_old > 0:
                    arr = np.array(theta_samples)
                    log(f"[RESET] 清除 {n_old} 帧 θ "
                        f"(mean={arr.mean():.2f} std={arr.std():.2f} "
                        f"min={arr.min():.2f} max={arr.max():.2f})")
                theta_samples.clear()
            elif key == ord("s"):
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = os.path.join(log_dir, f"theta_{ts}.png")
                cv2.imwrite(path, vis_panel)
                log(f"[SAVE] {path}")

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        n = len(theta_samples)
        if n > 0:
            arr = np.array(theta_samples)
            log(f"[SUMMARY] {n} 帧  θ mean={arr.mean():.2f}  std={arr.std():.2f}  "
                f"min={arr.min():.2f}  max={arr.max():.2f}")
        else:
            log("[SUMMARY] 无有效 θ 帧")
        if log_f is not None:
            log_f.close()
        detector.close()
        cv2.destroyAllWindows()
        if not args.no_log:
            print(f"[INFO] log 已保存: {args.log}")
        print("[INFO] 已退出")


if __name__ == "__main__":
    main()
