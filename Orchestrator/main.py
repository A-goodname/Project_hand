# -*- coding: utf-8 -*-
"""
Orchestrator — 系统入口
=======================
双相机 (眼在手上 + 眼在手外) 全流程 5 阶段顺序调度.

使用方法:
  python Orchestrator/main.py

阶段:
  greeting       Camera A: Phase 0 人检测+4步协议,  Camera B: 椅子标定(人在A前,椅子空)
  handle_detect  Camera A: Handle+Cup 全速,        Camera B: PAUSED
  seated_check   Camera B: 人就位(已有baseline),    Camera A: 已关闭
  hand_in_cup    Camera B: HandInCup全速 → CONTINUE/ABORT

退出: 按 'q' 键或 Ctrl+C
"""

# 禁用 MediaPipe clearcut 遥测 — 必须在任何 MediaPipe import 之前
import os as _os
_os.environ["GLOG_minloglevel"] = "3"
_os.environ["GLOG_stderrthreshold"] = "4"
_os.environ["GLOG_logtostderr"] = "0"
_os.environ["GRPC_VERBOSITY"] = "NONE"

import os
import sys
import threading
import time

import cv2
import numpy as np
import yaml

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入 ----
from RealSense_R.realsense_shared import load_camera_serial as load_camera_a_serial
from Orchestrator.state_machine import (
    SharedContext,
    GreetingHandler,
    HandleGraspAndPourHandler,
)
from EyeToHand.hand_in_cup import HandInCupDetector
from EyeToHand.person_detector import PersonDetector
from Orchestrator.logger import init_log, session_log, close_log


def load_yaml(path):
    """安全加载 YAML 文件."""
    full_path = os.path.join(_project_root, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        print(f"[FATAL] 配置文件未找到: {full_path}")
        sys.exit(1)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  Camera B 帧采集 (共享)
# ============================================================

def _get_camera_b_frame(ctx):
    pd = ctx.person_detector
    if pd is None:
        ctx.frames_b = None
        return None
    try:
        color_b = pd.get_frames()[0]
        ctx.frames_b = (color_b,)
        return color_b
    except Exception as e:
        print(f"[WARN] Camera B 获取帧异常: {e}")
        ctx.frames_b = None
        return None


# ============================================================
#  可视化
# ============================================================

def _visualize_camera_b(ctx, phase):
    if ctx.frames_b is None or ctx.frames_b[0] is None:
        return
    vis_b = ctx.frames_b[0].copy()
    h, w = vis_b.shape[:2]

    if phase == "hand_in_cup" and ctx.hand_in_cup_detector is not None:
        vis_b = ctx.hand_in_cup_detector.draw_visualization(vis_b)
        cv2.putText(vis_b, f"Phase: {phase}", (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    else:
        cv2.putText(vis_b, f"Phase: {phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pd2 = ctx.person_detector
        if pd2 is not None:
            if not pd2.has_chair:
                cv2.putText(vis_b, "Chair: --", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            elif phase in ("seated_detect", "seated_check"):
                status = "SEATED" if ctx.is_person_seated else "waiting"
                s_color = (0, 255, 0) if ctx.is_person_seated else (0, 165, 255)
                cv2.putText(vis_b, f"Person: {status}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)
            else:
                cv2.putText(vis_b, f"Person: paused ({phase})", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

        h_status = f"HandInCup: {'ON_CUP' if ctx.hand_on_cup else 'idle'}"
        h_color = (0, 255, 0) if ctx.hand_on_cup else (128, 128, 128)
        cv2.putText(vis_b, h_status, (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

    cv2.imshow("Orchestrator - Camera B (Eye-to-Hand)", vis_b)


def _visualize_camera_a(ctx, phase):
    if ctx.frames_a is not None and ctx.frames_a[0] is not None:
        vis_a = ctx.frames_a[0].copy()
        cv2.putText(vis_a, f"Phase: {phase}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        h_ok = "H:OK" if (ctx.handle_cup_detector and
                          ctx.handle_cup_detector.last_known_pose is not None) else "H:..."
        c_ok = "C:OK" if (ctx.handle_cup_detector and
                          getattr(ctx.handle_cup_detector, 'last_known_cup_pose', None) is not None) else "C:..."
        cv2.putText(vis_a, f"{h_ok} {c_ok}", (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.imshow("Orchestrator - Camera A (Eye-in-Hand)", vis_a)
    else:
        blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
        cv2.putText(blank, f"Phase: {phase} (Camera A waiting...)",
                    (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
        cv2.imshow("Orchestrator - Camera A (Eye-in-Hand)", blank)


# ============================================================
#  Phase runners
# ============================================================

def _run_greeting(ctx, greeting_h, pd):
    """greeting: Camera A Phase 0 人检测+4步协议, Camera B 椅子标定(人在Camera A前,椅子空)."""
    _get_camera_b_frame(ctx)
    greeting_h.update_preview(ctx)

    # Camera B 椅子标定 (人在 Camera A 前, 椅子空 → 自动建立 baseline)
    if ctx.frames_b is not None and ctx.frames_b[0] is not None:
        pd.detect_person_seated(ctx.frames_b[0])


def _run_seated_detect(ctx, pd):
    """seated_detect: Camera B 椅子标定+人就位全速. Camera A idle."""
    color_b = _get_camera_b_frame(ctx)
    if color_b is not None:
        seated = pd.detect_person_seated(color_b)
        if seated is not None:
            ctx.is_person_seated = seated["is_seated"]


def _run_handle_detect(ctx, handle_h):
    """handle_detect: Camera A Handle+Cup 全速. Camera B PAUSED."""
    handle_h.get_frames_and_detect(ctx)
    handle_h.accept_tcp(ctx)


def _run_seated_check(ctx, pd):
    """seated_check: Camera B 人就位恢复 (Phase B 结束, 人应已到椅子)."""
    color_b = _get_camera_b_frame(ctx)
    if color_b is not None:
        seated = pd.detect_person_seated(color_b)
        if seated is not None:
            ctx.is_person_seated = seated["is_seated"]


def _run_hand_in_cup(ctx, hic):
    """hand_in_cup: Camera B HandInCup 全速. TRUE→CONTINUE, FALSE→ABORT."""
    color_b = _get_camera_b_frame(ctx)
    if color_b is not None and hic is not None:
        elapsed = time.time() - ctx.hand_in_cup_start_time
        if elapsed > 10.0 and ctx.hand_on_cup is None:
            ctx.hand_on_cup = False
            session_log("HandInCup 超时 10s")
        elif ctx.hand_on_cup is None or not ctx.hand_on_cup:
            # 持续检测直到命中 (hand_on_cup=True 时停止, 避免冲掉结果)
            hic.update(color_b)
            ctx.hand_on_cup = hic.is_hand_on_cup


# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Orchestrator — 双相机 5 阶段顺序调度")
    print("  Camera A (眼在手上): Handle + Cup 6D")
    print("  Camera B (眼在手外): 人就位 + Hand-in-cup")
    print("=" * 60)
    print()

    # ---- [0] 初始化日志 ----
    init_log("orchestrator")
    session_log("Orchestrator 启动")

    # ---- [1] 加载配置 ----
    print("[INFO] 加载配置...")
    realsense_cfg = load_yaml("RealSense_R/config.yaml")
    estun_cfg = load_yaml("Estun/estun_config.yaml")["estun"]
    eye_to_hand_cfg = load_yaml("EyeToHand/config.yaml")

    # ---- [2] 构建共享上下文 ----
    camera_a_serial = load_camera_a_serial(
        os.path.join(_project_root, "RealSense_R/D435i_info.yaml")
    )
    print(f"[INFO] Camera A serial: {camera_a_serial}")

    ctx = SharedContext(
        eye_to_hand_config=eye_to_hand_cfg,
        estun_config=estun_cfg,
        realsense_config=realsense_cfg,
        eye_to_hand_serial="",
        camera_a_serial=camera_a_serial,
    )

    # ---- [3] Hand-in-cup 检测器 ----
    ctx.hand_in_cup_detector = HandInCupDetector()

    # ---- [4] Camera B PersonDetector ----
    print("[INFO] 启动 PersonDetector (UGREEN Camera B)...")
    ctx.person_detector = PersonDetector(
        camera_id=eye_to_hand_cfg.get("ugreen", {}).get("camera_id", 1),
        config_dict=eye_to_hand_cfg,
        start_camera=False,
    )

    # ---- [5] 创建 Phase handler ----
    greeting_h = GreetingHandler()
    handle_h = HandleGraspAndPourHandler()

    # ---- [6] 窗口 ----
    cv2.namedWindow("Orchestrator - Camera B (Eye-to-Hand)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orchestrator - Camera A (Eye-in-Hand)", cv2.WINDOW_NORMAL)

    # ---- [7] Phase 0: GREETING ----
    phase = "greeting"
    session_log(f"Phase: {phase}")
    greeting_h.start(ctx)

    pd = ctx.person_detector
    _pd_was_seated = False  # 去重日志

    try:
        while not ctx.stop_event.is_set():
            # ================================================================
            # Phase 调度
            # ================================================================
            if phase == "greeting":
                _run_greeting(ctx, greeting_h, pd)
                if ctx.greeting_done.is_set():
                    session_log("Phase greeting → handle_detect")
                    greeting_h.stop(ctx)
                    handle_h.start(ctx)
                    phase = "handle_detect"

            elif phase == "handle_detect":
                _run_handle_detect(ctx, handle_h)
                if ctx.targets_sent.is_set():
                    session_log("Phase handle_detect → seated_check")
                    phase = "seated_check"

            elif phase == "seated_check":
                _run_seated_check(ctx, pd)
                if ctx.ready_received.is_set():
                    session_log("Phase seated_check → hand_in_cup")
                    phase = "hand_in_cup"

            elif phase == "hand_in_cup":
                _run_hand_in_cup(ctx, ctx.hand_in_cup_detector)
                if ctx.hand_judged.is_set():
                    session_log("Phase hand_in_cup → done")
                    break

            # ================================================================
            # 可视化
            # ================================================================
            _visualize_camera_b(ctx, phase)
            _visualize_camera_a(ctx, phase)

            # ---- 退出检测 ----
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                print("[INFO] 用户按 q 退出")
                ctx.stop_event.set()
                break

    except KeyboardInterrupt:
        session_log("键盘中断 (Ctrl+C)")
    except Exception as e:
        session_log(f"主循环异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ---- 清理 ----
        session_log("正在清理资源...")
        ctx.stop_event.set()
        cv2.destroyAllWindows()

        if ctx.person_detector is not None:
            try:
                ctx.person_detector.stop()
                print("[INFO] PersonDetector + Camera B 已停止")
            except Exception as e:
                print(f"[WARN] PersonDetector 停止异常: {e}")

        # Camera A 关闭: 优先用延迟引用 (_camera_a_to_close 由 daemon 线程在 Phase B 后保存)
        cam_a = ctx._camera_a_to_close or ctx.handle_cup_detector
        if cam_a is not None:
            try:
                cam_a.close()
                print("[INFO] Camera A pipeline 已停止")
            except Exception as e:
                print(f"[WARN] Camera A 停止异常: {e}")
            ctx.handle_cup_detector = None
            ctx._camera_a_to_close = None

        if ctx.hand_in_cup_detector is not None:
            try:
                t = threading.Thread(
                    target=ctx.hand_in_cup_detector.close,
                    daemon=True,
                )
                t.start()
                t.join(timeout=2.0)
                if t.is_alive():
                    print("[INFO] HandInCup 检测器关闭中 (后台继续)...")
                else:
                    print("[INFO] HandInCup 检测器已关闭")
            except Exception as e:
                print(f"[WARN] HandInCup 检测器关闭异常: {e}")

        from Estun.utils_shared import is_hand_available, disable_hand
        if is_hand_available():
            try:
                disable_hand(skip_reset=True)
                print("[INFO] 灵巧手已关闭")
            except Exception as e:
                print(f"[WARN] 灵巧手关闭异常: {e}")

        print("[INFO] 程序已退出")


if __name__ == "__main__":
    main()
