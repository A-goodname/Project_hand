# -*- coding: utf-8 -*-
"""
Orchestrator — 系统入口
=======================
双相机 (眼在手上 + 眼在手外) 全流程状态机入口.

使用方法:
  python Orchestrator/main.py

流程:
  1. 加载配置
  2. 初始化 Camera B (PersonDetector, 眼在手外)
  3. 构建状态机 (IDLE → HANDLE_GRASP_AND_POUR → IDLE)
  4. 主循环: Camera B 帧 → Person/HandInCup 检测 → 驱动状态机 → 可视化
  5. Camera A 由 HandleGraspAndPourHandler 内部管理

退出: 按 'q' 键或 Ctrl+C
"""

import os
import sys
import time

import cv2
import numpy as np
import yaml

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入 ----
from RealSense.realsense_shared import load_camera_serial  # Camera B (新相机, rs.align)
from RealSense_R.realsense_shared import load_camera_serial as load_camera_a_serial  # Camera A (旧相机)
from Orchestrator.state_machine import (
    SharedContext,
    StateMachine,
    OrchestratorState,
    IdleHandler,
    GreetingHandler,
    HandleGraspAndPourHandler,
)
from EyeToHand.hand_in_cup import HandInCupDetector
from EyeToHand.person_detector import PersonDetector


def load_yaml(path):
    """安全加载 YAML 文件."""
    full_path = os.path.join(_project_root, path) if not os.path.isabs(path) else path
    if not os.path.exists(full_path):
        print(f"[FATAL] 配置文件未找到: {full_path}")
        sys.exit(1)
    with open(full_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ============================================================
#  主函数
# ============================================================

def main():
    print("=" * 60)
    print("  Orchestrator — 双相机全流程状态机")
    print("  Camera A (眼在手上, 手动像素映射): Handle + Cup 6D")
    print("  Camera B (眼在手外, rs.align): 人体检测 + Hand-in-cup")
    print("=" * 60)
    print()

    # ---- [1] 加载配置 ----
    print("[INFO] 加载配置...")
    realsense_cfg = load_yaml("RealSense_R/config.yaml")         # Camera A (Handle+Cup, 旧相机)
    estun_cfg = load_yaml("Estun/estun_config.yaml")["estun"]
    eye_to_hand_cfg = load_yaml("EyeToHand/config.yaml")
    eye_to_hand_serial = load_camera_serial(
        os.path.join(_project_root, "EyeToHand/D435i_info.yaml")
    )
    print(f"[INFO] Camera B serial: {eye_to_hand_serial}")

    # ---- [2] 构建共享上下文 ----
    camera_a_serial = load_camera_a_serial(
        os.path.join(_project_root, "RealSense_R/D435i_info.yaml")
    )
    print(f"[INFO] Camera A serial: {camera_a_serial}")

    ctx = SharedContext(
        eye_to_hand_config=eye_to_hand_cfg,
        estun_config=estun_cfg,
        realsense_config=realsense_cfg,
        eye_to_hand_serial=eye_to_hand_serial,
        camera_a_serial=camera_a_serial,
    )

    # ---- [3] Hand-in-cup 检测器 (全生命周期共享, Camera B 帧注入) ----
    ctx.hand_in_cup_detector = HandInCupDetector()

    # ---- [4] Camera B PersonDetector (全生命周期, 主循环统一管理) ----
    print("[INFO] 启动 PersonDetector (UGREEN Camera B)...")
    ctx.person_detector = PersonDetector(
        camera_id=eye_to_hand_cfg.get("ugreen", {}).get("camera_id", 1),
        config_dict=eye_to_hand_cfg,
        start_camera=False,
    )

    # ---- [5] 注册状态机 ----
    sm = StateMachine()
    sm.register(OrchestratorState.IDLE, IdleHandler())
    sm.register(OrchestratorState.GREETING, GreetingHandler())
    sm.register(OrchestratorState.HANDLE_GRASP_AND_POUR, HandleGraspAndPourHandler())
    sm.transition_to(ctx, OrchestratorState.IDLE)

    # ---- [6] 主循环 (Camera B 帧 + 人就位 + HandInCup + 状态机驱动 + 可视化) ----
    cv2.namedWindow("Orchestrator - Camera B (Eye-to-Hand)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orchestrator - Camera A (Eye-in-Hand)", cv2.WINDOW_NORMAL)

    try:
        while not ctx.stop_event.is_set():
            # ============================================================
            # Camera B: 帧采集 (UGREEN Camera, 全生命周期)
            # ============================================================
            pd = ctx.person_detector
            if pd is not None:
                try:
                    color_b = pd.get_frames()[0]
                    ctx.frames_b = (color_b,)
                except Exception as e:
                    print(f"[WARN] Camera B 获取帧异常: {e}")
                    ctx.frames_b = None
            else:
                ctx.frames_b = None

            # ---- 人就位检测 (GREETING 完成后启动, READY 时停止) ----
            if ctx.run_seated_detection and ctx.frames_b is not None:
                color_b = ctx.frames_b[0]
                if color_b is not None and pd is not None and pd.has_chair:
                    seated = pd.detect_person_seated(color_b)
                    if seated is not None:
                        ctx.is_person_seated = seated["is_seated"]

            # ---- Hand-in-cup 检测 (READY 后 ≤5s, TRUE 提前结束) ----
            if ctx.hand_in_cup_active and ctx.hand_in_cup_detector is not None:
                if time.time() - ctx.hand_in_cup_start_time > 5.0:
                    ctx.hand_on_cup = False
                    print("[INFO] HandInCup 超时 5s")
                elif ctx.frames_b is not None:
                    color_b = ctx.frames_b[0]
                    if color_b is not None:
                        ctx.hand_in_cup_detector.update(color_b)
                        ctx.hand_on_cup = ctx.hand_in_cup_detector.is_hand_on_cup

            # ============================================================
            # 驱动状态机
            # ============================================================
            sm.update(ctx)

            # ============================================================
            # 可视化 Camera B
            # ============================================================
            if ctx.frames_b is not None and ctx.frames_b[0] is not None:
                vis_b = ctx.frames_b[0].copy()
                state_name = sm.state.name
                h, w = vis_b.shape[:2]

                if ctx.hand_in_cup_active and ctx.hand_in_cup_detector is not None:
                    vis_b = ctx.hand_in_cup_detector.draw_visualization(vis_b)
                    cv2.putText(vis_b, f"State: {state_name}", (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    cv2.putText(vis_b, f"State: {state_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    pd2 = ctx.person_detector
                    if pd2 is not None:
                        if not pd2.has_chair:
                            cv2.putText(vis_b, "Chair: --", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                        elif ctx.run_seated_detection:
                            status = "SEATED" if ctx.is_person_seated else "waiting"
                            s_color = (0, 255, 0) if ctx.is_person_seated else (0, 165, 255)
                            cv2.putText(vis_b, f"Person: {status}", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)
                        else:
                            cv2.putText(vis_b, "Person: STOPPED", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

                    h_status = f"HandInCup: {'ON_CUP' if ctx.hand_on_cup else 'idle'}"
                    elapsed = time.time() - ctx.hand_in_cup_start_time if ctx.hand_in_cup_active else 0
                    if ctx.hand_in_cup_active and elapsed > 0:
                        h_status += f" ({elapsed:.1f}s)"
                    h_color = (0, 255, 0) if ctx.hand_on_cup else (128, 128, 128)
                    cv2.putText(vis_b, h_status, (10, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, h_color, 2)

                cv2.imshow("Orchestrator - Camera B (Eye-to-Hand)", vis_b)

            # ============================================================
            # 可视化 Camera A
            # ============================================================
            if ctx.frames_a is not None and ctx.frames_a[0] is not None:
                vis_a = ctx.frames_a[0].copy()
                cv2.putText(vis_a, f"State: {sm.state.name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                h_ok = "H:OK" if (ctx.handle_detector and
                                  ctx.handle_detector.last_known_pose is not None) else "H:..."
                c_ok = "C:OK" if (ctx.cup_detector and
                                  ctx.cup_detector.last_known_pose is not None) else "C:..."
                cv2.putText(vis_a, f"{h_ok} {c_ok}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.imshow("Orchestrator - Camera A (Eye-in-Hand)", vis_a)
            else:
                blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"State: {sm.state.name} (Camera A waiting...)",
                            (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                cv2.imshow("Orchestrator - Camera A (Eye-in-Hand)", blank)

            # ---- 退出检测 ----
            key = cv2.waitKey(25) & 0xFF
            if key == ord("q"):
                print("[INFO] 用户按 q 退出")
                ctx.stop_event.set()
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] 主循环异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # ---- 清理 ----
        print("[INFO] 正在清理资源...")
        ctx.stop_event.set()
        cv2.destroyAllWindows()

        if ctx.person_detector is not None:
            try:
                ctx.person_detector.stop()
                print("[INFO] PersonDetector + Camera B pipeline 已停止")
            except Exception as e:
                print(f"[WARN] PersonDetector 停止异常: {e}")

        if ctx.handle_detector is not None:
            try:
                ctx.handle_detector.pipeline.stop()
                print("[INFO] Camera A pipeline 已停止")
            except Exception as e:
                print(f"[WARN] Camera A 停止异常: {e}")

        if ctx.hand_in_cup_detector is not None:
            try:
                ctx.hand_in_cup_detector.close()
                print("[INFO] HandInCup 检测器已关闭")
            except Exception as e:
                print(f"[WARN] HandInCup 检测器关闭异常: {e}")

        from Estun.utils_shared import is_hand_available, disable_hand
        if is_hand_available():
            try:
                disable_hand()
                print("[INFO] 灵巧手已关闭")
            except Exception as e:
                print(f"[WARN] 灵巧手关闭异常: {e}")

        print("[INFO] 程序已退出")


if __name__ == "__main__":
    main()
