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
from Orchestrator.state_machine import (
    SharedContext,
    StateMachine,
    OrchestratorState,
    IdleHandler,
    GreetingHandler,
    HandleGraspAndPourHandler,
)
from EyeToHand.hand_in_cup import HandInCupDetector


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
    ctx = SharedContext(
        eye_to_hand_config=eye_to_hand_cfg,
        estun_config=estun_cfg,
        realsense_config=realsense_cfg,
        eye_to_hand_serial=eye_to_hand_serial,
    )

    # ---- [3] Hand-in-cup 检测器 (全生命周期共享, Camera B pipeline 复用) ----
    ctx.hand_in_cup_detector = HandInCupDetector()

    # ---- [4] 注册状态机 ----
    sm = StateMachine()
    sm.register(OrchestratorState.IDLE, IdleHandler())
    sm.register(OrchestratorState.GREETING, GreetingHandler())
    sm.register(OrchestratorState.HANDLE_GRASP_AND_POUR, HandleGraspAndPourHandler())
    sm.transition_to(ctx, OrchestratorState.IDLE)

    # ---- [5] 主循环 (Camera B 帧 + Person/HandInCup 检测 + 状态机驱动 + 可视化) ----
    cv2.namedWindow("Orchestrator - Camera B (Eye-to-Hand)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orchestrator - Camera A (Eye-in-Hand)", cv2.WINDOW_NORMAL)

    running = True
    try:
        while running and not ctx.stop_event.is_set():
            # ============================================================
            # Camera B: 帧采集 (pipeline 始终运行, 由 PersonDetector 管理)
            # ============================================================
            pd = ctx.person_detector
            if pd is not None:
                try:
                    color_b, depth_b, depth_cmap_b, depth_frame_b = pd.get_frames()
                    ctx.frames_b = (color_b, depth_b, depth_cmap_b, depth_frame_b)
                except Exception as e:
                    print(f"[WARN] Camera B 获取帧异常: {e}")
                    ctx.frames_b = None
            else:
                ctx.frames_b = None

            # ---- Camera B 持续运行 (椅子标定 + 人就位 由状态机内部驱动) ----
            # 主循环仅保持帧获取, 具体逻辑由各状态的 on_update 处理

            # ---- Hand-in-cup 检测 (READY → RELEASE) ----
            if ctx.hand_in_cup_active and ctx.hand_in_cup_detector is not None:
                if ctx.frames_b is not None:
                    color_b, depth_b, _, depth_frame_b = ctx.frames_b
                    if color_b is not None and depth_frame_b is not None:
                        ctx.hand_in_cup_detector.update(color_b, depth_frame_b)
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

                # Hand-in-cup 激活 → 使用 draw_visualization() 全覆盖,
                # state 名移到底部 (避免与 HandInCup 文字 (10,30)/(10,52) 重叠)
                if ctx.hand_in_cup_active and ctx.hand_in_cup_detector is not None:
                    vis_b = ctx.hand_in_cup_detector.draw_visualization(vis_b)
                    cv2.putText(vis_b, f"State: {state_name}", (10, h - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                else:
                    # 状态
                    cv2.putText(vis_b, f"State: {state_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Person / 椅子 / 就位 状态
                    pd2 = ctx.person_detector
                    if pd2 is not None:
                        if not pd2.is_chair_calibrated:
                            cv2.putText(vis_b, "Chair: calibrating...", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        elif not ctx.stop_person_detector:
                            sc = pd2.seated_stable_count
                            status = "SEATED" if pd2.is_person_seated else "waiting"
                            s_color = (0, 255, 0) if pd2.is_person_seated else (0, 165, 255)
                            cv2.putText(vis_b, f"Person: {status} ({sc})", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, s_color, 2)
                        else:
                            cv2.putText(vis_b, "Person: STOPPED", (10, 55),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)

                    # Hand-in-cup 状态
                    h_status = f"HandInCup: {'ON_CUP' if ctx.hand_on_cup else 'idle'}"
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
                running = False
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
                print("[INFO] PersonDetector 已停止")
            except Exception as e:
                print(f"[WARN] PersonDetector 停止异常: {e}")

        if ctx.handle_detector is not None:
            try:
                ctx.handle_detector.pipeline.stop()
                print("[INFO] Handle 检测器已停止")
            except Exception as e:
                print(f"[WARN] Handle 检测器停止异常: {e}")
        if ctx.cup_detector is not None:
            try:
                ctx.cup_detector.pipeline.stop()
                print("[INFO] Cup 检测器已停止")
            except Exception as e:
                print(f"[WARN] Cup 检测器停止异常: {e}")
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
