# -*- coding: utf-8 -*-
"""
Orchestrator — 系统入口
=======================
双相机 (眼在手上 + 眼在手外) 全流程状态机入口.

使用方法:
  python Orchestrator/main.py

流程:
  1. 加载配置
  2. 初始化眼在手外 PersonDetector
  3. 构建状态机 (IDLE 默认)
  4. 主循环: 获取 Camera B 帧 → 驱动状态机 → 可视化
  5. 状态机内部管理 Camera A (Handle+Cup) 的生命周期

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
from RealSense_R.realsense_shared import load_camera_serial
from Orchestrator.state_machine import (
    SharedContext,
    StateMachine,
    IdleHandler,
    HandleGraspAndPourHandler,
    LocatePersonHandler,
    ApproachPersonHandler,
    WaitAndReleaseHandler,
    RetreatHandler,
)


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
    print("  眼在手上: Handle + Cup 6D 抓取")
    print("  眼在手外: 人体检测 + 递送")
    print("=" * 60)
    print()

    # ---- [1] 加载配置 ----
    print("[INFO] 加载配置...")
    realsense_cfg = load_yaml("RealSense_R/config.yaml")
    estun_cfg = load_yaml("Estun/estun_config.yaml")["estun"]
    eye_to_hand_cfg = load_yaml("EyeToHand/config.yaml")
    eye_to_hand_serial = load_camera_serial(
        os.path.join(_project_root, "EyeToHand/D435i_info.yaml")
    )
    print(f"[INFO] 眼在手外相机 serial: {eye_to_hand_serial}")

    # ---- [2] 构建共享上下文 ----
    ctx = SharedContext(
        eye_to_hand_config=eye_to_hand_cfg,
        estun_config=estun_cfg,
        realsense_config=realsense_cfg,
        eye_to_hand_serial=eye_to_hand_serial,
    )

    # ---- [3] 注册状态机 ----
    from Orchestrator.state_machine import OrchestratorState

    sm = StateMachine()
    sm.register(OrchestratorState.IDLE, IdleHandler())
    sm.register(OrchestratorState.HANDLE_GRASP_AND_POUR, HandleGraspAndPourHandler())
    sm.register(OrchestratorState.LOCATE_PERSON, LocatePersonHandler())
    sm.register(OrchestratorState.APPROACH_PERSON, ApproachPersonHandler())
    sm.register(OrchestratorState.WAIT_AND_RELEASE, WaitAndReleaseHandler())
    sm.register(OrchestratorState.RETREAT, RetreatHandler())

    # 进入 IDLE 状态
    sm.transition_to(ctx, OrchestratorState.IDLE)

    # ---- [5] 主循环 ----
    cv2.namedWindow("Orchestrator - Eye-to-Hand", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Orchestrator - Eye-in-Hand", cv2.WINDOW_NORMAL)

    running = True
    try:
        while running and not ctx.stop_event.is_set():
            # ---- 采集 Camera B (眼在手外) ----
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

            # ---- 驱动状态机 ----
            sm.update(ctx)

            # ---- 可视化 Camera B ----
            if ctx.frames_b is not None and ctx.frames_b[0] is not None:
                vis_b = ctx.frames_b[0].copy()
                # 状态覆盖
                state_name = sm.state.name
                cv2.putText(vis_b, f"State: {state_name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if ctx.delivery_pose is not None:
                    xyz, abc = ctx.delivery_pose
                    cv2.putText(vis_b, f"Deliver: ({xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f})",
                                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(vis_b, f"ABC: ({abc[0]:.1f}, {abc[1]:.1f}, {abc[2]:.1f})",
                                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                cv2.imshow("Orchestrator - Eye-to-Hand", vis_b)

            # ---- 可视化 Camera A ----
            if ctx.frames_a is not None and ctx.frames_a[0] is not None:
                vis_a = ctx.frames_a[0].copy()
                cv2.putText(vis_a, f"State: {sm.state.name}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                h_ok = "H:OK" if ctx.handle_detector and ctx.handle_detector.last_known_pose is not None else "H:..."
                c_ok = "C:OK" if ctx.cup_detector and ctx.cup_detector.last_known_pose is not None else "C:..."
                cv2.putText(vis_a, f"{h_ok} {c_ok}", (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.imshow("Orchestrator - Eye-in-Hand", vis_a)
            else:
                # 无画面时显示空白提示
                blank = 255 * np.ones((480, 640, 3), dtype=np.uint8)
                cv2.putText(blank, f"State: {sm.state.name} (Camera A waiting...)", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
                cv2.imshow("Orchestrator - Eye-in-Hand", blank)

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
