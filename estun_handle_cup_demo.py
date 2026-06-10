# -*- coding: utf-8 -*-
"""
estun 把手+水杯融合位姿转换验证 Demo
=============================================
整合 Handle 6D 检测器与 Cup 6D 检测器 (共享 RealSense pipeline),
实现完整 6 点位姿计算: Handle P1/P2/P3_pour + Cup P1/P2/P3.

与 estun_handle_demo.py 的差异:
  - 双检测器: RealSenseHandle6DDetector + RealSenseCup6DDetector
  - Cup 检测器不启动 pipeline (start_pipeline=False), 复用 Handle 的帧
  - 新增倒水位置 P3_pour 计算 (基于 spout_offset 反算)
  - 新增水杯抓取位姿计算

输入:
  - RealSense_R/config.yaml            : 相机 / YOLO / SAM / cup 配置
  - estun_config.yaml                  : 机器人 / 把手偏移 / 水杯偏移配置
  - calib_data/calibration_result.yaml : 手眼标定矩阵

输出:
  - logs/estun_handle_cup_demo_*.log   : 6 个位姿 + 中间结果

流程:
  [Step 1] 加载配置 (标定矩阵 / 夹爪偏移 / 水杯偏移)
  [Step 2] 初始化 Handle 检测器 (启动 pipeline)
  [Step 3] 初始化 Cup 检测器 (不启动 pipeline)
  [Step 4] 主循环: 同步检测 Handle + Cup → cam2ee → 稳定化
  [Step 5] 控制台输入机械臂当前位姿
  [Step 6] 位姿转换: 计算 6 个位姿
  [Step 7] 写入日志
"""

import os
import sys
import yaml
import cv2
import numpy as np
from datetime import datetime
from typing import Any, Optional, Tuple


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入位姿转换工具 ----
from utils_shared import (
    euler_to_rotation_matrix,
    transform_pose_to_base,
    compute_gripper_offset_transform,
    compute_grasp_rotation,
    post_correct_angles,
)
from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles


# =============================================================
# 配置加载
# =============================================================

# ---- 加载 RealSense / YOLO 配置 ----
_cfg_path = os.path.join(_project_root, "RealSense_R", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---- 加载 ESTUN 配置 ----
_estun_cfg_path = os.path.join(_project_root, "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

CALIB_RESULT_PATH = os.path.join(_project_root, _cfg_estun["calibration"]["result_path"])
STABLE_WINDOW = _cfg_estun["stabilization"]["window"]
POS_THRESHOLD = _cfg_estun["stabilization"]["pos_threshold"]
ANGLE_THRESHOLD = _cfg_estun["stabilization"]["angle_threshold"]

# ---- 把手偏移 ----
HANDLE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["t_handle_offset"], dtype=float)
HANDLE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["r_handle_offset"], dtype=float)
HANDLE_PRE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["t_offset"], dtype=float)
HANDLE_PRE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["r_offset"], dtype=float)
HANDLE_TARGET_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["t_offset"], dtype=float)
HANDLE_TARGET_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 水杯偏移 ----
CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset"], dtype=float)
CUP_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["r_cup_offset"], dtype=float)
CUP_PRE_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["t_offset"], dtype=float)
CUP_PRE_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["r_offset"], dtype=float)
CUP_TARGET_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["t_offset"], dtype=float)
CUP_TARGET_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 壶嘴偏移 + 倒水偏移 ----
SPOUT_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["spout_offset"]["t_offset"], dtype=float)
POUR_T_OFFSET_BASE = np.array(_cfg_estun["cup_grasp"]["pour_offset"]["t_offset"], dtype=float)
POUR_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["pour_offset"]["r_offset"], dtype=float)

# ---- 方向模式 ----
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]
HANDLE_SIDE = _cfg["handle"]["handle_side"]

# 把手模式下: 抓取方向自动与 handle_side 一致
if HANDLE_SIDE != GRASP_DIRECTION:
    print(f"[INFO] grasp_direction ({GRASP_DIRECTION}) 自动切换为 handle_side ({HANDLE_SIDE})")
    GRASP_DIRECTION = HANDLE_SIDE

# ---- fixed 模式 ----
_fixed_abc = _cfg_estun.get("fixed_abc", {})
FIXED_ABC_CFG = _fixed_abc.get(GRASP_DIRECTION, [])
if len(FIXED_ABC_CFG) != 3:
    if ORIENTATION_MODE == "fixed":
        print(f"[WARN] {GRASP_DIRECTION} 方向未配置有效 fixed_abc, 回退到 free 模式")
        ORIENTATION_MODE = "free"
    FIXED_ABC = np.array([-90, 82, -98], dtype=float)
else:
    FIXED_ABC = np.array(FIXED_ABC_CFG, dtype=float)

SKIP_FRAMES = _cfg["handle"]["skip_frames"]

VALID_MODES = {"vertical", "horizontal", "free", "fixed"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE}, 将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现, 将使用 free 模式")
    ORIENTATION_MODE = "free"


def load_cam2ee(yaml_path: str) -> np.ndarray:
    """从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)."""
    I = np.eye(4)
    if not os.path.exists(yaml_path):
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}")
        print("[WARN] 将使用单位矩阵")
        return I

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    calib = data.get("calibration", {})
    hand_eye = calib.get("handeye", {})
    if not hand_eye:
        print("[WARN] 标定文件中未找到 calibration/handeye 章节, 使用单位矩阵")
        return I

    use_method = hand_eye.get("use_method", "Tsai")
    method_data = hand_eye.get(use_method, {})
    if not method_data:
        print(f"[WARN] 方法 {use_method} 无数据, 使用单位矩阵")
        return I

    T_raw = method_data.get("T_cam_to_gripper", None)
    if T_raw is None or len(T_raw) != 4 or len(T_raw[0]) != 4:
        print("[WARN] T_cam_to_gripper 格式错误, 使用单位矩阵")
        return I

    T = np.array(T_raw, dtype=float)
    if T.shape != (4, 4):
        print(f"[WARN] T_cam_to_gripper 形状 {T.shape}, 使用单位矩阵")
        return I

    t = T[:3, 3]
    print(f"[INFO] 已加载标定矩阵: {use_method}.T_cam_to_gripper (cam->ee), 平移: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
    return T


# =============================================================
# 位姿稳定器 (精简版)
# =============================================================

class PoseStabilizer:
    """缓存最近 N 帧位姿数据; 当位置和角度变化均低于阈值时, 认为位姿稳定."""

    def __init__(self, window: int = 5, pos_thresh: float = 0.005, angle_thresh: float = 3.0):
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self._buffer: list = []

    def add(self, pos: Tuple[float, float, float], euler: Tuple[float, float, float]):
        self._buffer.append((np.array(pos, dtype=float), np.array(euler, dtype=float)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self) -> bool:
        if len(self._buffer) < self.window:
            return False
        poses = np.array([p for p, _ in self._buffer])
        angles = np.array([a for _, a in self._buffer])
        pos_range = np.ptp(poses, axis=0).max()
        angle_range = np.ptp(angles, axis=0).max()
        stable = pos_range < self.pos_thresh and angle_range < self.angle_thresh
        if stable:
            print(f"[STABLE] 位置波动: {pos_range*1000:.1f}mm, 角度波动: {angle_range:.1f}deg")
        return stable

    def get_mean(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        if not self._buffer:
            return None, None
        poses = np.array([p for p, _ in self._buffer])
        angles = np.array([a for _, a in self._buffer])
        return poses.mean(axis=0), angles.mean(axis=0)

    def reset(self):
        self._buffer = []


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  estun 把手+水杯融合位姿转换验证 Demo")
    print("  标定: " + CALIB_RESULT_PATH)
    print("=" * 60, flush=True)

    # ---- [Step 1a] 加载手眼标定矩阵 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")

    # ---- [Step 1b] 预计算工具偏移 ----
    _, t_handle_effective = compute_gripper_offset_transform(HANDLE_T_OFFSET, HANDLE_R_OFFSET)
    _, t_cup_effective = compute_gripper_offset_transform(CUP_T_OFFSET, CUP_R_OFFSET)
    print(f"[INFO] 把手偏移 t_handle_effective: ({t_handle_effective[0]:.4f}, {t_handle_effective[1]:.4f}, {t_handle_effective[2]:.4f}) m")
    print(f"[INFO] 水杯偏移 t_cup_effective:     ({t_cup_effective[0]:.4f}, {t_cup_effective[1]:.4f}, {t_cup_effective[2]:.4f}) m")
    print(f"[INFO] 壶嘴偏移 spout_offset:        ({SPOUT_T_OFFSET[0]:.4f}, {SPOUT_T_OFFSET[1]:.4f}, {SPOUT_T_OFFSET[2]:.4f}) m")
    print(f"[INFO] 倒水防碰撞偏移:               ({POUR_T_OFFSET_BASE[0]:.4f}, {POUR_T_OFFSET_BASE[1]:.4f}, {POUR_T_OFFSET_BASE[2]:.4f}) m (基系 Z)")

    # ---- [Step 2] 初始化 Handle 检测器 (启动 pipeline) ----
    from RealSense_R.test_handle_6D import RealSenseHandle6DDetector
    print("[INFO] 初始化 RealSense Handle 6D 检测器...")
    try:
        handle_detector = RealSenseHandle6DDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        return

    # ---- [Step 3] 初始化 Cup 检测器 (不启动 pipeline, 复用帧) ----
    from RealSense_R.test_cup_6D import RealSenseCup6DDetector
    print("[INFO] 初始化 RealSense Cup 6D 检测器 (复用 pipeline)...")
    try:
        cup_detector = RealSenseCup6DDetector(config_dict=_cfg, start_pipeline=False)
    except Exception as e:
        print(f"[FATAL] Cup 检测器初始化失败: {e}")
        try:
            handle_detector.pipeline.stop()
        except Exception:
            pass
        return

    # ---- [Step 4] 主循环: 同步检测 + 稳定化 ----
    handle_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW, pos_thresh=POS_THRESHOLD, angle_thresh=ANGLE_THRESHOLD
    )
    cup_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW, pos_thresh=POS_THRESHOLD, angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("estun Handle+Cup Demo (RealSense)", cv2.WINDOW_NORMAL)

    handle_stable_pose_ee: Any = None  # (pos_ee, euler_ee) or None
    cup_stable_pose_cam: Any = None  # (pos_cam, euler_cam) or None
    cup_stable_rim_pour: Optional[Tuple[np.ndarray, np.ndarray, float]] = None  # (rim_pos, pour_target, rim_r)

    running = True
    frame_counter = 0
    cached_handle = None
    cached_cup = None

    print(f"\n[INFO] 正在同步检测把手+水杯并稳定位姿 (跳帧: 每 {SKIP_FRAMES} 帧检测一次)...")
    print("[INFO] 两者均稳定后自动停止, 请输入机械臂位姿")
    print("[INFO] 按 q 退出\n")

    try:
        while running:
            # ---- 获取帧 (从 Handle 检测器) ----
            color_image, depth_image, depth_colormap, depth_frame = handle_detector.get_frames()
            if color_image is None or depth_frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出", flush=True)
                    running = False
                    break
                continue

            frame_counter += 1
            display_img = color_image.copy()

            # ---- 每 SKIP_FRAMES 帧执行一次检测 ----
            if frame_counter % SKIP_FRAMES == 1:
                # === Handle 检测 ===
                h_result = handle_detector.detect_handle_6d(color_image, depth_frame)
                body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler, red_mask, red_center = h_result

                if h_pos is not None and h_rot is not None:
                    smooth_pos, smooth_rot = handle_detector._pose_smoother.update(h_pos, h_rot)
                    if smooth_rot is not None:
                        h_pos = smooth_pos
                        h_rot = smooth_rot
                        h_euler = rotation_matrix_to_euler_angles(h_rot)

                    assert h_pos is not None and h_euler is not None
                    handle_stabilizer.add(tuple(h_pos), tuple(h_euler))
                    if handle_stabilizer.is_stable():
                        mean_pos_cam, mean_euler_cam = handle_stabilizer.get_mean()
                        if mean_pos_cam is not None and mean_euler_cam is not None:
                            mean_rot_cam = euler_to_rotation_matrix(*mean_euler_cam, order="zyx")
                            pos_ee = T_cam2ee[:3, :3] @ mean_pos_cam + T_cam2ee[:3, 3]
                            rot_ee = T_cam2ee[:3, :3] @ mean_rot_cam
                            A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)
                            handle_stable_pose_ee = (pos_ee, (A_ee, B_ee, C_ee))
                            print(f"\n[INFO] Handle 位姿已稳定!", flush=True)
                else:
                    handle_detector._pose_smoother.reset()

                # === Cup 检测 (注入帧到 Cup 检测器) ===
                cup_detector.inject_frames(color_image, depth_image, depth_colormap, depth_frame)
                (c_bbox, c_pos, c_rot, c_euler, c_point_cloud,
                 c_rim_pos, c_pour_target, c_rim_r) = cup_detector.compute_cup_6d_pose(color_image, depth_frame)

                if c_pos is not None and c_rot is not None:
                    smooth_cpos, smooth_crot = cup_detector._pose_smoother.update(c_pos, c_rot)
                    if smooth_crot is not None:
                        c_pos = smooth_cpos
                        c_rot = smooth_crot
                        c_euler = rotation_matrix_to_euler_angles(c_rot)

                    assert c_pos is not None and c_euler is not None
                    cup_stabilizer.add(tuple(c_pos), tuple(c_euler))
                    if cup_stabilizer.is_stable():
                        mean_cpos_cam, mean_ceuler_cam = cup_stabilizer.get_mean()
                        if mean_cpos_cam is not None and mean_ceuler_cam is not None:
                            cup_stable_pose_cam = (mean_cpos_cam, mean_ceuler_cam)
                            if c_rim_pos is not None and c_pour_target is not None and c_rim_r is not None:
                                cup_stable_rim_pour = (c_rim_pos, c_pour_target, c_rim_r)
                            else:
                                cup_stable_rim_pour = None
                            print(f"[INFO] Cup 位姿已稳定!", flush=True)
                else:
                    cup_detector._pose_smoother.reset()

                cached_handle = (body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler)
                cached_cup = (c_bbox, c_pos, c_rot, c_euler, c_point_cloud, c_rim_pos, c_pour_target, c_rim_r)

                # ---- 两者都稳定则退出主循环 ----
                if handle_stable_pose_ee is not None and cup_stable_pose_cam is not None:
                    print("[INFO] Handle + Cup 均稳定, 停止检测", flush=True)
                    running = False
                    break
            else:
                if cached_handle is not None:
                    body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler = cached_handle
                else:
                    body_mask = handle_mask = None
                    h_stage = "none"
                    yolo_bbox = h_pos = h_rot = h_euler = None

                if cached_cup is not None:
                    c_bbox, c_pos, c_rot, c_euler, c_point_cloud, c_rim_pos, c_pour_target, c_rim_r = cached_cup
                else:
                    c_bbox = c_pos = c_rot = c_euler = c_point_cloud = c_rim_pos = c_pour_target = c_rim_r = None

            # ---- 可视化 (Handle + Cup) ----
            if cached_handle is not None:
                handle_detector.visualize(display_img, body_mask, handle_mask,
                                          h_stage, yolo_bbox, h_pos, h_rot, h_euler)

            # ---- Cup 可视化 ----
            if cached_cup is not None and c_bbox is not None:
                cup_detector.visualize_cup(display_img, c_bbox, c_pos, c_rot, c_euler,
                                           c_rim_pos, c_pour_target, c_rim_r)

            # ---- 状态文字 ----
            h_status = "H:OK" if handle_stable_pose_ee is not None else "H:..."
            c_status = "C:OK" if cup_stable_pose_cam is not None else "C:..."
            cv2.putText(display_img, f"Status: {h_status} {c_status}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # ---- 显示 ----
            if depth_colormap is not None:
                h_vis = display_img.shape[0]
                depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                vis_panel = np.hstack((display_img, depth_small))
            else:
                vis_panel = display_img
            cv2.imshow("estun Handle+Cup Demo (RealSense)", vis_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                running = False
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        cv2.destroyAllWindows()

    # ---- 检查是否获得了稳定位姿 ----
    if handle_stable_pose_ee is None or cup_stable_pose_cam is None:
        print("[ERROR] 未获得稳定位姿, 退出")
        try:
            handle_detector.pipeline.stop()
        except Exception:
            pass
        return

    mean_hpos_ee, mean_heuler_ee = handle_stable_pose_ee
    mean_cpos_cam, mean_ceuler_cam = cup_stable_pose_cam
    if cup_stable_rim_pour is not None:
        rim_pos, pour_target, rim_r = cup_stable_rim_pour
    else:
        print("[ERROR] 未获得倒水目标点 (rim_pour 为 None), 退出")
        try:
            handle_detector.pipeline.stop()
        except Exception:
            pass
        return

    print(f"\n[INFO] 稳定 Handle 末端位姿: pos=({mean_hpos_ee[0]:.3f},{mean_hpos_ee[1]:.3f},{mean_hpos_ee[2]:.3f}) "
          f"euler=({mean_heuler_ee[0]:.1f},{mean_heuler_ee[1]:.1f},{mean_heuler_ee[2]:.1f})")
    print(f"[INFO] 稳定 Cup 相机位姿:   pos=({mean_cpos_cam[0]:.3f},{mean_cpos_cam[1]:.3f},{mean_cpos_cam[2]:.3f}) "
          f"euler=({mean_ceuler_cam[0]:.1f},{mean_ceuler_cam[1]:.1f},{mean_ceuler_cam[2]:.1f})")
    if pour_target is not None:
        print(f"[INFO] 倒水目标点:          ({pour_target[0]:.3f},{pour_target[1]:.3f},{pour_target[2]:.3f})")

    # ---- [Step 5] 控制台输入机械臂位姿 ----
    print("\n" + "-" * 50)
    print("请从示教器读取机械臂当前末端位姿")
    print("格式: [X,Y,Z,A,B,C]  (X,Y,Z 单位 mm, A,B,C 单位 deg)")
    print("示例: [500.0,200.0,400.0,-30.0,10.0,45.0]")
    print("-" * 50)

    robot_input = input("输入机械臂位姿: ").strip().strip("[]")
    parts = robot_input.split(",")
    if len(parts) != 6:
        print(f"[ERROR] 格式错误, 需要 6 个数值, 收到 {len(parts)} 个")
        try:
            handle_detector.pipeline.stop()
        except Exception:
            pass
        return

    try:
        robot_xyz_mm = [float(p) for p in parts[:3]]
        robot_ABC = tuple(float(p) for p in parts[3:6])
    except ValueError as e:
        print(f"[ERROR] 数值解析失败: {e}")
        try:
            handle_detector.pipeline.stop()
        except Exception:
            pass
        return

    robot_xyz = (robot_xyz_mm[0] / 1000.0, robot_xyz_mm[1] / 1000.0, robot_xyz_mm[2] / 1000.0)
    print(f"\n[INPUT] 机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m "
          f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg")

    # ---- [Step 6] 位姿转换链路 ----
    print("\n[INFO] 执行位姿转换...")

    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")

    # ============================================================
    # Phase 1: Handle P1 / P2 / P3_pour
    # ============================================================

    # Handle 位置补偿
    h_pos_ee_compensated = np.array(mean_hpos_ee) - t_handle_effective

    # Handle 方向对齐
    R_obj_ee = euler_to_rotation_matrix(*mean_heuler_ee, order="zyx")
    R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, HANDLE_R_OFFSET, R_ee_base,
                                        grasp_direction=GRASP_DIRECTION)

    if ORIENTATION_MODE == "fixed":
        # --- fixed 模式: Handle P1/P2 使用固定 ABC ---
        R_base_fix = euler_to_rotation_matrix(*FIXED_ABC, order="zyx")
        R_grasp_ee = R_ee_base.T @ R_base_fix

        pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
        A1, B1, C1 = tuple(FIXED_ABC)

        pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ HANDLE_PRE_T_OFFSET
        pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
        A2, B2, C2 = tuple(FIXED_ABC)

        R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")
        # P3_pour: 倒水位置
        pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
        pos_ee_p3 = pos_pour_ee - R_grasp_ee @ SPOUT_T_OFFSET
        pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
        rot_base_3 = R_ee_base @ R_target_ee
        A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
        A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)
    else:
        # --- vertical/free 模式 ---
        rot_base_1 = R_ee_base @ R_grasp_ee
        pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
        A1, B1, C1 = rotation_matrix_to_euler_angles(rot_base_1)
        A1, B1, C1 = post_correct_angles(A1, B1, C1, robot_ABC)
        if GRASP_DIRECTION == "right" and not (-100 <= C1 <= -80):
            C1 = -90.0

        rot_base_1_corrected = euler_to_rotation_matrix(A1, B1, C1, order="zyx")
        R_grasp_ee_corrected = R_ee_base.T @ rot_base_1_corrected

        # P2
        pos_ee_pre = h_pos_ee_compensated + R_grasp_ee_corrected @ HANDLE_PRE_T_OFFSET
        pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
        A2, B2, C2 = A1, B1, C1

        # P3_pour: 倒水位置
        # spout_offset 是在抓取末端系下定义的
        # 倒水时壶嘴对齐 pour_target, 反算抓取点 = pour_pose - spout_offset
        pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
        pos_ee_p3 = pos_pour_ee - R_grasp_ee_corrected @ SPOUT_T_OFFSET
        pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
        # 旋转: 在 P1 修正姿态上叠加倒水旋转
        R_pour_ee = R_grasp_ee_corrected @ euler_to_rotation_matrix(*POUR_R_OFFSET, order="zyx")
        rot_base_3 = R_ee_base @ R_pour_ee
        A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
        A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)

    # ============================================================
    # Phase 2: Cup P1 / P2 / P3
    # ============================================================

    # Cup cam → ee
    c_pos_ee = T_cam2ee[:3, :3] @ np.array(mean_cpos_cam) + T_cam2ee[:3, 3]
    c_rot_ee = T_cam2ee[:3, :3] @ euler_to_rotation_matrix(*mean_ceuler_cam, order="zyx")

    # Cup 位置补偿 (使用水杯偏移)
    c_pos_ee_compensated = c_pos_ee - t_cup_effective

    # Cup 方向对齐 (使用水杯 r_cup_offset)
    c_R_grasp_ee = compute_grasp_rotation(c_rot_ee, ORIENTATION_MODE, CUP_R_OFFSET, R_ee_base,
                                          grasp_direction=GRASP_DIRECTION)

    # Cup P1
    c_rot_base_1 = R_ee_base @ c_R_grasp_ee
    c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)
    cA1, cB1, cC1 = rotation_matrix_to_euler_angles(c_rot_base_1)
    cA1, cB1, cC1 = post_correct_angles(cA1, cB1, cC1, robot_ABC)
    if GRASP_DIRECTION == "right" and not (-100 <= cC1 <= -80):
        cC1 = -90.0

    c_rot_base_1_corrected = euler_to_rotation_matrix(cA1, cB1, cC1, order="zyx")
    c_R_grasp_ee_corrected = R_ee_base.T @ c_rot_base_1_corrected

    # Cup P2
    c_pos_ee_pre = c_pos_ee_compensated + c_R_grasp_ee_corrected @ CUP_PRE_T_OFFSET
    c_pos_base_2 = transform_pose_to_base(c_pos_ee_pre, robot_xyz, robot_ABC)
    cA2, cB2, cC2 = cA1, cB1, cC1

    # Cup P3
    c_R_target_ee = c_R_grasp_ee_corrected @ euler_to_rotation_matrix(*CUP_TARGET_R_OFFSET, order="zyx")
    c_pos_ee_target = c_pos_ee_compensated + c_R_grasp_ee_corrected @ CUP_TARGET_T_OFFSET
    c_rot_base_3 = R_ee_base @ c_R_target_ee
    c_pos_base_3 = transform_pose_to_base(c_pos_ee_target, robot_xyz, robot_ABC)
    cA3, cB3, cC3 = rotation_matrix_to_euler_angles(c_rot_base_3)
    cA3, cB3, cC3 = post_correct_angles(cA3, cB3, cC3, robot_ABC)

    # ---- 输出 6 个位姿 ----
    msg_hp1 = f"[{pos_base_1[0]:.3f},{pos_base_1[1]:.3f},{pos_base_1[2]:.3f},{A1:.2f},{B1:.2f},{C1:.2f}]"
    msg_hp2 = f"[{pos_base_2[0]:.3f},{pos_base_2[1]:.3f},{pos_base_2[2]:.3f},{A2:.2f},{B2:.2f},{C2:.2f}]"
    msg_hp3 = f"[{pos_base_3[0]:.3f},{pos_base_3[1]:.3f},{pos_base_3[2]:.3f},{A3:.2f},{B3:.2f},{C3:.2f}]"

    msg_cp1 = f"[{c_pos_base_1[0]:.3f},{c_pos_base_1[1]:.3f},{c_pos_base_1[2]:.3f},{cA1:.2f},{cB1:.2f},{cC1:.2f}]"
    msg_cp2 = f"[{c_pos_base_2[0]:.3f},{c_pos_base_2[1]:.3f},{c_pos_base_2[2]:.3f},{cA2:.2f},{cB2:.2f},{cC2:.2f}]"
    msg_cp3 = f"[{c_pos_base_3[0]:.3f},{c_pos_base_3[1]:.3f},{c_pos_base_3[2]:.3f},{cA3:.2f},{cB3:.2f},{cC3:.2f}]"

    print("\n" + "=" * 60)
    print("  Phase 1 — Handle")
    print(f"  H P1 (抓取点):   {msg_hp1}")
    print(f"  H P2 (预抓取点): {msg_hp2}")
    print(f"  H P3 (倒水点):   {msg_hp3}")
    print("  Phase 2 — Cup")
    print(f"  C P1 (抓取点):   {msg_cp1}")
    print(f"  C P2 (预抓取点): {msg_cp2}")
    print(f"  C P3 (递杯点):   {msg_cp3}")
    print("=" * 60)

    # ---- 写入日志 ----
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"estun_handle_cup_demo_{ts_now}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  estun 把手+水杯融合位姿转换验证 Demo\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- 配置参数 ---\n")
        f.write(f"orientation_mode: {ORIENTATION_MODE}\n")
        f.write(f"grasp_direction:  {GRASP_DIRECTION}\n")
        f.write(f"handle_side:      {HANDLE_SIDE}\n\n")

        f.write("--- 偏移参数 ---\n")
        f.write(f"t_handle_effective: ({t_handle_effective[0]:.4f}, {t_handle_effective[1]:.4f}, {t_handle_effective[2]:.4f})\n")
        f.write(f"t_cup_effective:    ({t_cup_effective[0]:.4f}, {t_cup_effective[1]:.4f}, {t_cup_effective[2]:.4f})\n")
        f.write(f"spout_offset:       ({SPOUT_T_OFFSET[0]:.4f}, {SPOUT_T_OFFSET[1]:.4f}, {SPOUT_T_OFFSET[2]:.4f})\n")
        f.write(f"pour_t_offset_base: ({POUR_T_OFFSET_BASE[0]:.4f}, {POUR_T_OFFSET_BASE[1]:.4f}, {POUR_T_OFFSET_BASE[2]:.4f})\n\n")

        f.write("--- 检测结果 ---\n")
        f.write(f"[Handle cam] pos=({mean_hpos_ee[0]:.4f},{mean_hpos_ee[1]:.4f},{mean_hpos_ee[2]:.4f}) "
                f"euler=({mean_heuler_ee[0]:.1f},{mean_heuler_ee[1]:.1f},{mean_heuler_ee[2]:.1f})\n")
        f.write(f"[Cup cam]    pos=({mean_cpos_cam[0]:.4f},{mean_cpos_cam[1]:.4f},{mean_cpos_cam[2]:.4f}) "
                f"euler=({mean_ceuler_cam[0]:.1f},{mean_ceuler_cam[1]:.1f},{mean_ceuler_cam[2]:.1f})\n")
        if pour_target is not None:
            f.write(f"[Pour target] ({pour_target[0]:.4f},{pour_target[1]:.4f},{pour_target[2]:.4f})\n")
        f.write(f"\n")

        f.write("--- 机械臂位姿 ---\n")
        f.write(f"xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m, "
                f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg\n\n")

        f.write("--- 6 个目标位姿 ---\n")
        f.write(f"Phase 1 — Handle:\n")
        f.write(f"  H P1 (抓取点):   {msg_hp1}\n")
        f.write(f"  H P2 (预抓取点): {msg_hp2}\n")
        f.write(f"  H P3 (倒水点):   {msg_hp3}\n\n")
        f.write(f"Phase 2 — Cup:\n")
        f.write(f"  C P1 (抓取点):   {msg_cp1}\n")
        f.write(f"  C P2 (预抓取点): {msg_cp2}\n")
        f.write(f"  C P3 (递杯点):   {msg_cp3}\n")

    print(f"[LOG] 日志已保存: {log_path}")

    # 清理资源
    print("\n[INFO] 清理资源...")
    try:
        handle_detector.pipeline.stop()
        print("[INFO] RealSense 管道已停止")
    except Exception as e:
        print(f"[WARN] 停止管道失败: {e}")


if __name__ == "__main__":
    main()