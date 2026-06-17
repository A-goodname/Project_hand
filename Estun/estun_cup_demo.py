# -*- coding: utf-8 -*-
"""
estun 水杯位姿转换验证 Demo (RealSense Cup 版)
=============================================
基于 estun_handle_cup_demo.py 剥离 Handle 检测部分, 仅保留 Cup 单检测链路.
使用 RealSense/test_cup_6D.py 的 RealSenseCup6DDetector (独立启动 pipeline, rs.align 简化版).

与 estun_handle_cup_demo.py 的差异:
  - 单检测器: RealSenseCup6DDetector (start_pipeline=True, 独立 pipeline)
  - 输出 3 个位姿: Cup P1 (抓取点) / P2 (预抓取点) / P3 (递杯点)
  - 无 Handle 检测/壶嘴/倒水相关逻辑
  - 无跳帧: 每帧实时检测 (YOLOv8m 推理快, 无需跳帧)
  - 支持 fine_tune_mode: 手动调整后反算新 t_cup_offset

输入:
  - RealSense/config.yaml              : 相机 / YOLO / cup 配置 (新相机)
  - Estun/estun_config.yaml            : 机器人 / cup_grasp 偏移 / fine_tune_mode
  - calib_data/calibration_result.yaml : 手眼标定矩阵

输出:
  - logs/estun_cup_demo_*.log          : 3 个位姿 + 中间结果
  - (fine_tune_mode) 建议的新 t_cup_offset 值

流程:
  [Step 1] 加载配置 (标定矩阵 / cup_grasp 偏移 / fine_tune_mode)
  [Step 2] 初始化 Cup 检测器 (start_pipeline=True)
  [Step 3] 主循环: 每帧 get_frames -> compute_cup_6d_pose -> cam2ee -> 稳定化
  [Step 4] 控制台输入机械臂当前位姿 [X,Y,Z,A,B,C]
  [Step 5] 位姿转换: 补偿 -> 对齐 -> 计算 P1/P2/P3
  [Step 6] 写入日志
  [Step 7] (可选) fine_tune_mode: 手动调整 -> 反算新偏移 -> 输出建议
"""

import os
import sys
import yaml
import cv2
import numpy as np
from datetime import datetime
from typing import Any, Optional, Tuple


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入位姿转换工具 ----
from Estun.utils_shared import (
    euler_to_rotation_matrix,
    transform_pose_to_base,
    compute_gripper_offset_transform,
    compute_grasp_rotation,
    post_correct_angles,
)
from RealSense.realsense_shared import rotation_matrix_to_euler_angles


# =============================================================
# 配置加载
# =============================================================

# ---- 加载 RealSense / YOLO / cup 配置 ----
_cfg_path = os.path.join(_project_root, "RealSense", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---- 加载 ESTUN 配置 ----
_estun_cfg_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

CALIB_RESULT_PATH = os.path.join(_project_root, _cfg_estun["calibration"]["result_path"])
STABLE_WINDOW = _cfg_estun["stabilization"]["window"]
POS_THRESHOLD = _cfg_estun["stabilization"]["pos_threshold"]
ANGLE_THRESHOLD = _cfg_estun["stabilization"]["angle_threshold"]

# ---- handle_side (从 RealSense config 读取, 需在 CUP_T_OFFSET 之前) ----
HANDLE_SIDE = _cfg.get("cup", {}).get("handle_side", "back")
print(f"[INFO] handle_side: {HANDLE_SIDE}")

# ---- cup_grasp 偏移 (根据 handle_side 选择对应偏移) ----
if HANDLE_SIDE == "left":
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset_left"], dtype=float)
    print(f"[INFO] 使用 left 模式偏移 t_cup_offset_left")
else:
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset"], dtype=float)
    print(f"[INFO] 使用 back 模式偏移 t_cup_offset")
CUP_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["r_cup_offset"], dtype=float)
CUP_PRE_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["t_offset"], dtype=float)
CUP_PRE_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["r_offset"], dtype=float)
CUP_TARGET_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["t_offset"], dtype=float)
CUP_TARGET_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 微调模式开关 ----
FINE_TUNE_MODE = _cfg_estun["cup_grasp"]["fine_tune_mode"]

# ---- 方向模式 ----
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]

_VALID_MODES = {"vertical", "horizontal", "free", "fixed"}
if ORIENTATION_MODE not in _VALID_MODES:
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
    print("  estun 水杯位姿转换验证 Demo (RealSense Cup)")
    print("  标定: " + CALIB_RESULT_PATH)
    if FINE_TUNE_MODE:
        print("  [FINE TUNE] 微调模式已启用")
    print("=" * 60, flush=True)

    # ---- [Step 1a] 加载手眼标定矩阵 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")

    # ---- [Step 1b] 预计算工具偏移 ----
    _, t_cup_effective = compute_gripper_offset_transform(CUP_T_OFFSET, CUP_R_OFFSET)
    print(f"[INFO] 水杯偏移 t_cup_effective: ({t_cup_effective[0]:.4f}, {t_cup_effective[1]:.4f}, {t_cup_effective[2]:.4f}) m")

    # ---- [Step 2] 初始化 Cup 检测器 (启动独立 pipeline) ----
    from RealSense.test_cup_6D import RealSenseCup6DDetector
    print("[INFO] 初始化 RealSense Cup 6D 检测器 (新相机, rs.align 版)...")
    try:
        cup_detector = RealSenseCup6DDetector(config_dict=_cfg, start_pipeline=True)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        return

    # ---- [Step 3] 主循环: 每帧实时检测 + 稳定化 ----
    cup_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW, pos_thresh=POS_THRESHOLD, angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("estun Cup Demo (RealSense)", cv2.WINDOW_NORMAL)

    cup_stable_pose_cam: Any = None  # (pos_cam, euler_cam) or None

    running = True

    print("\n[INFO] 实时检测水杯位姿 (每帧检测)...")
    print("[INFO] 位姿稳定后自动停止, 请输入机械臂位姿")
    print("[INFO] 按 q 退出\n")

    try:
        while running:
            # ---- 获取帧 ----
            color_image, depth_image, depth_colormap, depth_frame = cup_detector.get_frames()
            if color_image is None or depth_frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出", flush=True)
                    running = False
                    break
                continue

            display_img = color_image.copy()

            # ---- 每帧执行 Cup 检测 ----
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
                        print(f"\n[INFO] Cup 位姿已稳定!", flush=True)
                        running = False
                        break
            else:
                cup_detector._pose_smoother.reset()

            # ---- 可视化 ----
            if c_bbox is not None:
                cup_detector.visualize_cup(display_img, c_bbox, c_pos, c_rot, c_euler,
                                           c_rim_pos, c_pour_target, c_rim_r)

            # ---- 状态文字 ----
            c_status = "OK" if cup_stable_pose_cam is not None else f"frame..."
            cv2.putText(display_img, f"Cup: {c_status}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # ---- 显示 ----
            if depth_colormap is not None:
                h_vis = display_img.shape[0]
                depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                vis_panel = np.hstack((display_img, depth_small))
            else:
                vis_panel = display_img
            cv2.imshow("estun Cup Demo (RealSense)", vis_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                running = False
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        cv2.destroyAllWindows()

    # ---- 检查是否获得了稳定位姿 ----
    if cup_stable_pose_cam is None:
        print("[ERROR] 未获得稳定位姿, 退出")
        try:
            cup_detector.pipeline.stop()
        except Exception:
            pass
        return

    mean_cpos_cam, mean_ceuler_cam = cup_stable_pose_cam

    print(f"\n[INFO] 稳定 Cup 相机位姿: pos=({mean_cpos_cam[0]:.3f},{mean_cpos_cam[1]:.3f},{mean_cpos_cam[2]:.3f}) "
          f"euler=({mean_ceuler_cam[0]:.1f},{mean_ceuler_cam[1]:.1f},{mean_ceuler_cam[2]:.1f})")

    # ---- [Step 4] 控制台输入机械臂位姿 ----
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
            cup_detector.pipeline.stop()
        except Exception:
            pass
        return

    try:
        robot_xyz_mm = [float(p) for p in parts[:3]]
        robot_ABC = tuple(float(p) for p in parts[3:6])
    except ValueError as e:
        print(f"[ERROR] 数值解析失败: {e}")
        try:
            cup_detector.pipeline.stop()
        except Exception:
            pass
        return

    robot_xyz = (robot_xyz_mm[0] / 1000.0, robot_xyz_mm[1] / 1000.0, robot_xyz_mm[2] / 1000.0)
    print(f"\n[INPUT] 机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m "
          f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg")

    # ---- [Step 5] 位姿转换链路 ----
    print("\n[INFO] 执行位姿转换...")

    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")

    # ============================================================
    # Phase: Cup P1 / P2 / P3
    # ============================================================

    # Cup cam -> ee
    c_pos_ee = T_cam2ee[:3, :3] @ np.array(mean_cpos_cam) + T_cam2ee[:3, 3]
    c_rot_ee = T_cam2ee[:3, :3] @ euler_to_rotation_matrix(*mean_ceuler_cam, order="zyx")

    # Cup 位置补偿 (使用水杯偏移)
    c_pos_ee_compensated = c_pos_ee - t_cup_effective

    # Cup 方向对齐 (使用水杯 r_cup_offset)
    c_R_grasp_ee = compute_grasp_rotation(c_rot_ee, ORIENTATION_MODE, CUP_R_OFFSET, R_ee_base,
                                          grasp_direction=GRASP_DIRECTION)

    # Cup P1 (抓取点)
    c_rot_base_1 = R_ee_base @ c_R_grasp_ee
    c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)
    cA1, cB1, cC1 = rotation_matrix_to_euler_angles(c_rot_base_1)
    cA1, cB1, cC1 = post_correct_angles(cA1, cB1, cC1, robot_ABC)
    # ---- 根据 handle_side 施加角度约束 ----
    if HANDLE_SIDE == "back":
        # back 模式: 把手背对相机, 灵巧手右侧抓取
        # B 角 clamp 到 [70, 90], 使灵巧手从水平方向接近水杯
        if not (70 <= cB1 <= 90):
            print(f"[CLAMP] Cup P1 B 角 {cB1:.1f} -> 80.0 (back 模式)")
            cB1 = 80.0
        # C 角 clamp 到 [-120, -100], 使灵巧手手指垂直于杯身
        if not (-120 <= cC1 <= -100):
            print(f"[CLAMP] Cup P1 C 角 {cC1:.1f} -> -110.0 (back 模式)")
            cC1 = -110.0
    else:
        # left 模式: 把手在相机左侧, 灵巧手右侧抓取 (递给人)
        # B 角暂不约束, 由 compute_grasp_rotation 自然决定 (杯身竖直时 B 角天然合理)
        # C 角 clamp 到 [-100, -80], 默认 -90, 参考水壶约束
        if not (-100 <= cC1 <= -80):
            print(f"[CLAMP] Cup P1 C 角 {cC1:.1f} -> -90.0 (left 模式, 参考水壶)")
            cC1 = -90.0

    c_rot_base_1_corrected = euler_to_rotation_matrix(cA1, cB1, cC1, order="zyx")
    c_R_grasp_ee_corrected = R_ee_base.T @ c_rot_base_1_corrected

    # Cup P2 (预抓取点)
    c_pos_ee_pre = c_pos_ee_compensated + c_R_grasp_ee_corrected @ CUP_PRE_T_OFFSET
    c_pos_base_2 = transform_pose_to_base(c_pos_ee_pre, robot_xyz, robot_ABC)
    cA2, cB2, cC2 = cA1, cB1, cC1

    # Cup P3 (递杯点)
    c_R_target_ee = c_R_grasp_ee_corrected @ euler_to_rotation_matrix(*CUP_TARGET_R_OFFSET, order="zyx")
    c_pos_ee_target = c_pos_ee_compensated + c_R_grasp_ee_corrected @ CUP_TARGET_T_OFFSET
    c_rot_base_3 = R_ee_base @ c_R_target_ee
    c_pos_base_3 = transform_pose_to_base(c_pos_ee_target, robot_xyz, robot_ABC)
    cA3, cB3, cC3 = rotation_matrix_to_euler_angles(c_rot_base_3)
    cA3, cB3, cC3 = post_correct_angles(cA3, cB3, cC3, robot_ABC)

    # ---- 输出 3 个位姿 ----
    msg_p1 = f"[{c_pos_base_1[0]:.3f},{c_pos_base_1[1]:.3f},{c_pos_base_1[2]:.3f},{cA1:.2f},{cB1:.2f},{cC1:.2f}]"
    msg_p2 = f"[{c_pos_base_2[0]:.3f},{c_pos_base_2[1]:.3f},{c_pos_base_2[2]:.3f},{cA2:.2f},{cB2:.2f},{cC2:.2f}]"
    msg_p3 = f"[{c_pos_base_3[0]:.3f},{c_pos_base_3[1]:.3f},{c_pos_base_3[2]:.3f},{cA3:.2f},{cB3:.2f},{cC3:.2f}]"

    print("\n" + "=" * 60)
    print("  Cup 位姿")
    print(f"  P1 (抓取点):   {msg_p1}")
    print(f"  P2 (预抓取点): {msg_p2}")
    print(f"  P3 (递杯点):   {msg_p3}")
    print("=" * 60)

    # ---- [Step 6] 写入日志 ----
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"estun_cup_demo_{ts_now}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  estun 水杯位姿转换验证 Demo\n")
        f.write("=" * 60 + "\n\n")

        f.write("--- 配置参数 ---\n")
        f.write(f"orientation_mode: {ORIENTATION_MODE}\n")
        f.write(f"grasp_direction:  {GRASP_DIRECTION}\n")
        f.write(f"fine_tune_mode:   {FINE_TUNE_MODE}\n")
        f.write(f"handle_side:      {HANDLE_SIDE}\n\n")

        f.write("--- 偏移参数 ---\n")
        f.write(f"t_cup_effective: ({t_cup_effective[0]:.4f}, {t_cup_effective[1]:.4f}, {t_cup_effective[2]:.4f})\n")
        f.write(f"t_cup_offset:    ({CUP_T_OFFSET[0]:.4f}, {CUP_T_OFFSET[1]:.4f}, {CUP_T_OFFSET[2]:.4f})\n")
        f.write(f"pre_t_offset:    ({CUP_PRE_T_OFFSET[0]:.4f}, {CUP_PRE_T_OFFSET[1]:.4f}, {CUP_PRE_T_OFFSET[2]:.4f})\n")
        f.write(f"target_t_offset: ({CUP_TARGET_T_OFFSET[0]:.4f}, {CUP_TARGET_T_OFFSET[1]:.4f}, {CUP_TARGET_T_OFFSET[2]:.4f})\n\n")

        f.write("--- 检测结果 ---\n")
        f.write(f"[Cup cam] pos=({mean_cpos_cam[0]:.4f},{mean_cpos_cam[1]:.4f},{mean_cpos_cam[2]:.4f}) "
                f"euler=({mean_ceuler_cam[0]:.1f},{mean_ceuler_cam[1]:.1f},{mean_ceuler_cam[2]:.1f})\n\n")

        f.write("--- 机械臂位姿 ---\n")
        f.write(f"xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m, "
                f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg\n\n")

        f.write("--- 3 个目标位姿 ---\n")
        f.write(f"P1 (抓取点):   {msg_p1}\n")
        f.write(f"P2 (预抓取点): {msg_p2}\n")
        f.write(f"P3 (递杯点):   {msg_p3}\n")

    print(f"[LOG] 日志已保存: {log_path}")

    # ---- [Step 7] (可选) fine_tune_mode: 手动调整 -> 反算新偏移 ----
    if FINE_TUNE_MODE:
        print("\n" + "=" * 60)
        print("  [FINE TUNE] 微调模式已启用")
        print("=" * 60)
        print()
        print("请将机械臂移动到上述目标位姿, 然后手动微调")
        print("使灵巧手掌心正对水杯, 手指包覆杯身")
        print("输入微调后的实际机械臂位姿 [X,Y,Z,A,B,C]")
        print("(X,Y,Z 单位 mm, A,B,C 单位 deg, 仅使用 XYZ 计算, ABC 记录)")
        print("直接回车跳过微调")

        adj_input = input("微调后位姿: ").strip()
        if adj_input:
            adj_input = adj_input.strip("[]")
            adj_parts = adj_input.split(",")
            if len(adj_parts) == 6:
                try:
                    adj_xyz = np.array([float(p) / 1000.0 for p in adj_parts[:3]])

                    # 增量公式: t_eff_new = t_cup_effective + R_ee_base.T @ (P1_xyz - adj_xyz)
                    diff_xyz = np.array([c_pos_base_1[0], c_pos_base_1[1], c_pos_base_1[2]]) - adj_xyz
                    delta_ee = R_ee_base.T @ diff_xyz
                    t_effective_new = t_cup_effective + delta_ee

                    # 反推标准偏移 (C=0 时的 t_cup_offset)
                    Rz = euler_to_rotation_matrix(0, 0, CUP_R_OFFSET[2], order="zyx")  # Rz(CUP_R_OFFSET[2])
                    t_offset_new = Rz.T @ t_effective_new

                    offset_name = "t_cup_offset_left" if HANDLE_SIDE == "left" else "t_cup_offset"
                    print(f"\n[FINE TUNE] 建议新偏移 ({HANDLE_SIDE} 模式):")
                    print(f"  t_effective_new = ({t_effective_new[0]:.4f}, {t_effective_new[1]:.4f}, {t_effective_new[2]:.4f})")
                    print(f"  {offset_name}    = ({t_offset_new[0]:.4f}, {t_offset_new[1]:.4f}, {t_offset_new[2]:.4f})")

                    # 追加写入日志
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("\n--- fine_tune 结果 ---\n")
                        f.write(f"当前 handle_side: {HANDLE_SIDE}\n")
                        f.write(f"调整后 XYZ: ({adj_xyz[0]:.4f},{adj_xyz[1]:.4f},{adj_xyz[2]:.4f}) m\n")
                        f.write(f"建议 t_effective_new: ({t_effective_new[0]:.4f},{t_effective_new[1]:.4f},{t_effective_new[2]:.4f})\n")
                        f.write(f"建议 {offset_name}:    ({t_offset_new[0]:.4f},{t_offset_new[1]:.4f},{t_offset_new[2]:.4f})\n")

                    print(f"[LOG] 微调结果已追加至: {log_path}")

                except ValueError as e:
                    print(f"[ERROR] 数值解析失败: {e}")
            else:
                print(f"[WARN] 格式错误, 跳过微调 (需要 6 个数值, 收到 {len(adj_parts)} 个)")

    # 清理资源
    print("\n[INFO] 清理资源...")
    try:
        cup_detector.pipeline.stop()
        print("[INFO] RealSense 管道已停止")
    except Exception as e:
        print(f"[WARN] 停止管道失败: {e}")


if __name__ == "__main__":
    main()