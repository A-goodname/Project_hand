# -*- coding: utf-8 -*-
"""
estun 把手位姿转换验证 Demo (RealSense Handle 版)
==================================================
剥离 estun_handle_run 的位姿转换模块, 保留检测 + 稳定化 + 位姿转换完整链路.
去除 TCP 服务器 / 灵巧手 / CRP 指令循环等硬件/UI 依赖.

与 estun_demo.py 的差异:
  - 检测器: RealSenseHandle6DDetector (YOLO-World + SAM + 扫描线分离把手)
  - 偏移:   使用 t_handle_offset (抓取把手), 独立于瓶身 t_offset
  - 跳帧:   每 SKIP_FRAMES 帧调用一次 detect_handle_6d
  - 微调:   fine_tune_mode=True 时, 输出目标位姿后等待手动调整, 反算新偏移

输入:
  - RealSense_R/config.yaml           : 相机 / YOLO-World / SAM 配置
  - estun_config.yaml                 : 把手偏移 / 抓取模式
  - calib_data/calibration_result.yaml: 手眼标定矩阵

输出:
  - logs/estun_handle_demo_*.log      : 转换结果日志
  - (fine_tune_mode) 建议的新 t_handle_offset 值

流程:
  [Step 1] 加载配置 (标定矩阵 / 夹爪偏移)
  [Step 2] 初始化 RealSense 相机 + Handle6D 检测器
  [Step 3] 主循环: get_frames -> detect_handle_6d -> cam2ee -> 稳定化
  [Step 4] 控制台输入机械臂当前位姿 [X,Y,Z,A,B,C]
  [Step 5] 位姿转换: 补偿 -> 对齐 -> base -> 可达性修正
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
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入位姿转换工具 (纯 numpy, 无硬件依赖) ----
from utils_shared import (
    euler_to_rotation_matrix,
    transform_pose_to_base,
    compute_gripper_offset_transform,
    compute_grasp_rotation,
    post_correct_angles,
)


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

# ---- 把手偏移 (独立于瓶身) ----
HANDLE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["t_handle_offset"], dtype=float)
HANDLE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["r_handle_offset"], dtype=float)

# ---- 把手抓取偏移参数 ----
HANDLE_PRE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["t_offset"], dtype=float)
HANDLE_PRE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["r_offset"], dtype=float)
HANDLE_TARGET_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["t_offset"], dtype=float)
HANDLE_TARGET_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 方向模式 ----
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]
HANDLE_SIDE = _cfg["handle"]["handle_side"]  # 视觉检测侧

# 把手模式下: 抓取方向自动与 handle_side 一致
if HANDLE_SIDE != GRASP_DIRECTION:
    print(f"[INFO] grasp_direction ({GRASP_DIRECTION}) 自动切换为 handle_side ({HANDLE_SIDE})")
    GRASP_DIRECTION = HANDLE_SIDE

# ---- 把手检测参数 (skip_frames 来自 RealSense_R/config.yaml) ----
FINE_TUNE_MODE = _cfg_estun["handle_detection"]["fine_tune_mode"]
SKIP_FRAMES = _cfg["handle"]["skip_frames"]

VALID_MODES = {"vertical", "horizontal", "free"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE}, 将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现, 将使用 free 模式")
    ORIENTATION_MODE = "free"


def load_cam2ee(yaml_path: str) -> np.ndarray:
    """
    从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)

    参数:
        yaml_path : str, 标定结果 YAML 路径

    返回:
        np.ndarray, 4x4 齐次矩阵; 文件缺失/格式错误时返回单位矩阵
    """
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
# 工具: 旋转矩阵 -> 欧拉角 (ZYX 顺序)
# =============================================================

def rotation_matrix_to_euler_angles(R: np.ndarray) -> Tuple[float, float, float]:
    """
    从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw)

    参数:
        R : np.ndarray, 3x3 旋转矩阵

    返回:
        tuple: (A, B, C) 角度值 (deg)
    """
    if abs(abs(R[2, 0]) - 1.0) < 1e-6:
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = 0.0
            yaw = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = 0.0
            yaw = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])

    return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)


# =============================================================
# 功能模块: 位姿稳定器 (精简版, 无代际检查)
# =============================================================

class PoseStabilizer:
    """
    缓存最近 N 帧位姿数据; 当位置和角度变化均低于阈值时,
    认为位姿稳定, 返回缓冲区均值作为稳定结果
    """

    def __init__(self, window: int = 5, pos_thresh: float = 0.005, angle_thresh: float = 3.0):
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self._buffer: list = []

    def add(self, pos: Tuple[float, float, float], euler: Tuple[float, float, float]):
        """
        添加一帧位姿数据

        参数:
            pos   : (float, float, float), 位置 (x, y, z), 米
            euler : (float, float, float), 欧拉角 (A, B, C), 度
        """
        self._buffer.append((np.array(pos, dtype=float), np.array(euler, dtype=float)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self) -> bool:
        """检查缓冲的位姿是否稳定"""
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
        """返回缓冲区内位姿均值"""
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
    print("  estun 把手位姿转换验证 Demo (RealSense Handle)")
    print("  标定: " + CALIB_RESULT_PATH)
    print("  fine_tune_mode:", FINE_TUNE_MODE)
    print("=" * 60, flush=True)

    # ---- [Step 1a] 加载手眼标定矩阵 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")
    else:
        print("[INFO] 标定矩阵已加载, 位姿将转换到末端坐标系")

    # ---- [Step 1b] 预计算把手工具偏移 ----
    _, t_handle_effective = compute_gripper_offset_transform(HANDLE_T_OFFSET, HANDLE_R_OFFSET)
    print(f"[INFO] 把手偏移 t_handle_effective: ({t_handle_effective[0]:.4f}, {t_handle_effective[1]:.4f}, {t_handle_effective[2]:.4f}) m")

    # ---- [Step 2] 初始化把手检测器 ----
    from RealSense_R.test_handle_6D import RealSenseHandle6DDetector

    print("[INFO] 初始化 RealSense Handle 6D 检测器...")
    try:
        detector = RealSenseHandle6DDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        return

    # ---- [Step 3] 主循环: 获取帧 -> 把手检测 -> cam2ee -> 稳定化 ----
    stabilizer = PoseStabilizer(
        window=STABLE_WINDOW,
        pos_thresh=POS_THRESHOLD,
        angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("estun Handle Demo (RealSense)", cv2.WINDOW_NORMAL)

    stable_pose_ee: Optional[Tuple[Any, Any]] = None
    running = True
    frame_counter = 0
    cached_handle = None  # (body_mask, handle_mask, stage, bbox, pos, rot, euler)

    print(f"\n[INFO] 正在检测把手并稳定位姿 (跳帧: 每 {SKIP_FRAMES} 帧检测一次)...")
    print("[INFO] 稳定后自动停止, 请输入机械臂位姿")
    print("[INFO] 按 q 退出\n")

    try:
        while running:
            # ---- [Step 3a] 获取帧 ----
            color_image, depth_image, depth_colormap, depth_frame = detector.get_frames()
            if color_image is None or depth_frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出", flush=True)
                    running = False
                    break
                continue

            frame_counter += 1
            display_img = color_image.copy()

            # ---- [Step 3b] 每 SKIP_FRAMES 帧执行一次检测 ----
            if frame_counter % SKIP_FRAMES == 1:
                result = detector.detect_handle_6d(color_image, depth_frame)
                body_mask, handle_mask, stage, yolo_bbox, handle_pos, handle_rot, handle_euler = result

                # 内部 EMA 平滑 (与 test_handle_6D 的 run() 一致)
                if handle_pos is not None and handle_rot is not None:
                    smooth_pos, smooth_rot = detector._pose_smoother.update(handle_pos, handle_rot)
                    if smooth_rot is not None:
                        handle_pos = smooth_pos
                        handle_rot = smooth_rot
                        handle_euler = rotation_matrix_to_euler_angles(handle_rot)
                else:
                    detector._pose_smoother.reset()

                cached_handle = (body_mask, handle_mask, stage, yolo_bbox, handle_pos, handle_rot, handle_euler)
            else:
                if cached_handle is not None:
                    body_mask, handle_mask, stage, yolo_bbox, \
                        handle_pos, handle_rot, handle_euler = cached_handle
                else:
                    body_mask = handle_mask = None
                    stage = "none"
                    yolo_bbox = handle_pos = handle_rot = handle_euler = None

            # ---- [Step 3c] 可视化 ----
            if cached_handle is not None:
                body_mask, handle_mask, _, _, _, _, _ = cached_handle
            detector.visualize(display_img, body_mask, handle_mask,
                               stage, yolo_bbox, handle_pos, handle_rot, handle_euler)

            # ---- [Step 3d] 有有效位姿时: cam -> ee -> 稳定化 ----
            if handle_pos is not None and handle_rot is not None:
                # cam -> ee
                pos_ee = T_cam2ee[:3, :3] @ np.array(handle_pos) + T_cam2ee[:3, 3]
                rot_ee = T_cam2ee[:3, :3] @ handle_rot
                A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)

                # 可视化 ee 系位姿信息
                text_pos = f"ee Pos: ({pos_ee[0]:.2f}, {pos_ee[1]:.2f}, {pos_ee[2]:.2f})"
                cv2.putText(display_img, text_pos, (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                text_euler = f"ee Euler: (A={A_ee:.1f}, B={B_ee:.1f}, C={C_ee:.1f})"
                cv2.putText(display_img, text_euler, (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 稳定化
                stabilizer.add(tuple(pos_ee), (A_ee, B_ee, C_ee))
                if stabilizer.is_stable():
                    mean_pos, mean_euler = stabilizer.get_mean()
                    if mean_pos is not None and mean_euler is not None:
                        stable_pose_ee = (mean_pos, mean_euler)
                        print("\n[INFO] 位姿已稳定, 准备接收机械臂位姿", flush=True)
                        running = False
                        break

                print(f"\r[Handle 6D] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                      f"euler=({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})    ", end="", flush=True)

            # ---- [Step 3e] 显示 ----
            if depth_colormap is not None:
                h_vis = display_img.shape[0]
                depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                vis_panel = np.hstack((display_img, depth_small))
            else:
                vis_panel = display_img
            cv2.imshow("estun Handle Demo (RealSense)", vis_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                running = False
                break

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断")
    finally:
        cv2.destroyAllWindows()

    # ---- 检查是否获得了稳定位姿 ----
    if stable_pose_ee is None:
        print("[ERROR] 未获得稳定位姿, 退出")
        try:
            detector.pipeline.stop()
        except Exception:
            pass
        return

    mean_pos_ee, mean_euler_ee = stable_pose_ee
    print(f"\n[INFO] 稳定末端位姿: pos=({mean_pos_ee[0]:.3f},{mean_pos_ee[1]:.3f},{mean_pos_ee[2]:.3f}) "
          f"euler=({mean_euler_ee[0]:.1f},{mean_euler_ee[1]:.1f},{mean_euler_ee[2]:.1f})")

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
            detector.pipeline.stop()
        except Exception:
            pass
        return

    try:
        robot_xyz_mm = [float(p) for p in parts[:3]]
        robot_ABC = tuple(float(p) for p in parts[3:6])
    except ValueError as e:
        print(f"[ERROR] 数值解析失败: {e}")
        try:
            detector.pipeline.stop()
        except Exception:
            pass
        return

    robot_xyz = (robot_xyz_mm[0] / 1000.0, robot_xyz_mm[1] / 1000.0, robot_xyz_mm[2] / 1000.0)
    print(f"\n[INPUT] 机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m "
          f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg")

    # ---- [Step 5] 位姿转换链路 (P1 抓取点 / P2 预抓取点 / P3 目标点) ----
    print("\n[INFO] 执行位姿转换...")

    # 5a. 位置补偿 (使用把手偏移)
    pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
    print(f"  [补偿]    pos_compensated=({pos_ee_compensated[0]:.3f},{pos_ee_compensated[1]:.3f},{pos_ee_compensated[2]:.3f})")

    # 5b. 方向对齐 (使用把手 r_handle_offset)
    R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
    R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, HANDLE_R_OFFSET, R_ee_base, grasp_direction=GRASP_DIRECTION)
    print(f"  [对齐]    mode={ORIENTATION_MODE}")

    # === P1: 抓取点 ===
    rot_base_1 = R_ee_base @ R_grasp_ee
    pos_base_1 = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
    A1, B1, C1 = rotation_matrix_to_euler_angles(rot_base_1)
    A1, B1, C1 = post_correct_angles(A1, B1, C1, robot_ABC)

    # === P2: 预抓取点 (末端系下先平移再旋转) ===
    R_pre_ee = R_grasp_ee @ euler_to_rotation_matrix(*HANDLE_PRE_R_OFFSET, order="zyx")
    pos_ee_pre = pos_ee_compensated + HANDLE_PRE_T_OFFSET
    rot_base_2 = R_ee_base @ R_pre_ee
    pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
    A2, B2, C2 = rotation_matrix_to_euler_angles(rot_base_2)
    A2, B2, C2 = post_correct_angles(A2, B2, C2, robot_ABC)

    # === P3: 目标点 (末端系下先平移再旋转) ===
    R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")
    pos_ee_target = pos_ee_compensated + HANDLE_TARGET_T_OFFSET
    rot_base_3 = R_ee_base @ R_target_ee
    pos_base_3 = transform_pose_to_base(pos_ee_target, robot_xyz, robot_ABC)
    A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
    A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)

    # ---- [Step 6] 输出三个位姿 ----
    msg_p1 = (
        f"[{pos_base_1[0]:.3f},{pos_base_1[1]:.3f},{pos_base_1[2]:.3f},"
        f"{A1:.2f},{B1:.2f},{C1:.2f}]"
    )
    msg_p2 = (
        f"[{pos_base_2[0]:.3f},{pos_base_2[1]:.3f},{pos_base_2[2]:.3f},"
        f"{A2:.2f},{B2:.2f},{C2:.2f}]"
    )
    msg_p3 = (
        f"[{pos_base_3[0]:.3f},{pos_base_3[1]:.3f},{pos_base_3[2]:.3f},"
        f"{A3:.2f},{B3:.2f},{C3:.2f}]"
    )

    print("\n" + "=" * 60)
    print(f"  P1 (抓取点):   {msg_p1}")
    print(f"  P2 (预抓取点): {msg_p2}")
    print(f"  P3 (目标点):   {msg_p3}")
    print("=" * 60)

    # 写入日志
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"estun_handle_demo_{ts_now}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  estun 把手位姿转换验证 Demo (RealSense Handle)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"检测位姿 (cam): X={mean_pos_ee[0]:.3f}, Y={mean_pos_ee[1]:.3f}, Z={mean_pos_ee[2]:.3f}, "
                f"A={mean_euler_ee[0]:.1f}, B={mean_euler_ee[1]:.1f}, C={mean_euler_ee[2]:.1f}\n\n")

        f.write(f"机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m, "
                f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg\n\n")

        f.write("--- 中间结果 ---\n")
        f.write(f"[cam->ee] pos_ee=({mean_pos_ee[0]:.4f},{mean_pos_ee[1]:.4f},{mean_pos_ee[2]:.4f}) "
                f"A_ee={mean_euler_ee[0]:.2f}, B_ee={mean_euler_ee[1]:.2f}, C_ee={mean_euler_ee[2]:.2f}\n")
        f.write(f"[补偿] pos_ee_compensated=({pos_ee_compensated[0]:.4f},{pos_ee_compensated[1]:.4f},{pos_ee_compensated[2]:.4f})\n")
        f.write(f"[对齐] mode={ORIENTATION_MODE}\n")
        f.write(f"\nP1 (抓取点):   {msg_p1}\n")
        f.write(f"P2 (预抓取点): {msg_p2}\n")
        f.write(f"P3 (目标点):   {msg_p3}\n")

    print(f"[LOG] 日志已保存: {log_path}")

    # ---- [Step 7] fine_tune_mode: 手动调整 -> 反算新偏移 ----
    if FINE_TUNE_MODE:
        print("\n" + "=" * 60)
        print("  [FINE TUNE] 微调模式已启用")
        print("=" * 60)
        print()
        print("请将机械臂移动到上述目标位姿, 然后手动微调")
        print("使灵巧手掌心正对把手, 手指穿过把手孔洞")
        print("输入微调后的实际机械臂位姿 [X,Y,Z,A,B,C]")
        print("(X,Y,Z 单位 mm, A,B,C 单位 deg)")
        print("直接回车跳过微调")

        adj_input = input("微调后位姿: ").strip()
        if adj_input:
            adj_input = adj_input.strip("[]")
            adj_parts = adj_input.split(",")
            if len(adj_parts) == 6:
                try:
                    adj_xyz = np.array([float(p) / 1000.0 for p in adj_parts[:3]])
                    adj_ABC = np.array([float(p) for p in adj_parts[3:6]])

                    # 反求新偏移:
                    # 物体在基系下位置 (不变) = R_ee_base_orig @ pos_ee_orig + robot_xyz
                    R_ee_base_orig = euler_to_rotation_matrix(*robot_ABC, order="zyx")
                    p_obj_base = R_ee_base_orig @ mean_pos_ee + np.array(robot_xyz)

                    # 物体在调整后末端系下位置
                    R_ee_base_adj = euler_to_rotation_matrix(*adj_ABC, order="zyx")
                    pos_ee_current = R_ee_base_adj.T @ (p_obj_base - adj_xyz)

                    # 正确偏移 = pos_ee_current (物体在当前末端系下位置)
                    t_effective_new = pos_ee_current

                    # 反推标准偏移 (C=0 时的 t_offset)
                    Rz = euler_to_rotation_matrix(0, 0, HANDLE_R_OFFSET[2], order="zyx")  # Rz(-90°)
                    t_offset_new = Rz.T @ t_effective_new  # = Rz(90°) @ t_effective_new

                    print(f"\n[FINE TUNE] 建议新偏移:")
                    print(f"  t_effective_new = ({t_effective_new[0]:.4f}, {t_effective_new[1]:.4f}, {t_effective_new[2]:.4f})")
                    print(f"  t_handle_offset = ({t_offset_new[0]:.4f}, {t_offset_new[1]:.4f}, {t_offset_new[2]:.4f})")

                    # 追加写入日志
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write("\n--- fine_tune 结果 ---\n")
                        f.write(f"调整后机械臂位姿: xyz=({adj_xyz[0]:.4f},{adj_xyz[1]:.4f},{adj_xyz[2]:.4f}) m, "
                                f"ABC=({adj_ABC[0]:.2f},{adj_ABC[1]:.2f},{adj_ABC[2]:.2f}) deg\n")
                        f.write(f"建议 t_effective_new: ({t_effective_new[0]:.4f},{t_effective_new[1]:.4f},{t_effective_new[2]:.4f})\n")
                        f.write(f"建议 t_handle_offset: ({t_offset_new[0]:.4f},{t_offset_new[1]:.4f},{t_offset_new[2]:.4f})\n")

                    print(f"[LOG] 微调结果已追加至: {log_path}")

                except ValueError as e:
                    print(f"[ERROR] 数值解析失败: {e}")
            else:
                print(f"[WARN] 格式错误, 跳过微调 (需要 6 个数值, 收到 {len(adj_parts)} 个)")

    # 清理资源
    print("\n[INFO] 清理资源...")
    try:
        detector.pipeline.stop()
        print("[INFO] RealSense 管道已停止")
    except Exception as e:
        print(f"[WARN] 停止管道失败: {e}")


if __name__ == "__main__":
    main()