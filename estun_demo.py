# -*- coding: utf-8 -*-
"""
estun 位姿转换验证 Demo (RealSense 版)
======================================
剥离 estun_run 的位姿转换模块, 保留检测 + 稳定化 + 位姿转换完整链路.
去除 TCP 服务器 / 灵巧手 / CRP 指令循环等硬件/UI 依赖.

与 estun_run 的差异:
  - 输入: 机械臂位姿由控制台 input 输入 (格式 [X,Y,Z,A,B,C], 单位 mm, deg)
  - 输出: 结果写入 logs/ 日志文件, 替代 TCP 发送
  - 保留: RealSense 实时流 / YOLO 检测 / PCA 6D / 位姿平滑 / 稳定化 / 位姿转换
  - 去除: socket / 灵巧手 / CRP 指令循环 / 多线程 / 代际检查

输入:
  - RealSense_R/config.yaml           : 相机 / YOLO 模型配置
  - estun_config.yaml                 : 夹爪偏移 / 抓取模式
  - calib_data/calibration_result.yaml: 手眼标定矩阵

输出:
  - logs/estun_demo_*.log             : 转换结果日志

流程:
  [Step 1] 加载配置 (标定矩阵 / 夹爪偏移 / YOLO 模型)
  [Step 2] 初始化 RealSense 相机 + YOLO 检测器
  [Step 3] 主循环: get_frames -> YOLO -> PCA 6D -> 平滑 -> 稳定化
  [Step 4] 控制台输入机械臂当前位姿 [X,Y,Z,A,B,C]
  [Step 5] 位姿转换: 补偿 -> 对齐 -> base -> 可达性修正
  [Step 6] 写入日志
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

MODEL_NAME = os.path.join(_project_root, _cfg["model"]["name"])
CONF_THRESHOLD = _cfg["model"]["conf_threshold"]

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
GRIPPER_T_OFFSET = np.array(_cfg_estun["gripper_offset"]["t_offset"], dtype=float)
GRIPPER_R_OFFSET = np.array(_cfg_estun["gripper_offset"]["r_offset"], dtype=float)
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]

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
    print("  estun 位姿转换验证 Demo (RealSense)")
    print("  标定: " + CALIB_RESULT_PATH)
    print("=" * 60, flush=True)

    # ---- [Step 1a] 加载手眼标定矩阵 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")
    else:
        print("[INFO] 标定矩阵已加载, 位姿将转换到末端坐标系")

    # ---- [Step 1b] 预计算工具偏移 ----
    _, t_fl2gr = compute_gripper_offset_transform(GRIPPER_T_OFFSET, GRIPPER_R_OFFSET)
    print(f"[INFO] 工具偏移 t_fl2gr: ({t_fl2gr[0]:.4f}, {t_fl2gr[1]:.4f}, {t_fl2gr[2]:.4f}) m")

    # ---- [Step 2] 初始化检测器 ----
    from RealSense_R.test_6D import RealSenseYOLO6DPoseDetector

    print("[INFO] 初始化 RealSense YOLO 检测器...")
    try:
        detector = RealSenseYOLO6DPoseDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        return

    # ---- [Step 3] 主循环: 获取帧 -> 检测 -> 稳定化 ----
    stabilizer = PoseStabilizer(
        window=STABLE_WINDOW,
        pos_thresh=POS_THRESHOLD,
        angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("estun Demo (RealSense)", cv2.WINDOW_NORMAL)

    stable_pose_ee: Optional[Tuple[Any, Any]] = None
    running = True

    print("\n[INFO] 正在检测并稳定位姿...")
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

            # ---- [Step 3b] 缓存原始深度帧 ----
            try:
                frames = detector.pipeline.wait_for_frames()
                detector._cached_raw_depth = frames.get_depth_frame()
            except Exception:
                detector._cached_raw_depth = None

            # ---- [Step 3c] YOLO 检测 + PCA 6D ----
            display_img = color_image.copy()
            detections = detector.detect_targets(color_image)

            if detections:
                best_det = max(detections, key=lambda d: d[4])
                position, rotation, point_cloud = detector.compute_6d_pose(best_det, depth_frame)

                if position is not None and rotation is not None:
                    # 位姿平滑 (经 Pylance 类型收窄)
                    smooth_pos, smooth_rot = detector._pose_smoother.update(position, rotation)
                    if smooth_pos is not None and smooth_rot is not None:
                        # cam -> ee
                        pos_ee = T_cam2ee[:3, :3] @ np.array(smooth_pos) + T_cam2ee[:3, 3]
                        rot_ee = T_cam2ee[:3, :3] @ smooth_rot
                        A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)

                        # 稳定化
                        stabilizer.add(tuple(pos_ee), (A_ee, B_ee, C_ee))
                        if stabilizer.is_stable():
                            mean_pos, mean_euler = stabilizer.get_mean()
                            if mean_pos is not None and mean_euler is not None:
                                stable_pose_ee = (mean_pos, mean_euler)
                                print("\n[INFO] 位姿已稳定, 准备接收机械臂位姿", flush=True)
                                running = False
                                break

                        # 绘制检测框和位姿信息
                        x1, y1, x2, y2, conf = map(int, best_det[:5])
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        detector.visualize_pose(display_img, smooth_pos, smooth_rot, scale=0.08)

                        text_pos = f"Pos: ({pos_ee[0]:.2f}, {pos_ee[1]:.2f}, {pos_ee[2]:.2f})"
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        text_euler = f"Euler: (A={A_ee:.1f}, B={B_ee:.1f}, C={C_ee:.1f})"
                        cv2.putText(display_img, text_euler, (x1, y1 - 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        print(f"\r[6D] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                              f"euler=({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})    ", end="", flush=True)
                else:
                    detector._pose_smoother.reset()
            else:
                detector._pose_smoother.reset()

            # ---- [Step 3d] 显示 ----
            if depth_colormap is not None and display_img.shape[:2] == depth_colormap.shape[:2]:
                images: Any = np.hstack((display_img, depth_colormap))
            else:
                images = display_img
            cv2.imshow("estun Demo (RealSense)", images)
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

    # ---- [Step 5] 位姿转换链路 ----
    print("\n[INFO] 执行位姿转换...")

    # 5a. 位置补偿
    pos_ee_compensated = np.array(mean_pos_ee) - t_fl2gr
    print(f"  [补偿]    pos_compensated=({pos_ee_compensated[0]:.3f},{pos_ee_compensated[1]:.3f},{pos_ee_compensated[2]:.3f})")

    # 5b. 方向对齐
    R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
    R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, GRIPPER_R_OFFSET, R_ee_base, grasp_direction=GRASP_DIRECTION)
    print(f"  [对齐]    mode={ORIENTATION_MODE}")

    # 5c. 转换到基坐标系
    rot_base = R_ee_base @ R_grasp_ee
    pos_base = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
    A_base, B_base, C_base = rotation_matrix_to_euler_angles(rot_base)
    print(f"  [ee->base] pos_base=({pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f}) "
          f"A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}")

    # 5d. 可达性修正
    A_base, B_base, C_base = post_correct_angles(A_base, B_base, C_base, robot_ABC)
    print(f"  [修正]    A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}")

    # ---- [Step 6] 输出 ----
    msg = (
        f"[{pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f},"
        f"{A_base:.2f},{B_base:.2f},{C_base:.2f}]"
    )

    print("\n" + "=" * 60)
    print(f"  [RESULT] 目标基座标: {msg}")
    print("=" * 60)

    # 写入日志
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"estun_demo_{ts_now}.log")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("  estun 位姿转换验证 Demo (RealSense)\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"检测位姿 (cam): X={mean_pos_ee[0]:.3f}, Y={mean_pos_ee[1]:.3f}, Z={mean_pos_ee[2]:.3f}, "
                f"A={mean_euler_ee[0]:.1f}, B={mean_euler_ee[1]:.1f}, C={mean_euler_ee[2]:.1f}\n\n")

        f.write(f"机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m, "
                f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg\n\n")

        f.write("--- 中间结果 ---\n")
        f.write(f"[cam->ee] pos_ee=({mean_pos_ee[0]:.4f},{mean_pos_ee[1]:.4f},{mean_pos_ee[2]:.4f}) "
                f"A_ee={mean_euler_ee[0]:.2f}, B_ee={mean_euler_ee[1]:.2f}, C_ee={mean_euler_ee[2]:.2f}\n")
        f.write(f"[补偿]    pos_ee_compensated=({pos_ee_compensated[0]:.4f},{pos_ee_compensated[1]:.4f},{pos_ee_compensated[2]:.4f})\n")
        f.write(f"[对齐]    mode={ORIENTATION_MODE}\n")
        f.write(f"[ee->base] pos_base=({pos_base[0]:.4f},{pos_base[1]:.4f},{pos_base[2]:.4f}) "
                f"A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}\n")
        f.write(f"[修正]    A_corrected={A_base:.2f}, B_corrected={B_base:.2f}, C_corrected={C_base:.2f}\n\n")

        f.write(f"结果: 目标基座标: {msg}\n")

    print(f"[LOG] 日志已保存: {log_path}")

    # 清理资源
    print("\n[INFO] 清理资源...")
    try:
        detector.pipeline.stop()
        print("[INFO] RealSense 管道已停止")
    except Exception as e:
        print(f"[WARN] 停止管道失败: {e}")


if __name__ == "__main__":
    main()