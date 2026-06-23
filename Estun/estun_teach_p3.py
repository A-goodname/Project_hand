# -*- coding: utf-8 -*-
"""
P3 倒水位置示教工具 (仅 Cup 检测)
=================================
仅使用 Cup 6D 检测器获取水杯位置, 人工示教 P3 倒水姿态,
自动计算 Cup P1 法兰到 P3 法兰的基系偏移.

流程:
  [Step 1] 加载配置 / 标定矩阵
  [Step 2] 初始化 Cup 检测器 (启动 pipeline, 跳帧加速)
  [Step 3] 仅检测 Cup, 稳定后计算基系位置 (用 Home 位姿转换)
  [Step 4] 提示用户输入 P3 末端位姿
  [Step 5] 计算偏移并显示
  [Step 6] 写入 log / 询问是否继续

输入:
  - RealSense/config.yaml            : 相机 / YOLO / cup 配置
  - Estun/estun_config.yaml          : 标定路径 / cup 偏移
  - calib_data/calibration_result.yaml : 手眼标定矩阵

输出:
  - logs/teach_p3_*.log              : 示教记录
"""

import os
import sys
import yaml
import cv2
import numpy as np
from datetime import datetime


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入工具 ----
from Estun.utils_shared import euler_to_rotation_matrix, transform_pose_to_base, compute_gripper_offset_transform
from RealSense.realsense_shared import rotation_matrix_to_euler_angles


# =============================================================
# 配置加载
# =============================================================

_cfg_path = os.path.join(_project_root, "RealSense", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_estun_cfg_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

CALIB_RESULT_PATH = os.path.join(_project_root, _cfg_estun["calibration"]["result_path"])

# ---- Home 位姿 (固定, 用于 Cup 检测后的位姿转换) ----
HOME_XYZ = (0.387292, 0.100580, 0.616543)  # 米
HOME_ABC = (-118.552, 0.803, -91.776)      # 度

# ---- 水杯偏移 ----
CUP_HANDLE_SIDE = _cfg.get("cup", {}).get("handle_side", "back")
if CUP_HANDLE_SIDE == "left":
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset_left"], dtype=float)
else:
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset"], dtype=float)
CUP_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["r_cup_offset"], dtype=float)

CUP_SKIP_FRAMES = 10


def load_cam2ee(yaml_path: str) -> np.ndarray:
    """从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)."""
    I = np.eye(4)
    if not os.path.exists(yaml_path):
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}, 使用单位矩阵")
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
    print(f"[INFO] 已加载标定矩阵: 平移=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
    return T


def compute_cup_base(cam_pos, cam_euler, T_cam2ee, home_xyz, home_abc, t_cup_eff):
    """
    将 Cup 相机系位姿转换到基坐标系下的法兰位置 (含灵巧手补偿).

    参数:
        cam_pos    : (x, y, z) 相机系位置 (m)
        cam_euler  : (A, B, C) 相机系欧拉角 (deg)
        T_cam2ee   : 4x4 相机→法兰矩阵
        home_xyz   : (x, y, z) Home 位姿位置 (m)
        home_abc   : (A, B, C) Home 位姿欧拉角 (deg)
        t_cup_eff  : (x, y, z) 水杯灵巧手有效偏移 (m)

    返回:
        c_pos_base : 基系下 Cup 法兰位置 (m)
    """
    c_pos_ee = T_cam2ee[:3, :3] @ np.array(cam_pos) + T_cam2ee[:3, 3]
    c_rot_ee = T_cam2ee[:3, :3] @ euler_to_rotation_matrix(*cam_euler, order="zyx")
    # 减去灵巧手偏移得到法兰位置 (与 estun_cup_demo.py 保持一致)
    c_pos_ee_compensated = c_pos_ee - t_cup_eff
    # 用 Home 位姿将末端系→基系
    return transform_pose_to_base(c_pos_ee_compensated, home_xyz, home_abc)


def main():
    print("=" * 60)
    print("  P3 倒水位置示教工具")
    print("  仅使用 Cup 检测, 人工示教 P3")
    print(f"  Home 位姿: xyz={HOME_XYZ} ABC={HOME_ABC}")
    print("=" * 60, flush=True)

    # ---- 加载手眼标定 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)

    # ---- 初始化 Cup 检测器 (启动 pipeline) ----
    from RealSense.test_cup_6D import RealSenseCup6DDetector
    print("[INFO] 初始化 Cup 6D 检测器...")
    try:
        cup_detector = RealSenseCup6DDetector(config_dict=_cfg, start_pipeline=True)
    except Exception as e:
        print(f"[FATAL] Cup 检测器初始化失败: {e}")
        return

    # ---- 位姿稳定器 ----
    cup_stabilizer_buffer = []
    STABLE_WINDOW = 3
    POS_THRESH = 0.01
    ANGLE_THRESH = 5.0

    # ---- 预计算水杯有效偏移 ----
    _, t_cup_effective = compute_gripper_offset_transform(CUP_T_OFFSET, CUP_R_OFFSET)

    # ---- 日志 ----
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"teach_p3_{ts_now}.log")
    log_lines = []
    log_lines.append(f"Home 位姿: xyz={HOME_XYZ} ABC={HOME_ABC}\n")
    log_lines.append(f"模式: {CUP_HANDLE_SIDE}\n")

    cv2.namedWindow("P3 Teach - Cup Detection", cv2.WINDOW_NORMAL)

    teach_count = 0
    running = True

    while running:
        # ---- 检测 Cup 直到稳定 ----
        print("\n[INFO] 正在检测水杯... (按 q 退出)", flush=True)
        cup_stable = False
        frame_counter = 0
        c_pos_cam = None
        c_euler_cam = None

        while not cup_stable:
            color_image, depth_image, depth_colormap, depth_frame = cup_detector.get_frames()
            if color_image is None or depth_frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出")
                    running = False
                    break
                continue

            display_img = color_image.copy()
            frame_counter += 1

            if frame_counter % CUP_SKIP_FRAMES == 1:
                (c_bbox, c_pos, c_rot, c_euler, c_point_cloud) = cup_detector.compute_cup_6d_pose(color_image, depth_frame)

                if c_pos is not None and c_rot is not None:
                    smooth_cpos, smooth_crot = cup_detector._pose_smoother.update(c_pos, c_rot)
                    if smooth_crot is not None:
                        c_pos = smooth_cpos
                        c_rot = smooth_crot
                        c_euler = rotation_matrix_to_euler_angles(c_rot)

                    if c_euler is not None:
                        cup_stabilizer_buffer.append((np.array(c_pos), np.array(c_euler)))
                        if len(cup_stabilizer_buffer) > STABLE_WINDOW:
                            cup_stabilizer_buffer.pop(0)
                        if len(cup_stabilizer_buffer) == STABLE_WINDOW:
                            poses = np.array([p for p, _ in cup_stabilizer_buffer])
                            angles = np.array([a for _, a in cup_stabilizer_buffer])
                            pos_range = np.ptp(poses, axis=0).max()
                            angle_range = np.ptp(angles, axis=0).max()
                            if pos_range < POS_THRESH and angle_range < ANGLE_THRESH:
                                c_pos_cam = poses.mean(axis=0)
                                c_euler_cam = angles.mean(axis=0)
                                cup_stable = True
                                print(f"\n[TEACH] Cup 稳定! pos_cam=({c_pos_cam[0]:.3f},{c_pos_cam[1]:.3f},{c_pos_cam[2]:.3f}) "
                                      f"euler=({c_euler_cam[0]:.1f},{c_euler_cam[1]:.1f},{c_euler_cam[2]:.1f})", flush=True)
                else:
                    cup_detector._pose_smoother.reset()

                if c_bbox is not None:
                    cup_detector.visualize_cup(display_img, c_bbox, c_pos, c_rot, c_euler)

            cv2.putText(display_img, f"Cup 稳定中... ({len(cup_stabilizer_buffer)}/{STABLE_WINDOW})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("P3 Teach - Cup Detection", display_img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 用户按 q 退出")
                running = False
                break

        if not running:
            break

        # ---- 计算 Cup P1 法兰在基系的位置 (含灵巧手补偿) ----
        cup_p1_base = compute_cup_base(c_pos_cam, c_euler_cam, T_cam2ee, HOME_XYZ, HOME_ABC, t_cup_effective)
        teach_count += 1
        print(f"\n{'='*50}")
        print(f"[TEACH] 第 {teach_count} 次示教")
        print(f"[TEACH] Cup P1 法兰 base = ({cup_p1_base[0]:.4f}, {cup_p1_base[1]:.4f}, {cup_p1_base[2]:.4f}) m")
        print(f"{'='*50}")

        # ---- 用户输入 P3 末端位姿 ----
        print("\n请从示教器读取机械臂当前末端位姿 (倒水姿态 P3)")
        print("格式: [X,Y,Z,A,B,C]  (X,Y,Z 单位 mm, A,B,C 单位 deg)")
        print("示例: [500.0,200.0,400.0,-30.0,10.0,45.0]")

        p3_input = input("输入 P3 位姿: ").strip().strip("[]")
        parts = p3_input.split(",")
        if len(parts) != 6:
            print(f"[ERROR] 格式错误, 需要 6 个数值, 收到 {len(parts)} 个")
            continue

        try:
            p3_xyz_mm = [float(p) for p in parts[:3]]
            p3_abc = [float(p) for p in parts[3:6]]
        except ValueError as e:
            print(f"[ERROR] 数值解析失败: {e}")
            continue

        p3_xyz = np.array([p3_xyz_mm[0] / 1000.0, p3_xyz_mm[1] / 1000.0, p3_xyz_mm[2] / 1000.0])

        # ---- 计算偏移 ----
        t_offset = p3_xyz - cup_p1_base
        r_abc = p3_abc

        print(f"\n[TEACH] 第 {teach_count} 次结果:")
        print(f"[TEACH]   t_offset = ({t_offset[0]:.4f}, {t_offset[1]:.4f}, {t_offset[2]:.4f}) m")
        print(f"[TEACH]   r_abc    = ({r_abc[0]:.1f}, {r_abc[1]:.1f}, {r_abc[2]:.1f}) deg")
        print(f"[TEACH]   (Cup P1法兰 + t_offset = P3法兰, 不包含 t_tcp_fixed)")

        # ---- 写入日志 ----
        log_line = (
            f"[{teach_count}] P3_teach: xyz=({p3_xyz[0]:.4f},{p3_xyz[1]:.4f},{p3_xyz[2]:.4f}) "
            f"ABC=({r_abc[0]:.1f},{r_abc[1]:.1f},{r_abc[2]:.1f}) | "
            f"Cup P1 base=({cup_p1_base[0]:.4f},{cup_p1_base[1]:.4f},{cup_p1_base[2]:.4f}) | "
            f"t_offset=({t_offset[0]:.4f},{t_offset[1]:.4f},{t_offset[2]:.4f}) "
            f"r_abc=({r_abc[0]:.1f},{r_abc[1]:.1f},{r_abc[2]:.1f})\n"
        )
        log_lines.append(log_line)
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        print(f"[TEACH] 已保存到日志: {log_path}")

        # ---- 询问是否继续 ----
        print()
        again = input("是否继续下一次示教？[y/N]: ").strip().lower()
        if again != "y":
            running = False

        # 重置稳定器
        cup_stabilizer_buffer = []

    # ---- 输出汇总 ----
    print(f"\n{'='*50}")
    print(f"  示教结束, 共采集 {teach_count} 次")
    print(f"  日志: {log_path}")
    if teach_count >= 2:
        print(f"\n  [提示] 请在日志中查看各组数据, 取平均后写入配置:")
        print("    estun_config.yaml 中 handle_grasp 下添加:")
        print("    p3_teach_offset:")
        print("      use_teach: true")
        print("      t_offset: [X, Y, Z]       # 平均 t_offset (m)")
        print("      r_abc: [A, B, C]          # 平均 r_abc (deg)")
    print(f"{'='*50}")

    # 清理
    cv2.destroyAllWindows()
    try:
        cup_detector.pipeline.stop()
        print("[INFO] RealSense 管道已停止")
    except Exception:
        pass
    print("[INFO] 示教工具退出")


if __name__ == "__main__":
    main()