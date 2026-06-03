# -*- coding: utf-8 -*-
"""
estun_epiceye 位姿转换验证 Demo
================================
剥离 estun_epiceye_run 的位姿转换模块, 保留检测 + 位姿转换完整链路.
去除 TCP 服务器 / 灵巧手 / 循环 / 窗口显示等硬件/UI 依赖.

与 estun_epiceye_run 的差异:
  - 输入: 机械臂位姿由控制台 input 输入 (格式 [X,Y,Z,A,B,C], 单位 mm, deg)
  - 输出: 结果写入 logs/ 日志文件, 替代 TCP 发送
  - 保留: 相机触发 / YOLO 检测 / PCA 6D / 位姿转换全链路
  - 去除: socket / 灵巧手 / cv2 / 多线程 / 主循环

输入:
  - estun_config.yaml               : TCP 配置 (仅用 gripper_offset / orientation_mode)
  - EpicEye/epiceye_config.yaml     : 相机 / YOLO / 深度 / 手眼矩阵 / 运行模式

输出:
  - logs/estun_epiceye_demo_*.log   : 转换结果日志

流程:
  [Step 1] 加载配置
  [Step 2] 获取图像+点云 (online 触发相机 / offline 加载 captures)
  [Step 3] YOLO 灰度检测 -> 最优 bbox
  [Step 4] PCA 6D 位姿 -> pos_cam, rot_cam
  [Step 5] 控制台输入机械臂当前位姿 [X,Y,Z,A,B,C]
  [Step 6] 位姿转换: cam->ee -> 补偿 -> 对齐 -> base -> 可达性修正
  [Step 7] 写入日志
"""

import os
import sys
import yaml
import numpy as np
from datetime import datetime


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
from EpicEye.test_6D import EpicEyeYOLO6DPoseDetector


# =============================================================
# 配置加载
# =============================================================

# ---- 加载 ESTUN 配置 ----
_estun_cfg_path = os.path.join(_project_root, "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

GRIPPER_T_OFFSET = np.array(_cfg_estun["gripper_offset"]["t_offset"], dtype=float)
GRIPPER_R_OFFSET = np.array(_cfg_estun["gripper_offset"]["r_offset"], dtype=float)
ORIENTATION_MODE = _cfg_estun["orientation_mode"]

# 防御性检查
VALID_MODES = {"vertical", "horizontal", "free"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE}, 将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现, 将使用 free 模式")
    ORIENTATION_MODE = "free"

# ---- 加载 EpicEye 配置 ----
_epiceye_cfg_path = os.path.join(_project_root, "EpicEye", "epiceye_config.yaml")
if not os.path.exists(_epiceye_cfg_path):
    print(f"[FATAL] EpicEye 配置文件未找到: {_epiceye_cfg_path}")
    sys.exit(1)
with open(_epiceye_cfg_path, "r", encoding="utf-8") as f:
    _cfg_eye = yaml.safe_load(f)

MODE_ONLINE = _cfg_eye.get("mode", {}).get("online", True)
HANDEYE_MATRIX_RAW = _cfg_eye.get("handeye_matrix", None)


def load_handeye_matrix():
    """
    从配置加载手眼矩阵 (cam -> ee)

    返回:
        np.ndarray, 4x4 齐次矩阵; 格式错误时返回单位矩阵
    """
    I = np.eye(4)
    if HANDEYE_MATRIX_RAW is None:
        print("[WARN] epiceye_config.yaml 中未找到 handeye_matrix, 使用单位矩阵")
        return I

    T = np.array(HANDEYE_MATRIX_RAW, dtype=float)
    if T.shape != (4, 4):
        print(f"[WARN] handeye_matrix 形状 {T.shape}, 应为 4x4, 使用单位矩阵")
        return I

    t = T[:3, 3]
    print(f"[INFO] 已加载手眼矩阵 (cam->ee), 平移: ({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
    return T


# =============================================================
# 工具: 旋转矩阵 -> 欧拉角 (ZYX 顺序)
# =============================================================

# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  estun_epiceye 位姿转换验证 Demo")
    print(f"  模式: {'online' if MODE_ONLINE else 'offline'}")
    print("=" * 60, flush=True)

    # ---- [Step 1a] 加载手眼矩阵 ----
    T_cam2ee = load_handeye_matrix()

    # ---- [Step 1b] 预计算工具偏移 ----
    _, t_fl2gr = compute_gripper_offset_transform(GRIPPER_T_OFFSET, GRIPPER_R_OFFSET)
    print(f"[INFO] 工具偏移 t_fl2gr: ({t_fl2gr[0]:.4f}, {t_fl2gr[1]:.4f}, {t_fl2gr[2]:.4f}) m")

    # ---- [Step 2] 初始化检测器 ----
    print("[INFO] 初始化检测器...")
    detector = EpicEyeYOLO6DPoseDetector()
    color_image = None
    point_data = None
    depth_data = None

    try:
        # ---- [Step 2a] 获取帧数据 ----
        if MODE_ONLINE:
            print("[INFO] online 模式: 触发相机抓拍...")
            color_image, point_data, depth_data = detector.trigger_and_save()
            if color_image is None:
                print("[FATAL] 相机抓拍失败")
                return
            print("[INFO] 相机抓拍完成")
        else:
            print("[INFO] offline 模式: 加载最新抓拍数据...")
            from EpicEye.test_6D import _get_captures_dir, _list_captures, _load_capture_offline
            captures_dir = _get_captures_dir()
            if not os.path.exists(captures_dir):
                print(f"[FATAL] captures 目录不存在: {captures_dir}")
                return
            timestamps = _list_captures(captures_dir)
            if not timestamps:
                print("[FATAL] captures 目录中没有找到抓拍数据")
                return
            ts = timestamps[-1]
            print(f"[INFO] 加载时间戳: {ts}")
            color_image, point_data, depth_data = _load_capture_offline(ts)
            if color_image is None:
                print("[FATAL] 数据加载失败")
                return

        # ---- [Step 3] YOLO 检测 ----
        print("[INFO] 执行 YOLO 灰度检测...")
        detections = detector.detect_targets(color_image)
        print(f"  YOLO 检测到 {len(detections)} 个目标")

        if not detections:
            print("[ERROR] 未检测到目标, 无法进行位姿转换")
            return

        # ---- [Step 4] PCA 6D 位姿 ----
        best_det = max(detections, key=lambda d: d[4])
        bbox = best_det[:4]

        print("[INFO] 计算 6D 位姿...")
        position, rotation, points = detector.compute_6d_pose(bbox, point_data, depth_data)

        if position is None or rotation is None:
            print("[ERROR] 点云不足, 无法计算 6D 位姿")
            return

        A_cam, B_cam, C_cam = EpicEyeYOLO6DPoseDetector.rotation_matrix_to_euler_angles(rotation)
        print(f"  检测位姿 (cam): X={position[0]:.3f}, Y={position[1]:.3f}, "
              f"Z={position[2]:.3f}, A={A_cam:.1f}, B={B_cam:.1f}, C={C_cam:.1f}")

        pos_cam = position
        rot_cam = rotation

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
            return

        try:
            robot_xyz_mm = [float(p) for p in parts[:3]]
            robot_ABC = tuple(float(p) for p in parts[3:6])
        except ValueError as e:
            print(f"[ERROR] 数值解析失败: {e}")
            return

        # 转换 mm -> m
        robot_xyz = (robot_xyz_mm[0] / 1000.0, robot_xyz_mm[1] / 1000.0, robot_xyz_mm[2] / 1000.0)
        print(f"\n[INPUT] 机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m "
              f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg")

        # ---- [Step 6] 位姿转换链路 ----
        print("\n[INFO] 执行位姿转换...")

        # 6a. 手眼矩阵: cam -> ee
        pos_ee = T_cam2ee[:3, :3] @ np.array(pos_cam) + T_cam2ee[:3, 3]
        rot_ee = T_cam2ee[:3, :3] @ rot_cam
        A_ee, B_ee, C_ee = EpicEyeYOLO6DPoseDetector.rotation_matrix_to_euler_angles(rot_ee)
        print(f"  [cam->ee]  pos_ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
              f"A_ee={A_ee:.1f}, B_ee={B_ee:.1f}, C_ee={C_ee:.1f}")

        # 6b. 位置补偿: t_grasp_ee = t_obj_ee - t_tool'_ee
        pos_ee_compensated = np.array(pos_ee) - t_fl2gr
        print(f"  [补偿]    pos_compensated=({pos_ee_compensated[0]:.3f},{pos_ee_compensated[1]:.3f},{pos_ee_compensated[2]:.3f})")

        # 6c. 方向对齐
        R_obj_ee = euler_to_rotation_matrix(A_ee, B_ee, C_ee, order="zyx")
        R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
        R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, GRIPPER_R_OFFSET, R_ee_base)
        print(f"  [对齐]    mode={ORIENTATION_MODE}")

        # 6d. 转换到基坐标系
        rot_base = R_ee_base @ R_grasp_ee
        pos_base = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)

        A_base, B_base, C_base = EpicEyeYOLO6DPoseDetector.rotation_matrix_to_euler_angles(rot_base)
        print(f"  [ee->base] pos_base=({pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f}) "
              f"A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}")

        # 6e. 可达性修正
        A_base, B_base, C_base = post_correct_angles(A_base, B_base, C_base, robot_ABC)
        print(f"  [修正]    A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}")

        # ---- [Step 7] 输出 ----
        # 格式化目标位姿 (与 estun_epiceye_run [SEND] 格式一致)
        msg = (
            f"[{pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f},"
            f"{A_base:.2f},{B_base:.2f},{C_base:.2f}]"
        )

        # 控制台输出
        print("\n" + "=" * 60)
        print(f"  [RESULT] 目标基座标: {msg}")
        print("=" * 60)

        # 写入日志
        log_dir = os.path.join(_project_root, "logs")
        os.makedirs(log_dir, exist_ok=True)
        ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"estun_epiceye_demo_{ts_now}.log")

        with open(log_path, "w", encoding="utf-8") as f:
            f.write("=" * 60 + "\n")
            f.write("  estun_epiceye 位姿转换验证 Demo\n")
            f.write(f"  模式: {'online' if MODE_ONLINE else 'offline'}\n")
            f.write("=" * 60 + "\n\n")

            f.write(f"检测位姿 (cam): X={pos_cam[0]:.3f}, Y={pos_cam[1]:.3f}, "
                    f"Z={pos_cam[2]:.3f}, A={A_cam:.1f}, B={B_cam:.1f}, C={C_cam:.1f}\n\n")

            f.write(f"机械臂位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) m, "
                    f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}) deg\n\n")

            f.write("--- 中间结果 ---\n")
            f.write(f"[cam->ee] pos_ee=({pos_ee[0]:.4f},{pos_ee[1]:.4f},{pos_ee[2]:.4f}) "
                    f"A_ee={A_ee:.2f}, B_ee={B_ee:.2f}, C_ee={C_ee:.2f}\n")
            f.write(f"[补偿]    pos_ee_compensated=({pos_ee_compensated[0]:.4f},{pos_ee_compensated[1]:.4f},{pos_ee_compensated[2]:.4f})\n")
            f.write(f"[对齐]    mode={ORIENTATION_MODE}\n")
            f.write(f"[ee->base] pos_base=({pos_base[0]:.4f},{pos_base[1]:.4f},{pos_base[2]:.4f}) "
                    f"A={A_base:.2f}, B={B_base:.2f}, C={C_base:.2f}\n")
            f.write(f"[修正]    A_corrected={A_base:.2f}, B_corrected={B_base:.2f}, C_corrected={C_base:.2f}\n\n")

            f.write(f"结果: 目标基座标: {msg}\n")

        print(f"[LOG] 日志已保存: {log_path}")

    finally:
        # 清理资源
        print("\n[INFO] 清理资源...")
        detector.close()


if __name__ == "__main__":
    main()