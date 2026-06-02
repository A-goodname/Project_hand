# -*- coding: utf-8 -*-
"""
手眼标定求解器 -- eye-in-hand
==============================
输入:
  - calib_data/images/       (标定板图片)
  - calib_data/poses.csv     (末端位姿)

输出:
  - calib_data/calibration_result.yaml   (标定结果, 含全部方法 + 最佳方法标记)
  - calib_data/board_in_base.yaml        (仅最佳方法对应的标定板在基坐标系位姿)

流程:
  1. 加载 calib_config.yaml 配置
  2. 读取 poses.csv: index, x, y, z, rx, ry, rz
  3. 遍历 images/ 检测棋盘格角点
  4. calibrateCamera() 求解相机内参
  5. 对每张图片 solvePnP() 得到 camHboard (标定板->相机)
  6. 根据重投影误差阈值剔除不合格数据
  7. 将 X-Y-Z 固定角转为旋转矩阵 (baseHee)
  8. 对 5 种手眼方法分别求解 calibrateHandEye()
  9. 用 Board-in-base 一致性误差评估各方法, 选最优
  10. 保存 calibration_result.yaml, board_in_base.yaml (仅最佳)

一致性评估原理:
  对同一物理标定板, 无论从哪个姿态观测,
  board_in_base_i = baseHee_i @ camHee @ camHboard_i 应为常数.
  平移标准差和旋转标准差越小 → 标定质量越好.
"""

import os
import sys
import cv2
import numpy as np
import yaml
from datetime import datetime

# ---- 脚本所在目录 (使路径不依赖 CWD) ----
_script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- 加载配置 ----
_config_path = os.path.join(_script_dir, "calib_config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_cb_cfg = _cfg["calibration"]["chessboard"]
PATTERN_SIZE = tuple(_cb_cfg["pattern_size"])   # (cols, rows)
SQUARE_SIZE = _cb_cfg["square_size"]             # meter

_cam_cfg = _cfg["calibration"]["camera"]
RESOLUTION = tuple(_cam_cfg["resolution"])       # (width, height)

_data_cfg = _cfg["calibration"]["data"]
SAVE_DIR = os.path.join(_script_dir, _data_cfg["save_dir"])
IMAGE_DIR = os.path.join(SAVE_DIR, "images")
POSE_CSV = os.path.join(SAVE_DIR, "poses.csv")
IMAGE_FORMAT = _data_cfg.get("image_format", "png")
REPROJ_THRESHOLD = _data_cfg.get("reprojection_error_threshold", 0.5)


# =============================================================
# 辅助函数: 欧拉角 / 旋转矩阵 / 四元数 转换
# =============================================================

def euler_zyx_to_rotation_matrix(rx, ry, rz):
    """
    将 X-Y-Z 固定角 (Euler ZYX) 转为 3x3 旋转矩阵.

    参数:
        rx, ry, rz : 欧拉角 (角度制), 分别绕固定 X, Y, Z 轴旋转
    返回:
        3x3 旋转矩阵 R = Rz(rz) @ Ry(ry) @ Rx(rx)
    """
    a = np.radians(rx)
    b = np.radians(ry)
    c = np.radians(rz)

    Rx = np.array([[1, 0, 0],
                   [0, np.cos(a), -np.sin(a)],
                   [0, np.sin(a), np.cos(a)]])
    Ry = np.array([[np.cos(b), 0, np.sin(b)],
                   [0, 1, 0],
                   [-np.sin(b), 0, np.cos(b)]])
    Rz = np.array([[np.cos(c), -np.sin(c), 0],
                   [np.sin(c), np.cos(c), 0],
                   [0, 0, 1]])

    return Rz @ Ry @ Rx


def rotation_matrix_to_euler_zyx(R):
    """
    将 3x3 旋转矩阵转为 X-Y-Z 固定角 (Euler ZYX), 输出角度制.

    参数:
        R : 3x3 旋转矩阵
    返回:
        (rx, ry, rz) : 角度制欧拉角
    """
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        rx = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        rx = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        ry = np.degrees(np.arctan2(-R[2, 0], sy))
        rz = 0.0
    return (rx, ry, rz)


def rotation_matrix_to_quaternion(R):
    """
    将 3x3 旋转矩阵转为四元数 [x, y, z, w].
    """
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    return np.array([qx, qy, qz, qw])


def make_2d_list(rows, cols, flat_data):
    """将一维列表转为 rows x cols 的嵌套列表."""
    return [flat_data[i * cols:(i + 1) * cols] for i in range(rows)]


# =============================================================
# 核心评估函数: Board-in-base 一致性误差
# =============================================================

def compute_board_in_base_consistency(
    R_gripper2base_list, t_gripper2base_list,
    R_target2cam_list, t_target2cam_list,
    R_cam2ee, t_cam2ee
):
    """
    用 Board-in-base 一致性评估手眼标定质量.

    原理: 对同一物理标定板, board_in_base = baseHee @ camHee @ camHboard
          在所有观测中应为常数.其离散程度反映标定误差.

    参数:
        R_gripper2base_list : 末端->基座旋转矩阵列表 (每项 3x3)
        t_gripper2base_list : 末端->基座平移向量列表 (每项 3,)
        R_target2cam_list   : 标定板->相机旋转矩阵列表 (每项 3x3)
        t_target2cam_list   : 标定板->相机平移向量列表 (每项 3,)
        R_cam2ee            : 手眼旋转矩阵 (3x3)
        t_cam2ee            : 手眼平移向量 (3,)

    返回:
        dict {
            "board_in_base_list": 每组的位姿数据列表,
            "translation_std":    平移标准差均值 (m),
            "rotation_std_deg":   旋转欧拉角标准差均值 (度),
            "combined_score":     综合评分 (越小越好)
        }
    """
    # 构建 camHee
    camHee = np.eye(4)
    camHee[:3, :3] = R_cam2ee
    camHee[:3, 3] = t_cam2ee.reshape(3)

    translations = []
    eulers_deg = []

    for i in range(len(R_gripper2base_list)):
        # baseHee_i
        baseHee = np.eye(4)
        baseHee[:3, :3] = R_gripper2base_list[i]
        baseHee[:3, 3] = t_gripper2base_list[i].reshape(3)

        # camHboard_i
        camHboard = np.eye(4)
        camHboard[:3, :3] = R_target2cam_list[i]
        camHboard[:3, 3] = t_target2cam_list[i].reshape(3)

        # board_in_base = baseHee @ camHee @ camHboard
        board_in_base = baseHee @ camHee @ camHboard

        t = board_in_base[:3, 3]
        R = board_in_base[:3, :3]
        euler = rotation_matrix_to_euler_zyx(R)

        translations.append(t)
        eulers_deg.append(euler)

    trans = np.array(translations)        # (N, 3)
    eulers = np.array(eulers_deg)         # (N, 3)

    t_std = trans.std(axis=0)             # (3,) 每个轴的平移标准差
    r_std = eulers.std(axis=0)            # (3,) 每个轴的旋转标准差 (度)

    t_std_mean = float(t_std.mean())      # 平均平移标准差 (m)
    r_std_mean = float(r_std.mean())      # 平均旋转标准差 (deg)

    # 综合评分: 旋转标准差按 0.001 m/° 折算为等效平移误差
    ANGLE_TO_TRANS_RATIO = 0.001          # 1° ≈ 1mm
    combined_score = t_std_mean + ANGLE_TO_TRANS_RATIO * r_std_mean

    # 构建 board_in_base_list (用于保存)
    board_in_base_list = []
    for i in range(len(R_gripper2base_list)):
        baseHee_i = np.eye(4)
        baseHee_i[:3, :3] = R_gripper2base_list[i]
        baseHee_i[:3, 3] = t_gripper2base_list[i].reshape(3)

        camHboard_i = np.eye(4)
        camHboard_i[:3, :3] = R_target2cam_list[i]
        camHboard_i[:3, 3] = t_target2cam_list[i].reshape(3)

        bib = baseHee_i @ camHee @ camHboard_i

        R_b2b = bib[:3, :3]
        t_b2b = bib[:3, 3]
        euler_b = rotation_matrix_to_euler_zyx(R_b2b)
        quat_b = rotation_matrix_to_quaternion(R_b2b)

        board_in_base_list.append({
            "translation": [round(float(v), 6) for v in t_b2b.tolist()],
            "rotation": [round(float(v), 6) for v in R_b2b.flatten().tolist()],
            "euler_zyx_deg": [round(float(v), 4) for v in euler_b],
            "quaternion": [round(float(v), 6) for v in quat_b.tolist()],
        })

    # 统计: 标注在 board_in_base_list 的 index 字段由外部处理
    return {
        "board_in_base_list": board_in_base_list,
        "translation_std": t_std_mean,
        "rotation_std_deg": r_std_mean,
        "combined_score": combined_score,
    }


# =============================================================
# 功能函数: 标定数据组装
# =============================================================

def assemble_handeye_data(obj_points, img_points, camera_matrix, dist_coeffs, poses, valid_indices):
    """
    对每张图片执行 solvePnP 并组装手眼标定所需的输入列表.

    solvePnP 得到 board->camera (target2cam);
    由 pose 解析 ee->base (gripper2base);
    根据重投影误差阈值过滤不合格数据.

    参数:
        obj_points     : 世界坐标系角点列表 (用于 solvePnP)
        img_points     : 图像坐标系角点列表
        camera_matrix  : 相机内参 (3x3)
        dist_coeffs    : 畸变系数
        poses          : dict {index: (x, y, z, rx, ry, rz)}
        valid_indices  : 成功检测到棋盘格的 index 列表

    返回:
        R_gripper2base_list  : 末端->基座旋转矩阵列表
        t_gripper2base_list  : 末端->基座平移向量列表
        R_target2cam_list    : 标定板->相机旋转矩阵列表
        t_target2cam_list    : 标定板->相机平移向量列表
        filtered_indices     : 通过阈值过滤的 index 列表
        filtered_errors      : 对应重投影误差列表
    """
    R_gripper2base_list = []
    t_gripper2base_list = []
    R_target2cam_list = []
    t_target2cam_list = []
    filtered_indices = []
    filtered_errors = []

    print("[INFO] 求解 PnP 并过滤不合格数据:")

    # 棋盘格 3D 角点 (单张)
    pattern_cols, pattern_rows = PATTERN_SIZE
    objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    for i, idx in enumerate(valid_indices):
        # solvePnP: 求解 board->camera 的旋转/平移
        _, rvec, tvec = cv2.solvePnP(objp, img_points[i], camera_matrix, dist_coeffs)

        # 计算重投影误差
        imgpoints2, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img_points[i], imgpoints2.reshape(-1, 2), cv2.NORM_L2) / len(objp)

        if error > REPROJ_THRESHOLD:
            print(f"  [REJECT] img_{idx:03d}: 重投影误差 "
                  f"{error:.3f} px > {REPROJ_THRESHOLD} px")
            continue

        print(f"  [ACCEPT] img_{idx:03d}: 重投影误差 {error:.3f} px")

        # board->camera (target2cam)
        R_board2cam, _ = cv2.Rodrigues(rvec)
        R_target2cam_list.append(R_board2cam)
        t_target2cam_list.append(tvec.reshape(3))

        # ee->base (gripper2base)
        pose = poses[idx]
        R_ee2base = euler_zyx_to_rotation_matrix(pose[3], pose[4], pose[5])
        R_gripper2base_list.append(R_ee2base)
        t_gripper2base_list.append(np.array(pose[:3]))

        filtered_indices.append(idx)
        filtered_errors.append(error)

    return (R_gripper2base_list, t_gripper2base_list,
            R_target2cam_list, t_target2cam_list,
            filtered_indices, filtered_errors)


# =============================================================
# 功能函数: 结果保存
# =============================================================

def save_calibration_result(save_dir, camera_matrix, dist_coeffs,
                             R_cam2ee, t_cam2ee, euler_cam2ee,
                             filtered_indices, filtered_errors, reproj_error,
                             method_results, best_method):
    """
    保存 calibration_result.yaml, 包含相机内参,手眼矩阵,过滤信息等.
    输出格式兼容 estun_run.py 的 load_cam2gripper() 读取逻辑:
      calibration.hand_eye.use_method -> "Tsai"
      calibration.hand_eye.Tsai.T_cam_to_gripper -> 4x4 齐次矩阵

    参数:
        save_dir          : 保存目录
        camera_matrix     : 相机内参 (3x3)
        dist_coeffs       : 畸变系数
        R_cam2ee          : 最佳手眼旋转矩阵 (3x3)
        t_cam2ee          : 最佳手眼平移向量 (3,)
        euler_cam2ee      : 最佳手眼欧拉角 (rx, ry, rz)
        filtered_indices  : 通过阈值过滤的 index 列表
        filtered_errors   : 对应重投影误差列表
        reproj_error      : 相机标定总体重投影误差
        method_results    : dict, name -> {"R", "t", "combined_score", "translation_std", "rotation_std_deg"}
        best_method       : 最佳手眼标定方法名
    """
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    # 构建 hand_eye 章节: use_method + 各方法数据
    hand_eye_data = {
        "use_method": best_method,
    }
    for name, res in method_results.items():
        R = res["R"]
        t = res["t"].reshape(3)
        T_4x4 = np.eye(4)
        T_4x4[:3, :3] = R
        T_4x4[:3, 3] = t
        hand_eye_data[name] = {
            "rotation": make_2d_list(3, 3, [
                round(v, 8) for row in R.tolist() for v in row
            ]),
            "translation": [round(v, 8) for v in t.tolist()],
            "T_cam_to_gripper": make_2d_list(4, 4, [
                round(v, 8) for row in T_4x4.tolist() for v in row
            ]),
            "combined_score": round(res["combined_score"], 10),
            "translation_std_m": round(res["translation_std"], 8),
            "rotation_std_deg": round(res["rotation_std_deg"], 6),
        }

    result = {
        "calibration": {
            "method": "OpenCV_HandEye",
            "timestamp": timestamp,
            "num_samples": len(filtered_indices),
            "reprojection_error_px": round(reproj_error, 6),
            "camera_matrix": make_2d_list(3, 3, [
                round(v, 6) for row in camera_matrix.tolist() for v in row
            ]),
            "distortion_coefficients": [
                round(v, 6) for v in dist_coeffs.ravel().tolist()
            ],
            "handeye": hand_eye_data,
            "filtered_indices": filtered_indices,
            "reprojection_errors_per_view": [round(e, 4) for e in filtered_errors],
        }
    }

    result_path = os.path.join(save_dir, "calibration_result.yaml")
    with open(result_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
    print(f"[INFO] 标定结果已保存: {result_path}")


def save_board_in_base(save_dir, board_in_base_list, index_list=None):
    """
    保存 board_in_base.yaml, 包含每张图片对应的标定板在基坐标系下的位姿,
    以及位姿的统计评估 (均值/标准差/最小值/最大值).

    参数:
        save_dir           : 保存目录
        board_in_base_list : 位姿列表 (含 translation, euler_zyx_deg)
        index_list         : 可选, 对应原始图片 index 列表, 用于填充 index 字段
    """
    # 填入 index 字段
    for i, item in enumerate(board_in_base_list):
        if "index" not in item:
            if index_list is not None and i < len(index_list):
                item["index"] = index_list[i]
            else:
                item["index"] = i

    # 收集 translation 和 euler 数据用于统计
    trans = np.array([p["translation"] for p in board_in_base_list])
    eulers = np.array([p["euler_zyx_deg"] for p in board_in_base_list])

    stats = {
        "num_samples": len(board_in_base_list),
        "translation_m": {
            "mean": [round(float(v), 6) for v in trans.mean(axis=0)],
            "std" : [round(float(v), 6) for v in trans.std(axis=0)],
            "min" : [round(float(v), 6) for v in trans.min(axis=0)],
            "max" : [round(float(v), 6) for v in trans.max(axis=0)],
        },
        "euler_zyx_deg": {
            "mean": [round(float(v), 4) for v in eulers.mean(axis=0)],
            "std" : [round(float(v), 4) for v in eulers.std(axis=0)],
            "min" : [round(float(v), 4) for v in eulers.min(axis=0)],
            "max" : [round(float(v), 4) for v in eulers.max(axis=0)],
        },
    }

    board_data = {
        "statistics": stats,
        "board_in_base": board_in_base_list,
    }

    board_path = os.path.join(save_dir, "board_in_base.yaml")
    with open(board_path, "w", encoding="utf-8") as f:
        yaml.dump(board_data, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
    print(f"[INFO] 标定板位姿已保存: {board_path}")

    # 打印统计摘要
    t_mean = stats["translation_m"]["mean"]
    t_std  = stats["translation_m"]["std"]
    e_mean = stats["euler_zyx_deg"]["mean"]
    e_std  = stats["euler_zyx_deg"]["std"]
    print(f"[INFO] board_in_base 统计评估 ({len(board_in_base_list)} 组):")
    print(f"  平移 [m]     mean: x={t_mean[0]:.6f}, y={t_mean[1]:.6f}, z={t_mean[2]:.6f}")
    print(f"               std:  x={t_std[0]:.6f}, y={t_std[1]:.6f}, z={t_std[2]:.6f}")
    print(f"  Euler [deg]  mean: rx={e_mean[0]:.4f}, ry={e_mean[1]:.4f}, rz={e_mean[2]:.4f}")
    print(f"               std:  rx={e_std[0]:.4f}, ry={e_std[1]:.4f}, rz={e_std[2]:.4f}")


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  手眼标定求解器 -- eye-in-hand")
    print("  棋盘格内角点: %s, 方格边长: %s m" % (str(PATTERN_SIZE), str(SQUARE_SIZE)))
    print("=" * 60)

    # ---- [Step 1] 加载位姿数据 ----
    print("[INFO] 加载位姿数据: %s" % POSE_CSV)
    poses = {}  # index -> (x, y, z, rx, ry, rz)
    with open(POSE_CSV, "r") as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(",")
        idx = int(parts[0])
        x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
        rx, ry, rz = float(parts[4]), float(parts[5]), float(parts[6])
        poses[idx] = (x, y, z, rx, ry, rz)

    print("[INFO] 加载了 %d 组末端位姿" % len(poses))

    # ---- [Step 2] 检测棋盘格角点 ----
    print("[INFO] 检测棋盘格角点:")
    pattern_cols, pattern_rows = PATTERN_SIZE
    objp = np.zeros((pattern_cols * pattern_rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_cols, 0:pattern_rows].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    indices = sorted(poses.keys())

    obj_points = []    # 世界坐标系角点集合
    img_points = []    # 图像坐标系角点集合
    valid_indices = [] # 成功检测到棋盘格的 index 列表

    for idx in indices:
        img_path = os.path.join(IMAGE_DIR, f"img_{idx:03d}.{IMAGE_FORMAT}")
        if not os.path.exists(img_path):
            print("  [SKIP] 图片不存在: img_%03d.%s" % (idx, IMAGE_FORMAT))
            continue

        img = cv2.imread(img_path)
        if img is None:
            print("  [SKIP] 无法读取: img_%03d.%s" % (idx, IMAGE_FORMAT))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, PATTERN_SIZE, None)

        if not found:
            print("  [SKIP] 未检测到棋盘格: img_%03d.%s" % (idx, IMAGE_FORMAT))
            continue

        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        obj_points.append(objp)
        img_points.append(corners_refined.reshape(-1, 2))
        valid_indices.append(idx)

    print("[INFO] 成功检测到棋盘格的图片: %d 张" % len(valid_indices))
    if len(valid_indices) < 5:
        print("[ERROR] 有效图片太少, 无法进行标定")
        return

    # ---- [Step 3] 相机内参标定 ----
    print("[INFO] 执行相机内参标定 (calibrateCamera)...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, RESOLUTION, None, None
    )
    print("[INFO] 重投影误差: %.4f px" % ret)
    print("[INFO] 内参矩阵:\n%s" % str(camera_matrix))
    print("[INFO] 畸变系数: %s\n" % str(dist_coeffs.ravel()))

    # ---- [Step 4] 组装手眼标定数据 ----
    (R_gripper2base_list, t_gripper2base_list,
     R_target2cam_list, t_target2cam_list,
     filtered_indices, filtered_errors) = assemble_handeye_data(
        obj_points, img_points, camera_matrix, dist_coeffs, poses, valid_indices
    )

    n_filtered = len(filtered_indices)
    print("[INFO] 过滤后有效数据对: %d" % n_filtered)
    if n_filtered < 5:
        print("[ERROR] 有效数据对太少, 无法进行手眼标定")
        return

    # ---- [Step 5] 手眼标定 + Board-in-base 一致性评估 ----
    print("[INFO] 执行手眼标定 (calibrateHandEye)...")

    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]

    # 保存所有方法的结果: name -> {"R", "t", "combined_score", "translation_std", "rotation_std_deg", "board_in_base_list"}
    method_results = {}

    for method, name in methods:
        try:
            R_cam2ee, t_cam2ee = cv2.calibrateHandEye(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                method=method
            )
            # 用 Board-in-base 一致性评估
            consistency = compute_board_in_base_consistency(
                R_gripper2base_list, t_gripper2base_list,
                R_target2cam_list, t_target2cam_list,
                R_cam2ee, t_cam2ee
            )
            method_results[name] = {
                "R": R_cam2ee,
                "t": t_cam2ee,
                "combined_score": consistency["combined_score"],
                "translation_std": consistency["translation_std"],
                "rotation_std_deg": consistency["rotation_std_deg"],
                "board_in_base_list": consistency["board_in_base_list"],
            }
            print("  [INFO] %s 方法: 求解成功"
                  "  t_std=%.6f m  r_std=%.4f deg  score=%.8f"
                  % (name, consistency["translation_std"],
                     consistency["rotation_std_deg"],
                     consistency["combined_score"]))
        except Exception as e:
            print("  [WARN] %s 方法: 求解失败 - %s" % (name, str(e)))

    if not method_results:
        print("[ERROR] 所有手眼标定方法均失败")
        return

    # 按综合评分 (combined_score) 升序排序, 取最小者为最佳
    sorted_methods = sorted(method_results.items(),
                            key=lambda kv: kv[1]["combined_score"])
    best_method = sorted_methods[0][0]
    best_R = method_results[best_method]["R"]
    best_t = method_results[best_method]["t"]
    best_board_in_base = method_results[best_method]["board_in_base_list"]

    R_cam2ee = best_R
    t_cam2ee = best_t.reshape(3)
    euler_cam2ee = rotation_matrix_to_euler_zyx(R_cam2ee)

    print("[INFO] 手眼标定方法排名 (按 Board-in-base 一致性评分):")
    for rank, (name, res) in enumerate(sorted_methods, 1):
        marker = " * [BEST]" if name == best_method else "   "
        print("  %s #%d %s: score=%.8f  (t_std=%.6fm, r_std=%.4f°)"
              % (marker, rank, name,
                 res["combined_score"],
                 res["translation_std"],
                 res["rotation_std_deg"]))
    print("[INFO] 使用 %s 方法的结果" % best_method)
    print("[INFO] 手眼矩阵 (cam->ee):")
    print("  旋转矩阵:\n%s" % str(R_cam2ee))
    print("  平移向量: %s" % str(t_cam2ee))
    print("  Euler ZYX [deg]: rx=%.4f, ry=%.4f, rz=%.4f" % euler_cam2ee)

    # ---- [Step 6] 保存结果 ----
    save_calibration_result(
        SAVE_DIR, camera_matrix, dist_coeffs,
        R_cam2ee, t_cam2ee, euler_cam2ee,
        filtered_indices, filtered_errors, ret,
        method_results, best_method
    )
    save_board_in_base(SAVE_DIR, best_board_in_base,
                       index_list=filtered_indices)

    # ---- 打印摘要 ----
    print()
    print("=" * 60)
    print("  手眼标定完成")
    print("  有效数据对: %d" % n_filtered)
    print("  相机标定重投影误差: %.4f px" % ret)
    print("  Board-in-base 平移标准差: %.6f m" % method_results[best_method]["translation_std"])
    print("  Board-in-base 旋转标准差: %.4f deg" % method_results[best_method]["rotation_std_deg"])
    print("  手眼标定方法: %s" % best_method)
    print("  手眼矩阵 Euler ZYX [deg]: rx=%.4f, ry=%.4f, rz=%.4f" % euler_cam2ee)
    print("  手眼矩阵平移 [m]: x=%.6f, y=%.6f, z=%.6f" % tuple(t_cam2ee))
    print("=" * 60)


if __name__ == "__main__":
    main()