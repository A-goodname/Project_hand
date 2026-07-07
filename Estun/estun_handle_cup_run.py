# -*- coding: utf-8 -*-
"""
ESTUN RealSense_R Handle + Cup 6D 融合抓取系统 (旧相机版)
==============================================
整合 Handle 6D 检测器 (微调 YOLOv8n + SAM + 扫描线分离把手) 与 Cup 6D 检测器 (YOLOv8m COCO),
手眼标定, TCP 服务器与灵巧手, 实现全流程: 抓壶倒水 → 抓杯递杯.

使用 RealSense_R 旧模块 (手动像素映射, 无 rs.align).

与 estun_handle_run.py 的差异:
  - 双检测器: RealSenseHandle6DDetector + RealSenseCup6DDetector (共享 pipeline, RealSense_R)
  - 两阶段: Phase 1 (Handle 抓取+倒水) + Phase 2 (Cup 抓取+递杯)
  - TCP Phase C 指令: Handle 阶段用 PREGRIP/GRIPDONE/RESET, Cup 阶段用 RELEASE/GRIP
  - P3_pour 倒水位置: 基于 Cup TCP 基系位置 + 杯几何偏移反算回相机系
  - 手势:   使用 utils_shared 函数式接口
        把手阶段: hand_pre_grip/hand_grip_done/hand_reset
        水杯阶段: release_object(PAPER)/grip_object(GRIP)

使用说明:
  1. 连接 RealSense 相机
  2. 连接灵巧手 (USB 串口)
  3. 运行 python Estun/estun_handle_cup_run.py
  4. 机器人作为 TCP 客户端连接本服务器

输入:
  - RealSense_R/config.yaml          : 相机 / YOLO / SAM / cup 配置 (旧相机)
  - Estun/estun_config.yaml          : 机器人 / 把手偏移 / 水杯偏移配置
  - calib_data/calibration_result.yaml : 手眼标定矩阵 (可选)

输出:
  - 6 个稳定目标位姿 [x,y,z,A,B,C] 分两阶段发送至机器人 (TCP)
  - 实时显示画面 (Handle 6D + Cup 6D)

流程:
  主线程:
    [Init] 加载标定 / 手 / 检测器 / TCP -> 主循环
    [Step1] 获取帧
    [Step2] Handle 6D + Cup 6D 同步检测 (跳帧)
    [Step3] cam2ee -> 稳定化 -> last_known_pose 更新
    [Step4] 显示画面
    [Step5] 非阻塞 accept -> 子线程 handle_client

  子线程 handle_client (daemon=True):
    [Phase A] 接收机器人初始位姿 (5s 超时)
    [Phase B] 轮询 last_known_pose ->
              Phase 1a: Handle P1/P2 (不依赖倒水口)
              Phase 2:  Cup P1/P2/P3 (得到 Cup TCP 基系位置)
              Phase 1b: 基于 Cup TCP 计算倒水口 → Handle P3
              发送 Handle 三个位姿
    [Phase C] 指令循环: PREGRIP / GRIPDONE / RESET / CUP / RELEASE / GRIP
              CUP 时回复 Cup P1/P2/P3, 继续指令循环
"""

import os
import socket
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入自定义包 ----
from RealSense_R.test_handle_6D import RealSenseHandle6DDetector
from RealSense_R.test_cup_6D import RealSenseCup6DDetector
from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles
from Estun.utils_shared import (
    euler_to_rotation_matrix, transform_pose_to_base,
    compute_gripper_offset_transform, compute_grasp_rotation,
    post_correct_angles,
    init_hand, is_hand_available,
    hand_pre_grip, hand_grip_done, hand_reset, disable_hand,
    grip_object, release_object,
)

# ---- 全局变量 ----
stop_event = threading.Event()
_hand_available = False  # 灵巧手是否可用, 在 main 中初始化
_poses_cleared = False       # daemon 线程在机械臂断开时置 True
_waiting_for_stable = False  # 主循环在等待重稳定时置 True

# ---- 加载 RealSense / YOLO 配置 ----
_cfg_path = os.path.join(_project_root, "RealSense_R", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---- 加载 ESTUN 配置 ----
_estun_cfg_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

HOST = _cfg_estun["tcp_server"]["host"]
PORT = _cfg_estun["tcp_server"]["port"]
HAND_PORT = _cfg_estun["hand"]["port"]
CALIB_RESULT_PATH = os.path.join(_project_root, _cfg_estun["calibration"]["result_path"])
STABLE_WINDOW = _cfg_estun["stabilization"]["window"]
POS_THRESHOLD = _cfg_estun["stabilization"]["pos_threshold"]
ANGLE_THRESHOLD = _cfg_estun["stabilization"]["angle_threshold"]

# ---- 把手偏移 ----
HANDLE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["t_handle_offset"], dtype=float)
HANDLE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["r_handle_offset"], dtype=float)
HANDLE_T_TCP_FIXED = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["t_tcp_fixed"], dtype=float)  # 末端系下固定 TCP 偏移, 仅用于 P3 倒水口计算
HANDLE_PRE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["t_offset"], dtype=float)
HANDLE_PRE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["r_offset"], dtype=float)
HANDLE_TARGET_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["t_offset"], dtype=float)
HANDLE_TARGET_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- cup handle_side (从 RealSense_R config 的 cup 段读取) ----
CUP_HANDLE_SIDE = _cfg.get("cup", {}).get("handle_side", "back")
print(f"[INFO] cup handle_side: {CUP_HANDLE_SIDE}")

# ---- 水杯偏移 (根据 handle_side 选择对应偏移) ----
if CUP_HANDLE_SIDE == "left":
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset_left"], dtype=float)
    print(f"[INFO] 使用 left 模式偏移 t_cup_offset_left")
else:
    CUP_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset"], dtype=float)
    print(f"[INFO] 使用 back 模式偏移 t_cup_offset")
CUP_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["gripper_offset"]["r_cup_offset"], dtype=float)
CUP_PRE_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["t_offset"], dtype=float)
# ---- P3 示教偏移 (根据 cup handle_side + pour_angle 选择) ----
P3_TEACH_ENABLE = _cfg_estun["handle_grasp"]["p3_teach_offset"]["enable"]
P3_POUR_ANGLE = _cfg_estun["handle_grasp"]["p3_teach_offset"].get("pour_angle", 30)
P3_TEACH_SIDE = "back" if CUP_HANDLE_SIDE == "back" else "left"
_angle_key = str(P3_POUR_ANGLE)
_side_cfg_raw = _cfg_estun["handle_grasp"]["p3_teach_offset"][P3_TEACH_SIDE]
_side_cfg = {str(k): v for k, v in _side_cfg_raw.items()}  # YAML 整数键→字符串
if _angle_key not in _side_cfg:
    _angle_key = list(_side_cfg.keys())[0]  # 回退到第一个可用角度
P3_TEACH_T_OFFSET = np.array(_side_cfg[_angle_key]["t_offset"], dtype=float)
P3_TEACH_R_ABC = np.array(_side_cfg[_angle_key]["r_abc"], dtype=float)
if P3_TEACH_ENABLE:
    print(f"[INFO] P3 示教偏移: side={P3_TEACH_SIDE}, pour_angle={_angle_key}°, "
          f"t_offset=({P3_TEACH_T_OFFSET[0]:.4f},{P3_TEACH_T_OFFSET[1]:.4f},{P3_TEACH_T_OFFSET[2]:.4f}) "
          f"r_abc=({P3_TEACH_R_ABC[0]:.1f},{P3_TEACH_R_ABC[1]:.1f},{P3_TEACH_R_ABC[2]:.1f})")
CUP_PRE_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["pre_offset"]["r_offset"], dtype=float)
CUP_TARGET_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["t_offset"], dtype=float)
CUP_TARGET_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 壶嘴偏移 + 倒水偏移 ----
SPOUT_T_OFFSET = np.array(_cfg_estun["cup_grasp"]["spout_offset"]["t_offset"], dtype=float)
POUR_T_OFFSET_BASE = np.array(_cfg_estun["cup_grasp"]["pour_offset"]["t_offset"], dtype=float)
POUR_R_OFFSET = np.array(_cfg_estun["cup_grasp"]["pour_offset"]["r_offset"], dtype=float)

# ---- 杯几何参数 (用于倒水口计算) ----
CUP_RIM_RADIUS = _cfg.get("cup", {}).get("rim_radius", 0.035)
CUP_HEIGHT = _cfg.get("cup", {}).get("height", 0.100)

# ---- 方向模式 ----
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]
HANDLE_SIDE = _cfg["handle"]["side"]

if HANDLE_SIDE != GRASP_DIRECTION:
    print(f"[INFO] grasp_direction ({GRASP_DIRECTION}) 自动切换为 handle_side ({HANDLE_SIDE})")
    GRASP_DIRECTION = HANDLE_SIDE

# ---- 把手检测参数 ----
SKIP_FRAMES = _cfg["handle"]["skip_frames"]

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

VALID_MODES = {"vertical", "horizontal", "free", "fixed"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE}, 将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现, 将使用 free 模式")
    ORIENTATION_MODE = "free"


def fmt_pose(xyz, abc):
    """格式化单个位姿: "x,y,z,A,B,C" (位置 m, 角度 deg)"""
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def pack_three(p1, p2, p3):
    """打包三个位姿: <P1>[P2]{P3} — 与 ESTUN TranStrToCpos 括号标记约定匹配"""
    return f"<{fmt_pose(p1[0], p1[1])}>[{fmt_pose(p2[0], p2[1])}]{{{fmt_pose(p3[0], p3[1])}}}"


def load_cam2ee(yaml_path):
    """从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)"""
    I = np.eye(4)
    path = Path(yaml_path)
    if not path.exists():
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}")
        print("[WARN] 将直接发送相机坐标系位姿 (无灵巧手变换)")
        return I

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    calib = data.get("calibration", {})
    hand_eye = calib.get("handeye", {})
    if not hand_eye:
        print("[WARN] 标定文件中未找到 calibration/handeye 章节")
        print("[WARN] 使用单位矩阵")
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
    print(f"[INFO] 已加载标定矩阵: {use_method}.T_cam_to_gripper (cam->ee), 平移: {t} m")
    print("[INFO] 注意: 还需叠加 handle_gripper_offset 才得到最终灵巧手 TCP 位姿")
    return T


# =============================================================
# 功能模块: 代际计数器与位姿稳定器
# =============================================================

_gf = 0  # 全局代际

class PoseStabilizer:
    """
    缓存最近 N 帧位姿数据; 当位置和角度变化均低于阈值时,
    认为位姿稳定, 返回缓冲区均值作为稳定结果
    使用代际计数器 (_sf) 确保稳定后的数据只属于当前客户端连接周期
    """

    def __init__(self, window=5, pos_thresh=0.005, angle_thresh=3.0):
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self._buffer = []
        self._sf = -1

    def capture_generation(self):
        self._sf = _gf

    @staticmethod
    def on_disconnect():
        global _gf
        _gf += 1

    def is_generation_valid(self):
        return self._sf == _gf

    def add(self, pos, euler):
        self._buffer.append((np.array(pos, dtype=float), np.array(euler, dtype=float)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self):
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

    def get_mean(self):
        if not self._buffer:
            return None, None
        poses = np.array([p for p, _ in self._buffer])
        angles = np.array([a for _, a in self._buffer])
        return poses.mean(axis=0), angles.mean(axis=0)

    def reset(self):
        self._buffer = []


# =============================================================
# 功能模块: TCP 客户端处理
# =============================================================

def _handle_client_impl(conn, addr, detector, T_cam2ee, cup_detector):
    """
    实际的客户端处理逻辑

    参数:
        conn         : socket, 客户端连接
        addr         : (str, int), 客户端地址
        detector     : RealSenseHandle6DDetector
        T_cam2ee     : np.ndarray, 4x4 相机 -> 末端法兰变换矩阵
        cup_detector : RealSenseCup6DDetector (共享 pipeline, 注入帧)
    """
    global _hand_available
    print(f"[INFO] 客户端已连接: {addr}")

    # ---- 预计算工具偏移 ----
    _, t_handle_effective = compute_gripper_offset_transform(HANDLE_T_OFFSET, HANDLE_R_OFFSET)
    _, t_cup_effective = compute_gripper_offset_transform(CUP_T_OFFSET, CUP_R_OFFSET)
    cup_pose_data = None  # pack_three 格式 <P1>[P2]{P3}, 在 Phase B 中获取, Phase C 使用

    # ---- [Phase A] 接收机器人初始末端位姿 (5s 超时) ----
    robot_xyz = None
    robot_ABC = None
    try:
        conn.settimeout(5.0)
        init_data = conn.recv(1024)
        if not init_data:
            print("[WARN] 机器人未发送初始位姿, 断开连接", flush=True)
            return
        pose_str = init_data.decode("utf-8").strip().strip("[]")
        parts = pose_str.split(",")
        if len(parts) == 6:
            robot_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
            robot_ABC = (float(parts[3]), float(parts[4]), float(parts[5]))
            print(f"[INFO] 机器人当前末端位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                  f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})", flush=True)
        else:
            print(f"[WARN] 初始位姿格式错误: {init_data.decode('utf-8').strip()}", flush=True)
            return
    except socket.timeout:
        print("[WARN] 等待机器人初始位姿超时, 断开连接", flush=True)
        return
    except Exception as e:
        print(f"[WARN] 接收初始位姿异常: {e}", flush=True)
        return

    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
    t_ee_base = np.array(robot_xyz, dtype=float)

    # ---- [Phase B] 等待稳定位姿 -> 计算位姿 -> 发送 ----
    print("[INFO] 等待稳定目标位姿 (Handle + Cup)...", flush=True)
    target_sent = False
    wait_start = time.time()
    while not target_sent and not stop_event.is_set() and (time.time() - wait_start) < 30.0:
        handle_pose = detector.last_known_pose
        cup_pose = cup_detector.last_known_pose if hasattr(cup_detector, 'last_known_pose') else None

        if handle_pose is not None and cup_pose is not None:
            mean_pos_ee, mean_euler_ee = handle_pose
            cup_pos_cam, cup_euler_cam = cup_pose

            # ============================================================
            # Phase 1a: Handle P1 / P2 (不依赖倒水口, 先计算)
            # ============================================================
            h_pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
            R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
            R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, HANDLE_R_OFFSET, R_ee_base,
                                                grasp_direction=GRASP_DIRECTION)

            if ORIENTATION_MODE == "fixed":
                # --- fixed 模式 ---
                R_base_fix = euler_to_rotation_matrix(*FIXED_ABC, order="zyx")
                R_grasp_ee = R_ee_base.T @ R_base_fix

                pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = tuple(FIXED_ABC)

                pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ HANDLE_PRE_T_OFFSET
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = tuple(FIXED_ABC)

                # P3_pour 延后到 Phase 1b (需要 Cup TCP 计算倒水口)
                R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")

                # fixed 模式变量
                _fixed_R_grasp_ee = R_grasp_ee
                _fixed_R_target_ee = R_target_ee
            else:
                # --- vertical/free 模式 ---
                rot_base_1 = R_ee_base @ R_grasp_ee
                pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = rotation_matrix_to_euler_angles(rot_base_1)
                A1, B1, C1 = post_correct_angles(A1, B1, C1, robot_ABC)
                # vertical 模式修正 B/C 角; free/fixed 模式仅可达性修复
                if ORIENTATION_MODE == "vertical" and GRASP_DIRECTION == "right":
                    # C 角 clamp 到 [-100, -80], 使灵巧手手指垂直于把手方向
                    if not (-100 <= C1 <= -80):
                        print(f"[CLAMP] Handle P1 C 角 {C1:.1f} -> -90.0 (vertical)")
                        C1 = -90.0
                    # B 角 clamp 到 [84, 86], 超出则赋中值 85 (右手抓取)
                    if not (84 <= B1 <= 86):
                        print(f"[CLAMP] Handle P1 B 角 {B1:.1f} -> 85.0")
                        B1 = 85.0

                rot_base_1_corrected = euler_to_rotation_matrix(A1, B1, C1, order="zyx")
                R_grasp_ee_corrected = R_ee_base.T @ rot_base_1_corrected

                pos_ee_pre = h_pos_ee_compensated + R_grasp_ee_corrected @ HANDLE_PRE_T_OFFSET
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = A1, B1, C1

                R_pour_ee = R_grasp_ee_corrected @ euler_to_rotation_matrix(*POUR_R_OFFSET, order="zyx")

                # vertical/free 模式变量
                _R_grasp_ee_corrected = R_grasp_ee_corrected
                _R_pour_ee = R_pour_ee

            # ============================================================
            # Phase 2: Cup P1 / P2 / P3 (计算 Cup TCP 基系位置 c_pos_base_1)
            # ============================================================
            c_pos_ee = T_cam2ee[:3, :3] @ np.array(cup_pos_cam) + T_cam2ee[:3, 3]
            c_rot_ee = T_cam2ee[:3, :3] @ euler_to_rotation_matrix(*cup_euler_cam, order="zyx")
            c_pos_ee_compensated = c_pos_ee - t_cup_effective
            c_R_grasp_ee = compute_grasp_rotation(c_rot_ee, ORIENTATION_MODE, CUP_R_OFFSET, R_ee_base,
                                                  grasp_direction=GRASP_DIRECTION)

            # Cup P1
            c_rot_base_1 = R_ee_base @ c_R_grasp_ee
            c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)
            cA1, cB1, cC1 = rotation_matrix_to_euler_angles(c_rot_base_1)
            cA1, cB1, cC1 = post_correct_angles(cA1, cB1, cC1, robot_ABC)
            # ---- 根据 cup handle_side 施加角度约束 (仅 vertical 模式; free 无限制, 仅可达性修复) ----
            if ORIENTATION_MODE == "vertical":
                if CUP_HANDLE_SIDE == "back":
                    if cB1 < 90:
                        print(f"[CLAMP] Cup P1 B 角 {cB1:.1f} -> 89.0 (vertical back)")
                        cB1 = 89.0
                    elif cB1 > 90:
                        print(f"[CLAMP] Cup P1 B 角 {cB1:.1f} -> 91.0 (vertical back)")
                        cB1 = 91.0
                    if not (-120 <= cC1 <= -100):
                        print(f"[CLAMP] Cup P1 C 角 {cC1:.1f} -> -110.0 (back 模式)")
                        cC1 = -110.0
                else:
                    if cB1 < 90:
                        print(f"[CLAMP] Cup P1 B 角 {cB1:.1f} -> 89.0 (vertical left)")
                        cB1 = 89.0
                    elif cB1 > 90:
                        print(f"[CLAMP] Cup P1 B 角 {cB1:.1f} -> 91.0 (vertical left)")
                        cB1 = 91.0
                    if not (-100 <= cC1 <= -80):
                        print(f"[CLAMP] Cup P1 C 角 {cC1:.1f} -> -90.0 (left 模式)")
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

            # 缓存 cup 位姿供 Phase C 使用
            cup_pose_data = pack_three(
                ((c_pos_base_1, (cA1, cB1, cC1))),
                ((c_pos_base_2, (cA2, cB2, cC2))),
                ((c_pos_base_3, (cA3, cB3, cC3)))
            )

            # ============================================================
            # Phase 1b: 基于 Cup TCP 计算倒水口 → Handle P3
            # ============================================================
            if P3_TEACH_ENABLE:
                # ---- 使用示教偏移 (基系) ----
                pos_base_3 = c_pos_base_1 + P3_TEACH_T_OFFSET
                A3, B3, C3 = tuple(P3_TEACH_R_ABC)
                rot_base_3 = euler_to_rotation_matrix(A3, B3, C3, order="zyx")
                print(f"[POUR] 使用示教偏移: t_offset=({P3_TEACH_T_OFFSET[0]:.4f},{P3_TEACH_T_OFFSET[1]:.4f},{P3_TEACH_T_OFFSET[2]:.4f}) "
                      f"r_abc=({P3_TEACH_R_ABC[0]:.1f},{P3_TEACH_R_ABC[1]:.1f},{P3_TEACH_R_ABC[2]:.1f})")
            else:
                # ---- 基于视觉计算倒水口 ----
                _tcp_offset_base = R_ee_base @ HANDLE_T_TCP_FIXED
                _tcp_base = c_pos_base_1 + _tcp_offset_base
                if CUP_HANDLE_SIDE == "back":
                    pour_target_base = _tcp_base + np.array([CUP_RIM_RADIUS, 0.0, CUP_HEIGHT / 2.0])
                else:  # left
                    pour_target_base = _tcp_base + np.array([0.0, CUP_RIM_RADIUS, CUP_HEIGHT / 2.0])

                pour_target_ee_calc = R_ee_base.T @ (pour_target_base - t_ee_base)
                pour_target = T_cam2ee[:3, :3].T @ (pour_target_ee_calc - T_cam2ee[:3, 3])
                print(f"[POUR] 倒水口 (视觉): "
                      f"base=({pour_target_base[0]:.4f},{pour_target_base[1]:.4f},{pour_target_base[2]:.4f}) "
                      f"cam=({pour_target[0]:.4f},{pour_target[1]:.4f},{pour_target[2]:.4f})")

                if ORIENTATION_MODE == "fixed":
                    pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
                    pos_ee_p3 = pos_pour_ee - _fixed_R_grasp_ee @ SPOUT_T_OFFSET - t_handle_effective
                    pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
                    rot_base_3 = R_ee_base @ _fixed_R_target_ee
                    A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                    A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)
                else:
                    pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
                    spout_rotated = _R_pour_ee @ SPOUT_T_OFFSET
                    pos_ee_p3 = pos_pour_ee - spout_rotated - t_handle_effective
                    pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + POUR_T_OFFSET_BASE
                    rot_base_3 = R_ee_base @ _R_pour_ee
                    A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                    A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)

            # ---- 发送 Handle 三个位姿 ----
            try:
                msg = pack_three(
                    ((pos_base_1, (A1, B1, C1))),
                    ((pos_base_2, (A2, B2, C2))),
                    ((pos_base_3, (A3, B3, C3)))
                )
                conn.sendall(msg.encode("utf-8"))
                print(f"[SEND] Handle 合并位姿: {msg}", flush=True)
                target_sent = True
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[WARN] 发送位姿失败: {e}", flush=True)
                return
        else:
            time.sleep(0.1)

    if not target_sent:
        print("[WARN] 超时: 未获得稳定目标位姿, 断开连接", flush=True)
        return

    # ---- [Phase C] 指令循环: PREGRIP / GRIPDONE / RESET / RELEASE / GRIP / CUP ----
    conn.settimeout(2.0)
    try:
        while not stop_event.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue

            if not data:
                print("[INFO] 连接已关闭 (机器人执行了 SocketClose)", flush=True)
                break

            cmd = data.decode("utf-8").strip()
            print(f"[RECV] {cmd}", flush=True)

            if cmd == "PREGRIP":
                if _hand_available:
                    hand_pre_grip()
                print("[INFO] 灵巧手: PREGRIP", flush=True)

            elif cmd == "GRIPDONE":
                if _hand_available:
                    hand_grip_done()
                print("[INFO] 灵巧手: GRIPDONE", flush=True)

            elif cmd == "RESET":
                if _hand_available:
                    hand_reset()
                print("[INFO] 灵巧手: RESET", flush=True)

            elif cmd == "RELEASE":
                if _hand_available:
                    release_object()
                print("[INFO] 灵巧手: RELEASE (PAPER)", flush=True)

            elif cmd == "GRIP":
                if _hand_available:
                    grip_object()
                print("[INFO] 灵巧手: GRIP", flush=True)

            elif cmd == "CUP":
                # 回复 cup 三个位姿, 格式与 Phase B 一致
                if cup_pose_data is not None:
                    try:
                        conn.sendall(cup_pose_data.encode("utf-8"))
                        print(f"[SEND] Cup 合并位姿: {cup_pose_data}", flush=True)
                    except (BrokenPipeError, ConnectionResetError) as e:
                        print(f"[WARN] 发送 cup 位姿失败: {e}", flush=True)
                        break
                else:
                    print("[WARN] CUP 命令时 cup_pose_data 为 None, 忽略", flush=True)

            else:
                print(f"[INFO] 忽略非指令数据: {cmd}", flush=True)

    except ConnectionResetError:
        print(f"[INFO] 连接重置: {addr}", flush=True)
    except Exception as e:
        print(f"[ERROR] 客户端处理异常: {e}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[INFO] 客户端已断开: {addr}", flush=True)


def handle_client(conn, addr, detector, T_cam2ee, cup_detector):
    """TCP 客户端处理入口 (wrapper) — 断开时清零位姿, 触发主循环重稳定"""
    global _poses_cleared
    try:
        _handle_client_impl(conn, addr, detector, T_cam2ee, cup_detector)
    finally:
        PoseStabilizer.on_disconnect()
        # 断开时清零位姿缓存, 通知主循环重新检测
        detector.last_known_pose = None
        cup_detector.last_known_pose = None
        _poses_cleared = True


# =============================================================
# 主函数
# =============================================================

def main():
    global _hand_available, _poses_cleared, _waiting_for_stable
    print("=" * 60)
    print("  ESTUN RealSense_R Handle + Cup 6D 融合抓取系统 (旧相机版)")
    print("  模型: Handle=微调YOLOv8n+SAM | Cup=YOLOv8m COCO")
    print("  流程: 抓壶倒水 → 抓杯递杯")
    print("=" * 60, flush=True)

    # ---- [Init 1] 加载手眼标定矩阵 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")

    # ---- [Init 2] 初始化灵巧手 (使用 utils_shared 函数式接口) ----
    print(f"\n[INFO] 正在初始化灵巧手...")
    init_hand(hand_port=HAND_PORT, initial_gesture="RESET")
    _hand_available = is_hand_available()
    if _hand_available:
        print("[INFO] 灵巧手已准备就绪")
    else:
        print("[WARN] 灵巧手未就绪, 将模拟所有手势操作")

    # ---- [Init 3] 初始化 Handle 6D 检测器 (启动 pipeline) ----
    print(f"\n[INFO] 正在初始化 RealSense_R Handle 6D 检测器 (旧相机)...")
    try:
        detector = RealSenseHandle6DDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        return

    # ---- [Init 4] 初始化 Cup 6D 检测器 (不启动 pipeline, 复用帧) ----
    print(f"\n[INFO] 正在初始化 RealSense_R Cup 6D 检测器 (复用 pipeline)...")
    try:
        cup_detector = RealSenseCup6DDetector(config_dict=_cfg, start_pipeline=False)
        # 复用 pipeline 时不调用 _init_realsense, 从 Handle 检测器拷贝所有相机参数
        cup_detector.color_intrinsics = detector.color_intrinsics
        cup_detector.depth_intrinsics = detector.depth_intrinsics
        cup_detector.depth_scale = detector.depth_scale
        cup_detector.color_to_depth_extrinsics = detector.color_to_depth_extrinsics
        cup_detector.depth_to_color_extrinsics = detector.depth_to_color_extrinsics
    except Exception as e:
        print(f"[FATAL] Cup 检测器初始化失败: {e}")
        try:
            detector.pipeline.stop()
        except Exception:
            pass
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        return

    # ---- [Init 5] 启动 TCP 服务器 (非阻塞) ----
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"\n[INFO] 服务器已启动: {HOST}:{PORT}")
    print("[INFO] 等待机器人连接...", flush=True)

    detector.last_known_pose = None
    cup_detector.last_known_pose = None  # type: ignore[assignment]

    main_handle_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW, pos_thresh=POS_THRESHOLD, angle_thresh=ANGLE_THRESHOLD
    )
    main_cup_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW, pos_thresh=POS_THRESHOLD, angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("ESTUN Handle+Cup 6D (RealSense_R)", cv2.WINDOW_NORMAL)

    # ---- 跳帧控制 ----
    frame_counter = 0
    cached_handle = None
    cached_cup = None

    running = True
    try:
        while running:
            # ---- [Step 1] 获取帧 ----
            color_image, depth_image, depth_colormap, depth_frame = detector.get_frames()
            if color_image is None or depth_frame is None:
                time.sleep(0.03)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出", flush=True)
                    stop_event.set()
                    running = False
                    break
                continue

            frame_counter += 1
            display_img = color_image.copy()

            # ---- [Step 2] 每 SKIP_FRAMES 帧执行一次检测 ----
            if frame_counter % SKIP_FRAMES == 1:
                # === Handle 检测 ===
                h_result = detector.detect_handle_6d(color_image, depth_frame)
                body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler, red_mask, red_center = h_result

                if h_pos is not None and h_rot is not None:
                    smooth_pos, smooth_rot = detector._pose_smoother.update(h_pos, h_rot)
                    if smooth_rot is not None:
                        h_pos = smooth_pos
                        h_rot = smooth_rot
                        h_euler = rotation_matrix_to_euler_angles(h_rot)

                    assert h_pos is not None and h_euler is not None
                    main_handle_stabilizer.add(tuple(h_pos), tuple(h_euler))
                    if main_handle_stabilizer.is_stable():
                        if not main_handle_stabilizer.is_generation_valid():
                            main_handle_stabilizer.reset()
                            main_handle_stabilizer.capture_generation()
                        else:
                            mean_pos_cam, mean_euler_cam = main_handle_stabilizer.get_mean()
                            if mean_pos_cam is not None and mean_euler_cam is not None:
                                mean_rot_cam = euler_to_rotation_matrix(*mean_euler_cam, order="zyx")
                                pos_ee = T_cam2ee[:3, :3] @ mean_pos_cam + T_cam2ee[:3, 3]
                                rot_ee = T_cam2ee[:3, :3] @ mean_rot_cam
                                A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)
                                detector.last_known_pose = (pos_ee, (A_ee, B_ee, C_ee))
                                print(f"\n[Handle STABLE] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                                      f"euler=({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})", flush=True)
                else:
                    detector._pose_smoother.reset()

                # === Cup 检测 (注入帧) ===
                cup_detector.inject_frames(color_image, depth_image, depth_colormap, depth_frame)
                (c_bbox, c_pos, c_rot, c_euler, c_point_cloud) = cup_detector.compute_cup_6d_pose(color_image, depth_frame)

                if c_pos is not None and c_rot is not None:
                    smooth_cpos, smooth_crot = cup_detector._pose_smoother.update(c_pos, c_rot)
                    if smooth_crot is not None:
                        c_pos = smooth_cpos
                        c_rot = smooth_crot
                        c_euler = rotation_matrix_to_euler_angles(c_rot)

                    assert c_pos is not None and c_euler is not None
                    main_cup_stabilizer.add(tuple(c_pos), tuple(c_euler))
                    if main_cup_stabilizer.is_stable():
                        if not main_cup_stabilizer.is_generation_valid():
                            main_cup_stabilizer.reset()
                            main_cup_stabilizer.capture_generation()
                        else:
                            mean_cpos_cam, mean_ceuler_cam = main_cup_stabilizer.get_mean()
                            if mean_cpos_cam is not None and mean_ceuler_cam is not None:
                                cup_detector.last_known_pose = (mean_cpos_cam, mean_ceuler_cam)  # type: ignore[assignment]
                                print(f"\n[Cup STABLE]    pos=({mean_cpos_cam[0]:.3f},{mean_cpos_cam[1]:.3f},{mean_cpos_cam[2]:.3f}) "
                                      f"euler=({mean_ceuler_cam[0]:.1f},{mean_ceuler_cam[1]:.1f},{mean_ceuler_cam[2]:.1f})", flush=True)
                else:
                    cup_detector._pose_smoother.reset()

                # ---- 重稳定完成通知: 双检测器均稳定后才提示 ----
                if _waiting_for_stable:
                    if detector.last_known_pose is not None and cup_detector.last_known_pose is not None:
                        _waiting_for_stable = False
                        print("\n" + "=" * 60)
                        print("[INFO] Handle + Cup 位姿已稳定, 可以启动机械臂")
                        print("=" * 60, flush=True)

                cached_handle = (body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler)
                cached_cup = (c_bbox, c_pos, c_rot, c_euler, c_point_cloud)
            else:
                if cached_handle is not None:
                    body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler = cached_handle
                else:
                    body_mask = handle_mask = None
                    h_stage = "none"
                    yolo_bbox = h_pos = h_rot = h_euler = None

                if cached_cup is not None:
                    c_bbox, c_pos, c_rot, c_euler, c_point_cloud = cached_cup
                else:
                    c_bbox = c_pos = c_rot = c_euler = c_point_cloud = None

            # ---- 可视化 ----
            if cached_handle is not None:
                detector.visualize(display_img, body_mask, handle_mask,
                                   h_stage, yolo_bbox, h_pos, h_rot, h_euler)
            if cached_cup is not None and c_bbox is not None:
                cup_detector.visualize_cup(display_img, c_bbox, c_pos, c_rot, c_euler)

            # ---- ee 系位姿文字 ----
            if h_pos is not None and h_rot is not None and cached_handle is not None:
                pos_ee = T_cam2ee[:3, :3] @ np.array(h_pos) + T_cam2ee[:3, 3]
                rot_ee = T_cam2ee[:3, :3] @ h_rot
                A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)
                cv2.putText(display_img, f"H ee: ({pos_ee[0]:.2f},{pos_ee[1]:.2f},{pos_ee[2]:.2f})",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(display_img, f"H euler: ({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # ---- 状态文字 ----
            h_status = "H:OK" if detector.last_known_pose is not None else "H:..."
            c_status = "C:OK" if cup_detector.last_known_pose is not None else "C:..."
            cv2.putText(display_img, f"Status: {h_status} {c_status}",
                        (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

            # ---- [Step 4] 显示 ----
            if depth_colormap is not None:
                h_vis = display_img.shape[0]
                depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                vis_panel = np.hstack((display_img, depth_small))
            else:
                vis_panel = display_img
            cv2.imshow("ESTUN Handle+Cup 6D (RealSense_R)", vis_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                stop_event.set()
                running = False
                break

            # ---- [Step 5] 重稳定检测 + 非阻塞 accept -> 子线程 handle_client ----
            # 机械臂断开后 _poses_cleared 被 daemon 线程置 True, 在此处处理
            if _poses_cleared:
                _poses_cleared = False
                _waiting_for_stable = True
                main_handle_stabilizer.reset()
                main_cup_stabilizer.reset()
                main_handle_stabilizer.capture_generation()
                main_cup_stabilizer.capture_generation()
                print("\n" + "=" * 60)
                print("[INFO] 机械臂已断开, 位姿缓存已清除, 正在重新检测...")
                print("[INFO] (可更换抓取物体位置, 位姿稳定后将自动提示)")
                print("=" * 60, flush=True)

            try:
                conn, addr = server.accept()
            except BlockingIOError:
                continue

            main_handle_stabilizer.capture_generation()
            main_cup_stabilizer.capture_generation()
            # 不再清零 last_known_pose — 位姿已在断开时通过重稳定流程准备就绪

            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, detector, T_cam2ee, cup_detector),
                daemon=True
            )
            t.start()
            print("\n[INFO] 等待下一次连接...", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] 服务器异常: {e}")
    finally:
        cv2.destroyAllWindows()
        print("\n[INFO] 正在清理资源...")
        server.close()
        try:
            detector.pipeline.stop()
            print("[INFO] RealSense 管道已停止")
        except Exception as e:
            print(f"[WARN] 停止管道失败: {e}")
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        print("[INFO] 程序已退出", flush=True)


if __name__ == "__main__":
    main()