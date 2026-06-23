# -*- coding: utf-8 -*-
"""
ESTUN RealSense Handle 6D 抓取系统
===================================
整合 RealSense Handle 6D 检测器 (微调 YOLOv8n + SAM + 扫描线分离把手),
手眼标定, TCP 服务器与灵巧手, 实现实时把手检测, 位姿转换与机器人抓取控制

与 estun_run.py 的差异:
  - 检测器: RealSenseHandle6DDetector (微调 YOLOv8n + SAM + 把手分离)
  - 偏移:   使用 t_handle_offset (抓取把手), 独立于瓶身 t_offset
  - 跳帧:   每 SKIP_FRAMES 帧调用一次 detect_handle_6d
  - 3点位姿: 同时发送 P1(抓取点)/P2(预抓取点)/P3(目标点)
  - 手势:   使用 utils_shared 函数式接口 (hand_pre_grip/hand_grip_done/hand_reset)

使用说明:
  1. 连接 RealSense 相机
  2. 连接灵巧手 (USB 串口)
  3. 运行 python Estun/estun_handle_run.py
  4. 机器人作为 TCP 客户端连接本服务器

输入:
  - RealSense_R/config.yaml         : 相机 / 微调 YOLOv8n / SAM 配置
  - Estun/estun_config.yaml         : ESTUN 机器人 / 灵巧手 / 把手偏移配置
  - calib_data/calibration_result.yaml : 手眼标定矩阵 (可选)

输出:
  - 3个稳定目标位姿 [x,y,z,A,B,C] 发送至机器人 (TCP)
  - 实时显示画面 (彩色+深度, Handle 6D坐标系)

流程:
  主线程:
    [Init] 加载标定 / 手 / 检测器 / TCP -> 主循环
    [Step1] 获取帧
    [Step2] Handle 6D 检测 (跳帧)
    [Step3] cam2ee -> 稳定化 -> last_known_pose 更新
    [Step4] 显示画面
    [Step5] 非阻塞 accept -> 子线程 handle_client

  子线程 handle_client (daemon=True):
    [Phase A] 接收机器人初始位姿 (5s 超时)
    [Phase B] 轮询 last_known_pose -> 计算 P1/P2/P3 -> 依次发送
    [Phase C] 指令循环: PREGRIP / GRIPDONE / RESET
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
from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles
from Estun.utils_shared import (
    euler_to_rotation_matrix, transform_pose_to_base,
    compute_gripper_offset_transform, compute_grasp_rotation,
    post_correct_angles,
    init_hand, hand_pre_grip, hand_grip_done, hand_reset, disable_hand,
)

# ---- 全局变量 ----
stop_event = threading.Event()
_hand_available = False  # 灵巧手是否可用, 在 main 中初始化

# ---- 加载 RealSense / YOLO 配置 ----
_cfg_path = os.path.join(_project_root, "RealSense_R", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# ---- 加载 ESTUN 配置 (独立文件) ----
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

# ---- 把手偏移 (独立于瓶身) ----
HANDLE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["t_handle_offset"], dtype=float)
HANDLE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["gripper_offset"]["r_handle_offset"], dtype=float)

# ---- 方向模式 ----
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]
HANDLE_SIDE = _cfg["handle"]["side"]  # 视觉检测侧

# 把手模式下: 抓取方向自动与 handle_side 一致
if HANDLE_SIDE != GRASP_DIRECTION:
    print(f"[INFO] grasp_direction ({GRASP_DIRECTION}) 自动切换为 handle_side ({HANDLE_SIDE})")
    GRASP_DIRECTION = HANDLE_SIDE

# ---- 把手抓取参数 ----
# 预抓取偏移: 先平移再旋转
HANDLE_PRE_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["t_offset"], dtype=float)
HANDLE_PRE_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["pre_offset"]["r_offset"], dtype=float)
# 目标点偏移: 先平移再旋转
HANDLE_TARGET_T_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["t_offset"], dtype=float)
HANDLE_TARGET_R_OFFSET = np.array(_cfg_estun["handle_grasp"]["target_offset"]["r_offset"], dtype=float)

# ---- 把手检测参数 (skip_frames 来自 RealSense_R/config.yaml) ----
SKIP_FRAMES = _cfg["handle"]["skip_frames"]

# ---- fixed 模式: 读取固定 ABC ----
_fixed_abc = _cfg_estun.get("fixed_abc", {})
FIXED_ABC_CFG = _fixed_abc.get(GRASP_DIRECTION, [])
if len(FIXED_ABC_CFG) != 3:
    if ORIENTATION_MODE == "fixed":
        print(f"[WARN] {GRASP_DIRECTION} 方向未配置有效 fixed_abc, 回退到 free 模式")
        ORIENTATION_MODE = "free"
    FIXED_ABC = np.array([-90, 82, -98], dtype=float)  # fallback, 不被使用
else:
    FIXED_ABC = np.array(FIXED_ABC_CFG, dtype=float)

# ---- 防御性检查: 确保 orientation_mode 合法 ----
VALID_MODES = {"vertical", "horizontal", "free", "fixed"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE}, 将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现, 将使用 free 模式")
    ORIENTATION_MODE = "free"

def load_cam2ee(yaml_path):
    """
    从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)

    参数:
        yaml_path : str, 标定结果 YAML 路径

    返回:
        np.ndarray, 4x4 齐次矩阵; 文件缺失/格式错误时返回单位矩阵
    """
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
    t_str = f"({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
    print(f"[INFO] 已加载标定矩阵: {use_method}.T_cam_to_gripper (cam->ee), 平移: {t_str} m")
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

def _handle_client_impl(conn, addr, detector, T_cam2ee):
    """
    实际的客户端处理逻辑 (不负责 on_disconnect, 由外层 wrapper 兜底)

    参数:
        conn     : socket, 客户端连接
        addr     : (str, int), 客户端地址
        detector : RealSenseHandle6DDetector, 检测器实例
        T_cam2ee : np.ndarray, 4x4 相机 -> 末端法兰变换矩阵
    """
    global _hand_available
    print(f"[INFO] 客户端已连接: {addr}")

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

    # ---- [Phase B] 等待稳定位姿 -> 计算3个位姿 -> 依次发送 ----
    _, t_handle_effective = compute_gripper_offset_transform(HANDLE_T_OFFSET, HANDLE_R_OFFSET)
    print("[INFO] 等待稳定目标位姿...", flush=True)
    target_sent = False
    wait_start = time.time()
    while not target_sent and not stop_event.is_set() and (time.time() - wait_start) < 30.0:
        if detector.last_known_pose is not None:
            mean_pos_ee, mean_euler_ee = detector.last_known_pose

            # 基础: 位置补偿 + 方向对齐
            pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
            R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
            R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
            R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, HANDLE_R_OFFSET, R_ee_base, grasp_direction=GRASP_DIRECTION)

            if ORIENTATION_MODE == "fixed":
                # === Fixed 模式: 用配置的固定 ABC 覆盖姿态 ===
                R_base_fix = euler_to_rotation_matrix(*FIXED_ABC, order="zyx")
                R_grasp_ee = R_ee_base.T @ R_base_fix

                # P1: 固定姿态
                pos_base_1 = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = tuple(FIXED_ABC)

                # P2: 平移 PRE_T_OFFSET (在固定姿态末端系下定义), 姿态同 P1
                pos_ee_pre = pos_ee_compensated + R_grasp_ee @ HANDLE_PRE_T_OFFSET
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = tuple(FIXED_ABC)

                # P3: 叠加 target 旋转 + 平移 (在固定姿态末端系下定义)
                R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")
                rot_base_3 = R_ee_base @ R_target_ee
                pos_ee_target = pos_ee_compensated + R_grasp_ee @ HANDLE_TARGET_T_OFFSET
                pos_base_3 = transform_pose_to_base(pos_ee_target, robot_xyz, robot_ABC)
                A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)
            else:
                # === Vertical / Free 模式: 原有逻辑 ===
                # P1: 抓取点 (固定 C 值到合理范围)
                rot_base_1 = R_ee_base @ R_grasp_ee
                pos_base_1 = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = rotation_matrix_to_euler_angles(rot_base_1)
                A1, B1, C1 = post_correct_angles(A1, B1, C1, robot_ABC)
                # vertical 模式修正 B/C 角; free/fixed 模式仅可达性修复
                if ORIENTATION_MODE == "vertical" and GRASP_DIRECTION == "right":
                    # C 角 clamp 到 [-100, -80], 使灵巧手手指垂直于把手方向
                    if not (-100 <= C1 <= -80):
                        print(f"[CLAMP] Handle P1 C 角 {C1:.1f} -> -90.0 (vertical)")
                        C1 = -90.0
                    # B 角防跳变: <90->89, >90->91, ==90 不修正
                    if B1 < 90:
                        print(f"[CLAMP] Handle P1 B 角 {B1:.1f} -> 89.0 (vertical, <90)")
                        B1 = 89.0
                    elif B1 > 90:
                        print(f"[CLAMP] Handle P1 B 角 {B1:.1f} -> 91.0 (vertical, >90)")
                        B1 = 91.0

                # 从修正后 P1 反推末端系下修正抓取旋转, 供 P2/P3 继承
                rot_base_1_corrected = euler_to_rotation_matrix(A1, B1, C1, order="zyx")
                R_grasp_ee_corrected = R_ee_base.T @ rot_base_1_corrected

                # P2: 预抓取点 (继承 P1 修正姿态, 偏移在 P1 末端系下定义)
                pos_ee_pre = pos_ee_compensated + R_grasp_ee_corrected @ HANDLE_PRE_T_OFFSET
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = A1, B1, C1  # 继承 P1 修正后的姿态

                # P3: 目标点 (在 P1 修正姿态上叠加 target 旋转 + target 平移, 偏移在 P1 末端系下定义)
                R_target_ee = R_grasp_ee_corrected @ euler_to_rotation_matrix(*HANDLE_TARGET_R_OFFSET, order="zyx")
                pos_ee_target = pos_ee_compensated + R_grasp_ee_corrected @ HANDLE_TARGET_T_OFFSET
                rot_base_3 = R_ee_base @ R_target_ee
                pos_base_3 = transform_pose_to_base(pos_ee_target, robot_xyz, robot_ABC)
                A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)

            # === 合并一次发送 3 个位姿 (x,y,z,A,B,C)[x,y,z,A,B,C]{x,y,z,A,B,C} ===
            try:
                p1_str = f"({pos_base_1[0]:.3f},{pos_base_1[1]:.3f},{pos_base_1[2]:.3f},{A1:.2f},{B1:.2f},{C1:.2f})"
                p2_str = f"[{pos_base_2[0]:.3f},{pos_base_2[1]:.3f},{pos_base_2[2]:.3f},{A2:.2f},{B2:.2f},{C2:.2f}]"
                p3_str = f"{pos_base_3[0]:.3f},{pos_base_3[1]:.3f},{pos_base_3[2]:.3f},{A3:.2f},{B3:.2f},{C3:.2f}"
                msg = p1_str + p2_str + "{" + p3_str + "}"
                conn.sendall(msg.encode("utf-8"))
                print(f"[SEND] 合并位姿: {msg}", flush=True)
                target_sent = True
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[WARN] 发送位姿失败: {e}", flush=True)
                return
        else:
            time.sleep(0.1)

    if not target_sent:
        print("[WARN] 超时: 未获得稳定目标位姿, 断开连接", flush=True)
        return

    # ---- [Phase C] 指令循环: PREGRIP / GRIPDONE / RESET ----
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


def handle_client(conn, addr, detector, T_cam2ee):
    """TCP 客户端处理入口 (wrapper)"""
    try:
        _handle_client_impl(conn, addr, detector, T_cam2ee)
    finally:
        PoseStabilizer.on_disconnect()


# =============================================================
# 主函数
# =============================================================

def main():
    global _hand_available
    print("=" * 60)
    print("  ESTUN RealSense Handle 6D 抓取系统")
    print("  模型: 微调 YOLOv8n + SAM + Handle 扫描线")
    print("=" * 60, flush=True)

    # ---- [Init 1] 加载手眼标定矩阵 (cam -> ee) ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")
    else:
        print("[INFO] 标定矩阵已加载, 位姿将转换到末端坐标系")
        print("[INFO] 后续还将叠加 handle_gripper_offset 得到灵巧手 TCP 位姿")

    # ---- [Init 2] 初始化灵巧手 (使用 utils_shared 函数式接口) ----
    print(f"\n[INFO] 正在初始化灵巧手...")
    init_hand(hand_port=HAND_PORT, initial_gesture="RESET")
    _hand_available = True
    print("[INFO] 灵巧手已准备就绪")

    # ---- [Init 3] 初始化 Handle 6D 检测器 ----
    print(f"\n[INFO] 正在初始化 RealSense Handle 6D 检测器...")
    try:
        detector = RealSenseHandle6DDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        return

    # ---- [Init 4] 启动 TCP 服务器 (非阻塞) ----
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"\n[INFO] 服务器已启动: {HOST}:{PORT}")
    print("[INFO] 等待机器人连接...", flush=True)

    detector.last_known_pose = None
    main_stabilizer = PoseStabilizer(
        window=STABLE_WINDOW,
        pos_thresh=POS_THRESHOLD,
        angle_thresh=ANGLE_THRESHOLD
    )

    cv2.namedWindow("ESTUN Handle 6D Pose (RealSense)", cv2.WINDOW_NORMAL)

    # ---- 跳帧控制 ----
    frame_counter = 0
    cached_handle = None

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

            # ---- [Step 2] 每 SKIP_FRAMES 帧执行一次 Handle 6D 检测 ----
            if frame_counter % SKIP_FRAMES == 1:
                result = detector.detect_handle_6d(color_image, depth_frame)
                body_mask, handle_mask, stage, yolo_bbox, pos, rot, handle_euler, red_mask, red_center = result

                if pos is not None and rot is not None:
                    smooth_pos, smooth_rot = detector._pose_smoother.update(pos, rot)
                    if smooth_rot is not None:
                        pos = smooth_pos
                        rot = smooth_rot
                        handle_euler = rotation_matrix_to_euler_angles(rot)

                    # 相机系稳定化 (只有真实检测帧才 add)
                    assert pos is not None and handle_euler is not None
                    main_stabilizer.add(tuple(pos), tuple(handle_euler))
                    if main_stabilizer.is_stable():
                        if not main_stabilizer.is_generation_valid():
                            main_stabilizer.reset()
                            main_stabilizer.capture_generation()
                        else:
                            mean_pos_cam, mean_euler_cam = main_stabilizer.get_mean()
                            if mean_pos_cam is not None and mean_euler_cam is not None:
                                # 稳定后做一次 cam→ee
                                mean_rot_cam = euler_to_rotation_matrix(*mean_euler_cam, order="zyx")
                                pos_ee = T_cam2ee[:3, :3] @ mean_pos_cam + T_cam2ee[:3, 3]
                                rot_ee = T_cam2ee[:3, :3] @ mean_rot_cam
                                A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)
                                detector.last_known_pose = (pos_ee, (A_ee, B_ee, C_ee))
                else:
                    detector._pose_smoother.reset()

                cached_handle = (body_mask, handle_mask, stage, yolo_bbox, pos, rot, handle_euler)
                is_detection_frame = True
            else:
                if cached_handle is not None:
                    body_mask, handle_mask, stage, yolo_bbox, pos, rot, handle_euler = cached_handle
                else:
                    body_mask = handle_mask = None
                    stage = "none"
                    yolo_bbox = None
                    pos = rot = handle_euler = None
                is_detection_frame = False

            # ---- 可视化 ----
            if cached_handle is not None:
                body_mask, handle_mask, _, _, _, _, _ = cached_handle
            detector.visualize(display_img, body_mask, handle_mask,
                               stage, yolo_bbox, pos, rot, handle_euler)

            # ---- [Step 3] 可视化 ee 系位姿 (稳定化已在检测帧内完成) ----
            if pos is not None and rot is not None and cached_handle is not None:
                pos_ee = T_cam2ee[:3, :3] @ np.array(pos) + T_cam2ee[:3, 3]
                rot_ee = T_cam2ee[:3, :3] @ rot
                A_ee, B_ee, C_ee = rotation_matrix_to_euler_angles(rot_ee)

                cv2.putText(display_img, f"ee Pos: ({pos_ee[0]:.2f}, {pos_ee[1]:.2f}, {pos_ee[2]:.2f})",
                            (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                cv2.putText(display_img, f"ee Euler: (A={A_ee:.1f}, B={B_ee:.1f}, C={C_ee:.1f})",
                            (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                if is_detection_frame:
                    print(f"\r[Handle 6D] cam=({pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}) "
                          f"ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                          f"euler=({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})\n", flush=True)

            # ---- [Step 4] 显示 ----
            if depth_colormap is not None:
                h_vis = display_img.shape[0]
                depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
                vis_panel = np.hstack((display_img, depth_small))
            else:
                vis_panel = display_img
            cv2.imshow("ESTUN Handle 6D Pose (RealSense)", vis_panel)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                stop_event.set()
                running = False
                break

            # ---- [Step 5] 非阻塞 accept -> 子线程 handle_client ----
            try:
                conn, addr = server.accept()
            except BlockingIOError:
                continue

            main_stabilizer.capture_generation()
            detector.last_known_pose = None

            t = threading.Thread(
                target=handle_client,
                args=(conn, addr, detector, T_cam2ee),
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
        # 关闭灵巧手
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        print("[INFO] 程序已退出", flush=True)


if __name__ == "__main__":
    main()