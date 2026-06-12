# -*- coding: utf-8 -*-
"""
ESTUN RealSense 6D 抓取系统
===========================
整合 RealSense YOLO 6D 检测器,手眼标定,TCP 服务器与灵巧手,实现
实时水瓶检测,位姿转换与机器人抓取控制

使用说明:
  1. 连接 RealSense 相机
  2. 连接灵巧手 (USB 串口)
  3. 运行 python estun_run.py
  4. 机器人作为 TCP 客户端连接本服务器

输入:
  - RealSense_R/config.yaml         : 相机 / YOLO 模型配置
  - estun_config.yaml               : ESTUN 机器人 / 灵巧手 / 标定配置
  - calib_data/calibration_result.yaml : 手眼标定矩阵 (可选)

输出:
  - 稳定目标位姿 [x,y,z,A,B,C] 发送至机器人 (TCP)
  - 实时显示画面 (彩色+深度, 6D 坐标系)

流程:
  主线程:
    [Init] 加载标定 / 手 / 检测器 / TCP -> 主循环
    [Step1] 获取帧
    [Step2] 缓存原始深度帧
    [Step3] YOLO 检测 -> 6D 位姿 -> cam2ee 转换 -> 稳定化
    [Step4] 显示画面
    [Step5] 非阻塞 accept -> 子线程 handle_client

  子线程 handle_client (daemon=True):
    [Phase A] 接收机器人初始位姿 (5s 超时)
    [Phase B] 轮询 last_known_pose -> 夹爪偏移补偿 -> 基坐标转换 -> 发送目标
    [Phase C] CRP 指令循环: GRIP / RELEASE

架构:
  - 辅助函数: 纯函数, 无副作用
  - 代际计数器 / PoseStabilizer: 确保稳定位姿属于当前连接周期
  - 灵巧手控制: 封装 OmniHand 操作, 导入失败时模拟
  - 姿态后修正: 可达性 + 方向模式 (vertical / horizontal / free)
  - TCP 客户端处理: wrapper 保证 on_disconnect 兜底
  - 主函数: 按步骤串联完整流程, finally 清理资源
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


# ---- 将项目根目录加入 sys.path,以导入 RealSense_R / Omnihand 包 ----
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入自定义包 ----
from RealSense_R.test_6D import RealSenseYOLO6DPoseDetector
from utils_shared import (
    euler_to_rotation_matrix, transform_pose_to_base,
    compute_gripper_offset_transform, compute_grasp_rotation,
    post_correct_angles,
    init_hand, grip_object, release_object, disable_hand,
)

# ---- 全局退出事件 ----
stop_event = threading.Event()

# ---- 加载 RealSense / YOLO 配置 ----
_cfg_path = os.path.join(_project_root, "RealSense_R", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

MODEL_NAME = os.path.join(_project_root, _cfg["model"]["name"])
CONF_THRESHOLD = _cfg["model"]["conf_threshold"]

# ---- 加载 ESTUN 配置 (独立文件) ----
_estun_cfg_path = os.path.join(_project_root, "estun_config.yaml")
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
GRIPPER_T_OFFSET = np.array(_cfg_estun["gripper_offset"]["t_offset"], dtype=float)
GRIPPER_R_OFFSET = np.array(_cfg_estun["gripper_offset"]["r_offset"], dtype=float)
ORIENTATION_MODE = _cfg_estun["orientation_mode"]
GRASP_DIRECTION = _cfg_estun["grasp_direction"]

# ---- 防御性检查:确保 orientation_mode 合法 ----
VALID_MODES = {"vertical", "horizontal", "free"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE},将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现,将使用 free 模式")
    ORIENTATION_MODE = "free"


# 辅助函数从 utils_shared 导入


def load_cam2ee(yaml_path):
    """
    从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)

    注意: 此矩阵是 cam -> 末端法兰 (ee), 不是 cam -> 夹爪 TCP
    还需叠加 gripper_offset 变换才能得到最终夹爪 TCP 位姿

    读取 calibration_result.yaml 中 use_method 指定的方法对应的
    T_cam_to_gripper (实际含义是 cam -> ee)

    参数:
        yaml_path : str, 标定结果 YAML 路径

    返回:
        np.ndarray, 4x4 齐次矩阵; 文件缺失/格式错误时返回单位矩阵
    """
    I = np.eye(4)
    path = Path(yaml_path)
    if not path.exists():
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}")
        print("[WARN] 将直接发送相机坐标系位姿 (无夹爪变换)")
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
    print("[INFO] 注意: 还需叠加 gripper_offset 变换才得到最终夹爪 TCP 位姿")
    return T


# =============================================================
# 功能模块: 代际计数器与位姿稳定器
# =============================================================

_gf = 0  # 全局代际 (Global Flag), 每次客户端断开 +1

class PoseStabilizer:
    """
    缓存最近 N 帧位姿数据; 当位置和角度变化均低于阈值时,
    认为位姿稳定, 返回缓冲区均值作为稳定结果

    使用代际计数器 (_sf) 确保稳定后的数据只属于当前客户端连接周期,
    避免旧数据污染 last_known_pose
    """

    def __init__(self, window=5, pos_thresh=0.005, angle_thresh=3.0):
        """
        初始化稳定器

        参数:
            window       : int, 缓冲区大小 (帧数)
            pos_thresh   : float, 位置稳定阈值 (米)
            angle_thresh : float, 角度稳定阈值 (度)
        """
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self._buffer = []   # [(pos_np3, euler_np3), ...]
        self._sf = -1       # 稳定化代际 (Stable Flag), 初始无效

    def capture_generation(self):
        """客户端连接时调用: 快照当前全局代际, 后续稳定化只接受本轮数据"""
        self._sf = _gf

    @staticmethod
    def on_disconnect():
        """客户端断开时调用: 递增全局代际, 使旧 _sf 自动失效"""
        global _gf
        _gf += 1

    def is_generation_valid(self):
        """当前缓存的稳定数据是否属于当前连接周期"""
        return self._sf == _gf

    def add(self, pos, euler):
        """
        添加一帧位姿数据

        参数:
            pos   : (float, float, float), 位置 (x, y, z), 米
            euler : (float, float, float), 欧拉角 (A, B, C), 度
        """
        self._buffer.append((np.array(pos, dtype=float), np.array(euler, dtype=float)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self):
        """
        检查缓冲的位姿是否稳定

        返回:
            bool, True 表示缓冲区已满且波动低于阈值
        """
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
        """
        返回缓冲区内位姿均值

        返回:
            pos_mean   : np.ndarray or None, 位置均值
            euler_mean : np.ndarray or None, 欧拉角均值
        """
        if not self._buffer:
            return None, None
        poses = np.array([p for p, _ in self._buffer])
        angles = np.array([a for _, a in self._buffer])
        return poses.mean(axis=0), angles.mean(axis=0)

    def reset(self):
        """清空缓冲区"""
        self._buffer = []


# 灵巧手控制与post_correct_angles从utils_shared导入


# =============================================================
# 功能模块: TCP 客户端处理
# =============================================================

def _handle_client_impl(conn, addr, detector, T_cam2ee):
    """
    实际的客户端处理逻辑 (不负责 on_disconnect, 由外层 wrapper 兜底)

    参数:
        conn     : socket, 客户端连接
        addr     : (str, int), 客户端地址
        detector : RealSenseYOLO6DPoseDetector, 检测器实例
        T_cam2ee : np.ndarray, 4x4 相机 -> 末端法兰变换矩阵
    """
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

    # ---- [Phase B] 等待主循环产生稳定位姿 -> 基坐标转换 -> 补偿偏移 (30s 超时) ----
    # 预计算夹爪偏移变换 (常量, 在循环外执行一次)
    _, t_fl2gr = compute_gripper_offset_transform(GRIPPER_T_OFFSET, GRIPPER_R_OFFSET)
    print("[INFO] 等待稳定目标位姿...", flush=True)
    target_sent = False
    wait_start = time.time()
    while not target_sent and not stop_event.is_set() and (time.time() - wait_start) < 30.0:
        if detector.last_known_pose is not None:
            mean_pos_ee, mean_euler_ee = detector.last_known_pose

            # 位置补偿: t_grasp_ee = t_obj_ee - t_tool'_ee  (第二部分 Step 5)
            pos_ee_compensated = np.array(mean_pos_ee) - t_fl2gr

            # 方向对齐: 根据物体朝向和抓取模式计算 R_grasp_ee (第二部分 Step 5)
            R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
            R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
            R_grasp_ee = compute_grasp_rotation(R_obj_ee, ORIENTATION_MODE, GRIPPER_R_OFFSET, R_ee_base, grasp_direction=GRASP_DIRECTION)

            # 转换到基坐标系 (第二部分 Step 6)
            rot_base = R_ee_base @ R_grasp_ee
            pos_base = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
            A_base, B_base, C_base = RealSenseYOLO6DPoseDetector.rotation_matrix_to_euler_angles(
                rot_base
            )

            # 可达性修正: 将 A/C 加减 180 度, 使其偏离当前末端位姿最小
            A_base, B_base, C_base = post_correct_angles(
                A_base, B_base, C_base, robot_ABC
            )

            # [Phase B.1] 发送目标位姿 (仅一次)
            msg = (
                f"[{pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f},"
                f"{A_base:.2f},{B_base:.2f},{C_base:.2f}]"
            )
            try:
                conn.sendall(msg.encode("utf-8"))
                print(f"[SEND] 目标基座标: {msg.strip()}", flush=True)
                target_sent = True
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[WARN] 发送目标位姿失败: {e}", flush=True)
                return
        else:
            time.sleep(0.1)

    if not target_sent:
        print("[WARN] 超时: 未获得稳定目标位姿, 断开连接", flush=True)
        return

    # ---- [Phase C] CRP 指令循环: GRIP / RELEASE / 断连 ----
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

            # 忽略机器人发来的循环位姿更新
            if cmd.startswith("[") and cmd.endswith("]"):
                if cmd.count(",") == 5:
                    continue

            print(f"[RECV] {cmd}", flush=True)

            if "GRIP" in cmd:
                grip_object()
                conn.sendall(b"111")
                print("[INFO] 已回复 111 (抓取完成)", flush=True)

            elif "RELEASE" in cmd:
                release_object()
                conn.sendall(b"222")
                print("[INFO] 已回复 222 (释放完成)", flush=True)
                break

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
    """
    TCP 客户端处理入口 (wrapper)

    确保无论哪种退出路径 (Phase A 提早 return / Phase C 正常结束),
    都执行 on_disconnect() 递增代际

    参数:
        conn     : socket, 客户端连接
        addr     : (str, int), 客户端地址
        detector : RealSenseYOLO6DPoseDetector, 检测器实例
        T_cam2ee : np.ndarray, 4x4 相机 -> 末端法兰变换矩阵
    """
    try:
        _handle_client_impl(conn, addr, detector, T_cam2ee)
    finally:
        PoseStabilizer.on_disconnect()


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  ESTUN RealSense 6D 抓取系统")
    print("  模型: " + MODEL_NAME)
    print("=" * 60, flush=True)

    # ---- [Init 1] 加载手眼标定矩阵 (cam -> ee, 相机 -> 末端法兰) ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 使用单位矩阵 (直接发送相机坐标系位姿)")
    else:
        print("[INFO] 标定矩阵已加载, 位姿将转换到末端坐标系")
        print("[INFO] 后续还将叠加 gripper_offset 得到夹爪 TCP 位姿")

    # ---- [Init 2] 初始化灵巧手 (OmniHand / 模拟) ----
    init_hand()

    # ---- [Init 3] 初始化 RealSense YOLO 6D 检测器 ----
    print(f"\n[INFO] 正在初始化 RealSense YOLO 检测器...")
    try:
        detector = RealSenseYOLO6DPoseDetector(config_dict=_cfg)
    except Exception as e:
        print(f"[FATAL] RealSense 初始化失败: {e}")
        print("[FATAL] 请确认已连接 RealSense D400 系列相机")
        disable_hand()
        return

    # ---- [Init 4] 启动 TCP 服务器 (非阻塞, 主线程专注于显示/检测) ----
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

    cv2.namedWindow("ESTUN RealSense 6D Pose", cv2.WINDOW_NORMAL)

    running = True
    try:
        while running:
            # ---- [Step 1] 获取帧: 等待 RealSense 对齐后的彩色/深度帧 ----
            color_image, depth_image, depth_colormap, depth_frame = detector.get_frames()
            if color_image is None or depth_frame is None:
                time.sleep(0.03)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出", flush=True)
                    stop_event.set()
                    running = False
                    break
                continue

            # ---- [Step 2] 缓存原始深度帧 (用于 anchor 映射) ----
            try:
                frames = detector.pipeline.wait_for_frames()
                detector._cached_raw_depth = frames.get_depth_frame()
            except Exception:
                detector._cached_raw_depth = None

            # ---- [Step 3] YOLO 检测 -> 6D 位姿 -> 稳定化 -> 更新 last_known_pose ----
            display_img = color_image.copy()
            det = detector._detect_targets(color_image)

            if det is not None:
                position, rotation, point_cloud = detector.compute_6d_pose(det["bbox"], depth_frame)

                if position is not None and rotation is not None:
                    smooth_pos, smooth_rot = detector._pose_smoother.update(position, rotation)
                    if smooth_pos is not None:
                        assert smooth_rot is not None
                        # 转换到末端法兰坐标系 (cam -> ee)
                        pos_ee = T_cam2ee[:3, :3] @ np.array(smooth_pos) + T_cam2ee[:3, 3]
                        rot_ee = T_cam2ee[:3, :3] @ smooth_rot
                        A_ee, B_ee, C_ee = (
                            RealSenseYOLO6DPoseDetector.rotation_matrix_to_euler_angles(rot_ee)
                        )

                        main_stabilizer.add(pos_ee, (A_ee, B_ee, C_ee))
                        if main_stabilizer.is_stable():
                            # 代际检查: 只接受属于当前客户端连接周期的稳定数据
                            if not main_stabilizer.is_generation_valid():
                                # 代际不匹配 -> 数据混入了旧帧, 丢弃重新积累
                                main_stabilizer.reset()
                                main_stabilizer.capture_generation()
                                continue
                            mean_pos, mean_euler = main_stabilizer.get_mean()
                            if mean_pos is not None:
                                detector.last_known_pose = (mean_pos, mean_euler)

                        # 在彩色图上绘制检测框和位姿信息
                        x1, y1, x2, y2 = det["bbox"]
                        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        K = np.array([
                            [detector.depth_intrinsics.fx, 0, detector.depth_intrinsics.ppx],
                            [0, detector.depth_intrinsics.fy, detector.depth_intrinsics.ppy],
                            [0, 0, 1]
                        ], dtype=float)
                        detector.visualize_pose(display_img, smooth_pos, smooth_rot, scale=0.08)

                        text_pos = f"Pos: ({pos_ee[0]:.2f}, {pos_ee[1]:.2f}, {pos_ee[2]:.2f})"
                        cv2.putText(display_img, text_pos, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        text_euler = f"Euler: (A={A_ee:.1f}, B={B_ee:.1f}, C={C_ee:.1f})"
                        cv2.putText(display_img, text_euler, (x1, y1 - 28),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        print(f"\r[6D] cam=({smooth_pos[0]:.3f},{smooth_pos[1]:.3f},{smooth_pos[2]:.3f}) "
                              f"ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                              f"euler=({A_ee:.1f},{B_ee:.1f},{C_ee:.1f})\n", flush=True)
                else:
                    detector._pose_smoother.reset()
            else:
                detector._pose_smoother.reset()

            # ---- [Step 4] 显示: 拼接彩色/深度图, 绘制 6D 坐标系和位置信息 ----
            if depth_colormap is not None and display_img.shape[:2] == depth_colormap.shape[:2]:
                images = np.hstack((display_img, depth_colormap))
            else:
                images = display_img
            cv2.imshow("ESTUN RealSense 6D Pose", images)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                stop_event.set()
                running = False
                break

            # ---- [Step 5] 非阻塞 accept -> 子线程 handle_client, 主循环继续 ----
            try:
                conn, addr = server.accept()
            except BlockingIOError:
                continue

            # 快照本轮代际, 后续稳定化只接受本轮连接的新数据
            main_stabilizer.capture_generation()
            detector.last_known_pose = None  # 清除旧数据, 强制等待新稳定位姿

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
        disable_hand()
        print("[INFO] 程序已退出", flush=True)


if __name__ == "__main__":
    main()