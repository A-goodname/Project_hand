# -*- coding: utf-8 -*-
"""
单相机 Handle 数据采集状态机 (Camera A, 旧 D435i) — VLA 训练数据采集版
============================================================
运动流程与 estun_handle_cup_sm.py 的 Handle 阶段完全一致:
  Phase A(接收位姿) → Phase B(计算+发送Handle P1/P2/P3) → Phase C(PREGRIP/GRIPDONE/RESET)

与 estun_handle_sm.py 的区别:
  - 检测器仍含 Cup 检测 (HandleCupDetector), 用杯位计算 Handle P3 倒水点
    (P3 = Cup P1 基系 + p3_teach_offset 示教偏移)
  - P3 是倒水点, 不是 Z-up 抬起

与 estun_handle_cup_sm.py 的区别:
  - 无 Cup 阶段: 不响应 CUP/RELEASE/GRIP, 不抓杯
  - CRP 端负责: P2 → P1(抓) → DCp1(回退点, 避免碰撞) → P3(倒水) → DCp1 → P1 → RESET → P2 → Home
  - Python 依旧只发 3 个位姿 (pack_three), DCp1 由 CRP 内 CposOffset 动态计算

用途: 采集"抓壶→倒水"动作数据, 用于训练 VLA 模型.

使用方法:
  python Orchestrator/estun_handle_data_sm.py
"""

import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import yaml

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles, load_camera_serial
from Orchestrator.handle_cup_detector import HandleCupDetector
from Estun.utils_shared import (
    euler_to_rotation_matrix,
    transform_pose_to_base,
    compute_gripper_offset_transform,
    post_correct_angles,
    init_hand, is_hand_available, disable_hand,
    hand_pre_grip, hand_grip_done, hand_reset,
)


# ============================================================
#  位姿稳定器 (与 state_machine.py 相同)
# ============================================================

_gf = 0  # 全局代际


class PoseStabilizer:
    """缓存最近 N 帧; 位置+角度波动低于阈值时输出均值.
    代际计数器: 客户端断开重连时自增, 防止旧帧污染新连接."""

    def __init__(self, window=5, pos_thresh=0.005, angle_thresh=3.0):
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self._buffer = []
        self._sf = -1

    def capture_generation(self):
        global _gf
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
        return pos_range < self.pos_thresh and angle_range < self.angle_thresh

    def get_mean(self):
        if not self._buffer:
            return None, None
        poses = np.array([p for p, _ in self._buffer])
        angles = np.array([a for _, a in self._buffer])
        return poses.mean(axis=0), angles.mean(axis=0)

    def reset(self):
        self._buffer = []


# ============================================================
#  工具函数 (与 state_machine.py 相同)
# ============================================================

def fmt_pose(xyz, abc):
    """格式化单个位姿: "x,y,z,A,B,C" (位置 m, 角度 deg)"""
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def pack_three(p1, p2, p3):
    """打包三个位姿: <P1>[P2]{P3} — 与 ESTUN TranStrToCpos 括号标记约定匹配"""
    return f"<{fmt_pose(p1[0], p1[1])}>[{fmt_pose(p2[0], p2[1])}]{{{fmt_pose(p3[0], p3[1])}}}"


def load_cam2ee(yaml_path):
    """从标定结果加载 相机→末端法兰 变换矩阵 (4x4 齐次矩阵)"""
    I = np.eye(4)
    if not os.path.exists(yaml_path):
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}")
        return I
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    calib = data.get("calibration", {})
    hand_eye = calib.get("handeye", {})
    if not hand_eye:
        return I
    use_method = hand_eye.get("use_method", "Tsai")
    method_data = hand_eye.get(use_method, {})
    if not method_data:
        return I
    T_raw = method_data.get("T_cam_to_gripper", None)
    if T_raw is None or len(T_raw) != 4 or len(T_raw[0]) != 4:
        return I
    T = np.array(T_raw, dtype=float)
    if T.shape != (4, 4):
        return I
    t = T[:3, 3]
    print(f"[INFO] 已加载标定矩阵: {use_method}.T_cam_to_gripper, 平移: {t} m")
    return T


# ============================================================
#  HandleDataSM — 单相机 Handle 数据采集状态机
# ============================================================

class HandleDataSM:
    """单相机 Handle 数据采集状态机 (Camera A, 旧 D435i).

    核心移植自 Orchestrator/state_machine.py 的 HandleGraspAndPourHandler:
      - 检测器: HandleCupDetector (YOLO+HSV, 含 Cup 检测 → P3 倒水点)
      - 位姿计算: 全部固定 ABC (Orchestrator/config.yaml)
      - 稳定器: PoseStabilizer + 代际计数器

    运动流程移植自 estun_handle_cup_sm.py 的 Handle 阶段:
      - Phase A: 接收机器人初始位姿
      - Phase B: 等待稳定 → 计算 Handle P1/P2/P3 (P3 = 倒水点) → 发送 Handle
      - Phase C: 指令循环 (PREGRIP/GRIPDONE/RESET, 无 CUP)

    CRP 运动序列 (estun_handle_data_sm.crp):
      Home → P2(预抓取) → P1(抓壶) → DCp1(回退避碰) → P3(倒水) → DCp1 → P1 → P2 → Home
    DCp1 由 CRP 内 CposOffset 动态计算 (H_P1 + Δ(H_P3−H_P1)), Python 只发 3 个位姿.
    """

    def __init__(self):
        # ---- 运行时对象 ----
        self._detector = None          # HandleCupDetector
        self._server = None            # TCP socket
        self._hand_available = False
        self._T_cam2ee = None          # 4x4 手眼标定

        # ---- 跳帧 + 帧缓存 ----
        self._frame_counter = 0
        self._last_color = None
        self._last_depth_frame = None
        self._last_depth_image = None
        self._last_depth_colormap = None

        # ---- 位姿稳定器 ----
        self._handle_stabilizer = None
        self._cup_stabilizer = None

        # ---- 偏移参数 (_load_configs 填充) ----
        self._skip_frames = 5
        # Handle
        self._handle_t_offset = None
        self._handle_r_offset = None
        self._handle_t_tcp_fixed = None
        self._handle_pre_t_offset = None
        self._handle_pre_r_offset = None
        self._handle_target_t_offset = None
        self._handle_target_r_offset = None
        self._handle_fixed_abc = None
        # Cup (仅用于计算 Handle P3 倒水点)
        self._cup_handle_side = "back"
        self._cup_t_offset = None
        self._cup_r_offset = None
        self._cup_pre_t_offset = None
        self._cup_pre_r_offset = None
        self._cup_target_t_offset = None
        self._cup_target_r_offset = None
        self._cup_fixed_abc = None
        # P3 teach / pour
        self._p3_teach_enable = True
        self._p3_teach_t_offset = None
        self._p3_teach_r_abc = None
        self._spout_t_offset = None
        self._pour_t_offset_base = None
        self._pour_r_offset = None
        # Cup geometry
        self._cup_rim_radius = 0.035
        self._cup_height = 0.100

        # ---- TCP / Hand ----
        self._host = "0.0.0.0"
        self._port = 8080
        self._hand_port = "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00"
        self._phase_b_timeout = 60.0

        # ---- 退出信号 ----
        self._stop_event = threading.Event()

    # ================================================================
    #  生命周期
    # ================================================================

    def start(self):
        """初始化全部资源."""
        self._load_configs()
        self._init_hand()
        self._init_detector()
        self._init_stabilizer_and_calib()
        self._init_tcp()
        print("[HandleDataSM] 状态机已启动, 等待机器人连接...")

    def run(self):
        """主循环: 帧获取 → 检测 → 可视化 → accept TCP."""
        cv2.namedWindow("HandleDataSM | Camera A", cv2.WINDOW_NORMAL)

        try:
            while not self._stop_event.is_set():
                # [Step 1] 获取帧
                color, depth_img, depth_cmap, depth_frame = self._get_frames()
                if color is None or depth_frame is None:
                    time.sleep(0.03)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        print("[INFO] 用户按 q 退出")
                        self._stop_event.set()
                        break
                    continue

                # [Step 2] 跳帧检测 + 稳定化
                self._frame_counter += 1
                self._detect()

                # [Step 3] 可视化
                self._visualize(color.copy(), depth_cmap)

                # [Step 4] 退出检测
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出")
                    self._stop_event.set()
                    break

                # [Step 5] 非阻塞 accept
                self._accept_tcp()

        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断 (Ctrl+C)")
        except Exception as e:
            print(f"[ERROR] 主循环异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()

    def stop(self):
        """清理所有资源."""
        self._stop_event.set()
        cv2.destroyAllWindows()

        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

        if self._detector is not None:
            try:
                self._detector.close()
                print("[INFO] Camera A pipeline 已停止")
            except Exception as e:
                print(f"[WARN] 停止 pipeline 失败: {e}")

        if self._hand_available:
            try:
                disable_hand()
                print("[INFO] 灵巧手已关闭")
            except Exception as e:
                print(f"[WARN] 灵巧手关闭异常: {e}")

        print("[HandleDataSM] 已退出")

    # ================================================================
    #  配置加载 (移植自 state_machine._load_offsets + _start)
    # ================================================================

    def _load_configs(self):
        """加载 3 个配置文件 + 固定 ABC + 偏移参数."""
        # ---- Orchestrator 独立配置 (检测参数 / 固定 ABC / 稳定器) ----
        orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self._orch_cfg = {}
        try:
            with open(orch_path, "r", encoding="utf-8") as f:
                self._orch_cfg = yaml.safe_load(f)
        except Exception:
            print("[WARN] Orchestrator/config.yaml 读取失败, 使用默认值")

        # ---- ESTUN 配置 (偏移 / TCP / 灵巧手) ----
        estun_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
        if not os.path.exists(estun_path):
            print(f"[FATAL] ESTUN 配置文件未找到: {estun_path}")
            sys.exit(1)
        with open(estun_path, "r", encoding="utf-8") as f:
            estun = yaml.safe_load(f)["estun"]

        oc = self._orch_cfg

        # ================================================================
        #  Handle 偏移
        # ================================================================
        hg = estun["handle_grasp"]
        self._handle_t_offset = np.array(hg["gripper_offset"]["t_handle_offset"], dtype=float)
        self._handle_r_offset = np.array(hg["gripper_offset"]["r_handle_offset"], dtype=float)
        self._handle_t_tcp_fixed = np.array(hg["gripper_offset"]["t_tcp_fixed"], dtype=float)
        self._handle_pre_t_offset = np.array(hg["pre_offset"]["t_offset"], dtype=float)
        self._handle_pre_r_offset = np.array(hg["pre_offset"]["r_offset"], dtype=float)
        self._handle_target_t_offset = np.array(hg["target_offset"]["t_offset"], dtype=float)
        self._handle_target_r_offset = np.array(hg["target_offset"]["r_offset"], dtype=float)

        # ================================================================
        #  Cup 偏移 (仅用于计算 P3 倒水点)
        # ================================================================
        self._cup_handle_side = oc.get("cup_handle_side", "back")
        print(f"[INFO] cup handle_side: {self._cup_handle_side}")
        cg = estun["cup_grasp"]
        if self._cup_handle_side == "left":
            self._cup_t_offset = np.array(cg["gripper_offset"]["t_cup_offset_left"], dtype=float)
        else:
            self._cup_t_offset = np.array(cg["gripper_offset"]["t_cup_offset"], dtype=float)
        self._cup_r_offset = np.array(cg["gripper_offset"]["r_cup_offset"], dtype=float)
        self._cup_pre_t_offset = np.array(cg["pre_offset"]["t_offset"], dtype=float)
        self._cup_pre_r_offset = np.array(cg["pre_offset"]["r_offset"], dtype=float)
        self._cup_target_t_offset = np.array(cg["target_offset"]["t_offset"], dtype=float)
        self._cup_target_r_offset = np.array(cg["target_offset"]["r_offset"], dtype=float)

        # ================================================================
        #  P3 示教偏移 (Handle 倒水)
        # ================================================================
        p3 = hg["p3_teach_offset"]
        self._p3_teach_enable = p3.get("enable", True)
        pour_angle = p3.get("pour_angle", 30)
        p3_side = "back" if self._cup_handle_side == "back" else "left"
        side_raw = p3[p3_side]
        side_cfg = {str(k): v for k, v in side_raw.items()}
        angle_key = str(pour_angle)
        if angle_key not in side_cfg:
            angle_key = list(side_cfg.keys())[0]
        self._p3_teach_t_offset = np.array(side_cfg[angle_key]["t_offset"], dtype=float)
        self._p3_teach_r_abc = np.array(side_cfg[angle_key]["r_abc"], dtype=float)
        if self._p3_teach_enable:
            print(f"[INFO] P3 示教: side={p3_side} angle={angle_key}° "
                  f"t=({self._p3_teach_t_offset[0]:.4f},{self._p3_teach_t_offset[1]:.4f},{self._p3_teach_t_offset[2]:.4f}) "
                  f"abc=({self._p3_teach_r_abc[0]:.1f},{self._p3_teach_r_abc[1]:.1f},{self._p3_teach_r_abc[2]:.1f})")

        # ================================================================
        #  Spout + Pour 偏移 + Cup 几何
        # ================================================================
        self._spout_t_offset = np.array(cg["spout_offset"]["t_offset"], dtype=float)
        self._pour_t_offset_base = np.array(cg["pour_offset"]["t_offset"], dtype=float)
        self._pour_r_offset = np.array(cg["pour_offset"]["r_offset"], dtype=float)
        self._cup_rim_radius = float(oc.get("cup_rim_radius", 0.035))
        self._cup_height = float(oc.get("cup_height", 0.100))

        # ================================================================
        #  固定 ABC (从 Orchestrator config, 与 state_machine 完全一致)
        # ================================================================
        _h_abc = oc.get("handle_euler", [-90, 89, -90])
        self._handle_fixed_abc = tuple(_h_abc)
        _cup_key = "cup_euler_left" if self._cup_handle_side == "left" else "cup_euler_back"
        _c_abc = oc.get(_cup_key, [-90, 89, -110])
        self._cup_fixed_abc = tuple(_c_abc)
        print(f"[INFO] Handle P1 ABC (基系): {self._handle_fixed_abc}")
        print(f"[INFO] Cup   P1 ABC (基系): {self._cup_fixed_abc}")

        # ================================================================
        #  跳帧 / TCP / Hand
        # ================================================================
        self._skip_frames = oc.get("skip_frames", 5)
        self._host = estun["tcp_server"]["host"]
        self._port = estun["tcp_server"]["port"]
        self._phase_b_timeout = float(estun["tcp_server"].get("phase_b_timeout", 60.0))
        self._hand_port = estun["hand"]["port"]

    # ================================================================
    #  灵巧手
    # ================================================================

    def _init_hand(self):
        print(f"[INFO] 初始化灵巧手 (port={self._hand_port})...")
        init_hand(hand_port=self._hand_port, initial_gesture="RESET")
        self._hand_available = is_hand_available()
        if self._hand_available:
            print("[INFO] 灵巧手已就绪")
        else:
            print("[WARN] 灵巧手未就绪, 将模拟所有手势")

    # ================================================================
    #  检测器 (HandleCupDetector — 轻量版, 含 Cup 检测)
    # ================================================================

    def _init_detector(self):
        """创建 HandleCupDetector (YOLO kettle + 白色把手HSV + 红色胶带 + YOLO cup)."""
        cam_a_serial = load_camera_serial(
            os.path.join(_project_root, "RealSense_R/D435i_info.yaml")
        )
        print(f"[INFO] Camera A serial: {cam_a_serial}")

        import torch
        _dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._detector = HandleCupDetector(
            orch_config=self._orch_cfg,
            serial=cam_a_serial,
            device=_dev,
        )
        print("[INFO] HandleCupDetector pipeline 已启动")

    # ================================================================
    #  稳定器 + 标定 (移植自 state_machine._start)
    # ================================================================

    def _init_stabilizer_and_calib(self):
        stab_cfg = self._orch_cfg.get("stabilization", {})
        self._handle_stabilizer = PoseStabilizer(
            window=stab_cfg.get("window", 3),
            pos_thresh=stab_cfg.get("pos_threshold", 0.01),
            angle_thresh=stab_cfg.get("angle_threshold", 5.0),
        )
        self._cup_stabilizer = PoseStabilizer(
            window=stab_cfg.get("window", 3),
            pos_thresh=stab_cfg.get("pos_threshold", 0.01),
            angle_thresh=stab_cfg.get("angle_threshold", 5.0),
        )
        self._handle_stabilizer.capture_generation()
        self._cup_stabilizer.capture_generation()

        estun_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
        with open(estun_path, "r", encoding="utf-8") as f:
            estun = yaml.safe_load(f)["estun"]
        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml")
        )
        self._T_cam2ee = load_cam2ee(calib_path)
        print("[INFO] 稳定器+标定已就绪")

    # ================================================================
    #  TCP
    # ================================================================

    def _init_tcp(self):
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        self._server.bind((self._host, self._port))
        self._server.listen(5)
        print(f"[INFO] TCP 服务器: {self._host}:{self._port}")

    def _accept_tcp(self):
        if self._server is None:
            return
        assert self._handle_stabilizer is not None  # set in _init_stabilizer_and_calib
        assert self._cup_stabilizer is not None
        try:
            conn, addr = self._server.accept()
            print(f"[INFO] 机器人已连接: {addr}")
            self._handle_stabilizer.capture_generation()
            self._cup_stabilizer.capture_generation()
            t = threading.Thread(
                target=self._handle_session,
                args=(conn, addr),
                daemon=True,
            )
            t.start()
            print("[INFO] 等待下一次连接...")
        except BlockingIOError:
            pass

    # ================================================================
    #  Camera A 帧获取 (移植自 state_machine._get_frames)
    # ================================================================

    def _get_frames(self):
        det = self._detector
        if det is None:
            return None, None, None, None
        color, depth, depth_colormap, depth_frame = det.get_frames()
        self._last_color = color
        self._last_depth_image = depth
        self._last_depth_frame = depth_frame
        self._last_depth_colormap = depth_colormap
        return color, depth, depth_colormap, depth_frame

    # ================================================================
    #  检测 + 稳定化 (移植自 state_machine._detect)
    # ================================================================

    def _detect(self):
        """跳帧检测 Handle+Cup → PoseStabilizer → 更新 last_known_pose.
        与 state_machine.HandleGraspAndPourHandler._detect 完全一致."""
        if self._last_color is None or self._last_depth_frame is None:
            return

        det = self._detector
        T_cam2ee = self._T_cam2ee
        if det is None or T_cam2ee is None:
            return
        if self._handle_stabilizer is None or self._cup_stabilizer is None:
            return

        self._frame_counter += 1
        if self._frame_counter % self._skip_frames != 1:
            return

        result = det.detect(self._last_color, self._last_depth_frame)

        # === Handle 稳定 ===
        h = result["handle"]
        if h["pos_cam"] is not None:
            pos_cam = np.array(h["pos_cam"])
            euler = np.array(h["euler_cam"])
            self._handle_stabilizer.add(tuple(pos_cam), tuple(euler))
            if self._handle_stabilizer.is_stable():
                if not self._handle_stabilizer.is_generation_valid():
                    self._handle_stabilizer.reset()
                    self._handle_stabilizer.capture_generation()
                else:
                    mean_pos, _mean_euler = self._handle_stabilizer.get_mean()
                    if mean_pos is not None:
                        pos_ee = T_cam2ee[:3, :3] @ mean_pos + T_cam2ee[:3, 3]
                        det.last_known_pose = (pos_ee, self._handle_fixed_abc)
                        print(f"\n[Handle STABLE] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f})")

        # === Cup 稳定 ===
        c = result["cup"]
        if c["pos_cam"] is not None:
            pos_cam = np.array(c["pos_cam"])
            euler = np.array(c["euler_cam"])
            self._cup_stabilizer.add(tuple(pos_cam), tuple(euler))
            if self._cup_stabilizer.is_stable():
                if not self._cup_stabilizer.is_generation_valid():
                    self._cup_stabilizer.reset()
                    self._cup_stabilizer.capture_generation()
                else:
                    mean_cpos, _mean_ceuler = self._cup_stabilizer.get_mean()
                    if mean_cpos is not None:
                        det.last_known_cup_pose = (mean_cpos, self._cup_fixed_abc)
                        print(f"\n[Cup STABLE]    pos=({mean_cpos[0]:.3f},{mean_cpos[1]:.3f},{mean_cpos[2]:.3f})")

    # ================================================================
    #  可视化
    # ================================================================

    def _visualize(self, display_img, depth_colormap):
        if self._detector is not None:
            display_img = self._detector.visualize(display_img)

        if depth_colormap is not None:
            h_vis = display_img.shape[0]
            depth_small = cv2.resize(depth_colormap, (h_vis, h_vis))
            vis_panel = np.hstack((display_img, depth_small))
        else:
            vis_panel = display_img
        cv2.imshow("HandleDataSM | Camera A", vis_panel)

    # ================================================================
    #  TCP Session (daemon 线程入口)
    # ================================================================

    def _handle_session(self, conn, addr):
        """单 TCP Session: Phase A → Phase B → Phase C.
        位姿计算移植自 estun_handle_cup_sm.py (P3 = 倒水点),
        无 Cup 抓取阶段."""
        print(f"[INFO] 客户端已连接: {addr}")

        # ---- 预计算工具偏移 ----
        _, t_handle_effective = compute_gripper_offset_transform(
            self._handle_t_offset, self._handle_r_offset)
        _, t_cup_effective = compute_gripper_offset_transform(
            self._cup_t_offset, self._cup_r_offset)

        try:
            # ================================================================
            #  Phase A: 接收机器人初始末端位姿
            # ================================================================
            robot_xyz, robot_ABC = self._phase_a(conn)
            if robot_xyz is None:
                print("[INFO] Phase A 失败, 断开连接")
                return

            R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
            t_ee_base = np.array(robot_xyz, dtype=float)
            print(f"[INFO] Phase A 完成: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                  f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})")

            # ================================================================
            #  Phase B: 位姿计算 + 发送 Handle pack_three
            #  P3 = 倒水点 (基于 Cup P1 基系 + p3_teach_offset)
            # ================================================================
            det = self._detector
            T_cam2ee = self._T_cam2ee
            if det is None or T_cam2ee is None:
                print("[ERROR] Phase B: 检测器或标定未就绪")
                return

            # ---- type guards: config fields set by _load_configs ----
            assert self._handle_fixed_abc is not None
            assert self._cup_fixed_abc is not None
            assert self._handle_pre_t_offset is not None
            assert self._cup_pre_t_offset is not None
            assert self._handle_target_r_offset is not None
            assert self._handle_t_tcp_fixed is not None
            assert self._spout_t_offset is not None
            assert self._pour_t_offset_base is not None
            assert self._p3_teach_t_offset is not None
            assert self._p3_teach_r_abc is not None

            print("[INFO] Phase B: 等待稳定目标位姿 (Handle + Cup)...")
            target_sent = False
            wait_start = time.time()

            while not target_sent and not self._stop_event.is_set() \
                    and (time.time() - wait_start) < self._phase_b_timeout:
                handle_pose = det.last_known_pose
                cup_pose = det.last_known_cup_pose

                if handle_pose is None or cup_pose is None:
                    time.sleep(0.1)
                    continue

                mean_pos_ee, mean_euler_ee = handle_pose
                cup_pos_cam, cup_euler_cam = cup_pose

                # ============================================================
                #  Handle P1 / P2 (固定 ABC, 移植自 state_machine)
                # ============================================================
                h_pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective

                # 固定 ABC 直接构造旋转矩阵 (基系)
                R_base_fix = euler_to_rotation_matrix(*self._handle_fixed_abc, order="zyx")
                R_grasp_ee = R_ee_base.T @ R_base_fix

                # Handle P1
                pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = self._handle_fixed_abc

                # Handle P2
                pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ self._handle_pre_t_offset
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = self._handle_fixed_abc

                R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(
                    *self._handle_target_r_offset, order="zyx")

                # ============================================================
                #  Cup P1 基系 (用于计算 Handle P3 倒水点)
                # ============================================================
                c_pos_ee = T_cam2ee[:3, :3] @ np.array(cup_pos_cam) + T_cam2ee[:3, 3]
                c_R_base_fix = euler_to_rotation_matrix(*cup_euler_cam, order="zyx")
                c_R_grasp_ee = R_ee_base.T @ c_R_base_fix
                c_pos_ee_compensated = c_pos_ee - t_cup_effective

                # Cup P1 (基系 XYZ + 固定 ABC)
                c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)

                # ============================================================
                #  Handle P3 (倒水口, 基于 Cup P1 基系 + 示教偏移)
                #  移植自 state_machine / estun_handle_cup_sm
                # ============================================================
                if self._p3_teach_enable:
                    pos_base_3 = tuple(np.array(c_pos_base_1) + self._p3_teach_t_offset)
                    A3, B3, C3 = tuple(self._p3_teach_r_abc)
                    print(f"[POUR] 示教偏移: "
                          f"t=({self._p3_teach_t_offset[0]:.4f},{self._p3_teach_t_offset[1]:.4f},{self._p3_teach_t_offset[2]:.4f}) "
                          f"abc=({self._p3_teach_r_abc[0]:.1f},{self._p3_teach_r_abc[1]:.1f},{self._p3_teach_r_abc[2]:.1f})")
                else:
                    _tcp_offset_base = R_ee_base @ self._handle_t_tcp_fixed
                    _tcp_base = np.array(c_pos_base_1) + _tcp_offset_base
                    if self._cup_handle_side == "back":
                        pour_target_base = _tcp_base + np.array(
                            [self._cup_rim_radius, 0.0, self._cup_height / 2.0])
                    else:
                        pour_target_base = _tcp_base + np.array(
                            [0.0, self._cup_rim_radius, self._cup_height / 2.0])
                    pour_target_ee_calc = R_ee_base.T @ (pour_target_base - t_ee_base)
                    pour_target = T_cam2ee[:3, :3].T @ (pour_target_ee_calc - T_cam2ee[:3, 3])

                    pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
                    pos_ee_p3 = pos_pour_ee - R_grasp_ee @ self._spout_t_offset - t_handle_effective
                    pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) \
                        + self._pour_t_offset_base
                    rot_base_3 = R_ee_base @ R_target_ee
                    A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                    A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)

                # ---- 发送 Handle pack_three ----
                try:
                    msg = pack_three(
                        ((pos_base_1, (A1, B1, C1))),
                        ((pos_base_2, (A2, B2, C2))),
                        ((pos_base_3, (A3, B3, C3)))
                    )
                    conn.sendall(msg.encode("utf-8"))
                    print(f"[SEND] Phase B Handle: {msg}")
                    target_sent = True
                except (BrokenPipeError, ConnectionResetError) as e:
                    print(f"[WARN] Phase B: 发送失败: {e}")
                    return

            if not target_sent:
                print(f"[WARN] Phase B: 超时 {self._phase_b_timeout:.0f}s 未获得稳定位姿")
                return

            print("[INFO] Phase B 完成, 进入 Phase C")

            # ================================================================
            #  Phase C: 指令循环 (无 Cup 阶段)
            #  运动流程: PREGRIP / GRIPDONE / RESET
            #  CRP 负责: DCp1 回退点 + 倒水 + 回 Home
            # ================================================================
            self._phase_c(conn)

        except Exception as e:
            print(f"[ERROR] Session 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            PoseStabilizer.on_disconnect()
            if det is not None:
                det.last_known_pose = None
                det.last_known_cup_pose = None
            try:
                conn.close()
            except Exception:
                pass
            print(f"[INFO] 客户端已断开: {addr}")

    # ================================================================
    #  Phase A: 接收机器人初始末端位姿
    # ================================================================

    def _phase_a(self, conn):
        """Phase A: 5s 超时接收 [X,Y,Z,A,B,C] (m, deg)."""
        try:
            conn.settimeout(5.0)
            data = conn.recv(1024)
            if not data:
                print("[WARN] Phase A: 未收到数据")
                return None, None
            pose_str = data.decode("utf-8").strip().strip("[]")
            parts = pose_str.split(",")
            if len(parts) != 6:
                print(f"[WARN] Phase A: 格式错误: {pose_str}")
                return None, None
            robot_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
            robot_ABC = (float(parts[3]), float(parts[4]), float(parts[5]))
            return robot_xyz, robot_ABC
        except socket.timeout:
            print("[WARN] Phase A: 超时 (5s)")
            return None, None
        except Exception as e:
            print(f"[WARN] Phase A: 异常: {e}")
            return None, None

    # ================================================================
    #  Phase C: 指令循环 (PREGRIP / GRIPDONE / RESET)
    # ================================================================

    def _phase_c(self, conn):
        """Phase C: PREGRIP / GRIPDONE / RESET, SocketClose 退出.
        无 CUP/RELEASE/GRIP — 数据采集版不抓杯."""
        conn.settimeout(2.0)
        print("[INFO] Phase C: 等待机器人指令...")

        try:
            while not self._stop_event.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue

                if not data:
                    print("[INFO] Phase C: 连接已关闭 (SocketClose)")
                    break

                cmd = data.decode("utf-8").strip()
                print(f"[RECV] {cmd}")

                if cmd == "PREGRIP":
                    if self._hand_available:
                        hand_pre_grip()
                    print("[INFO] PREGRIP")

                elif cmd == "GRIPDONE":
                    if self._hand_available:
                        hand_grip_done()
                    print("[INFO] GRIPDONE")

                elif cmd == "RESET":
                    if self._hand_available:
                        hand_reset()
                    print("[INFO] RESET")

                else:
                    print(f"[INFO] 忽略: {cmd}")

        except ConnectionResetError:
            print("[INFO] Phase C: 连接重置")
        except Exception as e:
            print(f"[ERROR] Phase C: 异常: {e}")


# ============================================================
#  入口
# ============================================================

def main():
    print("=" * 60)
    print("  HandleDataSM — 单相机 Handle 数据采集状态机 (VLA)")
    print("  Camera A (RealSense_R 旧 D435i)")
    print("  检测器: HandleCupDetector (YOLO+HSV, 含 Cup → P3 倒水点)")
    print("  位姿: 全部固定 ABC — 移植自 state_machine")
    print("  流程: Phase A(接收位姿) → Phase B(发送Handle P1/P2/P3) → Phase C(指令循环)")
    print("  CRP:  P2 → P1(抓) → DCp1(避碰) → P3(倒水) → DCp1 → P1 → P2 → Home")
    print("=" * 60)

    sm = HandleDataSM()
    sm.start()
    sm.run()


if __name__ == "__main__":
    main()
