# -*- coding: utf-8 -*-
"""
单相机 Handle 抓取状态机 (Camera A, 旧 D435i)
==============================================
从 Orchestrator/state_machine.py 移植核心逻辑 (检测器 / 位姿计算 / 固定 ABC),
仅保留 estun_handle_run.py 的运动流程 (Phase A-C, 仅水壶).

检测器:     HandleCupDetector (只用 Handle 部分，忽略 Cup)
位姿计算:   全部固定 ABC (Orchestrator/config.yaml)
运动流程:   Phase A(接收位姿) → Phase B(计算+发送Handle P1/P2/P3) → Phase C(PREGRIP/GRIPDONE/RESET)
           P1(抓取) → P2(预抓取) → P3(Z轴抬起)

仅用 Camera A, 仅水壶 (无 Cup).

使用方法:
  python Orchestrator/estun_handle_sm.py
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
#  工具函数
# ============================================================

_gf = 0


class PoseStabilizer:
    """缓存最近 N 帧; 位置+角度波动低于阈值时输出均值."""

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


def fmt_pose(xyz, abc):
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def pack_three(p1, p2, p3):
    return f"<{fmt_pose(p1[0], p1[1])}>[{fmt_pose(p2[0], p2[1])}]{{{fmt_pose(p3[0], p3[1])}}}"


def load_cam2ee(yaml_path):
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
#  HandleSM — 单相机 Handle 抓取状态机
# ============================================================

class HandleSM:
    """单相机 Handle 抓取状态机 (Camera A, 旧 D435i).

    移植自 Orchestrator/state_machine.py + estun_handle_sm.py (handle+cup),
    精简为仅水壶: 无 Cup / Phase D / P3 倒水.
    """

    def __init__(self):
        self._detector = None
        self._server = None
        self._hand_available = False
        self._T_cam2ee = None

        self._frame_counter = 0
        self._last_color = None
        self._last_depth_frame = None

        self._handle_stabilizer = None
        self._skip_frames = 5

        # 偏移
        self._handle_t_offset = None
        self._handle_r_offset = None
        self._handle_pre_t_offset = None
        self._handle_pre_r_offset = None
        self._handle_fixed_abc = None
        self._handle_lift_z = 0.05

        self._host = "0.0.0.0"
        self._port = 8080
        self._hand_port = "COM5"
        self._phase_b_timeout = 60.0

        self._stop_event = threading.Event()

    # ================================================================
    #  生命周期
    # ================================================================

    def start(self):
        self._load_configs()
        self._init_hand()
        self._init_detector()
        self._init_stabilizer_and_calib()
        self._init_tcp()
        print("[HandleSM] 状态机已启动, 等待机器人连接...")

    def run(self):
        cv2.namedWindow("HandleSM | Camera A", cv2.WINDOW_NORMAL)
        try:
            while not self._stop_event.is_set():
                color, depth_img, depth_cmap, depth_frame = self._get_frames()
                if color is None or depth_frame is None:
                    time.sleep(0.03)
                    if cv2.waitKey(30) & 0xFF == ord("q"):
                        self._stop_event.set()
                        break
                    continue

                self._frame_counter += 1
                self._detect()
                self._visualize(color.copy(), depth_cmap)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self._stop_event.set()
                    break

                self._accept_tcp()
        except KeyboardInterrupt:
            print("\n[INFO] 键盘中断")
        finally:
            self.stop()

    def stop(self):
        self._stop_event.set()
        cv2.destroyAllWindows()
        if self._server:
            try: self._server.close()
            except Exception: pass
            self._server = None
        if self._detector:
            try: self._detector.close()
            except Exception: pass
        if self._hand_available:
            try: disable_hand()
            except Exception: pass
        print("[HandleSM] 已退出")

    # ================================================================
    #  配置
    # ================================================================

    def _load_configs(self):
        orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self._orch_cfg = {}
        try:
            with open(orch_path, "r", encoding="utf-8") as f:
                self._orch_cfg = yaml.safe_load(f)
        except Exception:
            print("[WARN] config.yaml 读取失败")

        estun_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
        with open(estun_path, "r", encoding="utf-8") as f:
            estun = yaml.safe_load(f)["estun"]

        oc = self._orch_cfg
        hg = estun["handle_grasp"]

        self._handle_t_offset = np.array(hg["gripper_offset"]["t_handle_offset"], dtype=float)
        self._handle_r_offset = np.array(hg["gripper_offset"]["r_handle_offset"], dtype=float)
        self._handle_pre_t_offset = np.array(hg["pre_offset"]["t_offset"], dtype=float)
        self._handle_pre_r_offset = np.array(hg["pre_offset"]["r_offset"], dtype=float)

        _h_abc = oc.get("handle_euler", [-90, 89, -90])
        self._handle_fixed_abc = tuple(_h_abc)
        print(f"[INFO] Handle P1 ABC (基系): {self._handle_fixed_abc}")

        self._handle_lift_z = float(oc.get("handle_lift_z", 0.05))
        print(f"[INFO] Handle P3 lift Z: {self._handle_lift_z:.3f}m")

        self._skip_frames = oc.get("skip_frames", 5)
        self._host = estun["tcp_server"]["host"]
        self._port = estun["tcp_server"]["port"]
        self._phase_b_timeout = float(estun["tcp_server"].get("phase_b_timeout", 60.0))
        self._hand_port = estun["hand"]["port"]

    # ================================================================
    #  灵巧手 / 检测器 / 稳定器 / TCP
    # ================================================================

    def _init_hand(self):
        init_hand(hand_port=self._hand_port, initial_gesture="RESET")
        self._hand_available = is_hand_available()
        print(f"[INFO] 灵巧手: {'就绪' if self._hand_available else '模拟'}")

    def _init_detector(self):
        serial = load_camera_serial(
            os.path.join(_project_root, "RealSense_R/D435i_info.yaml"))
        import torch
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._detector = HandleCupDetector(
            orch_config=self._orch_cfg, serial=serial, device=dev)
        print("[INFO] HandleCupDetector 已启动")

    def _init_stabilizer_and_calib(self):
        stab = self._orch_cfg.get("stabilization", {})
        self._handle_stabilizer = PoseStabilizer(
            window=stab.get("window", 3),
            pos_thresh=stab.get("pos_threshold", 0.01),
            angle_thresh=stab.get("angle_threshold", 5.0),
        )
        self._handle_stabilizer.capture_generation()

        estun_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
        with open(estun_path, "r", encoding="utf-8") as f:
            estun = yaml.safe_load(f)["estun"]
        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml"))
        self._T_cam2ee = load_cam2ee(calib_path)
        print("[INFO] 稳定器+标定已就绪")

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
        try:
            conn, addr = self._server.accept()
            print(f"[INFO] 机器人已连接: {addr}")
            self._handle_stabilizer.capture_generation()
            threading.Thread(target=self._handle_session, args=(conn, addr), daemon=True).start()
        except BlockingIOError:
            pass

    # ================================================================
    #  帧获取 + 检测 (移植自 state_machine._detect)
    # ================================================================

    def _get_frames(self):
        det = self._detector
        if det is None:
            return None, None, None, None
        color, depth, depth_cmap, depth_frame = det.get_frames()
        self._last_color = color
        self._last_depth_frame = depth_frame
        return color, depth, depth_cmap, depth_frame

    def _detect(self):
        if self._last_color is None or self._last_depth_frame is None:
            return
        det = self._detector
        T = self._T_cam2ee
        if det is None or T is None or self._handle_stabilizer is None:
            return

        self._frame_counter += 1
        if self._frame_counter % self._skip_frames != 1:
            return

        result = det.detect(self._last_color, self._last_depth_frame)
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
                    mean_pos, _ = self._handle_stabilizer.get_mean()
                    if mean_pos is not None:
                        pos_ee = T[:3, :3] @ mean_pos + T[:3, 3]
                        det.last_known_pose = (pos_ee, self._handle_fixed_abc)
                        print(f"\n[Handle STABLE] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f})")

    # ================================================================
    #  可视化
    # ================================================================

    def _visualize(self, display_img, depth_cmap):
        if self._detector is not None:
            display_img = self._detector.visualize(display_img)
        if depth_cmap is not None:
            h_vis = display_img.shape[0]
            depth_small = cv2.resize(depth_cmap, (h_vis, h_vis))
            vis_panel = np.hstack((display_img, depth_small))
        else:
            vis_panel = display_img
        cv2.imshow("HandleSM | Camera A", vis_panel)

    # ================================================================
    #  TCP Session
    # ================================================================

    def _handle_session(self, conn, addr):
        print(f"[INFO] 客户端已连接: {addr}")
        det = self._detector
        T = self._T_cam2ee

        _, t_eff = compute_gripper_offset_transform(self._handle_t_offset, self._handle_r_offset)

        try:
            # ---- Phase A ----
            robot_xyz, robot_ABC = self._phase_a(conn)
            if robot_xyz is None or robot_ABC is None:
                return

            R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
            print(f"[INFO] Phase A: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                  f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})")

            # ---- Phase B (固定 ABC, P3 = Z-up) ----
            if det is None or T is None:
                print("[ERROR] 检测器或标定未就绪")
                return

            print("[INFO] Phase B: 等待稳定位姿...")
            target_sent = False
            wait_start = time.time()

            while not target_sent and not self._stop_event.is_set() \
                    and (time.time() - wait_start) < self._phase_b_timeout:
                pose = det.last_known_pose
                if pose is None:
                    time.sleep(0.1)
                    continue

                assert self._handle_fixed_abc is not None  # set in _load_configs
                mean_pos_ee, _mean_euler_ee = pose
                h_pos_ee_compensated = np.array(mean_pos_ee) - t_eff
                R_base_fix = euler_to_rotation_matrix(*self._handle_fixed_abc, order="zyx")
                R_grasp_ee = R_ee_base.T @ R_base_fix

                # P1 (抓取点)
                pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                A1, B1, C1 = self._handle_fixed_abc

                # P2 (预抓取)
                pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ self._handle_pre_t_offset
                pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                A2, B2, C2 = self._handle_fixed_abc

                # P3 (Z轴抬起)
                pos_base_3 = (
                    pos_base_1[0],
                    pos_base_1[1],
                    pos_base_1[2] + self._handle_lift_z,
                )
                A3, B3, C3 = self._handle_fixed_abc

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
                    print(f"[WARN] 发送失败: {e}")
                    return

            if not target_sent:
                print(f"[WARN] Phase B: 超时 {self._phase_b_timeout:.0f}s")
                return

            # ---- Phase C (PREGRIP / GRIPDONE / RESET) ----
            self._phase_c(conn)

        except Exception as e:
            print(f"[ERROR] Session 异常: {e}")
            import traceback
            traceback.print_exc()
        finally:
            PoseStabilizer.on_disconnect()
            if det is not None:
                det.last_known_pose = None
            try:
                conn.close()
            except Exception:
                pass
            print(f"[INFO] 客户端已断开: {addr}")

    # ================================================================
    #  Phase A / Phase C
    # ================================================================

    def _phase_a(self, conn):
        try:
            conn.settimeout(5.0)
            data = conn.recv(1024)
            if not data:
                return None, None
            pose_str = data.decode("utf-8").strip().strip("[]")
            parts = pose_str.split(",")
            if len(parts) != 6:
                return None, None
            return (float(parts[0]), float(parts[1]), float(parts[2])), \
                   (float(parts[3]), float(parts[4]), float(parts[5]))
        except socket.timeout:
            print("[WARN] Phase A: 超时")
            return None, None

    def _phase_c(self, conn):
        conn.settimeout(2.0)
        print("[INFO] Phase C: 等待指令...")
        try:
            while not self._stop_event.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue
                if not data:
                    print("[INFO] SocketClose")
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
            print("[INFO] 连接重置")
        except Exception as e:
            print(f"[ERROR] Phase C: {e}")


# ============================================================
#  入口
# ============================================================

def main():
    print("=" * 60)
    print("  HandleSM — 单相机 Handle 抓取状态机 (仅水壶)")
    print("  P1(抓取) → P2(预抓取) → P3(Z-up)")
    print("=" * 60)
    sm = HandleSM()
    sm.start()
    sm.run()


if __name__ == "__main__":
    main()
