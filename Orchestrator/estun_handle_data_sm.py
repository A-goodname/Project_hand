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
from Orchestrator.d2_recorder import EpisodeRecorder, parse_pose_prefix  # D2 (2026-08-17)
from Orchestrator.d2_streams import (  # D2 raw/ 流记录 (2026-08-17)
    SceneRecorder, WristStreamRecorder,
)
from Orchestrator.tcp_framing import FramedLineReader                 # D2 (2026-08-17)
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
    """缓存最近 N 帧 (pos, euler[, theta]); 位置+角度(+θ) 波动低于阈值时输出均值.
    代际计数器: 客户端断开重连时自增, 防止旧帧污染新连接.

    2026-08-19 合并: 旋转 θ 并入稳定窗口 — 窗口内每帧同时具备 pos+euler+θ,
    稳定判定同时约束三者, 直接取同一窗口的 θ 中位数 (替代独立 θ deque,
    保 B3 "恰满窗+同代+一致才出候选" 语义). θ=None (cup 等无旋转用途)
    时退化为原 pos/angle 行为."""

    def __init__(self, window=5, pos_thresh=0.005, angle_thresh=3.0,
                 theta_thresh=5.0):
        self.window = window
        self.pos_thresh = pos_thresh
        self.angle_thresh = angle_thresh
        self.theta_thresh = theta_thresh
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

    def add(self, pos, euler, theta=None):
        self._buffer.append((np.array(pos, dtype=float),
                             np.array(euler, dtype=float),
                             (float(theta) if theta is not None else None)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def is_stable(self, theta_thresh=None):
        if len(self._buffer) < self.window:
            return False
        poses = np.array([p for p, _, _ in self._buffer])
        angles = np.array([a for _, a, _ in self._buffer])
        pos_range = np.ptp(poses, axis=0).max()
        angle_range = np.ptp(angles, axis=0).max()
        if all(t is not None for t in (x[2] for x in self._buffer)):
            # θ 模式 (旋转链): 窗口内全部带 θ, 纳入一致约束
            theta_range = np.ptp(np.array([x[2] for x in self._buffer]))
            thr = self.theta_thresh if theta_thresh is None else theta_thresh
            return (pos_range < self.pos_thresh
                    and angle_range < self.angle_thresh
                    and theta_range < thr)
        # 无 θ (cup 等): 仅位置+角度
        return pos_range < self.pos_thresh and angle_range < self.angle_thresh

    def get_mean(self):
        if not self._buffer:
            return None, None, None
        poses = np.array([p for p, _, _ in self._buffer])
        angles = np.array([a for _, a, _ in self._buffer])
        thetas = [x[2] for x in self._buffer]
        th_med = (float(np.median(thetas))
                  if all(t is not None for t in thetas) else None)
        return poses.mean(axis=0), angles.mean(axis=0), th_med

    def reset(self):
        self._buffer = []


# ============================================================
#  工具函数 (与 state_machine.py 相同)
# ============================================================

ROT_THETA_WIN = 5        # 旋转 θ 稳定窗口帧数 (2026-08-19: 并入 PoseStabilizer, 恰满窗才出候选)
ROT_THETA_SPREAD_MAX = 5.0   # 窗口内 θ 分歧上限 (deg, B3 2026-08-19: 极差超限 → 不稳定/REJECT)
ROT_THETA_FRESH_S = 1.5      # θ 观测新鲜窗口 (s, B3 2026-08-19: 超期 → REJECT)

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
        # handle_teach (P1 示教偏置, 基系直接加)
        self._handle_teach_enabled = False
        self._handle_teach_t_offset = None
        # 旋转档位 (2026-08-19): θ 并入 PoseStabilizer 稳定窗口 → band → p1_offsets 查表
        self._rot_band = None       # 当前稳定 band (B3: None = 未取得有效候选)
        self._rot_band_th = None    # 稳定时窗口内 θ 中位数 (deg)
        self._rot_valid = False     # B3: 有效候选 (恰 5 帧新鲜/同代/稳定/一致)
        self._rot_reject = None     # B3: 最近 REJECT 原因 (str|None)
        self._rot_obs_t = 0.0       # B3: 最近一次 θ 观测时刻 (monotonic, 新鲜性)
        self._p1_offsets = {}       # band → {p1_offset, p1_abc, ...} (estun_config handle_teach.p1_offsets)
        self._p1_z_check = None     # P1 Z 有效范围 (m, [lo,hi]), 超出警告 (2026-08-19)
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

        # ---- D2 episode 记录 (2026-08-17: 位姿事件 P0-P4/Home 前缀接收) ----
        self._ep_out_dir = os.path.join(_project_root, "logs", "d2")
        self._scene_batch_id = "canonical-a"
        self._episode_id = None          # None → 会话时间自动生成
        # D2 raw/ 流记录器 (会话期间由主循环/会话线程挂载)
        self._wrist_recorder = None      # 腕部 D435i (主循环挂钩)
        self._scene_recorder = None      # 场景 UGREEN (独立线程)
        self._session_thread = None      # 当前 TCP 会话线程 (退出时 join 收尾)

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
        self._install_signal_handlers()
        print("[HandleDataSM] 状态机已启动, 等待机器人连接...")

    # ---- 退出信号接管 (2026-08-17) ----
    # 默认 SIGINT 会抛 KeyboardInterrupt: 若二次 Ctrl+C 落在 finally 清理期间,
    # 异常未捕获直接杀进程 → daemon 会话线程被斩, recorder.close() 丢失
    # (session_end + 双 manifest). 接管信号后二次触发幂等置位, 主循环自然
    # 退出并走 finally 的 _join_session 收尾. SIGHUP 覆盖终端关闭场景.

    def _install_signal_handlers(self):
        import signal  # noqa: PLC0415

        signal.signal(signal.SIGINT, self._on_exit_signal)
        signal.signal(signal.SIGTERM, self._on_exit_signal)
        signal.signal(signal.SIGHUP, self._on_exit_signal)

    def _on_exit_signal(self, signum, _frame):
        if not self._stop_event.is_set():
            print(f"\n[INFO] 信号 {signum}: 置位退出, 等待会话收尾...")
            self._stop_event.set()
        # 幂等: 主线程从 handler 返回后主循环 while 检查退出 → finally 收尾

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
                # D2: 会话期间记录腕部 D435i 彩色/深度流 (raw/wrist_*)
                if self._wrist_recorder is not None:
                    self._wrist_recorder.record_frame(
                        color, depth_img, self._frame_counter)
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
            # 顺序: 先置位 (快) → join 会话收尾 (写 manifest) → 再慢清理.
            # join 前置, 避免 stop() 清 D435i/串口耗时期间二次 Ctrl+C 杀进程.
            self._stop_event.set()
            self._join_session()
            self.stop()

    def _join_session(self, timeout=8.0):
        """等待当前 TCP 会话线程收尾, 保证 recorder.close() 完成再退出进程."""
        t, self._session_thread = self._session_thread, None
        if t is not None:
            t.join(timeout=timeout)
            print(f"[INFO] 会话线程收尾: alive={t.is_alive()} "
                  f"(false = recorder.close() 已执行)")

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
        #  handle_teach (P1 示教偏置, 参考 logs/teach_handle_p1_*.log)
        #  语义: P1_base = detect_base + t_offset (基系直接加, 含 TCP 偏移)
        #  注意: 与 t_handle_effective ee 系补偿链不能混用 — enabled 二选一
        # ================================================================
        ht = estun.get("handle_teach", {})
        self._handle_teach_enabled = bool(ht.get("enabled", False))
        self._handle_teach_t_offset = (
            np.array(ht.get("p1_offset", [0.0, 0.0, 0.0]), dtype=float)
            if ht else np.zeros(3))
        # 按档位 t_offset 表 (2026-08-19): band → {p1_offset, p1_abc, ...}
        self._p1_offsets = (ht.get("p1_offsets", {})
                            if isinstance(ht.get("p1_offsets", {}), dict) else {})
        # P1 Z 有效范围检查 (2026-08-19 实机标定 [0.3599,0.3629])
        _zchk = ht.get("p1_z_check") if ht else None
        self._p1_z_check = (list(map(float, _zchk)) if isinstance(_zchk, (list, tuple)) and len(_zchk) == 2
                            else None)
        if self._handle_teach_enabled:
            print(f"[INFO] handle_teach 启用: P1 = detect_base + "
                  f"({self._handle_teach_t_offset[0]:.4f},"
                  f"{self._handle_teach_t_offset[1]:.4f},"
                  f"{self._handle_teach_t_offset[2]:.4f}) m (基系直接加) "
                  f"| p1_offsets 档位: {sorted(self._p1_offsets.keys())}")

        # ================================================================
        #  固定 ABC (从 Orchestrator config, 与 state_machine 完全一致)
        #  handle_teach 启用且含 p1_abc 时, 用示教 ABC 覆盖 (参考示教 log)
        # ================================================================
        _h_abc = oc.get("handle_euler", [-90, 89, -90])
        if self._handle_teach_enabled:
            _teach_abc = ht.get("p1_abc")
            if _teach_abc:
                _h_abc = _teach_abc
                print(f"[INFO] handle_teach: 固定 ABC 覆盖为 {tuple(_h_abc)} (示教 log p1_abc)")
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
        # 2026-08-19: 旋转 θ 并入 handle 稳定窗口 (B3 恰 5 帧, 默认 window=ROT_THETA_WIN);
        #   θ 极差阈值 = ROT_THETA_SPREAD_MAX. cup 无旋转, 退化为 pos/angle 行为
        self._handle_stabilizer = PoseStabilizer(
            window=stab_cfg.get("window", ROT_THETA_WIN),
            pos_thresh=stab_cfg.get("pos_threshold", 0.01),
            angle_thresh=stab_cfg.get("angle_threshold", 5.0),
            theta_thresh=ROT_THETA_SPREAD_MAX,
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
            self._session_thread = t     # 供主循环退出时 join, 保证 recorder.close() 完成
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

    # ================================================================
    #  旋转档位解析 (2026-08-19)
    # ================================================================

    def _resolve_rot_band(self, th_deg):
        """θ → 档位 (5 档纯 θ 区间, 与 handle_cup_detector._classify_rotation 一致).

        [-8,8]=0, (8,22]=15, >22=30, 负侧对称; <-40 不可达.
        """
        if th_deg > 22.0:
            return 30
        if th_deg > 8.0:
            return 15
        if th_deg < -40.0:
            return None
        if th_deg < -22.0:
            return -30
        if th_deg < -8.0:
            return -15
        return 0

    def _resolve_p1_offsets(self, band):
        """band → (p1_offset, p1_abc). 缺档回退 0° → 平铺字段.

        (查表用; Phase B 在 teach 启用且有 θ 时改用 _interp_p1_offset 插值 X/Y/Z)
        """
        entry = self._p1_offsets.get(str(band)) if self._p1_offsets else None
        if entry is None:
            entry = self._p1_offsets.get("0") if self._p1_offsets else None
        if entry is None:
            return self._handle_teach_t_offset, self._handle_fixed_abc  # 回退平铺
        off = np.array(entry.get("p1_offset", self._handle_teach_t_offset), dtype=float)
        abc = tuple(entry.get("p1_abc", self._handle_fixed_abc))
        return off, abc

    def _interp_p1_offset(self, th_deg):
        """θ 档内线性插值 X/Y/Z (2026-08-19, 消除档边界跳变).

        相邻档中心 (p1_offsets 键) 线性混合; θ 超出 [min,max] 钳到端点档.
        返回 np.ndarray (3,) offset; 无表/无 θ 时回退平铺.
        """
        if not self._p1_offsets:
            return self._handle_teach_t_offset
        centers = sorted(int(k) for k in self._p1_offsets)
        if len(centers) == 0:
            return self._handle_teach_t_offset
        if th_deg <= centers[0]:
            return np.array(self._p1_offsets[str(centers[0])]["p1_offset"], dtype=float)
        if th_deg >= centers[-1]:
            return np.array(self._p1_offsets[str(centers[-1])]["p1_offset"], dtype=float)
        for i in range(len(centers) - 1):
            lo, hi = centers[i], centers[i + 1]
            if lo <= th_deg <= hi:
                Olo = np.array(self._p1_offsets[str(lo)]["p1_offset"], dtype=float)
                Ohi = np.array(self._p1_offsets[str(hi)]["p1_offset"], dtype=float)
                t = (th_deg - lo) / (hi - lo) if hi != lo else 0.0
                return Olo + t * (Ohi - Olo)
        return self._handle_teach_t_offset

    def _clamp_p1_z(self, pos_base_1):
        """P1 Z 有效范围钳位 (2026-08-19 实机标定 [360,362]mm; 超出钳到最近边界).

        Phase B 先钳位再推 P2, 保证 P1/P2 满足同一 pre_offset 几何 (修 B5).
        返回钳位后的位置 (np.ndarray 原对象就地修改并返回); 无配置时不改.
        """
        if self._p1_z_check is None:
            return pos_base_1
        _zlo, _zhi = self._p1_z_check
        if pos_base_1[2] < _zlo:
            print(f"[WARN] P1 Z={pos_base_1[2]*1000:.1f}mm < 下限"
                  f"{_zlo*1000:.1f}mm, 钳到下限", flush=True)
            pos_base_1[2] = _zlo
        elif pos_base_1[2] > _zhi:
            print(f"[WARN] P1 Z={pos_base_1[2]*1000:.1f}mm > 上限"
                  f"{_zhi*1000:.1f}mm, 钳到上限", flush=True)
            pos_base_1[2] = _zhi
        return pos_base_1

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

        if self._frame_counter % self._skip_frames != 1:
            return

        result = det.detect(self._last_color, self._last_depth_frame)

        # === Handle 稳定 ===
        h = result["handle"]
        if h["pos_cam"] is not None:
            pos_cam = np.array(h["pos_cam"])
            euler = np.array(h["euler_cam"])
            theta = h.get("theta_deg")
            # 2026-08-19 合并: 位姿+θ 同一稳定窗口 —
            #   仅收 pos/euler/θ 齐全帧 (θ非None ⟺ fitEllipse 成功 ⟺ euler动态),
            #   排除姿态退化/不可达无 θ 帧参与稳定统计.
            #   同一代累积; 代际失效作废候选 (B3)
            if theta is not None:
                self._handle_stabilizer.add(tuple(pos_cam), tuple(euler), float(theta))
                self._rot_obs_t = time.monotonic()
            if not self._handle_stabilizer.is_generation_valid():
                self._rot_valid = False
                self._rot_band = None
                self._rot_band_th = None
                self._rot_reject = "观测代际失效 (会话重连)"
            if self._handle_stabilizer.is_stable():
                if not self._handle_stabilizer.is_generation_valid():
                    self._handle_stabilizer.reset()
                    self._handle_stabilizer.capture_generation()
                else:
                    mean_pos, _mean_euler, th_med = self._handle_stabilizer.get_mean()
                    if mean_pos is not None and th_med is not None:
                        pos_ee = T_cam2ee[:3, :3] @ mean_pos + T_cam2ee[:3, 3]
                        det.last_known_pose = (pos_ee, self._handle_fixed_abc)
                        # 稳定 = 同一窗口内 pos/euler/θ 全部一致 (θ 极差<阈值),
                        # 直接取 θ 中位数 → band (B3: 恰满窗+同代+一致才出候选)
                        band = self._resolve_rot_band(th_med)
                        if band is None:   # θ < -40 不可达
                            self._rot_valid = False
                            self._rot_reject = f"θ={th_med:.1f}° < -40 不可达"
                            print(f"\n[ROT-REJECT] {self._rot_reject}", flush=True)
                        else:
                            self._rot_band_th = th_med
                            self._rot_band = band
                            self._rot_valid = True
                            self._rot_reject = None
                            print(f"\n[Handle STABLE] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f})"
                                  f" θ_med={th_med:.1f}° band={band}")

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
                    mean_cpos, _mean_ceuler, _ = self._cup_stabilizer.get_mean()
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

        # ---- D2 episode 记录器 (2026-08-17) ----
        episode_id = self._episode_id or (
            f"tea-{self._scene_batch_id}-{time.strftime('%Y%m%dT%H%M%S')}")
        recorder = EpisodeRecorder(self._ep_out_dir, episode_id,
                                   self._scene_batch_id)
        recorder.record_receipt("socket_connected", remote=repr(addr))

        # ---- D2 raw/ 流记录 (场景 UGREEN 线程 + 腕部 D435i 挂钩) ----
        self._scene_recorder = SceneRecorder(recorder.ep_dir)
        self._scene_recorder.start()
        self._wrist_recorder = WristStreamRecorder(recorder.ep_dir)
        self._wrist_recorder.start()     # 缺此调用 record_frame 每帧直接 return (修复 2026-08-17)

        # ---- 预计算工具偏移 ----
        _, t_handle_effective = compute_gripper_offset_transform(
            self._handle_t_offset, self._handle_r_offset)
        _, t_cup_effective = compute_gripper_offset_transform(
            self._cup_t_offset, self._cup_r_offset)

        try:
            # ================================================================
            #  Phase A: 接收机器人初始末端位姿 (P0_ACTUAL 记录, 2026-08-17)
            # ================================================================
            robot_xyz, robot_ABC = self._phase_a(conn)
            if robot_xyz is None:
                recorder.note_invalid("Phase A 失败 (初始位姿未收到)")
                print("[INFO] Phase A 失败, 断开连接")
                return
            recorder.record_pose(
                "P0_ACTUAL", robot_xyz, robot_ABC,
                raw_payload=f"[{robot_xyz[0]},{robot_xyz[1]},{robot_xyz[2]},"
                            f"{robot_ABC[0]},{robot_ABC[1]},{robot_ABC[2]}]")

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
                #  Handle P1 / P2 (θ 插值 offset + C 解析式, 2026-08-19)
                #  位置双链 (二选一, 不能混用):
                #    handle_teach 启用 → P1 = detect_base + interp_offset(θ_med)
                #      (θ 5 帧中位数, X/Y/Z 相邻档线性插值, ABC 的 C 用 C=C0-θ 解析式)
                #    handle_teach 关闭 → 旧链 (ee 系减 t_eff + 固定 ABC) 对照
                # ============================================================
                # 解析 band → offset/abc (teach 启用时; 否则回退固定值)
                band_off = self._handle_teach_t_offset
                band_abc = self._handle_fixed_abc
                _band = 0
                if self._handle_teach_enabled:
                    # B3 (2026-08-19): 无有效 band → REJECT, 不得默认 0° offset;
                    # 等待有效观测或 Phase B 超时 → 不发计划 → 受控停止
                    if not self._rot_valid:
                        print(f"[ROT-REJECT] band 无效: {self._rot_reject or '观测缺失'}; "
                              f"不发送 P1/P2", flush=True)
                        time.sleep(0.1)
                        continue
                    if (time.monotonic() - self._rot_obs_t) > ROT_THETA_FRESH_S:
                        self._rot_valid = False
                        self._rot_reject = "θ 观测过期"
                        print(f"[ROT-REJECT] {self._rot_reject}; 不发送 P1/P2", flush=True)
                        time.sleep(0.1)
                        continue
                    _band = self._rot_band if self._rot_band is not None else 0
                    # X/Y/Z: 有 θ 用插值 (平滑), 否则查表
                    if self._rot_band_th is not None:
                        band_off = self._interp_p1_offset(self._rot_band_th)
                        _, band_abc = self._resolve_p1_offsets(_band)  # A/B
                    else:
                        band_off, band_abc = self._resolve_p1_offsets(_band)
                    # C 解析式: C = C0 - θ_med (平滑姿态补偿; C0 = 0° 档 p1_abc.C)
                    if self._rot_band_th is not None:
                        _c0 = -21.1
                        if self._p1_offsets:
                            _e0 = self._p1_offsets.get("0", {}) or {}
                            _c0 = _e0.get("p1_abc", [-115.2, 85.0, -21.1])[2]
                        band_abc = (band_abc[0], band_abc[1], float(_c0 - self._rot_band_th))
                # band ABC 直接构造旋转矩阵 (基系)
                R_base_fix = euler_to_rotation_matrix(*band_abc, order="zyx")
                R_grasp_ee = R_ee_base.T @ R_base_fix

                if self._handle_teach_enabled:
                    # P1 = detect_base + p1_offsets[band].p1_offset (基系直接加)
                    detect_base = transform_pose_to_base(
                        np.array(mean_pos_ee), robot_xyz, robot_ABC)
                    pos_base_1 = detect_base + band_off
                    # P1 Z 有效范围钳位 → 先钳位再推 P2, 保证 P1/P2 满足同一
                    # pre_offset 几何 (修 B5, 2026-08-19)
                    pos_base_1 = self._clamp_p1_z(pos_base_1)
                    # P2 = P1(钳位后) + R_base_fix @ pre_offset
                    # (推导: P2_base = P1_base + R_ee_base@R_grasp_ee@pre
                    #        = P1_base + R_base_fix@pre)
                    pos_base_2 = pos_base_1 + R_base_fix @ self._handle_pre_t_offset
                    _src = ("θ插值" if self._rot_band_th is not None else f"查表[{_band}]")
                    print(f"[TEACH] band={_band}({_src}) P1 = detect_base + offset = "
                          f"({pos_base_1[0]:.4f},{pos_base_1[1]:.4f},{pos_base_1[2]:.4f}) m"
                          f" ABC={tuple(round(x,1) for x in band_abc)}", flush=True)
                else:
                    # 旧链: ee 系减 t_handle_effective + 固定 ABC
                    h_pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
                    # Handle P1
                    pos_base_1 = transform_pose_to_base(
                        h_pos_ee_compensated, robot_xyz, robot_ABC)
                    # Handle P2
                    pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ self._handle_pre_t_offset
                    pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)

                A1, B1, C1 = band_abc
                A2, B2, C2 = band_abc

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
                    # P3 姿态: 完整固定 ABC (p3_teach_r_abc, 示教值)
                    # 与 P1/P2 姿态不要求一致 (用户约定 2026-08-17) — 逆解失败
                    # 根因是旧 Home 导致 P3 位置不可达, 重标后姿态直接用示教值
                    A3, B3, C3 = tuple(self._p3_teach_r_abc)
                    print(f"[POUR] 示教偏移: "
                          f"t=({self._p3_teach_t_offset[0]:.4f},{self._p3_teach_t_offset[1]:.4f},{self._p3_teach_t_offset[2]:.4f}) "
                          f"abc=({A3:.1f},{B3:.1f},{C3:.1f})")
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
            #  D2 (2026-08-17): 分号分帧接收位姿事件 (P0-P4/Home 前缀) → 记录
            # ================================================================
            self._phase_c(conn, recorder)

        except Exception as e:
            print(f"[ERROR] Session 异常: {e}")
            import traceback
            traceback.print_exc()
            recorder.note_invalid(f"Session 异常: {type(e).__name__}: {e}")
        finally:
            PoseStabilizer.on_disconnect()
            # D2: 停止 raw/ 流记录 (场景线程 + 腕部挂钩)
            if self._scene_recorder is not None:
                self._scene_recorder.stop()
                self._scene_recorder.join(timeout=3.0)
                self._scene_recorder = None
            if self._wrist_recorder is not None:
                self._wrist_recorder.close()
                self._wrist_recorder = None
            recorder.close()
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

    def _dispatch(self, recorder, occurrences, line):
        """分发一条消息 (分号帧或裸命令):
        位姿帧 P0-P4/Home 前缀 → D2 事件记录; PREGRIP/GRIPDONE/RESET → 动作."""
        if line in ("PREGRIP", "GRIPDONE", "RESET"):
            if line == "PREGRIP" and self._hand_available:
                hand_pre_grip()
            elif line == "GRIPDONE" and self._hand_available:
                hand_grip_done()
            elif line == "RESET" and self._hand_available:
                hand_reset()
            print(f"[INFO] {line}")
            recorder.record_receipt(line.lower(), command=line)
            recorder.record_hand(command=line, hand_available=self._hand_available)
            return
        try:
            event, xyz, abc = parse_pose_prefix(line, occurrences)
        except ValueError as e:
            reason = f"位姿帧解析失败: {e} | raw={line!r}"
            recorder.note_invalid(reason)
            recorder.record_transition("protocol_error", reason=reason)
            print(f"[WARN] {reason}")
            return
        recorder.record_pose(event, xyz, abc, raw_payload=line)
        print(f"[RECV] {event}: xyz={xyz} abc={abc}")

    def _phase_c(self, conn, recorder):
        """Phase C: ']' 分帧接收 — 位姿事件 (P0-P4/Home, 格式 P2[..] 等) +
        指令 (PREGRIP/GRIPDONE/RESET). 裸命令通过 pop_pending 回退兼容;
        SocketClose 退出."""
        reader = FramedLineReader(conn, delimiter=b"]")
        occurrences = {}
        print("[INFO] Phase C: 等待机器人指令/位姿事件...")

        try:
            while not self._stop_event.is_set():
                line = reader.read_line(timeout=2.0)
                if line is None:
                    if reader.eof:
                        print("[INFO] Phase C: 连接已关闭 (SocketClose)")
                        break
                    # 超时: 裸命令回退 (旧格式无结束符)
                    bare = reader.pop_pending()
                    if bare:
                        self._dispatch(recorder, occurrences, bare)
                    continue
                if not line:
                    continue
                self._dispatch(recorder, occurrences, line)

        except ConnectionResetError:
            print("[INFO] Phase C: 连接重置")
        except Exception as e:
            print(f"[ERROR] Phase C: 异常: {e}")
            recorder.note_invalid(f"Phase C 异常: {type(e).__name__}: {e}")


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

    import argparse  # noqa: PLC0415
    ap = argparse.ArgumentParser(description="HandleDataSM (D2 采集版)")
    ap.add_argument("--out", default=None, help="episode 输出根目录 (默认 logs/d2)")
    ap.add_argument("--scene-batch-id", default="canonical-a")
    ap.add_argument("--episode-id", default=None)
    args = ap.parse_args()

    sm = HandleDataSM()
    if args.out:
        sm._ep_out_dir = args.out
    sm._scene_batch_id = args.scene_batch_id
    sm._episode_id = args.episode_id
    sm.start()
    sm.run()


if __name__ == "__main__":
    main()
