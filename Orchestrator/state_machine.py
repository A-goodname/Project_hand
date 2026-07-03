# -*- coding: utf-8 -*-
"""
Orchestrator 状态机
===================
双相机全流程: IDLE → HANDLE_GRASP_AND_POUR → IDLE

状态说明:
  - IDLE:                   Camera B 等待检测到人 → 触发
  - HANDLE_GRASP_AND_POUR:  单 TCP Session 全流程 (抓壶倒水 → 抓杯递送 → 松手退避)

与 estun_handle_cup_run.py 的关键差异:
  - Cup P3 的 X 来自 Camera B 人体检测 + EyeToHand 标定 (非固定相机偏移)
  - 递送后增加 Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
  - 全程单 TCP 连接, SocketClose 后回 IDLE
"""

import os
import socket
import sys
import threading
import time
from abc import ABC, abstractmethod
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import yaml

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入 ----
from RealSense.realsense_shared import rotation_matrix_to_euler_angles
from RealSense.test_handle_6D import RealSenseHandle6DDetector
from RealSense.test_cup_6D import RealSenseCup6DDetector
from EyeToHand.person_detector import PersonDetector
from EyeToHand.detector import load_calibration, camera_x_to_base_x
from EyeToHand.hand_in_cup import HandInCupDetector
from Estun.utils_shared import (
    euler_to_rotation_matrix,
    transform_pose_to_base,
    compute_gripper_offset_transform,
    compute_grasp_rotation,
    post_correct_angles,
    init_hand, is_hand_available, disable_hand,
    hand_pre_grip, hand_grip_done, hand_reset,
    grip_object, release_object,
)


# ============================================================
# 通用工具
# ============================================================

def fmt_pose(xyz, abc):
    """格式化单个位姿: "x,y,z,A,B,C" (位置 m, 角度 deg)"""
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def pack_three(p1, p2, p3):
    """打包三个位姿: <P1>[P2]{P3}"""
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
# 位姿稳定器 (与 estun_handle_cup_run.py 一致)
# ============================================================

_gf = 0  # 全局代际


class PoseStabilizer:
    """位姿稳定器 — 缓存最近 N 帧, 位置+角度波动低于阈值时输出均值."""

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
# 状态定义
# ============================================================

class OrchestratorState(Enum):
    IDLE = auto()
    HANDLE_GRASP_AND_POUR = auto()


# ============================================================
# 共享上下文
# ============================================================

@dataclass
class SharedContext:
    """跨状态共享的数据与依赖.

    配置字段 (main.py 填充):
      - eye_to_hand_config: EyeToHand/config.yaml
      - estun_config:       Estun/estun_config.yaml["estun"]
      - realsense_config:   RealSense/config.yaml (Camera A, Handle+Cup)
      - eye_to_hand_serial: Camera B serial

    运行时字段 (状态/主循环 写入):
      - person_detector, stop_event
      - latest_person_3d, stop_person_detector
      - hand_in_cup_active, hand_on_cup
      - handle_cup_complete, frames_a, frames_b
    """
    # ---- 配置 (初始化时填入) ----
    eye_to_hand_config: dict
    estun_config: dict
    realsense_config: dict
    eye_to_hand_serial: str

    # ---- 运行时 ----
    person_detector: Optional[PersonDetector] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Person 检测 (main loop 写入, TCP handler 读取)
    latest_person_3d: Optional[tuple] = None       # (x, y, z) Camera B 系
    stop_person_detector: bool = False              # CUP 时置 True
    person_stable_x_cam: Optional[float] = None     # CUP 时缓存: 最新稳定 X_cam

    # Hand-in-cup 检测 (TCP handler 控制开关, main loop 更新结果)
    hand_in_cup_active: bool = False
    hand_on_cup: Optional[bool] = None              # True/False/None(未就绪)

    # HANDLE_GRASP_AND_POUR 完成标志
    handle_cup_complete: threading.Event = field(default_factory=threading.Event)

    # Camera A 检测器 (HandleGraspAndPourHandler 写入)
    handle_detector: Optional[RealSenseHandle6DDetector] = None
    cup_detector: Optional[RealSenseCup6DDetector] = None
    T_cam2ee: Optional[np.ndarray] = None

    # 可视化帧缓存 (main loop 写入)
    frames_a: Optional[tuple] = None  # Camera A (Handle)
    frames_b: Optional[tuple] = None  # Camera B (Person/HandInCup)


# ============================================================
# 状态处理器基类 & 状态机引擎
# ============================================================

class StateHandler(ABC):
    """状态处理器抽象基类."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def on_enter(self, ctx: SharedContext):
        """进入状态."""

    @abstractmethod
    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        """每 tick 调用. 返回下一状态或 None."""

    def on_exit(self, ctx: SharedContext):
        """离开状态."""

    def on_error(self, ctx: SharedContext, msg: str):
        print(f"[STATE_ERROR] {self.name}: {msg}")


class StateMachine:
    """线性状态机."""

    def __init__(self):
        self._state: OrchestratorState = OrchestratorState.IDLE
        self._handlers: dict[OrchestratorState, StateHandler] = {}

    def register(self, state: OrchestratorState, handler: StateHandler):
        self._handlers[state] = handler

    @property
    def state(self) -> OrchestratorState:
        return self._state

    def transition_to(self, ctx: SharedContext, target: OrchestratorState):
        old = self._state
        if old in self._handlers:
            try:
                self._handlers[old].on_exit(ctx)
            except Exception as e:
                print(f"[WARN] {old.name}.on_exit 异常: {e}")
        self._state = target
        print(f"[STATE] {old.name} → {target.name}")
        if target in self._handlers:
            try:
                self._handlers[target].on_enter(ctx)
            except Exception as e:
                print(f"[ERROR] {target.name}.on_enter 异常: {e}")
                self._handle_error(ctx, f"on_enter 失败: {e}")

    def update(self, ctx: SharedContext):
        h = self._handlers.get(self._state)
        if h is None:
            return
        try:
            result = h.on_update(ctx)
            if result is not None and result != self._state:
                self.transition_to(ctx, result)
        except Exception as e:
            print(f"[ERROR] {self._state.name}.on_update 异常: {e}")
            self._handle_error(ctx, str(e))

    def _handle_error(self, ctx: SharedContext, msg: str):
        h = self._handlers.get(self._state)
        if h is not None:
            try:
                h.on_error(ctx, msg)
            except Exception:
                pass
        self.transition_to(ctx, OrchestratorState.IDLE)


# ============================================================
# IDLE 处理器
# ============================================================

class IdleHandler(StateHandler):
    """IDLE: Camera B PersonDetector 等待检测到人 → 触发."""

    def __init__(self):
        super().__init__("IDLE")

    def on_enter(self, ctx: SharedContext):
        # 启动 PersonDetector (若未启动)
        if ctx.person_detector is None:
            serial = ctx.eye_to_hand_serial
            print(f"[INFO] 启动 PersonDetector (serial: {serial})...")
            ctx.person_detector = PersonDetector(
                config_dict=ctx.eye_to_hand_config,
                serial_number=serial,
            )
        # 重置所有运行时标志
        ctx.stop_person_detector = False
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        ctx.latest_person_3d = None
        ctx.person_stable_x_cam = None
        ctx.handle_cup_complete.clear()
        print("[IDLE] 等待检测到人体...")

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if ctx.person_detector is None:
            return None
        if ctx.frames_b is None:
            return None
        color, depth, _, depth_frame = ctx.frames_b
        if color is None or depth_frame is None:
            return None

        result = ctx.person_detector.detect_person_stable(color, depth_frame)
        if result is not None and result["pos_3d_cam"] is not None:
            pos = result["pos_3d_cam"]
            ctx.latest_person_3d = pos
            ctx.person_stable_x_cam = float(pos[0])
            print(f"[IDLE] 检测到人体 ✅ X_cam={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
            return OrchestratorState.HANDLE_GRASP_AND_POUR
        return None

    def on_exit(self, ctx: SharedContext):
        # PersonDetector pipeline 保持运行, 由 main loop 继续取帧
        pass


# ============================================================
# HANDLE_GRASP_AND_POUR 处理器
# ============================================================

class HandleGraspAndPourHandler(StateHandler):
    """HANDLE_GRASP_AND_POUR: 单 TCP Session 全流程.

    内部包含 Handle+Cup 检测 → TCP 通信 → 递送确认.
    全程使用一个 TCP 连接, SocketClose 后完成.
    """

    def __init__(self):
        super().__init__("HANDLE_GRASP_AND_POUR")

        # ---- 运行时对象 ----
        self._server: Optional[socket.socket] = None
        self._hand_available = False
        self._hand_in_cup: Optional[HandInCupDetector] = None

        # ---- 跳帧控制 ----
        self._frame_counter = 0
        self._cached_handle = None
        self._cached_cup = None
        self._last_color = None
        self._last_depth_frame = None
        self._last_depth_image = None
        self._last_depth_colormap = None

        # ---- 位姿稳定器 ----
        self._handle_stabilizer: Optional[PoseStabilizer] = None
        self._cup_stabilizer: Optional[PoseStabilizer] = None

        # ---- 偏移参数 (start 时从 config 加载) ----
        self._skip_frames = 90
        # Handle
        self._handle_t_offset = None
        self._handle_r_offset = None
        self._handle_t_tcp_fixed = None
        self._handle_pre_t_offset = None
        self._handle_pre_r_offset = None
        self._handle_target_t_offset = None
        self._handle_target_r_offset = None
        # Cup
        self._cup_handle_side = "back"
        self._cup_t_offset = None
        self._cup_r_offset = None
        self._cup_pre_t_offset = None
        self._cup_pre_r_offset = None
        self._cup_target_t_offset = None
        self._cup_target_r_offset = None
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
        # Orientation
        self._orientation_mode = "vertical"
        self._grasp_direction = "right"
        self._handle_side = "right"
        self._fixed_abc = None

        # ---- EyeToHand 标定 + 递送参数 (start 时加载) ----
        self._calib = None
        self._y_teach = 0.0
        self._z_teach = 0.0
        self._a_teach = 0.0
        self._b_teach = 0.0
        self._c_teach = 0.0

    # ================================================================
    # 生命周期
    # ================================================================

    def on_enter(self, ctx: SharedContext):
        self._start(ctx)
        ctx.handle_cup_complete.clear()

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        # 获取 Camera A 帧
        color, depth, depth_colormap, depth_frame = self._get_frames(ctx)
        ctx.frames_a = (color, depth, depth_colormap, depth_frame)

        # 检测 Handle+Cup
        self._detect(ctx)

        # 非阻塞 accept TCP
        self._accept_tcp(ctx)

        # 完成?
        if ctx.handle_cup_complete.is_set():
            print(f"[HANDLE_GRASP_AND_POUR] 全流程完成 ✅")
            return OrchestratorState.IDLE
        return None

    def on_exit(self, ctx: SharedContext):
        self._stop(ctx)

    # ================================================================
    # 初始化 / 清理
    # ================================================================

    def _start(self, ctx: SharedContext):
        """初始化检测器 + 手 + TCP 服务器 + 加载全部偏移参数."""
        cfg = ctx.realsense_config
        estun = ctx.estun_config
        eye_cfg = ctx.eye_to_hand_config

        # ---- 加载偏移参数 (从 estun_config) ----
        self._load_offsets(cfg, estun)

        # ---- 加载 EyeToHand 标定 + 递送参数 ----
        self._calib = load_calibration()
        delivery = eye_cfg.get("delivery", {})
        self._y_teach = delivery.get("Y_TEACH", 0.0)
        self._z_teach = delivery.get("Z_TEACH", 0.0)
        self._a_teach = delivery.get("A_TEACH", 0.0)
        self._b_teach = delivery.get("B_TEACH", 0.0)
        self._c_teach = delivery.get("C_TEACH", 0.0)
        print(f"[INFO] EyeToHand 标定: scale={self._calib['scale']:.4f}, bias={self._calib['bias']:.4f}")
        print(f"[INFO] 递送预示教: Y={self._y_teach:.3f} Z={self._z_teach:.3f} ABC=({self._a_teach:.1f},{self._b_teach:.1f},{self._c_teach:.1f})")

        # ---- 灵巧手 ----
        hand_port = estun.get("hand", {}).get("port", "COM5")
        init_hand(hand_port=hand_port, initial_gesture="RESET")
        self._hand_available = is_hand_available()

        # ---- Handle 6D 检测器 (Camera A) ----
        print("[INFO] 初始化 Handle 6D 检测器 (Camera A)...")
        ctx.handle_detector = RealSenseHandle6DDetector(config_dict=cfg)
        ctx.handle_detector.last_known_pose = None

        # ---- Cup 6D 检测器 (复用 pipeline) ----
        print("[INFO] 初始化 Cup 6D 检测器 (复用 pipeline)...")
        ctx.cup_detector = RealSenseCup6DDetector(config_dict=cfg, start_pipeline=False)
        detector = ctx.handle_detector
        ctx.cup_detector.color_intrinsics = detector.color_intrinsics
        ctx.cup_detector.depth_scale = detector.depth_scale
        ctx.cup_detector.last_known_pose = None

        # ---- 手眼标定 ----
        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml")
        )
        ctx.T_cam2ee = load_cam2ee(calib_path)

        # ---- 位姿稳定器 ----
        stab_cfg = estun.get("stabilization", {})
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

        # ---- TCP 服务器 ----
        host = estun["tcp_server"]["host"]
        port = estun["tcp_server"]["port"]
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        self._server.bind((host, port))
        self._server.listen(5)
        print(f"[INFO] TCP 服务器: {host}:{port}")

        # ---- Hand-in-cup 检测器 ----
        self._hand_in_cup = HandInCupDetector()

        # ---- 重置运行时标志 ----
        ctx.stop_person_detector = False
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        self._frame_counter = 0

    def _load_offsets(self, cfg, estun):
        """加载全部偏移 / 几何参数 (移植自 estun_handle_cup_run.py 模块级)."""
        # Handle
        hg = estun["handle_grasp"]
        self._handle_t_offset = np.array(hg["gripper_offset"]["t_handle_offset"], dtype=float)
        self._handle_r_offset = np.array(hg["gripper_offset"]["r_handle_offset"], dtype=float)
        self._handle_t_tcp_fixed = np.array(hg["gripper_offset"]["t_tcp_fixed"], dtype=float)
        self._handle_pre_t_offset = np.array(hg["pre_offset"]["t_offset"], dtype=float)
        self._handle_pre_r_offset = np.array(hg["pre_offset"]["r_offset"], dtype=float)
        self._handle_target_t_offset = np.array(hg["target_offset"]["t_offset"], dtype=float)
        self._handle_target_r_offset = np.array(hg["target_offset"]["r_offset"], dtype=float)

        # Cup handle_side
        self._cup_handle_side = cfg.get("cup", {}).get("handle_side", "back")
        print(f"[INFO] cup handle_side: {self._cup_handle_side}")

        # Cup offsets
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

        # P3 teach
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

        # Spout + Pour
        self._spout_t_offset = np.array(cg["spout_offset"]["t_offset"], dtype=float)
        self._pour_t_offset_base = np.array(cg["pour_offset"]["t_offset"], dtype=float)
        self._pour_r_offset = np.array(cg["pour_offset"]["r_offset"], dtype=float)

        # Cup geometry
        self._cup_rim_radius = cfg.get("cup", {}).get("rim_radius", 0.035)
        self._cup_height = cfg.get("cup", {}).get("height", 0.100)

        # Orientation
        self._orientation_mode = estun["orientation_mode"]
        self._grasp_direction = estun["grasp_direction"]
        self._handle_side = cfg["handle"]["side"]
        if self._handle_side != self._grasp_direction:
            print(f"[INFO] grasp_direction ({self._grasp_direction}) → handle_side ({self._handle_side})")
            self._grasp_direction = self._handle_side

        # fixed mode
        fixed_abc_map = estun.get("fixed_abc", {})
        fixed_abc_cfg = fixed_abc_map.get(self._grasp_direction, [])
        if len(fixed_abc_cfg) == 3:
            self._fixed_abc = np.array(fixed_abc_cfg, dtype=float)
        else:
            self._fixed_abc = np.array([-90, 82, -98], dtype=float)
            if self._orientation_mode == "fixed":
                print(f"[WARN] {self._grasp_direction} 方向 fixed_abc 未配置, 使用默认值")

        # 校验
        valid_modes = {"vertical", "horizontal", "free", "fixed"}
        if self._orientation_mode not in valid_modes:
            print(f"[WARN] 未知 orientation_mode: {self._orientation_mode}, → free")
            self._orientation_mode = "free"
        elif self._orientation_mode == "horizontal":
            print("[WARN] horizontal 模式未实现, → free")
            self._orientation_mode = "free"

        # 跳帧
        self._skip_frames = cfg["handle"]["skip_frames"]

    def _stop(self, ctx: SharedContext):
        """清理资源."""
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        if ctx.handle_detector:
            try:
                ctx.handle_detector.pipeline.stop()
            except Exception:
                pass
        if self._hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        ctx.stop_person_detector = False
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        ctx.handle_detector = None
        ctx.cup_detector = None
        ctx.T_cam2ee = None

    # ================================================================
    # Camera A 帧获取 & 检测
    # ================================================================

    def _get_frames(self, ctx: SharedContext):
        det = ctx.handle_detector
        if det is None:
            return None, None, None, None
        color, depth, depth_colormap, depth_frame = det.get_frames()
        self._last_color = color
        self._last_depth_image = depth
        self._last_depth_frame = depth_frame
        self._last_depth_colormap = depth_colormap
        return color, depth, depth_colormap, depth_frame

    def _detect(self, ctx: SharedContext):
        """跳帧检测 Handle+Cup."""
        if self._last_color is None or self._last_depth_frame is None:
            return

        detector = ctx.handle_detector
        cup_detector = ctx.cup_detector
        T_cam2ee = ctx.T_cam2ee
        if detector is None or cup_detector is None or T_cam2ee is None:
            return
        assert self._handle_stabilizer is not None, "handle_stabilizer 未初始化"
        assert self._cup_stabilizer is not None, "cup_stabilizer 未初始化"

        self._frame_counter += 1
        if self._frame_counter % self._skip_frames != 1:
            return

        # === Handle 检测 ===
        h_result = detector.detect_handle_6d(self._last_color, self._last_depth_frame)
        if h_result is not None:
            h_items: tuple = h_result if len(h_result) >= 7 else (None, None, None, None, None, None, None)  # type: ignore[assignment]
            body_mask, handle_mask, h_stage, yolo_bbox, h_pos, h_rot, h_euler = h_items[:7]

            if h_pos is not None and h_rot is not None:
                smooth_pos, smooth_rot = detector._pose_smoother.update(h_pos, h_rot)
                if smooth_rot is not None:
                    h_pos, h_rot = smooth_pos, smooth_rot
                    h_euler = rotation_matrix_to_euler_angles(h_rot)
                if h_euler is None:
                    h_euler = np.zeros(3)

                self._handle_stabilizer.add(tuple(h_pos), tuple(h_euler))  # type: ignore[arg-type]
                if self._handle_stabilizer.is_stable():
                    if not self._handle_stabilizer.is_generation_valid():
                        self._handle_stabilizer.reset()
                        self._handle_stabilizer.capture_generation()
                    else:
                        mean_pos, mean_euler = self._handle_stabilizer.get_mean()
                        if mean_pos is not None and mean_euler is not None:
                            mean_rot = euler_to_rotation_matrix(*mean_euler, order="zyx")
                            pos_ee = T_cam2ee[:3, :3] @ mean_pos + T_cam2ee[:3, 3]
                            rot_ee = T_cam2ee[:3, :3] @ mean_rot
                            A, B, C = rotation_matrix_to_euler_angles(rot_ee)
                            detector.last_known_pose = (pos_ee, (A, B, C))
                            print(f"\n[Handle STABLE] ee=({pos_ee[0]:.3f},{pos_ee[1]:.3f},{pos_ee[2]:.3f})")
            else:
                detector._pose_smoother.reset()

            # === Cup 检测 (注入帧) ===
            if self._last_depth_image is not None and cup_detector is not None:
                cup_detector.inject_frames(
                    self._last_color, self._last_depth_image, self._last_depth_colormap,
                    self._last_depth_frame
                )

            c_result = cup_detector.compute_cup_6d_pose(self._last_color, self._last_depth_frame)
            if c_result is not None:
                c_bbox, c_pos, c_rot, c_euler, c_point_cloud = c_result
                if c_pos is not None and c_rot is not None:
                    smooth_cpos, smooth_crot = cup_detector._pose_smoother.update(c_pos, c_rot)
                    if smooth_crot is not None:
                        c_pos, c_rot = smooth_cpos, smooth_crot
                        c_euler = rotation_matrix_to_euler_angles(c_rot)
                    if c_euler is None:
                        c_euler = np.zeros(3)
                    self._cup_stabilizer.add(tuple(c_pos), tuple(c_euler))  # type: ignore[arg-type]
                    if self._cup_stabilizer.is_stable():
                        if not self._cup_stabilizer.is_generation_valid():
                            self._cup_stabilizer.reset()
                            self._cup_stabilizer.capture_generation()
                        else:
                            mean_cpos, mean_ceuler = self._cup_stabilizer.get_mean()
                            if mean_cpos is not None and mean_ceuler is not None:
                                cup_detector.last_known_pose = (mean_cpos, mean_ceuler)  # type: ignore[assignment,unused-ignore]
                                print(f"\n[Cup STABLE] pos=({mean_cpos[0]:.3f},{mean_cpos[1]:.3f},{mean_cpos[2]:.3f})")
                else:
                    cup_detector._pose_smoother.reset()

            self._cached_handle = h_result
            self._cached_cup = c_result

    # ================================================================
    # TCP
    # ================================================================

    def _accept_tcp(self, ctx: SharedContext):
        """非阻塞 accept."""
        if self._server is None:
            return
        try:
            conn, addr = self._server.accept()
            print(f"[INFO] 机器人已连接: {addr}")
            t = threading.Thread(
                target=self._handle_client_impl,
                args=(conn, addr, ctx),
                daemon=True,
            )
            t.start()
        except BlockingIOError:
            pass

    # ================================================================
    # TCP 客户端处理 (单 Session 完整协议)
    # ================================================================

    def _handle_client_impl(self, conn: socket.socket, addr, ctx: SharedContext):
        """单 TCP Session 全流程: 移植自 estun_handle_cup_run._handle_client_impl.

        关键差异:
          - Cup P3 的 X 来自 Camera B 人体检测 + EyeToHand 标定
          - Phase D: 递送确认 + Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
        """
        print(f"[INFO] 客户端已连接: {addr}")

        # ---- 预计算工具偏移 ----
        _, t_handle_effective = compute_gripper_offset_transform(
            self._handle_t_offset, self._handle_r_offset)
        _, t_cup_effective = compute_gripper_offset_transform(
            self._cup_t_offset, self._cup_r_offset)
        cup_pose_data = None        # pack_three 格式, Phase B 计算, Phase C 使用
        cup_p1_base = None          # Cup P1 基系位置, ABORT 时用于放回

        # ---- [Phase A] 接收机器人初始末端位姿 (5s 超时) ----
        robot_xyz, robot_ABC = None, None
        try:
            conn.settimeout(5.0)
            init_data = conn.recv(1024)
            if not init_data:
                print("[WARN] 机器人未发送初始位姿, 断开连接")
                return
            pose_str = init_data.decode("utf-8").strip().strip("[]")
            parts = pose_str.split(",")
            if len(parts) == 6:
                robot_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
                robot_ABC = (float(parts[3]), float(parts[4]), float(parts[5]))
                print(f"[INFO] 机器人当前末端: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                      f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})")
            else:
                print(f"[WARN] 初始位姿格式错误: {pose_str}")
                return
        except socket.timeout:
            print("[WARN] 等待机器人初始位姿超时, 断开连接")
            return
        except Exception as e:
            print(f"[WARN] 接收初始位姿异常: {e}")
            return

        R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
        t_ee_base = np.array(robot_xyz, dtype=float)

        # ---- [Phase B] 等待稳定 → 计算位姿 → 发送 Handle ----
        print("[INFO] 等待稳定目标位姿 (Handle + Cup)...")
        detector = ctx.handle_detector
        cup_detector = ctx.cup_detector
        T_cam2ee = ctx.T_cam2ee
        assert detector is not None, "handle_detector 未初始化"
        assert cup_detector is not None, "cup_detector 未初始化"
        assert T_cam2ee is not None, "T_cam2ee 未加载"

        target_sent = False
        wait_start = time.time()
        while not target_sent and not ctx.stop_event.is_set() and (time.time() - wait_start) < 30.0:
            handle_pose = detector.last_known_pose
            cup_pose = cup_detector.last_known_pose

            if handle_pose is not None and cup_pose is not None:
                mean_pos_ee, mean_euler_ee = handle_pose  # type: ignore[misc]
                cup_pos_cam, cup_euler_cam = cup_pose  # type: ignore[misc]

                # ============================================================
                # Phase 1a: Handle P1 / P2 (不依赖倒水口)
                # ============================================================
                h_pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
                R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")
                R_grasp_ee = compute_grasp_rotation(
                    R_obj_ee, self._orientation_mode, self._handle_r_offset, R_ee_base,  # type: ignore[arg-type]
                    grasp_direction=self._grasp_direction)

                # 预声明分支变量 (Pylance narrowing)
                _fixed_R_grasp_ee: np.ndarray = np.eye(3)  # type: ignore[assignment]
                _fixed_R_target_ee: np.ndarray = np.eye(3)  # type: ignore[assignment]
                _R_grasp_ee_corrected: np.ndarray = np.eye(3)  # type: ignore[assignment]
                _R_pour_ee: np.ndarray = np.eye(3)  # type: ignore[assignment]

                if self._orientation_mode == "fixed":
                    assert self._fixed_abc is not None
                    R_base_fix = euler_to_rotation_matrix(*self._fixed_abc, order="zyx")
                    R_grasp_ee = R_ee_base.T @ R_base_fix
                    pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                    A1, B1, C1 = tuple(self._fixed_abc)  # type: ignore[arg-type]
                    pos_ee_pre = h_pos_ee_compensated + R_grasp_ee @ self._handle_pre_t_offset
                    pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                    A2, B2, C2 = tuple(self._fixed_abc)  # type: ignore[arg-type]
                    R_target_ee = R_grasp_ee @ euler_to_rotation_matrix(*self._handle_target_r_offset, order="zyx")  # type: ignore[arg-type]
                    _fixed_R_grasp_ee = R_grasp_ee
                    _fixed_R_target_ee = R_target_ee
                else:
                    rot_base_1 = R_ee_base @ R_grasp_ee
                    pos_base_1 = transform_pose_to_base(h_pos_ee_compensated, robot_xyz, robot_ABC)
                    A1, B1, C1 = rotation_matrix_to_euler_angles(rot_base_1)
                    A1, B1, C1 = post_correct_angles(A1, B1, C1, robot_ABC)
                    if self._orientation_mode == "vertical" and self._grasp_direction == "right":
                        if not (-100 <= C1 <= -80):
                            print(f"[CLAMP] Handle P1 C {C1:.1f} → -90.0")
                            C1 = -90.0
                        if not (84 <= B1 <= 86):
                            print(f"[CLAMP] Handle P1 B {B1:.1f} → 85.0")
                            B1 = 85.0

                    rot_base_1_corrected = euler_to_rotation_matrix(A1, B1, C1, order="zyx")
                    R_grasp_ee_corrected = R_ee_base.T @ rot_base_1_corrected
                    pos_ee_pre = h_pos_ee_compensated + R_grasp_ee_corrected @ self._handle_pre_t_offset
                    pos_base_2 = transform_pose_to_base(pos_ee_pre, robot_xyz, robot_ABC)
                    A2, B2, C2 = A1, B1, C1
                    R_pour_ee = R_grasp_ee_corrected @ euler_to_rotation_matrix(*self._pour_r_offset, order="zyx")  # type: ignore[arg-type]
                    _R_grasp_ee_corrected = R_grasp_ee_corrected
                    _R_pour_ee = R_pour_ee

                # ============================================================
                # Phase 2: Cup P1 / P2 / P3
                #   P1, P2: Camera A 检测 (同原版)
                #   P3:     delivery = (X_person, Y_TEACH, Z_TEACH, ABC_TEACH)
                # ============================================================
                c_pos_ee = T_cam2ee[:3, :3] @ np.array(cup_pos_cam) + T_cam2ee[:3, 3]
                c_rot_ee = T_cam2ee[:3, :3] @ euler_to_rotation_matrix(*cup_euler_cam, order="zyx")
                c_pos_ee_compensated = c_pos_ee - t_cup_effective
                c_R_grasp_ee = compute_grasp_rotation(
                    c_rot_ee, self._orientation_mode, self._cup_r_offset, R_ee_base,  # type: ignore[arg-type]
                    grasp_direction=self._grasp_direction)

                # Cup P1
                c_rot_base_1 = R_ee_base @ c_R_grasp_ee
                c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)
                cA1, cB1, cC1 = rotation_matrix_to_euler_angles(c_rot_base_1)
                cA1, cB1, cC1 = post_correct_angles(cA1, cB1, cC1, robot_ABC)
                if self._orientation_mode == "vertical":
                    if self._cup_handle_side == "back":
                        if cB1 < 90:
                            print(f"[CLAMP] Cup P1 B {cB1:.1f} → 89.0")
                            cB1 = 89.0
                        elif cB1 > 90:
                            print(f"[CLAMP] Cup P1 B {cB1:.1f} → 91.0")
                            cB1 = 91.0
                        if not (-120 <= cC1 <= -100):
                            print(f"[CLAMP] Cup P1 C {cC1:.1f} → -110.0")
                            cC1 = -110.0
                    else:
                        if cB1 < 90:
                            cB1 = 89.0
                        elif cB1 > 90:
                            cB1 = 91.0
                        if not (-100 <= cC1 <= -80):
                            print(f"[CLAMP] Cup P1 C {cC1:.1f} → -90.0")
                            cC1 = -90.0

                c_rot_base_1_corrected = euler_to_rotation_matrix(cA1, cB1, cC1, order="zyx")
                c_R_grasp_ee_corrected = R_ee_base.T @ c_rot_base_1_corrected

                # Cup P2
                c_pos_ee_pre = c_pos_ee_compensated + c_R_grasp_ee_corrected @ self._cup_pre_t_offset
                c_pos_base_2 = transform_pose_to_base(c_pos_ee_pre, robot_xyz, robot_ABC)
                cA2, cB2, cC2 = cA1, cB1, cC1

                # Cup P3 = delivery (EyeToHand)
                #   取 PersonDetector 最新稳定 X_cam → X_base
                #   YZ/ABC 用预示教值
                c_person_x = c_pos_base_1[0]  # fallback: 用 Cup P1 X
                if ctx.person_stable_x_cam is not None:
                    c_person_x = camera_x_to_base_x(ctx.person_stable_x_cam, self._calib)
                    print(f"[P3] X_cam={ctx.person_stable_x_cam:.3f} → X_base={c_person_x:.3f}")
                else:
                    print(f"[WARN] person_stable_x_cam=None, P3 X 用 Cup P1 X={c_person_x:.3f}")

                c_pos_base_3 = (
                    float(c_person_x),
                    self._y_teach,
                    self._z_teach,
                )
                cA3, cB3, cC3 = self._a_teach, self._b_teach, self._c_teach
                cup_p1_base = c_pos_base_1  # 保存供 ABORT 使用

                # 缓存 cup 位姿
                cup_pose_data = pack_three(
                    ((c_pos_base_1, (cA1, cB1, cC1))),
                    ((c_pos_base_2, (cA2, cB2, cC2))),
                    ((c_pos_base_3, (cA3, cB3, cC3)))
                )

                # ============================================================
                # Phase 1b: 基于 Cup TCP 计算倒水口 → Handle P3
                # ============================================================
                assert self._p3_teach_t_offset is not None
                assert self._p3_teach_r_abc is not None
                assert self._handle_t_tcp_fixed is not None
                assert self._spout_t_offset is not None
                assert self._pour_t_offset_base is not None
                assert self._pour_r_offset is not None

                if self._p3_teach_enable:
                    pos_base_3 = tuple(np.array(c_pos_base_1) + self._p3_teach_t_offset)  # type: ignore[arg-type]
                    A3, B3, C3 = tuple(self._p3_teach_r_abc)  # type: ignore[arg-type]
                    print(f"[POUR] 示教偏移: t=({self._p3_teach_t_offset[0]:.4f},{self._p3_teach_t_offset[1]:.4f},{self._p3_teach_t_offset[2]:.4f}) "
                          f"abc=({self._p3_teach_r_abc[0]:.1f},{self._p3_teach_r_abc[1]:.1f},{self._p3_teach_r_abc[2]:.1f})")
                else:
                    _tcp_offset_base = R_ee_base @ self._handle_t_tcp_fixed
                    _tcp_base = np.array(c_pos_base_1) + _tcp_offset_base
                    if self._cup_handle_side == "back":
                        pour_target_base = _tcp_base + np.array([self._cup_rim_radius, 0.0, self._cup_height / 2.0])
                    else:
                        pour_target_base = _tcp_base + np.array([0.0, self._cup_rim_radius, self._cup_height / 2.0])
                    pour_target_ee_calc = R_ee_base.T @ (pour_target_base - t_ee_base)
                    pour_target = T_cam2ee[:3, :3].T @ (pour_target_ee_calc - T_cam2ee[:3, 3])
                    if self._orientation_mode == "fixed":
                        pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
                        pos_ee_p3 = pos_pour_ee - _fixed_R_grasp_ee @ self._spout_t_offset - t_handle_effective
                        pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + self._pour_t_offset_base
                        rot_base_3 = R_ee_base @ _fixed_R_target_ee
                        A3, B3, C3 = rotation_matrix_to_euler_angles(rot_base_3)
                        A3, B3, C3 = post_correct_angles(A3, B3, C3, robot_ABC)
                    else:
                        pos_pour_ee = T_cam2ee[:3, :3] @ np.array(pour_target) + T_cam2ee[:3, 3]
                        spout_rotated = _R_pour_ee @ self._spout_t_offset
                        pos_ee_p3 = pos_pour_ee - spout_rotated - t_handle_effective
                        pos_base_3 = transform_pose_to_base(pos_ee_p3, robot_xyz, robot_ABC) + self._pour_t_offset_base
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
                    print(f"[SEND] Handle: {msg}")
                    target_sent = True
                except (BrokenPipeError, ConnectionResetError) as e:
                    print(f"[WARN] 发送 Handle 失败: {e}")
                    return
            else:
                time.sleep(0.1)

        if not target_sent:
            print("[WARN] 超时: 未获得稳定目标位姿, 断开连接")
            return

        # ---- [Phase C & D] 指令循环 ----
        conn.settimeout(2.0)
        cup_p3_sent = False        # CUP 触发后置 True
        hand_judged = False        # 第一次 RELEASE (hand-in-cup 判定) 已处理
        abort_mode = False         # TRUE=CONTINUE, FALSE=ABORT 进入放回模式

        try:
            while not ctx.stop_event.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    continue

                if not data:
                    print("[INFO] 连接已关闭 (SocketClose)")
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

                elif cmd == "CUP":
                    if not cup_p3_sent and cup_pose_data is not None:
                        ctx.stop_person_detector = True
                        print("[INFO] PersonDetector 检测已停止")
                        try:
                            conn.sendall(cup_pose_data.encode("utf-8"))
                            print(f"[SEND] Cup: {cup_pose_data}")
                            cup_p3_sent = True
                        except (BrokenPipeError, ConnectionResetError) as e:
                            print(f"[WARN] 发送 Cup 失败: {e}")
                            break
                    else:
                        print("[WARN] CUP: cup_pose_data 未就绪或已发送")

                elif cmd == "READY":
                    ctx.hand_in_cup_active = True
                    if self._hand_in_cup is not None:
                        self._hand_in_cup.reset()
                    print("[INFO] Hand-in-cup 启动")

                elif cmd == "RELEASE":
                    if not hand_judged:
                        # ---- 第一次 RELEASE: Hand-in-cup 判定 ----
                        hand_ok = ctx.hand_on_cup if ctx.hand_on_cup is not None else False
                        print(f"[INFO] Hand-in-cup 判定: {'TRUE' if hand_ok else 'FALSE'}")
                        hand_judged = True

                        if hand_ok:
                            # TRUE: 松手 → CONTINUE
                            if self._hand_available:
                                release_object()
                            print("[INFO] RELEASE (松杯)")
                            try:
                                # 发送格式与机器人端 TranStrToCpos("<",">") 配对
                                conn.sendall(b"<0,0,0,0,0,0>\n")
                                print("[SEND] CONTINUE (<0,0,0,0,0,0>)")
                            except Exception as e:
                                print(f"[WARN] 发送 CONTINUE 失败: {e}")
                                break
                            # 机器人: P4 → RESET → Home → SocketClose
                            # 循环继续, 等待后续 RESET
                        else:
                            # FALSE: 不松手 → ABORT
                            print("[INFO] 无人接杯 → ABORT")
                            abort_mode = True
                            try:
                                # 发送格式与机器人端 TranStrToCpos("[","]") 配对
                                conn.sendall(b"[0,0,0,0,0,0]\n")
                                print("[SEND] ABORT ([0,0,0,0,0,0])")
                            except Exception as e:
                                print(f"[WARN] 发送 ABORT 失败: {e}")
                                break
                            # 机器人: P1 → RELEASE → P2 → RESET → Home → SocketClose
                            # 循环继续, 等待后续 RELEASE + RESET
                    else:
                        # ---- FALSE 路径: 第二次 RELEASE (放回后松杯) ----
                        if abort_mode:
                            if self._hand_available:
                                release_object()
                            print("[INFO] RELEASE (放回后松杯)")

                else:
                    print(f"[INFO] 忽略: {cmd}")

        except ConnectionResetError:
            print(f"[INFO] 连接重置: {addr}")
        except Exception as e:
            print(f"[ERROR] 客户端处理异常: {e}")
        finally:
            # ---- 清理 ----
            ctx.hand_in_cup_active = False
            ctx.hand_on_cup = None
            try:
                conn.close()
            except Exception:
                pass
            print(f"[INFO] 客户端已断开: {addr}")

        # ---- 通知状态机完成 ----
        PoseStabilizer.on_disconnect()
        ctx.handle_cup_complete.set()
