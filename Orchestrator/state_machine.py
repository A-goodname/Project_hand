# -*- coding: utf-8 -*-
"""
Orchestrator 状态机
===================
驱动双相机全流程: IDLE → HANDLE_GRASP_AND_POUR → LOCATE_PERSON
                         → APPROACH_PERSON → WAIT_AND_RELEASE → RETREAT → IDLE

状态说明:
  - IDLE:               眼在手外等待检测到人 → 触发全流程
  - HANDLE_GRASP_AND_POUR:  眼在手上 抓壶倒水 → 抓杯
  - LOCATE_PERSON:      眼在手外 检测人 X 坐标 → 计算递送位姿
  - APPROACH_PERSON:    发送递送位姿给机器人 → 等待到达
  - WAIT_AND_RELEASE:   等待固定时长 → 松手
  - RETREAT:            机器人退回 → 回到 IDLE
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
from RealSense_R.realsense_shared import (
    load_camera_serial,
    rotation_matrix_to_euler_angles,
)
from RealSense_R.test_handle_6D import RealSenseHandle6DDetector
from RealSense_R.test_cup_6D import RealSenseCup6DDetector
from EyeToHand.person_detector import PersonDetector
from EyeToHand.detector import load_calibration, camera_x_to_base_x
from Estun.utils_shared import (
    euler_to_rotation_matrix, transform_pose_to_base,
    init_hand, is_hand_available, release_object, disable_hand,
)


# ============================================================
# 通用工具 (从 estun_handle_cup_run.py 复制, 不改原文件)
# ============================================================

def fmt_pose(xyz, abc):
    """格式化单个位姿: "x,y,z,A,B,C" (位置 m, 角度 deg)"""
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def pack_three(p1, p2, p3):
    """打包三个位姿: <P1>[P2]{P3} — 与 ESTUN TranStrToCpos 括号标记约定匹配"""
    return f"<{fmt_pose(p1[0], p1[1])}>[{fmt_pose(p2[0], p2[1])}]{{{fmt_pose(p3[0], p3[1])}}}"


def load_cam2ee(yaml_path):
    """从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)"""
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


_gf = 0  # 全局代际

class PoseStabilizer:
    """位姿稳定器 — 与 estun_handle_cup_run.py 一致"""
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
    LOCATE_PERSON = auto()
    APPROACH_PERSON = auto()
    WAIT_AND_RELEASE = auto()
    RETREAT = auto()


# ============================================================
# 共享上下文
# ============================================================

@dataclass
class SharedContext:
    """跨状态共享的数据与依赖.

    使用说明:
      - main.py 填充全部初始字段
      - 各 state handler 在运行时写入运行时字段
    """
    # ---- 配置 (初始化时填入) ----
    eye_to_hand_config: dict        # EyeToHand/config.yaml
    estun_config: dict              # Estun/estun_config.yaml["estun"]
    realsense_config: dict          # RealSense_R/config.yaml (Handle+Cup 用)
    eye_to_hand_serial: str         # EyeToHand 相机 serial

    # ---- 运行时字段 (state handler 写入/读取) ----
    person_detector: Optional[PersonDetector] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    # HANDLE_GRASP_AND_POUR 阶段
    handle_cup_complete: threading.Event = field(default_factory=threading.Event)
    handle_detector: Optional[RealSenseHandle6DDetector] = None
    cup_detector: Optional[RealSenseCup6DDetector] = None
    T_cam2ee: Optional[np.ndarray] = None

    # LOCATE_PERSON → DELIVERY 阶段
    delivery_pose: Optional[tuple] = None  # (xyz, abc)
    release_sent: bool = False

    # 可视化帧缓存 (main.py 每 tick 写入)
    frames_a: Optional[tuple] = None  # (color, depth, depth_colormap, depth_frame) — Camera A (Handle)
    frames_b: Optional[tuple] = None  # (color, depth, depth_colormap, depth_frame) — Camera B (Person)


# ============================================================
# 状态处理器基类
# ============================================================

class StateHandler(ABC):
    """状态处理器的抽象基类.

    每个状态实现 on_enter / on_update / on_exit / on_error.
    on_update 返回下一个状态 (或 None 保持当前).
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def on_enter(self, ctx: SharedContext):
        """进入状态时调用."""

    @abstractmethod
    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        """每 tick 调用. 返回下一个状态或 None."""

    def on_exit(self, ctx: SharedContext):
        """离开状态时调用 (可选重写)."""

    def on_error(self, ctx: SharedContext, msg: str):
        """状态内异常处理 (可选重写)."""
        print(f"[STATE_ERROR] {self.name}: {msg}")


# ============================================================
# 状态机引擎
# ============================================================

class StateMachine:
    """线性状态机.

    严格沿状态链正向推进. 异常恢复回 IDLE.
    """

    def __init__(self):
        self._state: OrchestratorState = OrchestratorState.IDLE
        self._handlers: dict[OrchestratorState, StateHandler] = {}

    def register(self, state: OrchestratorState, handler: StateHandler):
        """注册状态处理器."""
        self._handlers[state] = handler

    @property
    def state(self) -> OrchestratorState:
        return self._state

    def transition_to(self, ctx: SharedContext, target: OrchestratorState):
        """状态切换: 调旧状态的 on_exit → 更新 → 调新状态的 on_enter."""
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
        """每 tick 调用. 驱动当前状态."""
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
        """异常处理 — 通知当前 handler 后回 IDLE."""
        h = self._handlers.get(self._state)
        if h is not None:
            try:
                h.on_error(ctx, msg)
            except Exception:
                pass
        self.transition_to(ctx, OrchestratorState.IDLE)


# ============================================================
# Handle+Cup Runner (封装 eye-in-hand 全流程)
# ============================================================

class HandleCupRunner:
    """封装 Handle+Cup 全流程.

    在 HANDLE_GRASP_AND_POUR 状态中使用.
    逻辑源自 estun_handle_cup_run.py, 仅添加 completion event 通知.
    """

    def __init__(self, ctx: SharedContext):
        self._ctx = ctx
        self._hand_available = False

        # 跳帧控制
        self._frame_counter = 0
        self._skip_frames = ctx.realsense_config.get("handle", {}).get("skip_frames", 90)
        self._cached_handle = None
        self._cached_cup = None
        self._last_color = None
        self._last_depth_frame = None
        self._last_depth_image = None
        self._last_depth_colormap = None

        # 位姿稳定器
        stab_cfg = ctx.estun_config.get("stabilization", {})
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

        # TCP 服务器
        self._server = None

    # ------------------------------------------------------------
    def start(self):
        """初始化检测器 + TCP 服务器."""
        cfg = self._ctx.realsense_config
        estun = self._ctx.estun_config

        # ---- 灵巧手 ----
        hand_port = estun.get("hand", {}).get("port", "COM5")
        init_hand(hand_port=hand_port, initial_gesture="RESET")
        self._hand_available = is_hand_available()

        # ---- Handle 6D 检测器 ----
        print("[INFO] 初始化 Handle 6D 检测器 (Camera A)...")
        self._ctx.handle_detector = RealSenseHandle6DDetector(config_dict=cfg)
        self._ctx.handle_detector.last_known_pose = None

        # ---- Cup 6D 检测器 (复用 pipeline) ----
        print("[INFO] 初始化 Cup 6D 检测器 (复用 pipeline)...")
        detector = self._ctx.handle_detector
        self._ctx.cup_detector = RealSenseCup6DDetector(config_dict=cfg, start_pipeline=False)
        self._ctx.cup_detector.color_intrinsics = detector.color_intrinsics
        self._ctx.cup_detector.depth_scale = detector.depth_scale
        self._ctx.cup_detector.depth_intrinsics = detector.depth_intrinsics
        self._ctx.cup_detector.color_to_depth_extrinsics = detector.color_to_depth_extrinsics
        self._ctx.cup_detector.depth_to_color_extrinsics = detector.depth_to_color_extrinsics
        self._ctx.cup_detector.last_known_pose = None

        # ---- 手眼标定 ----
        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml")
        )
        self._ctx.T_cam2ee = load_cam2ee(calib_path)

        # ---- TCP 服务器 ----
        host = estun["tcp_server"]["host"]
        port = estun["tcp_server"]["port"]
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        self._server.bind((host, port))
        self._server.listen(5)
        print(f"[INFO] Handle+Cup TCP 服务器: {host}:{port}")

        self._handle_stabilizer.capture_generation()
        self._cup_stabilizer.capture_generation()

    # ------------------------------------------------------------
    def get_frames(self):
        """获取 Camera A 帧."""
        det = self._ctx.handle_detector
        if det is None:
            return None, None, None, None
        color, depth, depth_colormap, depth_frame = det.get_frames()
        self._last_color = color
        self._last_depth_image = depth
        self._last_depth_frame = depth_frame
        self._last_depth_colormap = depth_colormap
        return color, depth, depth_colormap, depth_frame

    # ------------------------------------------------------------
    def detect(self):
        """每 tick 调用: 跳帧检测 Handle+Cup."""
        if self._last_color is None or self._last_depth_frame is None:
            return

        detector = self._ctx.handle_detector
        cup_detector = self._ctx.cup_detector
        T_cam2ee = self._ctx.T_cam2ee
        if detector is None or cup_detector is None or T_cam2ee is None:
            return  # 检测器尚未初始化

        self._frame_counter += 1

        # ---- 跳帧检测 ----
        if self._frame_counter % self._skip_frames == 1:
            # === Handle 检测 ===
            h_result = detector.detect_handle_6d(self._last_color, self._last_depth_frame)
            if h_result is not None:
                h_items = h_result if len(h_result) >= 7 else (None, None, None, None, None, None, None)
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
                cd = cup_detector
                if self._last_depth_image is not None and cd is not None:
                    cd.inject_frames(
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
                                    cup_detector.last_known_pose = (mean_cpos, mean_ceuler)  # type: ignore[assignment]
                                    print(f"\n[Cup STABLE] pos=({mean_cpos[0]:.3f},{mean_cpos[1]:.3f},{mean_cpos[2]:.3f})")
                    else:
                        cup_detector._pose_smoother.reset()

                self._cached_handle = h_result
                self._cached_cup = c_result

    # ------------------------------------------------------------
    def get_frame_cache(self):
        """返回缓存的帧用于可视化."""
        return self._last_color, self._last_depth_colormap

    # ------------------------------------------------------------
    def accept_tcp(self):
        """非阻塞 accept TCP 连接."""
        if self._server is None:
            return
        try:
            conn, addr = self._server.accept()
            print(f"[INFO] 机器人已连接: {addr}")
            t = threading.Thread(
                target=self._handle_client_tcp,
                args=(conn, addr),
                daemon=True
            )
            t.start()
        except BlockingIOError:
            pass

    # ------------------------------------------------------------
    def _handle_client_tcp(self, conn, addr):
        """TCP 客户端处理."""
        # TODO: 从 estun_handle_cup_run.py 复制 _handle_client_impl,
        #       仅修改: 全流程结束时 set self._ctx.handle_cup_complete
        try:
            conn.close()
        except Exception:
            pass
        self._ctx.handle_cup_complete.set()
        PoseStabilizer.on_disconnect()

    # ------------------------------------------------------------
    def stop(self):
        """清理资源."""
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
        if self._ctx.handle_detector:
            try:
                self._ctx.handle_detector.pipeline.stop()
            except Exception:
                pass
        if is_hand_available():
            try:
                disable_hand()
            except Exception:
                pass

    # ------------------------------------------------------------
    def visualize(self, display_img):
        """在图像上叠加 Handle+Cup 检测结果 (可视化)."""
        # 调用 detector.visualize 的逻辑 (同 estun_handle_cup_run.py)
        pass


# ============================================================
# 各状态处理器
# ============================================================

class IdleHandler(StateHandler):
    """IDLE: 等待眼在手外相机稳定检测到人."""

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
        ctx.handle_cup_complete.clear()
        ctx.delivery_pose = None
        print("[IDLE] 等待检测到人体...")

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if ctx.person_detector is None:
            return None
        # 获取帧 (main.py 已缓存到 ctx.frames_b)
        if ctx.frames_b is None:
            return None
        color, depth, _, depth_frame = ctx.frames_b
        if color is None or depth_frame is None:
            return None

        result = ctx.person_detector.detect_person_stable(color, depth_frame)
        if result is not None:
            print(f"[IDLE] 检测到人体 ✅ pos_3d_cam=({result['pos_3d_cam'][0]:.3f}, "
                  f"{result['pos_3d_cam'][1]:.3f}, {result['pos_3d_cam'][2]:.3f})")
            return OrchestratorState.HANDLE_GRASP_AND_POUR
        return None


class HandleGraspAndPourHandler(StateHandler):
    """HANDLE_GRASP_AND_POUR: 抓壶倒水 → 抓杯."""

    def __init__(self):
        super().__init__("HANDLE_GRASP_AND_POUR")
        self._runner: Optional[HandleCupRunner] = None

    def on_enter(self, ctx: SharedContext):
        self._runner = HandleCupRunner(ctx)
        self._runner.start()
        ctx.handle_cup_complete.clear()

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if self._runner is None:
            return None
        # 获取帧 (内部缓存到 runner, 供跳帧检测用)
        color, depth, depth_colormap, depth_frame = self._runner.get_frames()
        ctx.frames_a = (color, depth, depth_colormap, depth_frame)

        # 跳帧检测 Handle+Cup
        self._runner.detect()
        # 非阻塞 accept TCP
        self._runner.accept_tcp()

        if ctx.handle_cup_complete.is_set():
            print(f"[HANDLE_GRASP_AND_POUR] Handle+Cup 全流程完成 ✅")
            return OrchestratorState.LOCATE_PERSON
        return None

    def on_exit(self, ctx: SharedContext):
        if self._runner:
            self._runner.stop()
            self._runner = None


class LocatePersonHandler(StateHandler):
    """LOCATE_PERSON: 检测人 X 坐标 → 计算递送位姿."""

    def __init__(self):
        super().__init__("LOCATE_PERSON")
        self._timeout = 15.0
        self._start_time = 0.0
        self._last_valid_x = None

    def on_enter(self, ctx: SharedContext):
        # 加载标定
        self._calib = load_calibration()
        self._delivery_cfg = ctx.eye_to_hand_config.get("delivery", {})
        self._y_teach = self._delivery_cfg.get("Y_TEACH", 0.0)
        self._z_teach = self._delivery_cfg.get("Z_TEACH", 0.0)
        self._a_teach = self._delivery_cfg.get("A_TEACH", 0.0)
        self._b_teach = self._delivery_cfg.get("B_TEACH", 0.0)
        self._c_teach = self._delivery_cfg.get("C_TEACH", 0.0)
        self._start_time = time.time()
        self._last_valid_x = None
        print(f"[LOCATE_PERSON] 开始检测人体 X 坐标 (标定: scale={self._calib['scale']:.4f}, bias={self._calib['bias']:.4f})")

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
            x_cam = result["pos_3d_cam"][0]
            x_base = camera_x_to_base_x(x_cam, self._calib)
            self._last_valid_x = x_base

            delivery_xyz = (float(x_base), self._y_teach, self._z_teach)
            delivery_abc = (self._a_teach, self._b_teach, self._c_teach)
            ctx.delivery_pose = (delivery_xyz, delivery_abc)

            print(f"[LOCATE_PERSON] X_cam={x_cam:.3f} → X_base={x_base:.3f} ✅")
            print(f"[LOCATE_PERSON] 递送位姿: xyz=({delivery_xyz[0]:.3f}, {delivery_xyz[1]:.3f}, {delivery_xyz[2]:.3f}) "
                  f"ABC=({delivery_abc[0]:.1f}, {delivery_abc[1]:.1f}, {delivery_abc[2]:.1f})")
            return OrchestratorState.APPROACH_PERSON

        # 超时处理
        if time.time() - self._start_time > self._timeout:
            if self._last_valid_x is not None:
                print(f"[WARN] 人体检测超时, 使用最近有效 X={self._last_valid_x:.3f}")
                ctx.delivery_pose = (
                    (self._last_valid_x, self._y_teach, self._z_teach),
                    (self._a_teach, self._b_teach, self._c_teach)
                )
                return OrchestratorState.APPROACH_PERSON
            else:
                print(f"[ERROR] 人体检测超时 {self._timeout}s, 无人检测到")
                ctx.delivery_pose = None
                return OrchestratorState.RETREAT
        return None

    def on_error(self, ctx: SharedContext, msg: str):
        print(f"[LOCATE_PERSON] 异常: {msg}")


class ApproachPersonHandler(StateHandler):
    """APPROACH_PERSON: 发送递送位姿给机器人 → 等待到达."""

    def __init__(self):
        super().__init__("APPROACH_PERSON")
        self._server = None
        self._conn = None
        self._pose_sent = False
        self._timeout = 30.0
        self._start_time = 0.0

    def on_enter(self, ctx: SharedContext):
        if ctx.delivery_pose is None:
            print("[ERROR] APPROACH_PERSON: delivery_pose 为空, 跳过")
            return

        host = ctx.estun_config.get("tcp_server", {}).get("host", "0.0.0.0")
        port = ctx.estun_config.get("tcp_server", {}).get("port", 8080)

        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        try:
            self._server.bind((host, port))
            self._server.listen(1)
            print(f"[APPROACH_PERSON] TCP 服务器: {host}:{port}")
        except Exception as e:
            print(f"[ERROR] 绑定 TCP 端口失败: {e}")
            self._server = None

        self._conn = None
        self._pose_sent = False
        self._start_time = time.time()
        ctx.release_sent = False

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if self._server is None or ctx.delivery_pose is None:
            return OrchestratorState.RETREAT

        # 接受连接
        if self._conn is None:
            try:
                self._conn, addr = self._server.accept()
                self._conn.settimeout(2.0)
                print(f"[APPROACH_PERSON] 机器人已连接: {addr}")
            except BlockingIOError:
                pass

        # 发送位姿
        if self._conn and not self._pose_sent:
            xyz, abc = ctx.delivery_pose
            msg = f"<{fmt_pose(xyz, abc)}>\n"
            try:
                self._conn.sendall(msg.encode("utf-8"))
                print(f"[APPROACH_PERSON] 发送递送位姿: {msg.strip()}")
                self._pose_sent = True
            except Exception as e:
                print(f"[ERROR] 发送递送位姿失败: {e}")
                return OrchestratorState.RETREAT

        # 等待 READY
        if self._conn and self._pose_sent:
            try:
                data = self._conn.recv(1024)
                if data:
                    cmd = data.decode("utf-8").strip()
                    print(f"[APPROACH_PERSON] 收到: {cmd}")
                    if cmd == "READY":
                        print("[APPROACH_PERSON] 机器人已到达递送位置 ✅")
                        return OrchestratorState.WAIT_AND_RELEASE
            except socket.timeout:
                pass
            except Exception as e:
                print(f"[ERROR] 接收异常: {e}")
                return OrchestratorState.RETREAT

        # 超时
        if time.time() - self._start_time > self._timeout:
            print(f"[ERROR] 等待机器人到达超时 {self._timeout}s")
            return OrchestratorState.RETREAT

        return None

    def on_exit(self, ctx: SharedContext):
        # 不关闭 conn, WAIT_AND_RELEASE 还需要用
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None

    def on_error(self, ctx: SharedContext, msg: str):
        self._cleanup()

    def _cleanup(self):
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None


class WaitAndReleaseHandler(StateHandler):
    """WAIT_AND_RELEASE: 等待固定时长 → 松手."""

    def __init__(self):
        super().__init__("WAIT_AND_RELEASE")
        self._enter_time = 0.0
        self._wait_seconds = 2.0
        self._done = False

    def on_enter(self, ctx: SharedContext):
        self._wait_seconds = ctx.eye_to_hand_config.get("delivery", {}).get("wait_seconds", 2.0)
        self._enter_time = time.time()
        self._done = False
        print(f"[WAIT_AND_RELEASE] 等待 {self._wait_seconds}s 后松手...")

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if self._done:
            return OrchestratorState.RETREAT
        elapsed = time.time() - self._enter_time
        if elapsed < self._wait_seconds:
            return None

        # 松手
        print(f"[WAIT_AND_RELEASE] 松手 ⏰ (已等待 {elapsed:.1f}s)")
        if is_hand_available():
            try:
                release_object()
                print("[WAIT_AND_RELEASE] release_object() ✅")
            except Exception as e:
                print(f"[WARN] release_object 异常: {e}")
        ctx.release_sent = True
        self._done = True
        return OrchestratorState.RETREAT

    def on_error(self, ctx: SharedContext, msg: str):
        pass


class RetreatHandler(StateHandler):
    """RETREAT: 等待机器人退回避让 → 回到 IDLE."""

    def __init__(self):
        super().__init__("RETREAT")
        self._enter_time = 0.0

    def on_enter(self, ctx: SharedContext):
        self._enter_time = time.time()
        print("[RETREAT] 等待机器人退回...")

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        if time.time() - self._enter_time > 3.0:
            print("[RETREAT] 复位完成 → IDLE")
            return OrchestratorState.IDLE
        return None

    def on_exit(self, ctx: SharedContext):
        """清空所有会话状态."""
        ctx.delivery_pose = None
        ctx.release_sent = False
        ctx.handle_cup_complete.clear()
