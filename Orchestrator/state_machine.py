# -*- coding: utf-8 -*-
"""
Orchestrator 状态机
===================
双相机 3 状态全流程: IDLE → GREETING → HANDLE_GRASP_AND_POUR → IDLE

状态说明:
  - IDLE:                   Camera B 椅子标定 → 人就位检测 → 触发
  - GREETING:               Camera A 独立 pipeline + TCP 8600 Phase 0 (招手→挥手→握手)
  - HANDLE_GRASP_AND_POUR:  单 TCP Session 8080 全流程 (抓壶倒水 → 抓杯递送 → HandInCup 判定)

与 estun_handle_cup_run.py 的关键差异:
  - Cup P3 XYZ/ABC 全示教固定 (EyeToHand/config.yaml delivery 段)
  - 递送后增加 Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
  - Phase 0 打招呼集成到 GREETING 状态 (TCP 8600)
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
from RealSense_R.realsense_shared import rotation_matrix_to_euler_angles
from RealSense_R.test_handle_6D import RealSenseHandle6DDetector
from RealSense_R.test_cup_6D import RealSenseCup6DDetector
from EyeToHand.person_detector import PersonDetector

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
    GREETING = auto()              # Phase 0 打招呼 (Camera A 独立 pipeline + TCP 8600)
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
      - hand_in_cup_active, hand_on_cup
      - handle_cup_complete, frames_a, frames_b
    """
    # ---- 配置 (初始化时填入) ----
    eye_to_hand_config: dict
    estun_config: dict
    realsense_config: dict
    eye_to_hand_serial: str         # Camera B serial
    camera_a_serial: str = ""       # Camera A serial (Phase 0 人进门检测)

    # ---- 运行时 ----
    person_detector: Optional[PersonDetector] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Hand-in-cup 检测 (TCP handler 控制开关, main loop 更新结果)
    hand_in_cup_active: bool = False
    hand_on_cup: Optional[bool] = None              # True/False/None(未就绪)
    hand_in_cup_detector: Optional["HandInCupDetector"] = None  # 共享实例, main.py 创建

    # GREETING 完成标志
    greeting_complete: threading.Event = field(default_factory=threading.Event)

    # HANDLE_GRASP_AND_POUR 完成标志
    handle_cup_complete: threading.Event = field(default_factory=threading.Event)

    # Camera A 检测器 (HandleGraspAndPourHandler 写入)
    handle_detector: Optional[RealSenseHandle6DDetector] = None
    cup_detector: Optional[RealSenseCup6DDetector] = None
    T_cam2ee: Optional[np.ndarray] = None

    # 人就位门控 (main loop 写入, TCP handler 读取)
    run_seated_detection: bool = False       # 主循环门控: 是否运行人就位检测
    is_person_seated: bool = False           # 每帧结果: 人是否就位

    # HandInCup 计时 (main loop 管理, TCP handler 读取)
    hand_in_cup_start_time: float = 0.0      # HandInCup 开始时间 (用于 ≤5s 超时)

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
    """IDLE: Camera B 椅子标定 → 人就位检测 → 触发 GREETING."""

    def __init__(self):
        super().__init__("IDLE")

    def on_enter(self, ctx: SharedContext):
        # PersonDetector 由 main.py 全生命周期管理, 此处仅做兜底
        if ctx.person_detector is None:
            cam_id = ctx.eye_to_hand_config.get("ugreen", {}).get("camera_id", 1)
            print(f"[INFO] 启动 PersonDetector (camera_id={cam_id})...")
            ctx.person_detector = PersonDetector(
                camera_id=cam_id,
                config_dict=ctx.eye_to_hand_config,
                start_camera=False,
            )
        # 重置运行时标志
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        ctx.handle_cup_complete.clear()
        ctx.greeting_complete.clear()
        ctx.run_seated_detection = False
        ctx.is_person_seated = False

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        # 直接进入 GREETING — Camera B 在此阶段才启动, 椅子/人就位在 GREETING 中检测
        return OrchestratorState.GREETING

    def on_exit(self, ctx: SharedContext):
        # PersonDetector pipeline 保持运行, Camera B 继续用于后续状态
        pass


# ============================================================
# GREETING 处理器 (Phase 0 打招呼)
# ============================================================

class GreetingHandler(StateHandler):
    """GREETING: Phase 0 打招呼 — 独立 Camera A pipeline + TCP 8600.

    4 步协议 (单连接, 移植自 estun_greeting_handle_cup_run.py):
      Step 1: recv 机器人位姿 → reply ZHAOSHOU1
      Step 2: recv 确认 → Camera A 等人 → close camera → reply ZHAOSHOU2
      Step 3: recv 确认 → 挥手 → reply WOSHOU
      Step 4: recv 确认 → 握手 → reply "Home"
    """

    def __init__(self):
        super().__init__("GREETING")
        self._server: Optional[socket.socket] = None
        self._pipeline = None        # Camera A 独立 pipeline (仅 color)
        self._yolo_model = None
        self._person_stable = 0
        self._person_required = 3    # 人进门检测连续帧数
        self._greeting_started = False

        # 打招呼位姿 (mm → m, start 时从 config 加载)
        self._zhaoshou1 = None
        self._zhaoshou2 = None
        self._woshou = None
        self._wave_cycles = 3
        self._wave_interval = 0.6
        # 握手压力检测 (从 config 读取)
        self._force_timeout = 30.0
        self._force_abs_threshold = 100
        self._force_delta_threshold = 100
        self._force_baseline_samples = 5
        self._force_poll_interval = 0.15
        self._force_release_timeout = 60.0
        self._force_release_delta = 50
        self._force_release_hold = 2.0

    # ================================================================
    # 生命周期
    # ================================================================

    def on_enter(self, ctx: SharedContext):
        self._start(ctx)
        ctx.greeting_complete.clear()

        # 启动 Camera B + 人就位检测 (GREETING 阶段开始, 人不在椅子附近)
        if ctx.person_detector is not None:
            ctx.person_detector.start_camera()
        ctx.run_seated_detection = True
        print("[GREETING] Camera B + 人就位检测已启动")

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        # 非阻塞 accept + 预览
        self._accept_tcp(ctx)

        if ctx.greeting_complete.is_set():
            print(f"[GREETING] Phase 0 完成 ✅")
            return OrchestratorState.HANDLE_GRASP_AND_POUR
        return None

    def on_exit(self, ctx: SharedContext):
        self._stop()
        print("[GREETING] 退出")

    # ================================================================
    # 初始化 / 清理
    # ================================================================

    def _start(self, ctx: SharedContext):
        estun = ctx.estun_config
        greeting_cfg = estun.get("greeting", {})

        # 加载打招呼位姿 (保持 mm, CRP TranStrToCpos lengthunit=mm)
        self._zhaoshou1 = [float(v) for v in greeting_cfg.get("zhaoshou1_pose", [0]*6)]
        self._zhaoshou2 = [float(v) for v in greeting_cfg.get("zhaoshou2_pose", [0]*6)]
        self._woshou = [float(v) for v in greeting_cfg.get("woshou_pose", [0]*6)]
        self._wave_cycles = int(greeting_cfg.get("wave_cycles", 3))
        self._wave_interval = float(greeting_cfg.get("wave_interval", 0.6))
        # 握手压力检测
        self._force_timeout = float(greeting_cfg.get("force_timeout", 30.0))
        self._force_abs_threshold = int(greeting_cfg.get("force_abs_threshold", 100))
        self._force_delta_threshold = int(greeting_cfg.get("force_delta_threshold", 100))
        self._force_baseline_samples = int(greeting_cfg.get("force_baseline_samples", 5))
        self._force_poll_interval = float(greeting_cfg.get("force_poll_interval", 0.15))
        self._force_release_timeout = float(greeting_cfg.get("force_release_timeout", 60.0))
        self._force_release_delta = int(greeting_cfg.get("force_release_delta", 50))
        self._force_release_hold = float(greeting_cfg.get("force_release_hold", 2.0))
        print(f"[GREETING] 位姿已加载: Z1={self._zhaoshou1} Z2={self._zhaoshou2} WS={self._woshou}")

        # ---- 灵巧手 (必须在 Phase 0 挥手/握手前初始化) ----
        hand_port = estun.get("hand", {}).get("port", "COM5")
        init_hand(hand_port=hand_port, initial_gesture="RESET")
        print("[GREETING] 灵巧手已初始化")

        # 打开 Camera A 独立 pipeline (仅 color, Phase 0 专用)
        # 注意: 必须用 Camera A serial (眼在手上), 不是 Camera B
        cam_a_serial = ctx.camera_a_serial or "342222071364"
        self._open_camera(cam_a_serial)

        # TCP 服务器 8600
        port = int(greeting_cfg.get("port", 8600))
        host = estun["tcp_server"]["host"]
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        self._server.bind((host, port))
        self._server.listen(1)
        print(f"[GREETING] TCP 服务器: {host}:{port}")

        self._greeting_started = False

    def _stop(self):
        """清理资源."""
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        self._close_camera()

    # ================================================================
    # Camera A 独立 pipeline (Phase 0 专用, 仅 color)
    # ================================================================

    def _open_camera(self, serial):
        """打开 Camera A 彩色流 (Phase 0 专用)."""
        import pyrealsense2 as rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self._pipeline.start(config)
        print("[GREETING] Camera A 彩色流已启动 (Phase 0)")

    def _get_camera_frame(self):
        """获取一帧 Camera A 彩色图像."""
        if self._pipeline is None:
            return None
        try:
            frames = self._pipeline.wait_for_frames(5000)
            color = frames.get_color_frame()
            if not color:
                return None
            return np.asanyarray(color.get_data())
        except Exception:
            return None

    def _close_camera(self):
        """关闭 Phase 0 Camera pipeline."""
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                print("[GREETING] Camera A 已关闭")
            except Exception:
                pass
            self._pipeline = None

    # ================================================================
    # YOLO 人检测 (进门)
    # ================================================================

    def _ensure_yolo(self):
        if self._yolo_model is not None:
            return self._yolo_model
        from ultralytics import YOLO
        model_path = os.path.join(_project_root, "model", "yolov8n.pt")
        if not os.path.exists(model_path):
            print("[GREETING] yolov8n.pt 未找到, 尝试自动下载...")
        self._yolo_model = YOLO(model_path)
        print("[GREETING] YOLOv8n 人检测模型已加载")
        return self._yolo_model

    def _person_detected(self, frame):
        """检测画面中是否有人 (class=0, 面积 ≥ 2%)."""
        model = self._ensure_yolo()
        result = model.predict(
            frame, classes=[0], conf=0.45, iou=0.45, imgsz=640, verbose=False,
        )[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return False
        h, w = frame.shape[:2]
        min_area = h * w * 0.02
        for box in boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy()
            x0, y0, x1, y1 = [float(v) for v in xyxy]
            if (x1 - x0) * (y1 - y0) >= min_area:
                return True
        return False

    def _wait_for_person(self, ctx: SharedContext, timeout=120.0):
        """YOLO 人检测, 连续 N 帧稳定则返回."""
        print("[GREETING] 等待人进门...")
        self._person_stable = 0
        deadline = time.time() + timeout
        while time.time() < deadline and not ctx.stop_event.is_set():
            frame = self._get_camera_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            detected = self._person_detected(frame)
            self._person_stable = self._person_stable + 1 if detected else 0
            print(f"\r[GREETING] 等人: stable={self._person_stable}/{self._person_required}", end="")
            if self._person_stable >= self._person_required:
                print("\n[GREETING] 检测到人进门!")
                return True
            time.sleep(0.03)
        print(f"\n[GREETING] 人检测超时 ({timeout}s)")
        return False

    # ================================================================
    # 挥手 + 握手 (复用 estun_greeting_handle_cup_run.py 逻辑)
    # ================================================================

    def _execute_wave(self, ctx: SharedContext):
        """Hello_1 ↔ Hello_2 ×N 挥手, 结束后 RESET 复位."""
        from Estun.utils_shared import _hand as _h
        from Omnihand.omnihand_gesture import Gesture

        if _h is not None:
            print(f"[GREETING] 挥手 {self._wave_cycles} 次 "
                  f"(间隔: {self._wave_interval}s)...")
            for i in range(self._wave_cycles):
                if ctx.stop_event.is_set():
                    print("[GREETING] 挥手被中断")
                    break
                print(f"  {i+1}/{self._wave_cycles}")
                _h.do_gesture(Gesture.HELLO_1, delay=0.5)
                time.sleep(self._wave_interval / 2)
                if ctx.stop_event.is_set():
                    break
                _h.do_gesture(Gesture.HELLO_2, delay=0.5)
                time.sleep(self._wave_interval / 2)

            print("[GREETING] RESET — 挥手结束, 复位")
            _h.do_gesture(Gesture.RESET, delay=0.5)
        else:
            print("[GREETING] 模拟挥手")
            time.sleep(self._wave_cycles * (0.5 + self._wave_interval / 2) * 2)

    def _read_hand_back_pressure(self):
        from Estun.utils_shared import _hand as _h
        if _h is None:
            return 0
        sensors = _h.serial.read_all_sensor_group_3()
        return int(np.asarray(sensors.get("back", []), dtype=np.int32).sum())

    def _wait_for_handshake_force(self, ctx: SharedContext):
        """等待手背压力 (基线+增量双阈值, 参数从 config 读取)."""
        baseline_samples = []
        for _ in range(max(1, self._force_baseline_samples)):
            baseline_samples.append(self._read_hand_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(baseline_samples, dtype=np.float64)))
        deadline = time.time() + self._force_timeout
        while time.time() < deadline and not ctx.stop_event.is_set():
            value = self._read_hand_back_pressure()
            delta = value - baseline
            abs_trigger = (baseline < self._force_abs_threshold * 0.5
                           and value >= self._force_abs_threshold)
            delta_trigger = (delta >= self._force_delta_threshold)
            print(f"[GREETING] 压力 back={value} baseline={baseline:.0f} delta={delta:.0f} "
                  f"abs_trig={abs_trigger} delta_trig={delta_trigger}")
            if abs_trigger or delta_trigger:
                print(f"[GREETING] 检测到抓握!")
                return True
            time.sleep(self._force_poll_interval)
        print(f"[GREETING] 压力超时 ({self._force_timeout}s)")
        return False

    def _wait_for_handshake_release(self, ctx: SharedContext):
        """等待松手: 连续 hold 秒内 delta < threshold (参数从 config 读取)."""
        baseline_samples = []
        for _ in range(max(1, self._force_baseline_samples)):
            baseline_samples.append(self._read_hand_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(baseline_samples, dtype=np.float64)))
        released_since = None
        deadline = time.time() + self._force_release_timeout
        while time.time() < deadline and not ctx.stop_event.is_set():
            value = self._read_hand_back_pressure()
            delta = value - baseline
            released = (delta <= self._force_release_delta)
            print(f"[GREETING] 压力 back={value} baseline={baseline:.0f} delta={delta:.0f} "
                  f"released={released}")
            now = time.time()
            if released:
                if released_since is None:
                    released_since = now
                if now - released_since >= self._force_release_hold:
                    print(f"[GREETING] 已松手!")
                    return True
            else:
                released_since = None
            time.sleep(self._force_poll_interval)
        print(f"[GREETING] 松手超时 ({self._force_release_timeout}s)")
        return False

    def _execute_handshake(self, ctx: SharedContext):
        """RESET → 等压力 → HAND_SHAKE → 等松手 → RESET."""
        from Estun.utils_shared import _hand as _h
        from Omnihand.omnihand_gesture import Gesture
        if _h is not None:
            _h.do_gesture(Gesture.RESET, delay=0.5)
            print("[GREETING] RESET — 等待握手...")
            self._wait_for_handshake_force(ctx)
            _h.do_gesture(Gesture.HAND_SHAKE, delay=1.0)
            print("[GREETING] HAND_SHAKE")
            self._wait_for_handshake_release(ctx)
            _h.do_gesture(Gesture.RESET, delay=0.5)
            print("[GREETING] RESET — 握手结束")
        else:
            print("[GREETING] 模拟握手")
            time.sleep(3.0)

    # ================================================================
    # TCP
    # ================================================================

    def _accept_tcp(self, ctx: SharedContext):
        """非阻塞 accept + 预览."""
        if self._server is None or self._greeting_started:
            return
        try:
            conn, addr = self._server.accept()
            print(f"[GREETING] 机器人已连接: {addr}")
            self._greeting_started = True
            t = threading.Thread(
                target=self._handle_session,
                args=(conn, addr, ctx),
                daemon=True,
            )
            t.start()
        except BlockingIOError:
            pass

    @staticmethod
    def _fmt_pose6(pose6):
        """6元素位姿 → "x,y,z,A,B,C" (mm, deg) — CRP TranStrToCpos lengthunit=mm."""
        return f"{pose6[0]:.1f},{pose6[1]:.1f},{pose6[2]:.1f},{pose6[3]:.2f},{pose6[4]:.2f},{pose6[5]:.2f}"

    def _handle_session(self, conn: socket.socket, addr, ctx: SharedContext):
        """4 步协议 (daemon 线程)."""
        print(f"[GREETING] 客户端: {addr}")
        try:
            # ---- Step 1: recv → reply ZHAOSHOU1 ----
            conn.settimeout(5.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 1: 未收到数据")
                return
            msg = data.decode("utf-8", errors="replace").strip()
            print(f"[GREETING] Step 1 recv: {msg}")
            r = self._fmt_pose6(self._zhaoshou1) + "\n"
            conn.sendall(r.encode("utf-8"))
            print(f"[GREETING] Step 1 reply ZHAOSHOU1: {r.strip()}")

            # ---- Step 2: recv → 等人 → close camera → reply ZHAOSHOU2 ----
            conn.settimeout(130.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 2: 未收到数据")
                return
            print(f"[GREETING] Step 2 recv: {data.decode('utf-8', errors='replace').strip()}")
            self._wait_for_person(ctx, timeout=120.0)
            self._close_camera()
            r = self._fmt_pose6(self._zhaoshou2) + "\n"
            conn.sendall(r.encode("utf-8"))
            print(f"[GREETING] Step 2 reply ZHAOSHOU2: {r.strip()}")

            # ---- Step 3: recv → 挥手 → reply WOSHOU ----
            conn.settimeout(30.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 3: 未收到数据")
                return
            print(f"[GREETING] Step 3 recv: {data.decode('utf-8', errors='replace').strip()}")
            self._execute_wave(ctx)
            r = self._fmt_pose6(self._woshou) + "\n"
            conn.sendall(r.encode("utf-8"))
            print(f"[GREETING] Step 3 reply WOSHOU: {r.strip()}")

            # ---- Step 4: recv → 握手 → reply "Home" ----
            conn.settimeout(90.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 4: 未收到数据")
                return
            print(f"[GREETING] Step 4 recv: {data.decode('utf-8', errors='replace').strip()}")
            self._execute_handshake(ctx)
            conn.sendall(b"Home\n")
            print("[GREETING] Step 4 reply: Home")

        except socket.timeout:
            print(f"[GREETING] Step 超时")
        except Exception as e:
            print(f"[GREETING] 异常: {e}")
        finally:
            self._close_camera()
            try:
                conn.close()
            except Exception:
                pass
            ctx.greeting_complete.set()
            print(f"[GREETING] 客户端已断开: {addr}")


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

        # ---- EyeToHand 递送参数 (全示教固定, start 时加载) ----
        self._x_teach = 0.0
        self._y_teach = 0.0
        self._z_teach = 0.0
        self._a_teach = 0.0
        self._b_teach = 0.0
        self._c_teach = 0.0

        # ---- TCP 超时参数 (start 时从 config 加载) ----
        self._phase_b_timeout = 30.0

    # ================================================================
    # 生命周期
    # ================================================================

    def on_enter(self, ctx: SharedContext):
        self._start(ctx)
        ctx.handle_cup_complete.clear()

    def on_update(self, ctx: SharedContext) -> Optional[OrchestratorState]:
        # 获取 Camera A 帧 (Phase B 期间有帧, 之后为 None)
        if ctx.handle_detector is not None:
            color, depth, depth_colormap, depth_frame = self._get_frames(ctx)
            ctx.frames_a = (color, depth, depth_colormap, depth_frame)
            self._detect(ctx)
        else:
            ctx.frames_a = None

        # 非阻塞 accept TCP
        self._accept_tcp(ctx)

        # 完成时退出程序 (stop_event 由 _handle_client_impl 设置)
        if ctx.stop_event.is_set():
            return None
        return None

    def on_exit(self, ctx: SharedContext):
        self._stop(ctx)

    # ================================================================
    # 初始化 / 清理
    # ================================================================

    def _start(self, ctx: SharedContext):
        """加载偏移参数 + 灵巧手 + TCP 8080. Camera A 检测器在 Phase B 按需打开."""
        cfg = ctx.realsense_config
        estun = ctx.estun_config
        eye_cfg = ctx.eye_to_hand_config

        # ---- 加载偏移参数 (从 estun_config) ----
        self._load_offsets(cfg, estun)

        # ---- 加载 EyeToHand 递送参数 (全示教固定) ----
        delivery = eye_cfg.get("delivery", {})
        self._x_teach = delivery.get("X_TEACH", 0.0)
        self._y_teach = delivery.get("Y_TEACH", 0.0)
        self._z_teach = delivery.get("Z_TEACH", 0.0)
        self._a_teach = delivery.get("A_TEACH", 0.0)
        self._b_teach = delivery.get("B_TEACH", 0.0)
        self._c_teach = delivery.get("C_TEACH", 0.0)
        print(f"[INFO] 递送位姿 (全示教): X={self._x_teach:.3f} Y={self._y_teach:.3f} Z={self._z_teach:.3f} ABC=({self._a_teach:.1f},{self._b_teach:.1f},{self._c_teach:.1f})")

        # ---- 灵巧手 (GREETING 已初始化, 不再重试使能) ----
        self._hand_available = is_hand_available()

        # ---- TCP 服务器 ----
        host = estun["tcp_server"]["host"]
        port = estun["tcp_server"]["port"]
        self._phase_b_timeout = float(estun["tcp_server"].get("phase_b_timeout", 30.0))
        self._server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._server.setblocking(False)
        self._server.bind((host, port))
        self._server.listen(5)
        print(f"[INFO] TCP 服务器: {host}:{port}")

        # ---- 重置运行时标志 ----
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        self._frame_counter = 0

        # Camera A / 稳定器 / 标定 在 _handle_client_impl Phase B 按需初始化

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
        # Camera A 已在 Phase B 关闭; 安全兜底
        if ctx.handle_detector is not None:
            try:
                ctx.handle_detector.pipeline.stop()
            except Exception:
                pass
        ctx.handle_detector = None
        ctx.cup_detector = None
        ctx.T_cam2ee = None
        if self._hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        ctx.hand_in_cup_active = False
        ctx.hand_on_cup = None
        ctx.run_seated_detection = False

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
          - Cup P3 XYZ/ABC 全示教固定 (EyeToHand/config.yaml delivery 段, 3D→1D 标定暂未接入)
          - Phase D: 递送确认 + Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
        """
        print(f"[INFO] 客户端已连接: {addr}")

        # ---- 嵌套清理函数 (确保任何退出路径都执行清理) ----
        def _cleanup():
            ctx.hand_in_cup_active = False
            ctx.hand_on_cup = None
            try:
                conn.close()
            except Exception:
                pass
            PoseStabilizer.on_disconnect()
            ctx.handle_cup_complete.set()
            print(f"[INFO] 客户端已断开: {addr}")

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
                _cleanup()
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
                _cleanup()
                return
        except socket.timeout:
            print("[WARN] 等待机器人初始位姿超时, 断开连接")
            _cleanup()
            return
        except Exception as e:
            print(f"[WARN] 接收初始位姿异常: {e}")
            _cleanup()
            return

        R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
        t_ee_base = np.array(robot_xyz, dtype=float)

        # ---- [Phase B] 打开 Camera A, 检测 Handle+Cup, 发送后关闭 ----
        print("[INFO] 打开 Camera A (Handle+Cup 检测)...")
        cfg = ctx.realsense_config
        estun = ctx.estun_config
        ctx.handle_detector = RealSenseHandle6DDetector(config_dict=cfg)
        ctx.handle_detector.last_known_pose = None
        ctx.cup_detector = RealSenseCup6DDetector(config_dict=cfg, start_pipeline=False)
        ctx.cup_detector.color_intrinsics = ctx.handle_detector.color_intrinsics
        ctx.cup_detector.depth_scale = ctx.handle_detector.depth_scale
        ctx.cup_detector.last_known_pose = None

        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml")
        )
        ctx.T_cam2ee = load_cam2ee(calib_path)

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

        print("[INFO] 等待稳定目标位姿 (Handle + Cup)...")
        detector = ctx.handle_detector
        cup_detector = ctx.cup_detector
        T_cam2ee = ctx.T_cam2ee
        if detector is None:
            print("[ERROR] handle_detector 未初始化")
            _cleanup()
            return
        if cup_detector is None:
            print("[ERROR] cup_detector 未初始化")
            _cleanup()
            return
        if T_cam2ee is None:
            print("[ERROR] T_cam2ee 未加载")
            _cleanup()
            return

        target_sent = False
        wait_start = time.time()
        while not target_sent and not ctx.stop_event.is_set() and (time.time() - wait_start) < self._phase_b_timeout:
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
                        if B1 < 90:
                            print(f"[CLAMP] Handle P1 B {B1:.1f} → 89.0")
                            B1 = 89.0
                        elif B1 > 90:
                            print(f"[CLAMP] Handle P1 B {B1:.1f} → 91.0")
                            B1 = 91.0

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
                #   P3:     delivery = (X_TEACH, Y_TEACH, Z_TEACH, ABC_TEACH)  # 全示教固定
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

                # Cup P3 = 递送位姿 (全示教固定值, 暂废弃标定)
                c_pos_base_3 = (
                    self._x_teach,
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
                    _cleanup()
                    return
            else:
                time.sleep(0.1)

        if not target_sent:
            print("[WARN] 超时: 未获得稳定目标位姿, 断开连接")
            _cleanup()
            return

        # ---- Camera A 已不再需要, 关闭 pipeline ----
        if ctx.handle_detector is not None:
            try:
                ctx.handle_detector.pipeline.stop()
                print("[INFO] Camera A pipeline 已关闭 (位姿已发送)")
            except Exception:
                pass
            ctx.handle_detector = None
            ctx.cup_detector = None
            ctx.T_cam2ee = None

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

                elif cmd == "GRIP":
                    if self._hand_available:
                        grip_object()
                    print("[INFO] GRIP")

                elif cmd == "CUP":
                    if not cup_p3_sent and cup_pose_data is not None:
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
                    # 停止人就位检测
                    ctx.run_seated_detection = False

                    if not ctx.is_person_seated:
                        # 无人就位 → 跳过 HandInCup, 直接 ABORT
                        print("[INFO] READY: 人未就位 → HandInCup=false, ABORT")
                        hand_judged = True
                        abort_mode = True
                        try:
                            conn.sendall(b"[0,0,0,0,0,0]")
                            print("[SEND] ABORT ([0,0,0,0,0,0])")
                        except Exception as e:
                            print(f"[WARN] 发送 ABORT 失败: {e}")
                            break
                    else:
                        # 人就位 → 启动 HandInCup, 最长检测 5s
                        print("[INFO] READY: 人就位 → 启动 HandInCup (≤5s)")
                        ctx.hand_in_cup_active = True
                        ctx.hand_in_cup_start_time = time.time()
                        if ctx.hand_in_cup_detector is not None:
                            ctx.hand_in_cup_detector.reset()
                        # 等待 main loop 完成检测 (≤5s, TRUE 提前结束)
                        HAND_IN_CUP_TIMEOUT = 5.0
                        while not ctx.stop_event.is_set():
                            if ctx.hand_on_cup:
                                print("[INFO] HandInCup: TRUE (提前检测到)")
                                break
                            if time.time() - ctx.hand_in_cup_start_time > HAND_IN_CUP_TIMEOUT:
                                ctx.hand_on_cup = False
                                print("[INFO] HandInCup: 超时 → FALSE")
                                break
                            time.sleep(0.05)
                        ctx.hand_in_cup_active = False
                        print(f"[INFO] HandInCup 结束: {'TRUE' if ctx.hand_on_cup else 'FALSE'}")

                elif cmd == "RELEASE":
                    if not ctx.hand_in_cup_active:
                        # ---- Phase C: 松把手 (READY 之前的 RELEASE) ----
                        if self._hand_available:
                            release_object()
                        print("[INFO] RELEASE (松把手)")
                    elif not hand_judged:
                        # ---- Phase D 第一次 RELEASE: Hand-in-cup 判定 ----
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
                                conn.sendall(b"<0,0,0,0,0,0>")
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
                                conn.sendall(b"[0,0,0,0,0,0]")
                                print("[SEND] ABORT ([0,0,0,0,0,0])")
                            except Exception as e:
                                print(f"[WARN] 发送 ABORT 失败: {e}")
                                break
                            # 机器人: P1 → RELEASE → P2 → RESET → Home → SocketClose
                            # 循环继续, 等待后续 RELEASE + RESET
                    else:
                        # ---- ABORT 路径第二次 RELEASE (放回后松杯) ----
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

        # ---- 全流程完成, 退出程序 ----
        PoseStabilizer.on_disconnect()
        ctx.handle_cup_complete.set()
        print("[INFO] 全流程完成, 退出程序")
        ctx.stop_event.set()
