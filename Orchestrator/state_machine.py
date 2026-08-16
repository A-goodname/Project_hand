# -*- coding: utf-8 -*-
"""
Orchestrator — 共享状态 + TCP 服务
==================================
双相机 5 阶段顺序调度: greeting → seated_detect → handle_detect → seated_check → hand_in_cup

TCP 服务:
  - GreetingHandler:   Phase 0 打招呼 (TCP 8600, 招手→挥手→握手)
  - HandleGraspAndPourHandler: Phase A→D 抓取倒水递送 (TCP 8080)

与 estun_handle_cup_run.py 的关键差异:
  - Cup P3 XYZ/ABC 全示教固定 (EyeToHand/config.yaml delivery 段)
  - 递送后增加 Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
  - Phase 0 打招呼 (TCP 8600)
  - 单 TCP 连接, SocketClose 后退出
"""

import os
import socket
import sys
import threading
import time
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
from Orchestrator.handle_cup_detector import HandleCupDetector
from Orchestrator.logger import session_log
from Orchestrator.rot_math import (  # B2 动态姿态链 (2026-08-16)
    quaternion_geodesic_deg,
    quaternion_mean,
    quaternion_to_rotation_matrix,
)
from Orchestrator.site_gate import run_site_gate  # audit 修复 1 (2026-08-16)
from Orchestrator.session_barrier import (
    BarrierReject,
    SessionBarrier,
    SessionState,
    WristObservation,
    calib_fingerprint,
)
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


class CalibrationError(RuntimeError):
    """标定失败 — 阻止启动 (D0 门禁). 调用方必须中止启动, 不得降级."""


# 标定质量门槛 (D0): 样本数下限 / 重投影误差上限
_CALIB_MIN_SAMPLES = 10
_CALIB_MAX_REPROJ_PX = 0.5


def load_cam2ee(yaml_path):
    """从标定结果加载 相机→末端法兰 变换矩阵 (4x4 齐次矩阵).

    D0 门禁: 任何失败抛 CalibrationError, 绝不返回 np.eye(4) 或默认 Tsai.
    检查项: 文件存在 / YAML 有效 / use_method 存在 / 方法存在 /
            非有限刚体 4x4 / 单位变换 / 样本与重投影质量.
    """
    if not os.path.exists(yaml_path):
        raise CalibrationError(f"标定文件不存在: {yaml_path}")
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        raise CalibrationError(f"标定 YAML 解析失败: {e}")
    if not isinstance(data, dict) or not data.get("calibration"):
        raise CalibrationError("标定 YAML 缺少 calibration 段")
    calib = data["calibration"]

    # 质量门槛 (样本数 / 重投影误差)
    num_samples = calib.get("num_samples")
    reproj = calib.get("reprojection_error_px")
    try:
        if num_samples is not None and int(num_samples) < _CALIB_MIN_SAMPLES:
            raise CalibrationError(f"标定样本数不足: {num_samples} < {_CALIB_MIN_SAMPLES}")
        if reproj is not None and float(reproj) > _CALIB_MAX_REPROJ_PX:
            raise CalibrationError(f"标定重投影误差超标: {reproj} > {_CALIB_MAX_REPROJ_PX}px")
    except (TypeError, ValueError) as exc:
        raise CalibrationError(f"标定质量字段非法: {exc}") from exc

    hand_eye = calib.get("handeye")
    if not isinstance(hand_eye, dict) or not hand_eye:
        raise CalibrationError("标定缺少 handeye 段")
    use_method = hand_eye.get("use_method")
    if not use_method:
        raise CalibrationError("标定缺少 handeye.use_method")
    method_data = hand_eye.get(use_method)
    if not isinstance(method_data, dict) or not method_data:
        raise CalibrationError(f"标定方法 {use_method!r} 不存在或为空")
    T_raw = method_data.get("T_cam_to_gripper")
    if T_raw is None:
        raise CalibrationError(f"标定方法 {use_method!r} 缺少 T_cam_to_gripper")
    try:
        T = np.array(T_raw, dtype=float)
    except Exception as e:
        raise CalibrationError(f"T_cam_to_gripper 无法转为数值: {e}")
    if T.shape != (4, 4):
        raise CalibrationError(f"T_cam_to_gripper 形状错误: {T.shape}")
    if not np.all(np.isfinite(T)):
        raise CalibrationError("T_cam_to_gripper 含非有限值")

    # 刚体检查: 旋转部分正交且行列式 ≈ 1
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if not (0.9 <= det <= 1.1):
        raise CalibrationError(f"旋转矩阵行列式异常: {det:.4f}")
    RRT = R @ R.T
    if np.max(np.abs(RRT - np.eye(3))) > 1e-3:
        raise CalibrationError("旋转矩阵非正交 (非刚体)")
    if np.max(np.abs(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6:
        raise CalibrationError("齐次变换最后一行异常")
    # 单位变换拒绝: 平移近零 + 旋转近单位
    if (np.max(np.abs(T[:3, 3])) < 1e-6 and
            np.max(np.abs(R - np.eye(3))) < 1e-6):
        raise CalibrationError("单位变换标定, 拒绝使用")

    t = T[:3, 3]
    print(f"[INFO] 已加载标定矩阵: {use_method}.T_cam_to_gripper, 平移: {t} m")
    return T


# ============================================================
# 位姿稳定器 (与 estun_handle_cup_run.py 一致)
# ============================================================

_gf = 0  # 全局代际


class PoseStabilizer:
    """位姿稳定器 — 缓存最近 N 帧, 位置+角度波动低于阈值时输出均值.

    B2 扩展 (2026-08-16): 支持四元数姿态缓冲, 以**测地距离**判定姿态连续性
    (PCA 180° 翻转/符号歧义场景下欧拉角范围判定会失真); get_mean_attitude
    输出 Markley 四元数均值旋转; continuity_report 供轨迹日志/现场证据.
    """

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

    def add(self, pos, euler, quat=None):
        """quat=(w,x,y,z) 可选; 提供时参与测地连续性判定."""
        self._buffer.append((np.array(pos, dtype=float),
                             np.array(euler, dtype=float),
                             None if quat is None else np.array(quat, dtype=float)))
        if len(self._buffer) > self.window:
            self._buffer.pop(0)

    def _max_geodesic_deg(self):
        """窗口内四元数两两测地距离最大值 (°); 不足两个或无四元数 → None."""
        quats = [q for _, _, q in self._buffer if q is not None]
        if len(quats) < 2:
            return None
        maxd = 0.0
        for i in range(len(quats)):
            for j in range(i + 1, len(quats)):
                d = quaternion_geodesic_deg(quats[i], quats[j])
                if d > maxd:
                    maxd = d
        return float(maxd)

    def is_stable(self):
        if len(self._buffer) < self.window:
            return False
        poses = np.array([p for p, _, _ in self._buffer])
        angles = np.array([a for _, a, _ in self._buffer])
        pos_range = np.ptp(poses, axis=0).max()
        angle_range = np.ptp(angles, axis=0).max()
        geo = self._max_geodesic_deg()
        if geo is not None and geo > self.angle_thresh:
            return False
        return pos_range < self.pos_thresh and angle_range < self.angle_thresh

    def get_mean(self):
        if not self._buffer:
            return None, None
        poses = np.array([p for p, _, _ in self._buffer])
        angles = np.array([a for _, a, _ in self._buffer])
        return poses.mean(axis=0), angles.mean(axis=0)

    def get_mean_attitude(self):
        """窗口四元数 Markley 均值 → (R 3x3, quat (w,x,y,z)); 无四元数 → (None, None)."""
        quats = [q for _, _, q in self._buffer if q is not None]
        if not quats:
            return None, None
        q_mean = quaternion_mean(quats)
        return quaternion_to_rotation_matrix(q_mean), q_mean

    def continuity_report(self):
        """连续性报告: 帧数/位置范围/欧拉范围/最大测地距离/判定 (供轨迹日志)."""
        if not self._buffer:
            return None
        poses = np.array([p for p, _, _ in self._buffer])
        angles = np.array([a for _, a, _ in self._buffer])
        geo = self._max_geodesic_deg()
        return {
            "frames": len(self._buffer),
            "pos_range_m": float(np.ptp(poses, axis=0).max()),
            "euler_range_deg": float(np.ptp(angles, axis=0).max()),
            "max_geodesic_deg": geo,
            "verdict": "stable" if self.is_stable() else "unstable",
        }

    def reset(self):
        self._buffer = []


# ============================================================
# 共享上下文
# ============================================================

@dataclass
class SharedContext:
    """主循环与 TCP daemon 线程间的共享状态.

    配置字段 (main.py 填充):
      - eye_to_hand_config: EyeToHand/config.yaml
      - estun_config:       Estun/estun_config.yaml["estun"]
      - realsense_config:   RealSense/config.yaml (Camera A, Handle+Cup)
      - eye_to_hand_serial: Camera B serial (已废弃)

    运行时字段:
      - person_detector:    Camera B 人就位检测器 (main.py 全生命周期)
      - hand_in_cup_detector: HandInCup 判定 (main.py 创建, 主循环注入帧)
      - handle_cup_detector: Camera A Handle+Cup 检测器 (GREETING 创建, HANDLE 复用)
      - stop_event:         程序退出信号
      - frames_a / frames_b: 可视化帧缓存
    """
    # ---- 配置 (初始化时填入) ----
    eye_to_hand_config: dict
    estun_config: dict
    realsense_config: dict
    eye_to_hand_serial: str         # Camera B serial (已废弃)
    camera_a_serial: str = ""       # Camera A serial

    # ---- 运行时 ----
    person_detector: Optional[PersonDetector] = None
    stop_event: threading.Event = field(default_factory=threading.Event)

    # Hand-in-cup 检测 (main loop 更新结果, TCP handler 读取)
    hand_on_cup: Optional[bool] = None              # True/False/None(未就绪)
    hand_in_cup_start_time: float = 0.0
    hand_in_cup_detector: Optional["HandInCupDetector"] = None

    # 人就位 (main loop 更新结果, TCP handler 读取)
    is_person_seated: bool = False
    _person_stable: int = 0           # 纯人检测连续帧计数 (only_person_detect 分支)

    # ---- Phase 过渡事件 (TCP daemon 线程写入, main loop 读取) ----
    greeting_done: threading.Event = field(default_factory=threading.Event)    # Phase 0 TCP 完成
    targets_sent: threading.Event = field(default_factory=threading.Event)     # Phase B pack_three 发送完成
    ready_received: threading.Event = field(default_factory=threading.Event)   # TCP 收到 READY
    hand_judged: threading.Event = field(default_factory=threading.Event)      # CONTINUE/ABORT 发送完成

    # ---- Camera A 检测器 (GreetingHandler 创建, HandleHandler 复用) ----
    handle_cup_detector: Optional[HandleCupDetector] = None
    _camera_a_to_close: Optional[HandleCupDetector] = None  # 延迟关闭引用, main 线程处理
    T_cam2ee: Optional[np.ndarray] = None

    # ---- 可视化帧缓存 (main loop 写入) ----
    frames_a: Optional[tuple] = None  # Camera A (Handle)
    frames_b: Optional[tuple] = None  # Camera B (Person/HandInCup)


# ============================================================
# GREETING TCP 服务 (Phase 0 打招呼)
# ============================================================

class GreetingHandler:
    """GREETING: Phase 0 打招呼 — 独立 Camera A pipeline + TCP 8600.

    4 步协议 (单连接, 移植自 estun_greeting_handle_cup_run.py):
      Step 1: recv 机器人位姿 → reply ZHAOSHOU1
      Step 2: recv 确认 → Camera A 等人 → close camera → reply ZHAOSHOU2
      Step 3: recv 确认 → 挥手 → reply WOSHOU
      Step 4: recv 确认 → 握手 → reply "Home"
    """

    def __init__(self):
        self._server: Optional[socket.socket] = None
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
    # 生命周期 (由 main.py 调用)
    # ================================================================

    def start(self, ctx: SharedContext):
        """初始化并启动 GREETING Phase 0. 由 main loop 在 greeting phase 开始时调用."""
        self._start(ctx)
        ctx.greeting_done.clear()
        ctx.targets_sent.clear()
        ctx.ready_received.clear()
        ctx.hand_judged.clear()
        ctx.is_person_seated = False
        ctx.hand_on_cup = None
        # Camera B 预加载 (UGREEN, CPU YOLO, 不抢 GPU)
        if ctx.person_detector is not None:
            ctx.person_detector.start_camera()
            print("[GREETING] Camera B 已启动")

    def update_preview(self, ctx: SharedContext):
        """每帧调用: 非阻塞 accept + Camera A 预览帧."""
        self._accept_tcp(ctx)

        det = ctx.handle_cup_detector
        if det is not None:
            try:
                color, depth, depth_cmap, _ = det.get_frames()
                ctx.frames_a = (color, depth, depth_cmap, None) if color is not None else None
            except Exception:
                pass

    def stop(self, ctx: SharedContext):
        """清理 GREETING 资源 (不关 Camera A pipeline)."""
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

        # Camera A — 复用 HandleCupDetector pipeline (color+depth), Phase 0 只用 color
        # 注意: 必须用 Camera A serial (眼在手上), 不是 Camera B
        cam_a_serial = ctx.camera_a_serial or "342222071364"
        if ctx.handle_cup_detector is None:
            import torch
            _dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            _orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
            _orch_cfg = {}
            try:
                with open(_orch_path, "r", encoding="utf-8") as f:
                    _orch_cfg = yaml.safe_load(f)
            except Exception:
                pass
            ctx.handle_cup_detector = HandleCupDetector(
                orch_config=_orch_cfg,
                serial=cam_a_serial, device=_dev,
            )
            print("[GREETING] HandleCupDetector pipeline 已启动 (Phase 0 复用 color)")

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
        # Camera A pipeline 不关闭 — 由 HANDLE 继续使用

    # ================================================================
    # Camera A pipeline — 复用 HandleCupDetector (Phase 0 只用 color)
    # ================================================================

    @staticmethod
    def _get_camera_frame(ctx):
        """从 HandleCupDetector pipeline 获取 color 帧."""
        det = ctx.handle_cup_detector
        if det is None:
            return None
        color, _, _, _ = det.get_frames()
        return color

    @staticmethod
    def _close_camera(ctx):
        """不关闭 pipeline — HandleCupDetector 由 HANDLE 继续使用."""
        pass

    # _open_camera 已移除, 由 _start 中创建 HandleCupDetector 替代
    _open_camera = None  # type: ignore[assignment]

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
            frame = self._get_camera_frame(ctx)
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
                time.sleep(self._wave_interval / 4)
                if ctx.stop_event.is_set():
                    break
                _h.do_gesture(Gesture.HELLO_2, delay=0.5)
                time.sleep(self._wave_interval / 4)

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
            self._close_camera(ctx)
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
            self._close_camera(ctx)
            try:
                conn.close()
            except Exception:
                pass
            ctx.greeting_done.set()
            print(f"[GREETING] 客户端已断开: {addr}")


# ============================================================
# HANDLE TCP 服务 (Phase A→D 抓取+倒水+递送)
# ============================================================

class HandleGraspAndPourHandler:
    """HANDLE: 单 TCP Session 全流程.

    内部包含 Handle+Cup 检测 → TCP 通信 → 递送确认.
    全程使用一个 TCP 连接, SocketClose 后完成.
    """

    def __init__(self):
        # ---- 运行时对象 ----
        self._server: Optional[socket.socket] = None
        self._hand_available = False

        # ---- 跳帧控制 ----
        self._frame_counter = 0
        self._cached_result = None
        self._last_color = None
        self._last_depth_frame = None
        self._last_depth_image = None
        self._last_depth_colormap = None

        # ---- 位姿稳定器 ----
        self._handle_stabilizer: Optional[PoseStabilizer] = None
        self._cup_stabilizer: Optional[PoseStabilizer] = None

        # ---- D0 会话屏障：只持有接入互斥，状态留在连接线程局部 ----
        self._session_lock = threading.Lock()
        self._session_active = False
        self._lease_epoch = 0

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
    # 生命周期 (由 main.py 调用)
    # ================================================================

    def start(self, ctx: SharedContext):
        """初始化并启动 HANDLE TCP 服务. 由 main loop 在 handle_detect phase 开始时调用."""
        self._start(ctx)

    def get_frames_and_detect(self, ctx: SharedContext):
        """每帧调用: 获取 Camera A 帧 + 跳帧检测 Handle+Cup."""
        if ctx.handle_cup_detector is not None:
            color, depth, depth_colormap, depth_frame = self._get_frames(ctx)
            ctx.frames_a = (color, depth, depth_colormap, depth_frame)
            self._detect(ctx)
        else:
            ctx.frames_a = None

    def accept_tcp(self, ctx: SharedContext):
        """非阻塞 accept TCP 8080."""
        self._accept_tcp(ctx)

    def stop(self, ctx: SharedContext):
        """清理 HANDLE 资源."""
        self._stop(ctx)

    # ================================================================
    # 初始化 / 清理
    # ================================================================

    def _start(self, ctx: SharedContext):
        """加载偏移参数 + 灵巧手 + TCP 8080. Camera A 检测器在 Phase B 按需打开."""
        cfg = ctx.realsense_config
        estun = ctx.estun_config
        eye_cfg = ctx.eye_to_hand_config

        # ---- 加载 Orchestrator 独立配置 ----
        _orch_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        self._orch_cfg = {}
        try:
            with open(_orch_path, "r", encoding="utf-8") as f:
                self._orch_cfg = yaml.safe_load(f)
        except Exception:
            pass

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
        session_log(f"灵巧手状态: {'可用' if self._hand_available else '不可用(模拟)'}")

        # ---- D0 门禁: 标定加载必须先于 TCP server (失败则阻止启动) ----
        calib_path = os.path.join(
            _project_root,
            estun.get("calibration", {}).get("result_path", "calib_data/calibration_result.yaml")
        )
        ctx.T_cam2ee = load_cam2ee(calib_path)
        session_log(f"标定已加载 (D0 门禁): {calib_path}")

        # ---- 现场门禁 site_gate (audit 修复 1): 标定后、bind 前, 只读检查 ----
        _sg = self._orch_cfg.get("site_gate", {})
        _sg_failures = run_site_gate(
            preparation_ok=bool(_sg.get("preparation_ok", False)),
            camera_a_serial=ctx.camera_a_serial or "",
            camera_b_path=_sg.get("camera_b_path", "usb-0:4.2"),
            hand_serial_path=_sg.get(
                "hand_serial_path",
                "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00"),
            eth_iface=_sg.get("eth_iface", "enp36s0f1"),
            block_link_down=bool(_sg.get("block_link_down", False)),
        )
        if _sg_failures:
            raise RuntimeError(
                f"site gate 阻止启动 ({len(_sg_failures)} 项): {_sg_failures[0]}")
        session_log("现场门禁通过 (相机身份/物理口/carrier 活体)")

        # ---- TCP 服务器 (标定通过后才监听运动 socket) ----
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
        ctx.hand_on_cup = None
        self._frame_counter = 0

        # ---- Camera A: 复用 GREETING 创建的 HandleCupDetector, 只补稳定器+标定 ----
        # 兜底: 如果 GREETING 没创建, 在此创建
        if ctx.handle_cup_detector is None:
            import torch
            _dev = "cuda:0" if torch.cuda.is_available() else "cpu"
            ctx.handle_cup_detector = HandleCupDetector(
                orch_config=self._orch_cfg,
                serial=ctx.camera_a_serial, device=_dev,
            )
            print("[INFO] Camera A HandleCupDetector 已创建 (兜底)")

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

        session_log("Camera A 稳定器已就绪 (标定已由 D0 门禁加载)")

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

        # Cup handle_side (从 orch config)
        self._cup_handle_side = self._orch_cfg.get("cup_handle_side", "back")
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

        # Cup geometry (从 orch config)
        self._cup_rim_radius = float(self._orch_cfg.get("cup_rim_radius", 0.035))
        self._cup_height = float(self._orch_cfg.get("cup_height", 0.100))

        # Orientation + 把手方向 (从 orch config)
        self._orientation_mode = self._orch_cfg.get("orientation_mode", "fixed")
        # B2 动态抓取 (2026-08-16): 姿态连续性门禁阈值 + 最近稳定动态姿态
        self._attitude_geodesic_deg = float(
            self._orch_cfg.get("attitude_geodesic_deg", 3.0))
        self._last_handle_attitude_ee = None       # 最近稳定动态姿态 R_ee (3x3)
        self._last_handle_geodesic_deg = None      # 稳定窗口最大测地距离 (°)
        self._handle_side = self._orch_cfg.get("handle_side", "right")
        self._grasp_direction = self._orch_cfg.get("grasp_direction", self._handle_side)
        if self._handle_side != self._grasp_direction:
            print(f"[INFO] grasp_direction ({self._grasp_direction}) → handle_side ({self._handle_side})")
            self._grasp_direction = self._handle_side
        print(f"[INFO] orientation_mode: {self._orientation_mode}")

        # fixed mode — ABC 从 orchestrator config 读取 (基系末端 ABC)
        _h_abc = self._orch_cfg.get("handle_euler")
        self._handle_fixed_abc = np.array(_h_abc if _h_abc else [-90, 89, -90], dtype=float)
        _cup_key = "cup_euler_left" if self._cup_handle_side == "left" else "cup_euler_back"
        _c_abc = self._orch_cfg.get(_cup_key)
        self._cup_fixed_abc = np.array(_c_abc if _c_abc else [-90, 89, -110], dtype=float)
        self._fixed_abc = self._handle_fixed_abc  # 兼容旧引用
        print(f"[INFO] Handle P1 ABC (基系): {tuple(self._handle_fixed_abc)}")
        print(f"[INFO] Cup   P1 ABC (基系): {tuple(self._cup_fixed_abc)}")

        # 校验
        valid_modes = {"vertical", "horizontal", "free", "fixed", "dynamic"}
        if self._orientation_mode not in valid_modes:
            print(f"[WARN] 未知 orientation_mode: {self._orientation_mode}, → free")
            self._orientation_mode = "free"
        elif self._orientation_mode == "horizontal":
            print("[WARN] horizontal 模式未实现, → free")
            self._orientation_mode = "free"
        elif self._orientation_mode == "dynamic":
            print(f"[INFO] dynamic 模式: 姿态连续性门禁 "
                  f"attitude_geodesic_deg={self._attitude_geodesic_deg}°")

        # 跳帧
        self._skip_frames = self._orch_cfg.get("skip_frames", 5)

    def _stop(self, ctx: SharedContext):
        """清理资源."""
        if self._server:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        # Camera A 已在 Phase B 关闭; 安全兜底
        if ctx.handle_cup_detector is not None:
            try:
                ctx.handle_cup_detector.close()
            except Exception:
                pass
        ctx.handle_cup_detector = None
        ctx.T_cam2ee = None
        if self._hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        ctx.hand_on_cup = None

    # ================================================================
    # Camera A 帧获取 & 检测
    # ================================================================

    def _get_frames(self, ctx: SharedContext):
        det = ctx.handle_cup_detector
        if det is None:
            return None, None, None, None
        color, depth, depth_colormap, depth_frame = det.get_frames()
        self._last_color = color
        self._last_depth_image = depth
        self._last_depth_frame = depth_frame
        self._last_depth_colormap = depth_colormap
        return color, depth, depth_colormap, depth_frame

    def _detect(self, ctx: SharedContext, *, force: bool = False):
        """跳帧检测 Handle+Cup (使用轻量 HandleCupDetector)."""
        if self._last_color is None or self._last_depth_frame is None:
            return

        det = ctx.handle_cup_detector
        T_cam2ee = ctx.T_cam2ee
        if det is None or T_cam2ee is None:
            return
        if self._handle_stabilizer is None or self._cup_stabilizer is None:
            return  # 稳定器尚未就绪, 等待

        self._frame_counter += 1
        if not force and self._frame_counter % self._skip_frames != 1:
            return

        result = det.detect(self._last_color, self._last_depth_frame)

        # === Handle 稳定 ===
        h = result["handle"]
        if h["pos_cam"] is not None:
            pos_cam = np.array(h["pos_cam"])
            euler = np.array(h["euler_cam"])  # 相机系 ZYX (动态) 或 config 回退, 窗口范围检查用
            quat = h.get("quat_cam") if h.get("source") == "dynamic" else None
            self._handle_stabilizer.add(tuple(pos_cam), tuple(euler), quat)  # type: ignore[arg-type]
            if self._handle_stabilizer.is_stable():
                if not self._handle_stabilizer.is_generation_valid():
                    self._handle_stabilizer.reset()
                    self._handle_stabilizer.capture_generation()
                else:
                    mean_pos, _mean_euler = self._handle_stabilizer.get_mean()
                    if mean_pos is not None:
                        pos_ee = T_cam2ee[:3, :3] @ mean_pos + T_cam2ee[:3, 3]
                        # B2: 动态链 — 旋转经 相机→末端 链 (修旧: 直接写 config ABC)
                        R_mean_cam, _q_mean = self._handle_stabilizer.get_mean_attitude()
                        if R_mean_cam is not None:
                            R_ee = T_cam2ee[:3, :3] @ R_mean_cam
                            euler_ee = tuple(rotation_matrix_to_euler_angles(R_ee))
                            self._last_handle_attitude_ee = R_ee
                            report = self._handle_stabilizer.continuity_report()
                            self._last_handle_geodesic_deg = (
                                report.get("max_geodesic_deg") if report else None)
                            print(f"\n[Handle STABLE dynamic] ee=({pos_ee[0]:.3f},"
                                  f"{pos_ee[1]:.3f},{pos_ee[2]:.3f}) "
                                  f"euler_ee={tuple(round(v, 2) for v in euler_ee)} "
                                  f"geodesic={self._last_handle_geodesic_deg}")
                        else:
                            euler_ee = tuple(self._handle_fixed_abc)  # 无动态姿态 → config 回退
                            self._last_handle_attitude_ee = None
                            self._last_handle_geodesic_deg = None
                            print(f"\n[Handle STABLE fallback] ee=({pos_ee[0]:.3f},"
                                  f"{pos_ee[1]:.3f},{pos_ee[2]:.3f}) fixed_abc")
                        det.last_known_pose = (pos_ee, euler_ee)
                        det.last_known_pose_ts = time.time()
                        det.last_known_pose_observation_id += 1
                        det.last_known_pose_at_monotonic_ns = time.monotonic_ns()
        # (else: 不稳定时保持上次稳定值, 不停清零)

        # === Cup 稳定 ===
        c = result["cup"]
        if c["pos_cam"] is not None:
            pos_cam = np.array(c["pos_cam"])
            euler = np.array(c["euler_cam"])
            self._cup_stabilizer.add(tuple(pos_cam), tuple(euler))  # type: ignore[arg-type]
            if self._cup_stabilizer.is_stable():
                if not self._cup_stabilizer.is_generation_valid():
                    self._cup_stabilizer.reset()
                    self._cup_stabilizer.capture_generation()
                else:
                    mean_cpos, _mean_ceuler = self._cup_stabilizer.get_mean()
                    if mean_cpos is not None:
                        # 存 (pos_cam, 基系 ABC) — ABC 来自 config
                        det.last_known_cup_pose = (mean_cpos, tuple(self._cup_fixed_abc))
                        print(f"\n[Cup STABLE] pos=({mean_cpos[0]:.3f},{mean_cpos[1]:.3f},{mean_cpos[2]:.3f})")

        self._cached_result = result

    # ================================================================
    # D0 屏障辅助: P2 复核新观测 / 延迟释放 Camera A
    # ================================================================

    def _refresh_handle_observation(self, ctx: SharedContext, *, after_ns: int, timeout: float = 6.0):
        """P2 复核: 清除旧稳定器缓存, 主动取新帧并检测, 等新稳定把手观测.

        返回 ``(observation_id, monotonic_ns)``；旧规划观察绝不复用。
        """
        det = ctx.handle_cup_detector
        if det is None or self._handle_stabilizer is None:
            return None
        old_observation_id = det.last_known_pose_observation_id
        det.last_known_pose = None
        det.last_known_pose_ts = 0.0
        det.last_known_pose_at_monotonic_ns = 0
        self._last_handle_attitude_ee = None    # B2: 姿态随观测一起作废
        self._last_handle_geodesic_deg = None
        self._handle_stabilizer.reset()
        self._handle_stabilizer.capture_generation()
        self._frame_counter = 0
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not ctx.stop_event.is_set():
            color, depth, _cmap, depth_frame = self._get_frames(ctx)
            if color is not None and depth_frame is not None:
                self._detect(ctx, force=True)
                observed_ns = det.last_known_pose_at_monotonic_ns
                if (
                    det.last_known_pose is not None
                    and det.last_known_pose_observation_id > old_observation_id
                    and observed_ns > after_ns
                ):
                    print(f"[BARRIER] P2 后新观测 id={det.last_known_pose_observation_id}")
                    return det.last_known_pose_observation_id, observed_ns
            time.sleep(0.05)
        print("[BARRIER] P2 复核超时: 未获得新稳定观测")
        return None

    def _arm_allowed(self):
        """B2 模式门禁: orientation_mode=fixed 禁 arm; dynamic 需姿态连续性达标.

        返回 (ok, reason). 门禁通过才允许 P2 授权/抓取下发.
        """
        if self._orientation_mode == "fixed":
            return False, "orientation_mode=fixed 禁 arm (B2: 动态姿态未用于抓取)"
        if self._orientation_mode == "dynamic":
            if self._last_handle_geodesic_deg is None:
                return False, "无动态姿态连续性判定 (未见稳定动态观测)"
            if self._last_handle_geodesic_deg > self._attitude_geodesic_deg:
                return (False, f"姿态连续性超阈 "
                        f"{self._last_handle_geodesic_deg:.1f}° > "
                        f"{self._attitude_geodesic_deg}°")
        return True, ""

    def _release_camera_a_late(self, ctx: SharedContext):
        """P2 复核结束后释放 Camera A 引用 (延迟交给 main 线程关闭)."""
        if ctx.handle_cup_detector is not None:
            ctx._camera_a_to_close = ctx.handle_cup_detector
            ctx.handle_cup_detector = None
            ctx.T_cam2ee = None
            session_log("Camera A 引用已释放 (P2 复核结束)")

    # ================================================================
    # TCP
    # ================================================================

    def _accept_tcp(self, ctx: SharedContext):
        """非阻塞 accept."""
        if self._server is None:
            return
        try:
            conn, addr = self._server.accept()
            with self._session_lock:
                if self._session_active:
                    print(f"[BARRIER] 拒绝并发 TCP 会话: {addr}")
                    conn.close()
                    return
                self._session_active = True
                self._lease_epoch += 1
                lease_epoch = self._lease_epoch
            print(f"[INFO] 机器人已连接: {addr}")
            session_log(f"TCP 8080 机器人已连接: {addr}")
            t = threading.Thread(
                target=self._handle_client_impl,
                args=(conn, addr, ctx, lease_epoch),
                daemon=True,
            )
            t.start()
        except BlockingIOError:
            pass

    def _release_session(self):
        with self._session_lock:
            self._session_active = False

    # ================================================================
    # TCP 客户端处理 (单 Session 完整协议)
    # ================================================================

    def _handle_client_impl(self, conn: socket.socket, addr, ctx: SharedContext, lease_epoch: int):
        """单 TCP Session 全流程: 移植自 estun_handle_cup_run._handle_client_impl.

        关键差异:
          - Cup P3 XYZ/ABC 全示教固定 (EyeToHand/config.yaml delivery 段, 3D→1D 标定暂未接入)
          - Phase D: 递送确认 + Hand-in-cup 判定 (TRUE→CONTINUE / FALSE→ABORT)
        """
        print(f"[INFO] 客户端已连接: {addr}")

        # ---- 嵌套清理函数 (确保任何退出路径都执行清理) ----
        def _cleanup():
            ctx.hand_on_cup = None
            ctx.targets_sent.set()   # 解除 main loop 阻塞
            ctx.stop_event.set()
            self._release_camera_a_late(ctx)   # 兜底: 会话结束释放 Camera A (幂等)
            try:
                conn.close()
            except Exception:
                pass
            PoseStabilizer.on_disconnect()
            self._release_session()
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
                if not all(np.isfinite(value) for value in (*robot_xyz, *robot_ABC)):
                    print("[WARN] 初始位姿含非有限值")
                    _cleanup()
                    return
                session_log(f"Phase A 初始位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                            f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})")
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

        # ---- D0 会话屏障: 本 TCP 会话即一个 run ----
        # P0 必须为本会话 Phase A 实际位姿, 米/度, 非全零 (全零原点不可作为有效回执)
        if all(abs(value) < 1e-9 for value in (*robot_xyz, *robot_ABC)):
            print("[WARN] Phase A 初始位姿为全零, 拒绝会话")
            _cleanup()
            return
        barrier = SessionBarrier(
            run_id=f"{time.monotonic_ns()}-{lease_epoch}",
            lease_epoch=lease_epoch,
        )
        grip_done_once = False

        R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
        t_ee_base = np.array(robot_xyz, dtype=float)

        # ---- [Phase B] Camera A 已预加载, 直接使用 ----
        det = ctx.handle_cup_detector
        T_cam2ee = ctx.T_cam2ee
        if det is None or T_cam2ee is None:
            print("[ERROR] Camera A 未预加载")
            _cleanup()
            return

        print("[INFO] 等待稳定目标位姿 (Handle + Cup)...")
        T_cam2ee = ctx.T_cam2ee
        if det is None:
            print("[ERROR] handle_cup_detector 未初始化")
            _cleanup()
            return
        if T_cam2ee is None:
            print("[ERROR] T_cam2ee 未加载")
            _cleanup()
            return

        target_sent = False
        wait_start = time.time()
        while not target_sent and not ctx.stop_event.is_set() and (time.time() - wait_start) < self._phase_b_timeout:
            handle_pose = det.last_known_pose
            cup_pose = det.last_known_cup_pose

            if handle_pose is not None and cup_pose is not None:
                mean_pos_ee, mean_euler_ee = handle_pose  # type: ignore[misc]
                cup_pos_cam, cup_euler_cam = cup_pose  # type: ignore[misc]

                # ============================================================
                # Phase 1a: Handle P1 / P2 (不依赖倒水口)
                # ============================================================
                h_pos_ee_compensated = np.array(mean_pos_ee) - t_handle_effective
                R_obj_ee = euler_to_rotation_matrix(*mean_euler_ee, order="zyx")

                # 预声明分支变量 (Pylance narrowing)
                _fixed_R_grasp_ee: np.ndarray = np.eye(3)
                _fixed_R_target_ee: np.ndarray = np.eye(3)
                _R_grasp_ee_corrected: np.ndarray = np.eye(3)
                _R_pour_ee: np.ndarray = np.eye(3)

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
                    R_grasp_ee = compute_grasp_rotation(
                        R_obj_ee, self._orientation_mode, self._handle_r_offset, R_ee_base,
                        grasp_direction=self._grasp_direction)
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
                # Phase 2: Cup P1 / P2 / P3 (fixed 模式: 基系 ABC 直接使用)
                #   P3: delivery = (X_TEACH, Y_TEACH, Z_TEACH, ABC_TEACH)  # 全示教固定
                # ============================================================
                c_pos_ee = T_cam2ee[:3, :3] @ np.array(cup_pos_cam) + T_cam2ee[:3, 3]
                # cup_euler_cam 是基系 ABC, 直接构造 R_base_fix (同 Handle fixed 分支)
                c_R_base_fix = euler_to_rotation_matrix(*cup_euler_cam, order="zyx")
                c_R_grasp_ee = R_ee_base.T @ c_R_base_fix
                c_pos_ee_compensated = c_pos_ee - t_cup_effective

                # Cup P1 (基系 ABC 直接使用)
                c_rot_base_1 = c_R_base_fix
                c_pos_base_1 = transform_pose_to_base(c_pos_ee_compensated, robot_xyz, robot_ABC)
                cA1, cB1, cC1 = cup_euler_cam  # type: ignore[misc]

                c_R_grasp_ee_corrected = c_R_grasp_ee  # fixed 模式无需修正

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
                    # D0 屏障: 记录下发目标 (基系 m/deg), 签发 pregrasp ticket
                    p2_target = (tuple(pos_base_2), (A2, B2, C2))
                    p1_target = (tuple(pos_base_1), (A1, B1, C1))
                    planning_observation_id = det.last_known_pose_observation_id
                    conn.sendall(msg.encode("utf-8"))
                    barrier.arm(
                        p0_pose=(robot_xyz, robot_ABC),
                        p2_target=p2_target,
                        p1_target=p1_target,
                        planning_observation_id=planning_observation_id,
                        calibration_fingerprint=calib_fingerprint(ctx.T_cam2ee),
                        now_ns=time.monotonic_ns(),
                    )
                    session_log(f"Phase B SEND Handle: {msg}")
                    session_log(f"[BARRIER] pregrasp ticket 签发: lease={lease_epoch}")
                    target_sent = True
                except (BrokenPipeError, ConnectionResetError, BarrierReject, ValueError) as e:
                    print(f"[WARN] 发送 Handle 失败: {e}")
                    _cleanup()
                    return
            else:
                time.sleep(0.1)

        if not target_sent:
            print("[WARN] 超时: 未获得稳定目标位姿, 断开连接")
            _cleanup()
            return

        # ---- Camera A: 通知 main 线程进入 seated_check, 但保留 pipeline ----
        # D0: D435i 必须保留至 P2 复核结束 (P2_ARRIVED 后取新帧/新稳定观测)
        # 释放延后到屏障通过 WAIT_GRIPDONE 或会话结束 (_release_camera_a_late)
        ctx.targets_sent.set()
        session_log("Phase B 完成, 通知 main 线程进入 seated_check (Camera A 保留至 P2 复核)")

        # ---- [Phase C & D] 指令循环 ----
        conn.settimeout(2.0)
        cup_p3_sent = False        # CUP 触发后置 True
        hand_judged = False        # 第一次 RELEASE (hand-in-cup 判定) 已处理
        abort_mode = False         # TRUE=CONTINUE, FALSE=ABORT 进入放回模式

        _last_heartbeat = time.time()
        try:
            while not ctx.stop_event.is_set():
                try:
                    data = conn.recv(1024)
                except socket.timeout:
                    try:
                        barrier.check_timeout(now_ns=time.monotonic_ns())
                    except BarrierReject as e:
                        print(f"[BARRIER] 超时拒绝 ({e.code}): {e.reason}")
                        _cleanup()
                        return
                    # 每 10s 心跳, 确认在等机器人指令
                    if time.time() - _last_heartbeat > 10:
                        print(f"[WAIT] 等待机器人指令... ({time.strftime('%H:%M:%S')})")
                        _last_heartbeat = time.time()
                    continue

                if not data:
                    print("[INFO] 连接已关闭 (SocketClose)")
                    break

                cmd = data.decode("utf-8").strip()
                print(f"[RECV] {cmd}")

                # ============================================================
                # D0 会话屏障 (P2/P1 双屏障): 屏障通过前所有指令走 barrier
                # ============================================================
                if barrier.state is not SessionState.COMPLETE:
                    if cmd == "PREGRIP":
                        if self._hand_available:
                            hand_pre_grip()
                        session_log("PREGRIP — 仅手势，不推进会话")
                        continue
                    try:
                        current_calibration = calib_fingerprint(ctx.T_cam2ee)
                        if barrier.state is SessionState.WAIT_P2_ARRIVED:
                            barrier.receive_p2(
                                cmd,
                                calibration_fingerprint=current_calibration,
                                now_ns=time.monotonic_ns(),
                            )
                            ok_arm, arm_reason = self._arm_allowed()
                            if not ok_arm:
                                raise BarrierReject(arm_reason, "ARM_GATE")
                            observation = self._refresh_handle_observation(
                                ctx, after_ns=barrier.p2_arrived_at_ns,
                            )
                            if observation is None:
                                raise BarrierReject("P2 后未获得新稳定 D435i 观测", "STALE_OBS")
                            observation_id, observed_at_ns = observation
                            payload = barrier.authorize_grasp(
                                WristObservation(observation_id, observed_at_ns),
                                calibration_fingerprint=calib_fingerprint(ctx.T_cam2ee),
                                now_ns=time.monotonic_ns(),
                            )
                            conn.sendall(payload.encode("utf-8"))
                            session_log(f"[BARRIER] SEND {payload}")
                            continue
                        if barrier.state is SessionState.WAIT_P1_ARRIVED:
                            payload = barrier.receive_p1(
                                cmd,
                                calibration_fingerprint=current_calibration,
                                now_ns=time.monotonic_ns(),
                            )
                            conn.sendall(payload.encode("utf-8"))
                            session_log(f"[BARRIER] SEND {payload}")
                            continue
                        if barrier.state is SessionState.WAIT_GRIPDONE:
                            barrier.receive_gripdone(cmd, now_ns=time.monotonic_ns())
                            if self._hand_available:
                                hand_grip_done()
                            grip_done_once = True
                            self._release_camera_a_late(ctx)
                            session_log("[BARRIER] GRIPDONE 通过，灵巧手闭合一次")
                            continue
                        raise BarrierReject(f"非法 barrier 状态: {barrier.state}", "STATE")
                    except BarrierReject as e:
                        print(f"[BARRIER] 拒绝 ({e.code}): {e.reason}")
                        session_log(f"[BARRIER] 拒绝 {e.code}: {e.reason}")
                        _cleanup()
                        return

                if cmd == "PREGRIP":
                    if self._hand_available:
                        hand_pre_grip()
                        session_log("PREGRIP — 灵巧手半闭合")
                    else:
                        session_log("PREGRIP — 灵巧手不可用, 跳过")

                elif cmd == "GRIPDONE":
                    # D0: 屏障通过后不得重复闭合
                    if not grip_done_once:
                        if self._hand_available:
                            hand_grip_done()
                    else:
                        print("[WARN] GRIPDONE 重复/越序, 忽略")
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
                            session_log(f"Phase C SEND Cup: {cup_pose_data}")
                            cup_p3_sent = True
                        except (BrokenPipeError, ConnectionResetError) as e:
                            print(f"[WARN] 发送 Cup 失败: {e}")
                            break
                    else:
                        print("[WARN] CUP: cup_pose_data 未就绪或已发送")

                elif cmd == "READY":
                    # 通知 main loop 进入 hand_in_cup phase
                    ctx.ready_received.set()

                    if not ctx.is_person_seated:
                        # 无人就位 → 跳过 HandInCup 等待, 直接标记 FALSE
                        print("[INFO] READY: 人未就位 → HandInCup=false, 等待 RELEASE")
                        ctx.hand_on_cup = False
                    else:
                        # 人就位 → 启动 HandInCup
                        _hic_cfg = ctx.eye_to_hand_config.get("hand_in_cup", {})
                        _timeout = float(_hic_cfg.get("timeout", 25.0))
                        print(f"[INFO] READY: 人就位 → 启动 HandInCup (≤{_timeout:.0f}s)")
                        ctx.hand_in_cup_start_time = time.time()
                        if ctx.hand_in_cup_detector is not None:
                            ctx.hand_in_cup_detector.reset()
                        # main loop 在 hand_in_cup phase 每帧运行检测
                        while not ctx.stop_event.is_set():
                            if ctx.hand_on_cup:
                                print("[INFO] HandInCup: TRUE (提前检测到)")
                                break
                            if time.time() - ctx.hand_in_cup_start_time > _timeout:
                                ctx.hand_on_cup = False
                                print(f"[INFO] HandInCup: 超时 {_timeout:.0f}s → FALSE")
                                break
                            time.sleep(0.05)
                        print(f"[INFO] HandInCup 结束: {'TRUE' if ctx.hand_on_cup else 'FALSE'}")

                elif cmd == "RELEASE":
                    if abort_mode:
                        # ---- ABORT 路径第二次 RELEASE (放回后松杯) ----
                        if self._hand_available:
                            release_object()
                        print("[INFO] RELEASE (放回后松杯)")
                    elif ctx.hand_on_cup is None:
                        # ---- Phase C: 松把手 (READY 之前, HandInCup 未启动) ----
                        if self._hand_available:
                            release_object()
                        print("[INFO] RELEASE (松把手)")
                    elif not hand_judged:
                        # ---- Phase D 第一次 RELEASE: Hand-in-cup 判定 ----
                        hand_ok = ctx.hand_on_cup
                        print(f"[INFO] Hand-in-cup 判定: {'TRUE' if hand_ok else 'FALSE'}")
                        hand_judged = True

                        if hand_ok:
                            # TRUE: 松手 → CONTINUE
                            if self._hand_available:
                                release_object()
                            print("[INFO] RELEASE (松杯)")
                            try:
                                # 发送可达位姿 (m), CRP TranStrToCpos lengthunit=m
                                sig = f"<{robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f},{robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}>"
                                conn.sendall(sig.encode("utf-8"))
                                session_log("Phase D CONTINUE (人手在杯)")
                                print(f"[SEND] CONTINUE ({sig})")
                            except Exception as e:
                                print(f"[WARN] 发送 CONTINUE 失败: {e}")
                                break
                            # 机器人: P4 → RESET → Home → SocketClose
                            # 循环继续, 等待后续 RESET (hand_judged 在最终清理时设置)
                        else:
                            # FALSE: 不松手 → ABORT
                            print("[INFO] 无人接杯 → ABORT")
                            abort_mode = True
                            try:
                                # 发送可达位姿 (m), CRP TranStrToCpos lengthunit=m
                                sig = f"[{robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f},{robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f}]"
                                conn.sendall(sig.encode("utf-8"))
                                session_log("Phase D ABORT (无人接杯)")
                                print(f"[SEND] ABORT ({sig})")
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
            ctx.hand_on_cup = None
            try:
                conn.close()
            except Exception:
                pass
            print(f"[INFO] 客户端已断开: {addr}")

        # ---- 全流程完成, 退出程序 ----
        PoseStabilizer.on_disconnect()
        ctx.hand_judged.set()  # 通知 main loop 退出
        ctx.stop_event.set()
        self._release_session()
        print("[INFO] 全流程完成, 退出程序")
