# -*- coding: utf-8 -*-
"""D2 原始流记录器 (2026-08-17) — 补齐九类流中的 raw/ 相机流与信号流.

对方 D2 采集要求 §3 九类流, 本模块覆盖:
  - SCENE_RGB   : UGREEN 场景相机 (Camera B, 物理口 usb-0:4.2) → mp4 + 时间戳 sidecar
  - WRIST_RGB   : 腕部 D435i 彩色 (data_sm 帧循环挂钩) → mp4 + 时间戳 sidecar
  - WRIST_DEPTH : 腕部 D435i 深度 (uint16 PNG 序列, 无损) + 时间戳 sidecar
  - HAND_STATE  : 灵巧手指令/可用性 (关节反馈无 API → UNAVAILABLE)
  - FORCE_TORQUE: 无真实力传感器 → UNAVAILABLE 健康记录 (禁填零)

纪律:
  - 禁硬编码 /dev/videoN (按 sysfs 名称枚举); 每流独立时间戳 (frame_idx +
    monotonic_ns + wall_utc), 不静默混用时钟
  - 相机 serial 随 manifest 记录; depth scale/对齐方式/无效约定写入 meta
"""
import datetime
import json
import os
import threading
import time

import cv2
import numpy as np

from Orchestrator.estun_d2_telemetry_sm import CLOCK_ID  # noqa: E402

_DEPTH_INVALID_CONVENTION = "z16 == 0 视为无效深度"


def _ts_row(frame_idx):
    return {
        "frame_idx": frame_idx,
        "monotonic_ns": time.monotonic_ns(),
        "wall_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "clock_id": CLOCK_ID,
    }


class SceneRecorder(threading.Thread):
    """UGREEN 场景相机 (Camera B) → raw/scene_rgb.mp4 + scene_rgb_timestamps.jsonl.

    按 sysfs 名称枚举 (EyeToHand.camera.find_camera_id), 禁 /dev/videoN 硬编码.
    """

    def __init__(self, ep_dir, fps=15.0, serial_note="UGREEN (SN0001 假序列号, "
                 "物理口 usb-0:4.2)"):
        super().__init__(daemon=True)
        self._ep_dir = ep_dir
        self._fps = float(fps)
        self._serial_note = serial_note
        # 修复 (2026-08-17): 属性名不得用 _stop — threading.Thread 内部同名
        # 方法, 覆盖后 join() 抛 "TypeError: 'Event' object is not callable",
        # 会话线程 finally 中断 → recorder.close() 不执行 (192820/195545 等
        # 无 manifest / 无 wrist sidecar 的根因)
        self._stop_event = threading.Event()
        self._cap = None
        self._writer = None
        self._ts = []
        self._count = 0
        self.error = None

    def _open_camera(self):
        from EyeToHand.camera import find_camera_id  # noqa: PLC0415
        cam_id = find_camera_id(keywords=["UGREEN"],
                                path_keyword="usb-0:4.2")
        if cam_id is None:
            raise RuntimeError("UGREEN 场景相机未枚举到 (usb-0:4.2)")
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"无法打开场景相机 video{cam_id}")
        return cap

    def run(self):
        raw = os.path.join(self._ep_dir, "raw")
        os.makedirs(raw, exist_ok=True)
        try:
            self._cap = self._open_camera()
            ok, frame = self._cap.read()
            if not ok:
                raise RuntimeError("场景相机首帧读取失败")
            h, w = frame.shape[:2]
            self._writer = cv2.VideoWriter(
                os.path.join(raw, "scene_rgb.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
            while not self._stop_event.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    break
                self._writer.write(frame)
                self._ts.append(_ts_row(self._count))
                self._count += 1
        except Exception as e:  # noqa: BLE001
            self.error = f"{type(e).__name__}: {e}"
        finally:
            if self._writer is not None:
                self._writer.release()
            if self._cap is not None:
                self._cap.release()
            self._flush_sidecar()

    def _flush_sidecar(self):
        raw = os.path.join(self._ep_dir, "raw")
        os.makedirs(raw, exist_ok=True)
        with open(os.path.join(raw, "scene_rgb_timestamps.jsonl"), "w",
                  encoding="utf-8") as f:
            for row in self._ts:
                f.write(json.dumps(row) + "\n")

    def stop(self):
        self._stop_event.set()


class WristStreamRecorder:
    """腕部 D435i 流记录器 (data_sm 帧循环挂钩, 非独立线程).

    record_frame(color, depth, frame_idx) 每帧调用:
      - 彩色 → raw/wrist_rgb.mp4 (fps 由调用方节流)
      - 深度 → raw/wrist_depth_<idx:06d>.png (uint16 无损) + 时间戳 sidecar
    """

    def __init__(self, ep_dir, fps=15.0, depth_skip=2,
                 serial="342222071364"):
        self._ep_dir = ep_dir
        self._fps = float(fps)
        self._depth_skip = max(1, int(depth_skip))
        self._serial = serial
        self._writer = None
        self._depth_count = 0
        self._rgb_count = 0            # 本实例彩色帧计数 (采样基准, 2026-08-17)
        self._ts_rgb = []
        self._ts_depth = []
        self._started = False
        self._io_lock = threading.RLock()

    def start(self):
        raw = os.path.join(self._ep_dir, "raw")
        os.makedirs(raw, exist_ok=True)
        self._started = True

    def record_frame(self, color, depth, frame_idx):
        with self._io_lock:
            self._record_frame_locked(color, depth, frame_idx)

    def _record_frame_locked(self, color, depth, frame_idx):
        if not self._started:
            return
        raw = os.path.join(self._ep_dir, "raw")
        if self._writer is None:
            h, w = color.shape[:2]
            self._writer = cv2.VideoWriter(
                os.path.join(raw, "wrist_rgb.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"), self._fps, (w, h))
            print(f"[D2-WRIST] 开始记录: {w}x{h}, fps={self._fps}, "
                  f"depth_skip={self._depth_skip}")
        self._writer.write(color)
        self._ts_rgb.append(_ts_row(frame_idx))
        # 修复 (2026-08-17): 采样基准用本实例计数而非调用方 frame_idx —
        # data_sm 主循环里 _frame_counter 每轮被加 2 次 (主循环 + _detect 跳帧),
        # frame_idx 恒为奇数, `frame_idx % depth_skip == 0` 永不成立 → 深度 0 张.
        if self._rgb_count % self._depth_skip == 0:
            png_path = os.path.join(
                raw, f"wrist_depth_{self._depth_count:06d}.png")
            ok = cv2.imwrite(
                png_path, depth, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            if not ok:
                print(f"[D2-WRIST] 警告: imwrite 返回 False frame={frame_idx} "
                      f"dtype={getattr(depth, 'dtype', None)} "
                      f"shape={getattr(depth, 'shape', None)}")
            if self._depth_count % 300 == 0:
                print(f"[D2-WRIST] 深度落盘: {os.path.basename(png_path)} "
                      f"ok={ok} 存在={os.path.exists(png_path)} "
                      f"frame={frame_idx} ep={os.path.basename(self._ep_dir)}")
            self._ts_depth.append(_ts_row(frame_idx))
            self._depth_count += 1
        self._rgb_count += 1

    def close(self):
        with self._io_lock:
            self._close_locked()

    def _close_locked(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        raw = os.path.join(self._ep_dir, "raw")
        os.makedirs(raw, exist_ok=True)   # 未 start 时目录不存在, sidecar 写入前补建
        with open(os.path.join(raw, "wrist_rgb_timestamps.jsonl"), "w",
                  encoding="utf-8") as f:
            for row in self._ts_rgb:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(raw, "wrist_depth_timestamps.jsonl"), "w",
                  encoding="utf-8") as f:
            for row in self._ts_depth:
                f.write(json.dumps(row) + "\n")
        with open(os.path.join(raw, "wrist_depth_meta.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "camera_serial": self._serial,
                "depth_dtype": "uint16",
                "depth_pixels": "640x480",
                "invalid_convention": _DEPTH_INVALID_CONVENTION,
                "alignment": "color_to_depth_pixel (d2c 外参, 见 D435i_info.yaml)",
                "depth_scale_source": "D435i_info.yaml / depth_sensor",
                "frames": self._depth_count,
                "fps_nominal": self._fps,
                "depth_skip": self._depth_skip,
            }, f, ensure_ascii=False, indent=2)
        self._started = False


def write_hand_state(ep_dir, **fields):
    """灵巧手状态事件 → raw 同级 signals/hand_state.jsonl (追加)."""
    raw = os.path.join(ep_dir, "signals")
    os.makedirs(raw, exist_ok=True)
    row = {"monotonic_ns": time.monotonic_ns(),
           "wall_utc": datetime.datetime.now(
               datetime.timezone.utc).isoformat(),
           "clock_id": CLOCK_ID, **fields}
    with open(os.path.join(raw, "hand_state.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_force_unavailable(ep_dir, reason="无真实力/力矩传感器 (硬件无此通道)"):
    """FORCE_TORQUE → UNAVAILABLE 健康记录 (禁填零伪造, D2 §3)."""
    raw = os.path.join(ep_dir, "signals")
    os.makedirs(raw, exist_ok=True)
    row = {
        "monotonic_ns": time.monotonic_ns(),
        "wall_utc": datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
        "clock_id": CLOCK_ID,
        "health": "UNAVAILABLE",
        "reason": reason,
        "unit": None,
        "values": None,
    }
    with open(os.path.join(raw, "force_torque.jsonl"), "a",
              encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
