#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase 0 打招呼独立服务 — 抽取自 Orchestrator/state_machine.py 的 GreetingHandler (2026-08-20).

配套 doc/estun_greeting_phase0.crp (CRP 侧 4 步 TCP 协议).
协议 (TCP 8600, 单 Socket 持久连接):
  Step 1: 机器人发当前位姿 → PC 回 ZHAOSHOU1 位姿 (mm,deg) → 机器人 MovL
  Step 2: 机器人发确认       → PC 迎宾相机 YOLO 等人 → PC 回 ZHAOSHOU2 → 机器人 MovL
  Step 3: 机器人发确认       → PC 挥手 (Hello_1↔Hello_2×N) → PC 回 WOSHOU → 机器人 MovL
  Step 4: 机器人发确认       → PC 握手 (压力检测) → PC 回 "Home" → 机器人 MovL → SocketClose

相机: 迎宾 UGREEN (USB 物理口, 默认 usb-0:4.1, EyeToHand/camera.py 按路径枚举) —
      人进门检测用它 (非 D435i).
灵巧手: Estun.utils_shared.init_hand + Gesture; 无手时自动模拟降级.

用法:
  conda run -n Proj python Estun/estun_greeting_phase0.py                     # 实机
  conda run -n Proj python Estun/estun_greeting_phase0.py --mock              # 不连灵巧手
  conda run -n Proj python Estun/estun_greeting_phase0.py --camera-path usb-0:4.1 --port 8600
按键: q 跳过等待 (机器人未连接时)
"""
import argparse
import os
import socket
import sys
import threading
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import cv2
import numpy as np
import yaml

from EyeToHand.camera import UGreenCamera

# 迎宾相机 USB 物理口 (用户定案 2026-08-20): 人进门检测用它, 非 D435i
DEFAULT_WELCOME_CAMERA_PATH = "usb-0:4.1"


def _load_estun_config():
    path = os.path.join(_project_root, "Estun", "estun_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["estun"]


class Phase0Greeting:
    """Phase 0 打招呼独立服务 (4 步 TCP)."""

    def __init__(self, estun_cfg, camera_path=DEFAULT_WELCOME_CAMERA_PATH,
                 camera_name="UGREEN Camera", camera_id=-1, preview=True,
                 mock=False, camera_worker=None):
        self._cfg = estun_cfg
        self._camera_path = camera_path
        self._camera_name = camera_name
        self._camera_id = int(camera_id)
        self._preview = preview
        self._mock = mock
        self._camera_worker = camera_worker
        self._stop_event = threading.Event()

        g = estun_cfg.get("greeting", {})
        self._zhaoshou1 = [float(v) for v in g.get("zhaoshou1_pose", [0] * 6)]
        self._zhaoshou2 = [float(v) for v in g.get("zhaoshou2_pose", [0] * 6)]
        self._woshou = [float(v) for v in g.get("woshou_pose", [0] * 6)]
        self._wave_cycles = int(g.get("wave_cycles", 3))
        self._wave_interval = float(g.get("wave_interval", 0.6))
        self._force_timeout = float(g.get("force_timeout", 30.0))
        self._force_abs_threshold = int(g.get("force_abs_threshold", 100))
        self._force_delta_threshold = int(g.get("force_delta_threshold", 100))
        self._force_baseline_samples = int(g.get("force_baseline_samples", 5))
        self._force_poll_interval = float(g.get("force_poll_interval", 0.15))
        self._force_release_timeout = float(g.get("force_release_timeout", 60.0))
        self._force_release_delta = int(g.get("force_release_delta", 50))
        self._force_release_hold = float(g.get("force_release_hold", 2.0))

        self._person_required = 3
        self._person_stable = 0
        self._yolo_model = None
        self._camera = None
        self._hand = None
        self._server = None
        self._completed = False

    # ================================================================
    #  生命周期
    # ================================================================

    def start(self):
        """初始化灵巧手 + 打开迎宾相机."""
        if not self._mock:
            self._init_hand()
        self._open_camera()
        return self

    def stop(self):
        """清理资源."""
        self._stop_event.set()
        self._close_camera()
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        # 灵巧手假失能 (2026-08-22): 只 RESET 手势, 不 disable/close 串口 — 让 PhaseA2D 接手时
        # 免重开/重使能 (避免 "使能未确认" 疯狂重试). 串口随进程退出自动释放.
        if self._hand is not None:
            try:
                from Estun.utils_shared import hand_reset
                hand_reset()
                print("[GREETING] 灵巧手 RESET (假失能, 不关串口)")
            except Exception as e:
                print(f"[GREETING] 灵巧手 RESET 失败: {e}")
            self._hand = None
        if self._preview:
            cv2.destroyAllWindows()
        print("[GREETING] 已退出")

    def _init_hand(self):
        """初始化灵巧手 (RESET). OMNIHAND 不可用时自动模拟降级."""
        from Estun.utils_shared import init_hand
        hand_port = self._cfg.get("hand", {}).get(
            "port", "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00")
        init_hand(hand_port=hand_port, initial_gesture="RESET")
        from Estun.utils_shared import _hand
        self._hand = _hand
        print("[GREETING] 灵巧手已初始化" if self._hand is not None
              else "[GREETING] 无灵巧手 → 模拟模式")

    # ================================================================
    #  迎宾相机 (UGREEN, 人进门检测)
    # ================================================================

    def _open_camera(self):
        if self._camera_worker is not None:
            self._camera_worker.start()
            print("[GREETING] 迎宾相机在独立进程中就绪")
            return
        self._camera = UGreenCamera(
            camera_id=self._camera_id, camera_name=self._camera_name,
            camera_path=self._camera_path,
            width=1280, height=720, fps=30)
        print("[GREETING] 迎宾相机已打开: "
              f"id={self._camera.camera_id} name={self._camera_name!r} "
              f"path={self._camera_path!r}")

    def _get_frame(self):
        if self._camera is None:
            return None
        try:
            (frame,) = self._camera.read()
            return frame
        except Exception:
            return None

    def _close_camera(self):
        if self._camera_worker is not None:
            self._camera_worker.close()
            return
        if self._camera is not None:
            self._camera.release()
            self._camera = None
            print("[GREETING] 迎宾相机关闭")

    # ================================================================
    #  YOLO 人检测 (进门)
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

    def _wait_for_person(self, timeout=120.0):
        """迎宾相机 YOLO 人检测, 连续 N 帧稳定则返回."""
        if self._camera_worker is not None:
            return self._camera_worker.wait_for_person(timeout)
        print("[GREETING] 等待人进门...")
        self._person_stable = 0
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop_event.is_set():
            frame = self._get_frame()
            if frame is None:
                time.sleep(0.05)
                continue
            detected = self._person_detected(frame)
            self._person_stable = self._person_stable + 1 if detected else 0
            if self._preview:
                disp = frame.copy()
                cv2.putText(disp, f"person={detected} stable={self._person_stable}/{self._person_required}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (0, 255, 0) if detected else (0, 0, 255), 2)
                cv2.imshow("Phase0 Greeting | Welcome Cam", disp)
                cv2.waitKey(1)
            if self._person_stable >= self._person_required:
                print("\n[GREETING] 检测到人进门!")
                return True
            time.sleep(0.03)
        print(f"\n[GREETING] 人检测超时 ({timeout}s)")
        return False

    # ================================================================
    #  挥手 + 握手 (Gesture 枚举, 无手模拟降级)
    # ================================================================

    def _execute_wave(self):
        from Omnihand.omnihand_gesture import Gesture
        h = self._hand
        if h is not None:
            print(f"[GREETING] 挥手 {self._wave_cycles} 次 (间隔 {self._wave_interval}s)...")
            for i in range(self._wave_cycles):
                if self._stop_event.is_set():
                    print("[GREETING] 挥手被中断")
                    break
                print(f"  {i + 1}/{self._wave_cycles}")
                h.do_gesture(Gesture.HELLO_1, delay=0.5)
                time.sleep(self._wave_interval / 4)
                if self._stop_event.is_set():
                    break
                h.do_gesture(Gesture.HELLO_2, delay=0.5)
                time.sleep(self._wave_interval / 4)
            print("[GREETING] RESET — 挥手结束, 复位")
            h.do_gesture(Gesture.RESET, delay=0.5)
        else:
            print("[GREETING] 模拟挥手")
            time.sleep(self._wave_cycles * (0.5 + self._wave_interval / 2) * 2)

    def _read_hand_back_pressure(self):
        h = self._hand
        if h is None:
            return 0
        sensors = h.serial.read_all_sensor_group_3()
        return int(np.asarray(sensors.get("back", []), dtype=np.int32).sum())

    def _wait_for_handshake_force(self):
        """等待手背压力 (基线+增量双阈值, 参数从 config 读取)."""
        baseline_samples = []
        for _ in range(max(1, self._force_baseline_samples)):
            baseline_samples.append(self._read_hand_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(baseline_samples, dtype=np.float64)))
        deadline = time.time() + self._force_timeout
        while time.time() < deadline and not self._stop_event.is_set():
            value = self._read_hand_back_pressure()
            delta = value - baseline
            abs_trigger = (baseline < self._force_abs_threshold * 0.5
                           and value >= self._force_abs_threshold)
            delta_trigger = (delta >= self._force_delta_threshold)
            print(f"[GREETING] 压力 back={value} baseline={baseline:.0f} "
                  f"delta={delta:.0f} abs={abs_trigger} delta_trig={delta_trigger}")
            if abs_trigger or delta_trigger:
                print("[GREETING] 检测到抓握!")
                return True
            time.sleep(self._force_poll_interval)
        print(f"[GREETING] 压力超时 ({self._force_timeout}s)")
        return False

    def _wait_for_handshake_release(self):
        """等待松手: 连续 hold 秒内 delta < threshold."""
        baseline_samples = []
        for _ in range(max(1, self._force_baseline_samples)):
            baseline_samples.append(self._read_hand_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(baseline_samples, dtype=np.float64)))
        released_since = None
        deadline = time.time() + self._force_release_timeout
        while time.time() < deadline and not self._stop_event.is_set():
            value = self._read_hand_back_pressure()
            delta = value - baseline
            released = (delta <= self._force_release_delta)
            now = time.time()
            if released:
                if released_since is None:
                    released_since = now
                if now - released_since >= self._force_release_hold:
                    print("[GREETING] 已松手!")
                    return True
            else:
                released_since = None
            time.sleep(self._force_poll_interval)
        print(f"[GREETING] 松手超时 ({self._force_release_timeout}s)")
        return False

    def _execute_handshake(self):
        """RESET → 等压力 → HAND_SHAKE → 等松手 → RESET."""
        from Omnihand.omnihand_gesture import Gesture
        h = self._hand
        if h is not None:
            h.do_gesture(Gesture.RESET, delay=0.5)
            print("[GREETING] RESET — 等待握手...")
            if not self._wait_for_handshake_force():
                h.do_gesture(Gesture.RESET, delay=0.5)
                return False
            h.do_gesture(Gesture.HAND_SHAKE, delay=1.0)
            print("[GREETING] HAND_SHAKE")
            released = self._wait_for_handshake_release()
            h.do_gesture(Gesture.RESET, delay=0.5)
            print("[GREETING] RESET — 握手结束")
            return released
        else:
            print("[GREETING] 模拟握手")
            time.sleep(3.0)
            return True

    # ================================================================
    #  TCP 4 步协议 (与 estun_greeting_phase0.crp 匹配)
    # ================================================================

    @staticmethod
    def _fmt_pose6(pose6):
        """6元素位姿 → "x,y,z,A,B,C" (mm, deg) — CRP TranStrToCpos lengthunit=mm."""
        return f"{pose6[0]:.1f},{pose6[1]:.1f},{pose6[2]:.1f}," \
               f"{pose6[3]:.2f},{pose6[4]:.2f},{pose6[5]:.2f}"

    def _handle_session(self, conn):
        """4 步协议 (单连接)."""
        print("[GREETING] 客户端已连接, 开始 4 步协议")
        try:
            # ---- Step 1: recv → reply ZHAOSHOU1 ----
            conn.settimeout(5.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 1: 未收到数据")
                return
            print(f"[GREETING] Step 1 recv: {data.decode('utf-8', errors='replace').strip()}")
            conn.sendall(self._fmt_pose6(self._zhaoshou1).encode() + b"\n")

            # ---- Step 2: recv → 等人 → close camera → reply ZHAOSHOU2 ----
            conn.settimeout(130.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 2: 未收到数据")
                return
            print(f"[GREETING] Step 2 recv: {data.decode('utf-8', errors='replace').strip()}")
            if not self._wait_for_person(timeout=120.0):
                print("[GREETING] 人检测未通过，不发送 ZHAOSHOU2 位姿")
                return
            self._close_camera()
            conn.sendall(self._fmt_pose6(self._zhaoshou2).encode() + b"\n")

            # ---- Step 3: recv → 挥手 → reply WOSHOU ----
            conn.settimeout(30.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 3: 未收到数据")
                return
            print(f"[GREETING] Step 3 recv: {data.decode('utf-8', errors='replace').strip()}")
            self._execute_wave()
            conn.sendall(self._fmt_pose6(self._woshou).encode() + b"\n")

            # ---- Step 4: recv → 握手 → reply "Home" ----
            conn.settimeout(90.0)
            data = conn.recv(4096)
            if not data:
                print("[GREETING] Step 4: 未收到数据")
                return
            print(f"[GREETING] Step 4 recv: {data.decode('utf-8', errors='replace').strip()}")
            if not self._execute_handshake():
                print("[GREETING] 握手未完成，不发送 Home 确认")
                return
            conn.sendall(b"Home\n")
            self._completed = True
            print("[GREETING] Step 4 reply: Home")
        except socket.timeout:
            print("[GREETING] Step 超时")
        except Exception as e:
            print(f"[GREETING] 异常: {e}")
        finally:
            self._close_camera()
            try:
                conn.close()
            except Exception:
                pass
            print("[GREETING] 打招呼阶段完成")

    def serve(self, port=8600):
        """非阻塞 accept + 迎宾相机预览循环; 单连接."""
        host = self._cfg.get("tcp_server", {}).get("host", "0.0.0.0")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, port))
        server.listen(1)
        server.setblocking(False)
        self._server = server
        print(f"[GREETING] TCP 服务器: {host}:{port}")
        print("[GREETING] 等待机器人连接... (按 'q' 跳过)")
        conn = None
        try:
            while conn is None and not self._stop_event.is_set():
                try:
                    conn, addr = server.accept()
                    print(f"[GREETING] 机器人已连接: {addr}")
                except BlockingIOError:
                    frame = self._get_frame()
                    if frame is not None and self._preview:
                        cv2.putText(frame, "Phase0: Waiting robot...", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Phase0 Greeting | Welcome Cam", frame)
                    if self._preview and cv2.waitKey(1) & 0xFF == ord("q"):
                        print("[GREETING] 用户跳过")
                        return False
                    time.sleep(0.05)
        finally:
            server.close()
            self._server = None
        if conn is not None:
            self._handle_session(conn)
        return self._completed


def main():
    ap = argparse.ArgumentParser(
        description="Phase 0 打招呼独立服务 (配套 doc/estun_greeting_phase0.crp)")
    ap.add_argument("--port", type=int, default=None,
                    help="TCP 端口 (默认 config greeting.port=8600)")
    ap.add_argument("--camera-path", default=None,
                    help="迎宾相机 USB 物理口 (默认 usb-0:4.1)")
    ap.add_argument("--camera-name", default=None,
                    help="迎宾相机设备名 (Windows 推荐使用完整 DSHOW 名称)")
    ap.add_argument("--camera-id", type=int, default=None,
                    help="迎宾相机 OpenCV 索引；指定后跳过名称/物理口探测")
    ap.add_argument("--no-preview", action="store_true", help="不显示预览窗口")
    ap.add_argument("--mock", action="store_true", help="不连接灵巧手 (模拟挥手/握手)")
    args = ap.parse_args()

    estun_cfg = _load_estun_config()
    g = estun_cfg.get("greeting", {})
    port = args.port if args.port is not None else int(g.get("port", 8600))
    camera_path = (args.camera_path if args.camera_path is not None
                   else g.get("camera_path", DEFAULT_WELCOME_CAMERA_PATH))
    camera_name = (args.camera_name if args.camera_name is not None
                   else g.get("camera_name", "UGREEN Camera"))
    camera_id = (args.camera_id if args.camera_id is not None
                 else int(g.get("camera_id", -1)))

    svc = Phase0Greeting(estun_cfg, camera_path=camera_path,
                         camera_name=camera_name, camera_id=camera_id,
                         preview=not args.no_preview, mock=args.mock)
    svc.start()
    try:
        completed = svc.serve(port=port)
    except KeyboardInterrupt:
        print("[GREETING] 用户中断")
        return 130
    finally:
        svc.stop()
    return 0 if completed else 2


if __name__ == "__main__":
    raise SystemExit(main())
