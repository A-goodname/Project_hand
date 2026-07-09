# -*- coding: utf-8 -*-
"""
模板项目名称 (中文注释 + 英文标点规范)
======================================
功能描述: 本次任务的具体功能.

使用说明:
  1. 连接硬件设备 (相机/串口等)
  2. 修改配置文件中的 host / port / serial 等参数
  3. 运行 python path/to/this_script.py
  4. 按 'q' 退出, Ctrl+C 中断

输入:
  - config.yaml          : 模块配置 (相机 / 模型 / 检测参数)
  - estun_config.yaml    : 机器人 / 灵巧手 / 偏移配置

输出:
  - TCP 位姿数据          : 发送至机器人的目标位姿
  - 实时画面               : OpenCV 窗口显示检测结果
  - logs/*.log            : 运行日志 (可选)

完整流程:
  Phase 0 (可选):   打招呼 / 初始化
  Phase A (Init):   接收机器人初始位姿
  Phase B (Detect): 视觉检测 → 位姿计算 → 发送
  Phase C (Command): TCP 指令循环
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


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入自定义包 ----
from Estun.utils_shared import (
    init_hand, is_hand_available, disable_hand,
    grip_object, release_object,
)

# ---- 全局变量 ----
stop_event = threading.Event()
_hand_available = False


# ---- 加载配置 ----
_cfg_path = os.path.join(_project_root, "ModuleName", "config.yaml")
with open(_cfg_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

_estun_cfg_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

HOST = _cfg_estun["tcp_server"]["host"]
PORT = _cfg_estun["tcp_server"]["port"]
HAND_PORT = _cfg_estun["hand"]["port"]


# =============================================================
# 辅助函数 (纯函数, 无副作用)
# =============================================================

def helper_function(param1, param2):
    """
    辅助函数的功能描述.

    参数:
        param1 : float, 说明
        param2 : float, 说明

    返回:
        float, 计算结果
    """
    return param1 + param2


# =============================================================
# 功能模块: 示例类
# =============================================================

class ExampleProcessor:
    """
    示例处理器 — 封装某阶段业务逻辑.

    使用:
        processor = ExampleProcessor(config, callback)
        processor.run()
    """

    def __init__(self, config, frame_callback=None):
        """
        参数:
            config          : dict, 配置字典
            frame_callback  : callable or None, 帧获取回调 () -> np.ndarray | None
        """
        self.config = config
        self.get_frame = frame_callback
        self._running = False

    def run(self):
        """主循环 (阻塞)."""
        self._running = True
        print("[INFO] ExampleProcessor 启动")
        try:
            while self._running and not stop_event.is_set():
                frame = self.get_frame() if self.get_frame else None
                if frame is None:
                    time.sleep(0.05)
                    continue
                # 处理帧...
        except KeyboardInterrupt:
            print("[INFO] 用户中断")
        finally:
            self._cleanup()

    def stop(self):
        """安全停止."""
        self._running = False

    def _cleanup(self):
        """资源清理."""
        print("[INFO] ExampleProcessor 已停止")


# =============================================================
# 功能模块: TCP 客户端处理 (daemon 线程)
# =============================================================

def handle_client(conn, addr):
    """
    TCP 客户端处理入口 (在 daemon 线程中运行).

    参数:
        conn : socket, 客户端连接
        addr : (str, int), 客户端地址
    """
    print(f"[INFO] 客户端已连接: {addr}")
    try:
        # ---- Phase A: 接收初始位姿 ----
        conn.settimeout(5.0)
        data = conn.recv(1024)
        if not data:
            print("[WARN] 机器人未发送初始位姿, 断开连接", flush=True)
            return
        print(f"[INFO] 收到初始位姿: {data.decode('utf-8').strip()}", flush=True)

        # ---- Phase B: 检测 → 发送位姿 ----
        # ... 视觉检测 + 位姿计算 ...
        # msg = pack_three(p1, p2, p3)
        # conn.sendall(msg.encode("utf-8"))

        # ---- Phase C: 指令循环 ----
        conn.settimeout(2.0)
        while not stop_event.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue
            if not data:
                print("[INFO] 连接已关闭", flush=True)
                break
            cmd = data.decode("utf-8").strip().upper()
            print(f"[RECV] {cmd}", flush=True)

            if cmd == "GRIP":
                if _hand_available:
                    grip_object()
            elif cmd == "RELEASE":
                if _hand_available:
                    release_object()
            else:
                print(f"[INFO] 未知指令: {cmd}", flush=True)

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


# =============================================================
# 主函数
# =============================================================

def main():
    global _hand_available

    print("=" * 60)
    print("  模板项目名称")
    print("=" * 60)

    # ---- [Init 1] 加载标定 ----
    # T_cam2ee = load_calibration(CALIB_PATH)

    # ---- [Init 2] 灵巧手 ----
    print(f"\n[INIT] 灵巧手...")
    init_hand(hand_port=HAND_PORT, initial_gesture="RESET")
    _hand_available = is_hand_available()

    # ---- [Init 3] 相机 / 检测器 ----
    # detector = SomeDetector(config_dict=_cfg)

    # ---- [Init 4] TCP 服务器 (非阻塞) ----
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"\n[INFO] 服务器已启动: {HOST}:{PORT}")
    print("[INFO] 等待机器人连接...", flush=True)

    cv2.namedWindow("Window Title", cv2.WINDOW_NORMAL)

    # ---- 主循环 ----
    running = True
    try:
        while running:
            # ---- 获取帧 + 检测 + 可视化 ----
            # color_image, depth_frame = detector.get_frames()
            # ... 检测逻辑 ...
            # cv2.imshow("Window Title", display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                stop_event.set()
                running = False
                break

            # ---- 非阻塞 accept ----
            try:
                conn, addr = server.accept()
            except BlockingIOError:
                continue

            print(f"\n[TCP] 客户端连接: {addr}")
            t = threading.Thread(
                target=handle_client,
                args=(conn, addr),
                daemon=True,
            )
            t.start()
            print("[TCP] 等待下一次连接...", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] 服务器异常: {e}")
    finally:
        # ---- 资源清理 ----
        cv2.destroyAllWindows()
        print("\n[INFO] 正在清理资源...")
        server.close()
        # detector.pipeline.stop()
        if _hand_available:
            try:
                disable_hand()
            except Exception:
                pass
        print("[INFO] 程序已退出", flush=True)


if __name__ == "__main__":
    main()
