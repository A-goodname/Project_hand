"""Isolated welcome-camera owner for the Windows Phase0+A2D fusion.

Only the child process opens UGREEN Camera 1080P and loads the Phase0 YOLO
model. The parent retains the robot socket and the single shared OmniHand.
"""
from __future__ import annotations

import multiprocessing as mp


def _camera_process(conn, cfg, camera_name: str, camera_path: str, camera_id: int):
    from Estun.estun_greeting_phase0 import Phase0Greeting

    service = Phase0Greeting(
        cfg, camera_name=camera_name, camera_path=camera_path,
        camera_id=camera_id, preview=False, mock=True,
    )
    try:
        service._open_camera()
        frame = service._get_frame()
        if frame is None:
            raise RuntimeError("迎宾相机已打开，但未读取到画面")
        service._person_detected(frame)  # load and warm YOLO before robot connects
        conn.send(("ready", service._camera.camera_id))
        command, timeout = conn.recv()
        if command == "detect":
            result = service._wait_for_person(timeout=float(timeout))
            conn.send(("detected", bool(result)))
    except Exception as exc:  # noqa: BLE001
        try:
            conn.send(("error", f"{type(exc).__name__}: {exc}"))
        except (BrokenPipeError, OSError):
            pass
    finally:
        service._close_camera()
        conn.close()


class WelcomeCameraWorker:
    """One-shot camera/YOLO subprocess; closes immediately after person check."""

    def __init__(self, cfg, *, camera_name="UGREEN Camera 1080P",
                 camera_path="", camera_id=-1):
        self._cfg = cfg
        self._camera_name = camera_name
        self._camera_path = camera_path
        self._camera_id = camera_id
        self._process = None
        self._conn = None

    def start(self):
        if self._process is not None:
            return
        context = mp.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(
            target=_camera_process,
            args=(child, self._cfg, self._camera_name,
                  self._camera_path, self._camera_id),
            name="phase0-welcome-camera", daemon=True,
        )
        try:
            process.start()
            child.close()
            self._process = process
            self._conn = parent
            if not parent.poll(45.0):
                raise TimeoutError("迎宾相机进程 45 秒内未就绪")
            kind, payload = parent.recv()
            if kind != "ready":
                raise RuntimeError(f"迎宾相机进程启动失败: {payload}")
            print(f"[GREETING] 迎宾相机子进程已预热: id={payload}")
        except Exception:
            self.close()
            raise

    def wait_for_person(self, timeout: float) -> bool:
        if self._process is None or self._conn is None:
            raise RuntimeError("迎宾相机进程未启动")
        try:
            self._conn.send(("detect", timeout))
            if not self._conn.poll(timeout + 10.0):
                raise TimeoutError("迎宾相机进程未在期限内返回识人结果")
            kind, payload = self._conn.recv()
            if kind != "detected":
                raise RuntimeError(f"迎宾相机识人失败: {payload}")
            return bool(payload)
        finally:
            self.close()

    def close(self):
        process, conn = self._process, self._conn
        self._process = None
        self._conn = None
        if process is not None:
            if process.is_alive() and conn is not None:
                try:
                    conn.send(("stop", 0))
                except (BrokenPipeError, OSError):
                    pass
            process.join(timeout=3.0)
            if process.is_alive():
                process.terminate()  # only the owned camera child, never robot/hand
                process.join(timeout=3.0)
        if conn is not None:
            conn.close()
