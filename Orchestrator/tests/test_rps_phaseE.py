# -*- coding: utf-8 -*-
"""Phase E RPS 机械臂接入 — mock TCP 协议测试 (无 ESTUN 控制器/相机/灵巧手).

覆盖 Estun/estun_rps_phaseE.py 的 PhaseERps 服务端在协议层面的行为:
  1. 正常流程: X1 → 回 P1(米,度); 注入 [RPS][RESULT] → 发 DONE; 对端关闭 → 优雅退出
  2. DONE 后意外数据忽略 + EOF: 不挂 15s
  3. --no-rps: 跳过日志等待, 立即发 DONE
  4. --mock: 模拟游戏短时后发 DONE
  5. play 位姿全零且非 mock → 拒绝启动 (防机器人跑原点)
  6. auto_invite FIFO: 服务端写 '\\n' 到 FIFO 触发邀请
"""
import os
import socket
import sys
import tempfile
import threading
import time
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from Estun.estun_rps_phaseE import PhaseERps  # noqa: E402

PLAY = (0.1234, -0.0567, 0.0890, -10.50, 20.25, 30.50)   # 米,度 (协议单位)
DONE_FLAG = "DONE"
RESULT_LINE = "[RPS][RESULT] player=rock robot=paper outcome=win\n"


def _make_cfg(voice_log, settle=0.0, max_wait=10.0, rounds=1,
              play=PLAY, auto_invite=False, fifo=None):
    return {
        "tcp_server": {"host": "127.0.0.1"},
        "rps": {
            "done_flag": DONE_FLAG,
            "voice_log_path": voice_log,
            "arm_settle_delay": settle,
            "max_game_wait": max_wait,
            "rounds": rounds,
            "poses": {"play": list(play)},
            "auto_invite": {"enabled": auto_invite,
                            "fifo_path": fifo or "/tmp/rps_voice_stdin"},
        },
    }


def _free_port():
    s = socket.socket()
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


def _recv_line(conn, timeout=3.0):
    conn.settimeout(timeout)
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    return buf


def _fmt(pose):
    return (f"{pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f},"
            f"{pose[3]:.2f},{pose[4]:.2f},{pose[5]:.2f}").encode()


class RpsPhaseETest(unittest.TestCase):

    def _start(self, cfg, port=None, **kwargs):
        port = port or _free_port()
        cfg["rps"]["port"] = port
        svc = PhaseERps(cfg, port=port, **kwargs)
        t = threading.Thread(target=svc.serve, daemon=True)
        t.start()
        # 轮询连接: 前几次失败是端口未就绪; 首次成功即本测试客户端
        deadline = time.time() + 3.0
        while time.time() < deadline:
            try:
                client = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                break
            except OSError:
                time.sleep(0.02)
        else:
            raise AssertionError("PhaseE 服务未就绪")
        return svc, t, client

    # ---------- 1. 正常流程 (Home 存储点直 movl, py 不回位姿) ----------

    def test_normal_flow_log_triggered(self):
        """X1→P1(米,度); 日志注入 [RPS][RESULT]→DONE; 对端关闭→优雅退出."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            open(log, "w").close()
            svc, t, client = self._start(_make_cfg(log, settle=0.0))
            try:
                client.sendall(b"1.0,2.0,3.0,4.0,5.0,6.0\n")   # X1
                self.assertEqual(_recv_line(client), _fmt(PLAY) + b"\n")
                # 注入结果 → 服务端应在 ~0.5s 内发 DONE
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(RESULT_LINE)
                line = _recv_line(client, timeout=3.0)
                self.assertEqual(line, DONE_FLAG.encode() + b"\n")
                # 机器人 movl Home(存储点) 后 socketclose → 客户端关闭
            finally:
                client.close()
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive(), "服务线程应正常结束")

    # ---------- 2. DONE 后额外数据忽略 + EOF ----------

    def test_stored_home_variant_eof_after_done(self):
        """DONE 后对端发意外数据 (忽略) 再关闭 → 优雅退出, 不挂 15s."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            open(log, "w").close()
            svc, t, client = self._start(_make_cfg(log, settle=0.0))
            try:
                client.sendall(b"1.0,2.0,3.0,4.0,5.0,6.0\n")
                self.assertEqual(_recv_line(client), _fmt(PLAY) + b"\n")
                with open(log, "a", encoding="utf-8") as fh:
                    fh.write(RESULT_LINE)
                self.assertEqual(_recv_line(client), DONE_FLAG.encode() + b"\n")
                client.sendall(b"9.0,9.0,9.0,9.0,9.0,9.0\n")   # 意外 X2, py 应忽略
            finally:
                client.close()
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive(), "DONE 后 EOF 应优雅退出 (非等 15s)")

    # ---------- 3. --no-rps ----------

    def test_no_rps_immediate_done(self):
        """--no-rps: 跳过日志等待, P1 后立即 DONE."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            svc, t, client = self._start(_make_cfg(log, settle=0.0), no_rps=True)
            try:
                client.sendall(b"1.0,2.0,3.0,4.0,5.0,6.0\n")
                self.assertEqual(_recv_line(client), _fmt(PLAY) + b"\n")
                t0 = time.monotonic()
                line = _recv_line(client, timeout=3.0)
                self.assertEqual(line, DONE_FLAG.encode() + b"\n")
                self.assertLess(time.monotonic() - t0, 1.5, "no_rps 应立即 DONE")
            finally:
                client.close()
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive())

    # ---------- 4. --mock ----------

    def test_mock_simulated_game(self):
        """--mock: 模拟游戏 ~3s 后发 DONE."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            svc, t, client = self._start(_make_cfg(log, settle=0.0), mock=True)
            try:
                client.sendall(b"1.0,2.0,3.0,4.0,5.0,6.0\n")
                self.assertEqual(_recv_line(client), _fmt(PLAY) + b"\n")
                line = _recv_line(client, timeout=6.0)
                self.assertEqual(line, DONE_FLAG.encode() + b"\n")
            finally:
                client.close()
            t.join(timeout=4.0)
            self.assertFalse(t.is_alive())

    # ---------- 5. play 全零占位拒绝 ----------

    def test_all_zero_play_rejected(self):
        """非 mock 下 play 全零 → ValueError (防机器人跑原点)."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            cfg = _make_cfg(log, play=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
            with self.assertRaises(ValueError):
                PhaseERps(cfg, port=_free_port(), mock=False)
            # mock 下允许 (协议自测)
            svc = PhaseERps(cfg, port=_free_port(), mock=True)
            self.assertIsNotNone(svc)

    # ---------- 6. auto_invite FIFO ----------

    def test_auto_invite_writes_fifo(self):
        """auto_invite: 服务端在 P1 后写 '\\n' 到 FIFO (触发 realtime_voice 邀请)."""
        with tempfile.TemporaryDirectory() as td:
            log = os.path.join(td, "voice.log")
            fifo = os.path.join(td, "voice_stdin")
            os.mkfifo(fifo)
            # 持写端 (模拟一键脚本 tail -f /dev/null > fifo), 防读端读到 EOF 停读
            # 顺序: 先开读端 (rfd), 再开写端 holder — 无读者时 O_WRONLY 会 ENXIO
            rfd = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
            holder = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
            try:
                svc, t, client = self._start(
                    _make_cfg(log, settle=0.0, auto_invite=True, fifo=fifo,
                              max_wait=0.5))
                try:
                    client.sendall(b"1.0,2.0,3.0,4.0,5.0,6.0\n")
                    self.assertEqual(_recv_line(client), _fmt(PLAY) + b"\n")
                    # 服务端写 '\\n' → 读端应读到
                    deadline = time.time() + 3.0
                    got = b""
                    while time.time() < deadline:
                        try:
                            got = os.read(rfd, 4096)
                            if got:
                                break
                        except BlockingIOError:
                            time.sleep(0.02)
                    self.assertEqual(got, b"\n", "FIFO 应收到回车触发邀请")
                    client.close()   # max_wait=0.5 兜底后发 DONE 时对端已关 (BrokenPipe 被捕获)
                finally:
                    os.close(rfd)
            finally:
                os.close(holder)
            t.join(timeout=3.0)
            self.assertFalse(t.is_alive())


if __name__ == "__main__":
    unittest.main()
