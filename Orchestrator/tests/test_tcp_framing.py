# -*- coding: utf-8 -*-
"""B4 TCP 层 loopback 测试: 分片 / 粘包 / 二次连接 / 重放 / 断链.

纯 loopback socket, 无 ESTUN 控制器/相机/机械臂 (CRP 不动).
覆盖 FramedLineReader + 最小协议会话 (真实 SessionBarrier 逻辑) 在
TCP 传输层伪象下的行为:
  1. 分片: 一条消息拆 1..N 次 send 到达, 帧读取器完整还原
  2. 粘包: 多条消息一次 send, 逐条读出
  3. 二次连接: 会话进行中第二连接无法注入/干扰第一会话
  4. 重放: 字节级重发同一消息, 帧层完整交付 (屏障层重放保护见
     d1a_protocol_fake_client.py: case_p1_replay / case_gripdone_repeat)
  5. 断链: 对端中途关闭/半条消息后关闭 → read_line 返回 None 不挂死
  6. 整流程: 分片+粘包混合下 P2→P1→GRIPDONE 会话正常完成
"""
import os
import socket
import sys
import threading
import time
import unittest

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from Orchestrator.session_barrier import (  # noqa: E402
    BarrierReject, SessionBarrier, SessionState, WristObservation,
    calib_fingerprint,
)
from Orchestrator.tcp_framing import FramedLineReader, send_line  # noqa: E402

# ---- 固定测试输入 (与 d1a_protocol_fake_client 同格式) ----
P0 = ((0.477, 0.231, 0.624), (-135.0, 4.5, -86.0))
P2_T = ((0.40, 0.10, 0.60), (-90.0, 89.0, -90.0))
P1_T = ((0.41, 0.11, 0.61), (-90.0, 89.0, -90.0))

CALIB = calib_fingerprint(np.eye(4))   # 合法格式指纹 (arm 校验格式)


def _fmt_pose(xyz, abc):
    return (f"{xyz[0]},{xyz[1]},{xyz[2]},"
            f"{abc[0]},{abc[1]},{abc[2]}")


def p2_cmd(pose=P2_T):
    return f"P2_ARRIVED|{_fmt_pose(pose[0], pose[1])}"


def p1_cmd(pose=P1_T):
    return f"P1_ARRIVED|{_fmt_pose(pose[0], pose[1])}"


def _recv_until_close(conn, timeout=5.0):
    """读至对端关闭 (测试辅助); 超时返回已收内容."""
    conn.settimeout(timeout)
    chunks = []
    while True:
        try:
            d = conn.recv(4096)
        except socket.timeout:
            break
        if not d:
            break
        chunks.append(d)
    return b"".join(chunks)


class TcpFramingTest(unittest.TestCase):
    """FramedLineReader 五项 TCP 伪象 + 整流程."""

    # ---------- 1. 分片 ----------

    def test_fragment_every_split(self):
        """一条 P2 消息在每个字节位置切开到达, 均完整还原."""
        msg = (p2_cmd() + "\n").encode("utf-8")
        for cut in range(1, len(msg)):
            srv, cli = socket.socketpair()
            reader = FramedLineReader(srv)
            cli.sendall(msg[:cut])
            time.sleep(0.001)
            cli.sendall(msg[cut:])
            line = reader.read_line(timeout=1.0)
            self.assertEqual(line, p2_cmd())
            cli.close()
            srv.close()

    def test_fragment_one_byte_chunks(self):
        """逐字节发送 (最极端分片), 仍完整还原."""
        msg = (p2_cmd() + "\n").encode("utf-8")
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        for b in msg:
            cli.sendall(bytes([b]))
        line = reader.read_line(timeout=1.0)
        self.assertEqual(line, p2_cmd())
        cli.close()
        srv.close()

    # ---------- 2. 粘包 ----------

    def test_coalesced_messages(self):
        """两条消息一次 send (粘包), 逐条读出, 无残留."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        cli.sendall((p2_cmd() + "\n" + "GRIPDONE\n").encode("utf-8"))
        self.assertEqual(reader.read_line(timeout=1.0), p2_cmd())
        self.assertEqual(reader.read_line(timeout=1.0), "GRIPDONE")
        cli.close()
        srv.close()

    def test_coalesced_three(self):
        """三条消息一次 send, 逐条读出."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        blob = (p2_cmd() + "\n" + p1_cmd() + "\n" + "GRIPDONE\n").encode("utf-8")
        cli.sendall(blob)
        self.assertEqual(reader.read_line(timeout=1.0), p2_cmd())
        self.assertEqual(reader.read_line(timeout=1.0), p1_cmd())
        self.assertEqual(reader.read_line(timeout=1.0), "GRIPDONE")
        cli.close()
        srv.close()

    # ---------- 3. 二次连接 (会话隔离) ----------

    def test_second_connection_isolated(self):
        """会话进行中, 第二连接的消息不影响第一会话的帧流."""
        srv1, cli1 = socket.socketpair()
        srv2, cli2 = socket.socketpair()
        reader1 = FramedLineReader(srv1)
        reader2 = FramedLineReader(srv2)
        # 第一会话收到 P2 (粘包中混入第二连接的噪声字节, 不应进入会话1)
        cli1.sendall((p2_cmd() + "\n").encode("utf-8"))
        cli2.sendall(b"\x00\x01 garbage not for session1\n")
        self.assertEqual(reader1.read_line(timeout=1.0), p2_cmd())
        # 会话1 后续仍正常
        cli1.sendall(b"GRIPDONE\n")
        self.assertEqual(reader1.read_line(timeout=1.0), "GRIPDONE")
        # 第二连接自己的帧流独立完整
        self.assertEqual(reader2.read_line(timeout=1.0),
                         "\x00\x01 garbage not for session1")
        cli1.close(); cli2.close(); srv1.close(); srv2.close()

    # ---------- 4. 重放 (帧层完整性) ----------

    def test_replay_same_bytes_delivered_twice(self):
        """同一消息字节重复发送, 帧层两次完整交付 (屏障层重放保护另测)."""
        msg = p2_cmd() + "\n"
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        cli.sendall(msg.encode("utf-8"))
        cli.sendall(msg.encode("utf-8"))
        self.assertEqual(reader.read_line(timeout=1.0), p2_cmd())
        self.assertEqual(reader.read_line(timeout=1.0), p2_cmd())
        cli.close()
        srv.close()

    # ---------- 5. 断链 ----------

    def test_disconnect_mid_line(self):
        """半条消息后对端关闭 → read_line 返回 None, 不挂死."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        cli.sendall(b"P2_ARRIVED|0.4,0.1,0.6,-90,89")
        cli.close()
        self.assertIsNone(reader.read_line(timeout=1.0))
        srv.close()

    def test_disconnect_clean(self):
        """完整会话后对端关闭 → 下一行 None."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        cli.sendall(b"GRIPDONE\n")
        self.assertEqual(reader.read_line(timeout=1.0), "GRIPDONE")
        cli.close()
        self.assertIsNone(reader.read_line(timeout=1.0))
        srv.close()

    def test_read_timeout_returns_none(self):
        """无数据超时 → None (不阻塞调用方)."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv)
        t0 = time.monotonic()
        self.assertIsNone(reader.read_line(timeout=0.3))
        self.assertLess(time.monotonic() - t0, 1.0)
        cli.close()
        srv.close()

    # ---------- 6. 整流程: 分片+粘包混合下真实屏障会话 ----------

    def _serve_one_session(self, conn):
        """最小协议会话: 初始位姿 → P2 → P1 → GRIPDONE (真实 SessionBarrier)."""
        reader = FramedLineReader(conn)
        barrier = SessionBarrier(run_id="tcp-framing-loopback",
                                 lease_epoch=1)
        barrier.arm(p0_pose=P0, p2_target=P2_T, p1_target=P1_T,
                    planning_observation_id=1000,
                    calibration_fingerprint=CALIB)
        results = {"completed": False, "error": None}
        try:
            init = reader.read_line(timeout=2.0)          # 初始位姿
            if init is None:
                results["error"] = "no init"
                return results
            # P2: 可接受分片/粘包到达
            line = reader.read_line(timeout=2.0)
            barrier.receive_p2(line, calibration_fingerprint=CALIB,
                               now_ns=time.monotonic_ns())
            obs = WristObservation(2001, time.monotonic_ns())   # 须为整数, ≠ planning 1000
            payload = barrier.authorize_grasp(obs,
                                              calibration_fingerprint=CALIB,
                                              now_ns=time.monotonic_ns())
            send_line(conn, payload)
            # P1
            line = reader.read_line(timeout=2.0)
            payload = barrier.receive_p1(line,
                                         calibration_fingerprint=CALIB,
                                         now_ns=time.monotonic_ns())
            send_line(conn, payload)
            # GRIPDONE
            line = reader.read_line(timeout=2.0)
            barrier.receive_gripdone(line, now_ns=time.monotonic_ns())
            results["completed"] = barrier.state is SessionState.COMPLETE
        except BarrierReject as e:
            results["error"] = f"{e.code}:{e.reason}"
        except Exception as e:  # noqa: BLE001
            results["error"] = repr(e)
        finally:
            try:
                conn.close()
            except OSError:
                pass
        return results

    def test_full_session_fragmented_coalesced(self):
        """分片+粘包混合: 完整 P2→P1→GRIPDONE 会话正常完成."""
        srv, cli = socket.socketpair()
        results = {}

        def serve():
            results.update(self._serve_one_session(srv))

        t = threading.Thread(target=serve, daemon=True)
        t.start()

        # 初始位姿: 分片发送
        init = (f"{P0[0][0]},{P0[0][1]},{P0[0][2]},"
                f"{P0[1][0]},{P0[1][1]},{P0[1][2]}\n").encode("utf-8")
        for b in init:
            cli.sendall(bytes([b]))
        # P2+P1 粘包一次发 (协议顺序: P2 回执后读 P1), GRIPDONE 逐字节分片.
        # TCP 保序: 全部发送后再读回执, 与服务端处理节奏解耦.
        cli.sendall((p2_cmd() + "\n" + p1_cmd() + "\n").encode("utf-8"))
        grip = b"GRIPDONE\n"
        for b in grip:
            cli.sendall(bytes([b]))
        data = _recv_until_close(cli).decode("utf-8")
        t.join(timeout=5.0)
        cli.close()
        self.assertFalse(t.is_alive(), "服务线程应正常结束")
        self.assertIsNone(results.get("error"), results.get("error"))
        self.assertTrue(results.get("completed"), results)
        self.assertIn("<0.477", data, f"P2 回执应为 <P0>: {data}")
        self.assertIn("[0.477", data, f"P1 回执应为 [P0]: {data}")

    # ---------- 超长行防护 ----------

    def test_oversized_line(self):
        """超过 max_line 的流 → None (防恶意/异常流)."""
        srv, cli = socket.socketpair()
        reader = FramedLineReader(srv, max_line=64)
        cli.sendall(b"P" * 100 + b"\n")
        self.assertIsNone(reader.read_line(timeout=1.0))
        cli.close()
        srv.close()


if __name__ == "__main__":
    unittest.main()
