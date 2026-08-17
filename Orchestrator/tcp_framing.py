# -*- coding: utf-8 -*-
"""TCP 行帧读取器 (B4 TCP 层, 2026-08-16).

审计 B4: TCP 层测试 — 分片 / 粘包 / 二次连接 / 重放 / 断链.
协议消息为换行终止文本行 (如 P2_ARRIVED|x,y,z,A,B,C, GRIPDONE).
裸 `conn.recv` 一次一条的读法在 TCP 分片 (一条消息拆到多次 recv) 与
粘包 (多条消息挤进一次 recv) 下会破坏协议; 本模块提供带缓冲的行帧读取:

- 分片: 缓冲区累积直到完整行
- 粘包: 行后余量保留, 下一条继续读
- 断链/超时: 返回 None, 不挂死
- 行超长: 视为协议错误返回 None (防恶意流)

用法:
    reader = FramedLineReader(conn)
    line = reader.read_line(timeout=2.0)   # str(去换行) 或 None
    send_line(conn, "P2_ARRIVED|0.4,0.1,0.6,-90,89,-90")

生产接入点: state_machine Phase C/D 指令循环 (:1422 裸 recv) 替换为
FramedLineReader.read_line —— 与 #9 Phase 1a 改造一并落地, 避免并发改动.
CRP 侧不动.
"""
import socket


class FramedLineReader:
    """TCP 行帧读取器: 分片拼接 + 粘包保留 + 超时/EOF/超长防护.

    delimiter: 帧结束符 (默认换行; D2 遥测协议用分号 ";").
    """

    def __init__(self, conn: socket.socket, max_line: int = 4096,
                 delimiter: bytes = b"\n") -> None:
        self._conn = conn
        self._buf = b""
        self._max_line = max_line
        self._delim = delimiter
        self._eof = False

    @property
    def pending_bytes(self) -> int:
        """未分帧的残留缓冲字节数 (断链残帧检测, PROTOCOL_TRUNCATED)."""
        return len(self._buf)

    @property
    def eof(self) -> bool:
        """对端已关闭/连接失效 (超时不置位, 仅真正断链置位)."""
        return self._eof

    def read_line(self, timeout: float | None = None) -> str | None:
        """读取一帧 (delimiter 结尾, 返回去结束符 str).

        返回 None 表示: 超时 (eof=False, 可继续等) / EOF (eof=True) /
        帧超长 (eof=False). 调用方必须区分三者, 不得把超时当断链.
        """
        if timeout is not None:
            self._conn.settimeout(timeout)
        while self._delim not in self._buf:
            try:
                chunk = self._conn.recv(4096)
            except socket.timeout:
                return None                      # 仅超时, 连接仍活
            except OSError:
                self._eof = True
                return None                      # 连接已失效
            if not chunk:
                self._eof = True
                return None                      # 对端关闭
            self._buf += chunk
            if len(self._buf) > self._max_line:
                return None                      # 帧超长, 协议错误
        line, self._buf = self._buf.split(self._delim, 1)
        return line.decode("utf-8", errors="replace").strip()


def send_line(conn: socket.socket, text: str) -> None:
    """发送一行 (自动补 \n)."""
    conn.sendall(text.encode("utf-8") + b"\n")
