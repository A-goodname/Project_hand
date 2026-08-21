#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Phase E 石头剪刀布 (RPS) 机械臂接入 — 单一入口 (2026-08-21).

RPS 语音/视觉/灵巧手出拳由 /data/Speech 服务负责 (rps_stream.py + realtime_voice.py --rps-game).
本脚本只负责机械臂: 按 CRP 协议把臂端到 P1 示教点位(玩家面前), 游戏期间保持不动,
游戏完成发 DONE 标志, 回 Home.

配套 CRP (用户编写, 端口 8700): open → P1(sendpos→getpos→movl) → While(等完成标志) → Home(存储点) → close.
TCP 协议 (8700, 单 Socket 持久连接, 位姿串 [x,y,z,a,b,c] 单位 米,度, 末尾换行):
  1. X1  ← 机器人当前位姿串 (getcurcpos→cpostostr→sendstr)
  2. P1  → py 回 config rps.poses.play (米,度) → 机器人 transtrtocpos→movl RPSpos (游戏进行, 臂保持)
  3. DONE→ py 检测到 [RPS][RESULT] 后发 (任意非空串, 内容不比) → CRP While(whileflag==1) 读到即置 0 break
  4.      → 机器人 movl Home (存储点直走, py 不回位姿) → socketclose → py 收 EOF 优雅退出
位姿串 = 当前用户 CRP: cpostostr/transtrtocpos 均 lengthunit=m angleunit=deg (非 Phase 0 的 mm).

约束:
  - 禁止 open 灵巧手 (RPS 服务独占 STM32 串口) / 禁止碰相机 (RPS 服务独占 /dev/video6)
  - 游戏完成信号 = tail realtime_voice stdout 日志 [RPS][RESULT] (不改同事代码; 一键脚本用 python -u 解缓冲)

多局制 (config rps.rounds):
  rounds=0  (推荐): 无限局 — 臂保持 P1, 玩家每局说 "开始" 续局, 说 "退出游戏/不玩了"
                    结束 (rps_voice_game 退出分支打日志裸行 "退出", 本脚本 tail 到即 DONE 回 Home)
  rounds=N: 固定 N 局 RESULT 后 DONE 回 Home
  两种模式均受 max_game_wait 总超时兜底.

用法:
  conda run -n Proj python Estun/estun_rps_phaseE.py                 # 实机 (人工在 RPS 语音终端按回车邀请)
  conda run -n Proj python Estun/estun_rps_phaseE.py --auto-invite  # 自动写 FIFO 触发邀请
  conda run -n Proj python Estun/estun_rps_phaseE.py --mock         # 协议自测: 不发真游戏等待, 短时后模拟完成
  conda run -n Proj python Estun/estun_rps_phaseE.py --no-rps       # 跳过日志等待, 立即发完成标志
按键: 终端 'q' 跳过 (机器人未连接时)
"""
import argparse
import errno
import os
import select
import socket
import sys
import threading
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import yaml


def _load_estun_config():
    path = os.path.join(_project_root, "Estun", "estun_config.yaml")
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["estun"]


class PhaseERps:
    """Phase E RPS 机械臂接入: TCP 8700 协议 + RPS 游戏协调 (单一入口)."""

    def __init__(self, estun_cfg, port=8700, mock=False, no_rps=False,
                 auto_invite=None):
        self._cfg = estun_cfg
        self._port = port
        self._mock = mock
        self._no_rps = no_rps
        self._stop_event = threading.Event()

        r = estun_cfg.get("rps", {})
        self._done_flag = str(r.get("done_flag", "DONE"))
        self._voice_log = str(r.get("voice_log_path", "/tmp/rps_voice_phaseE.log"))
        self._settle_delay = float(r.get("arm_settle_delay", 3.0))
        self._max_wait = float(r.get("max_game_wait", 180.0))
        # rounds>=1 = 固定 N 局后 DONE; rounds=0 = 等人退出 (玩家说退出关键词/超时 → DONE)
        self._rounds = int(r.get("rounds", 0))
        # 退出关键词的日志裸行 (rps_voice_game.py 退出分支 print("退出"), 行内容即 "退出")
        self._quit_keyword = str(r.get("quit_keyword", "退出"))

        # ---- P1 位姿 (米, 度) — Home 为 CRP 存储点直 movl, py 不回位姿 ----
        self._play_pose = [float(v) for v in r.get("poses", {}).get("play", [])]
        if len(self._play_pose) != 6:
            raise ValueError("rps.poses.play 缺失: 请先在 estun_config.yaml 示教填入 6 元素位姿 (米,度)")
        if all(abs(v) < 1e-9 for v in self._play_pose) and not self._mock:
            raise ValueError("rps.poses.play 为全零占位 (未示教). 实机请示教后填入; 协议测试请用 --mock")

        # ---- 自动邀请 (FIFO → realtime_voice stdin) ----
        ai = r.get("auto_invite", {})
        self._auto_invite = bool(auto_invite) if auto_invite is not None \
            else bool(ai.get("enabled", False))
        self._fifo_path = str(ai.get("fifo_path", "/tmp/rps_voice_stdin"))

        self._server = None

    # ================================================================
    #  位姿串 (米,度) — 与用户 CRP TranStrToCpos lengthunit=m 匹配
    # ================================================================

    @staticmethod
    def _fmt_pose6(pose6):
        """6元素位姿 → "x,y,z,a,b,c" (米,度) — CRP TranStrToCpos lengthunit=m angleunit=deg."""
        return f"{pose6[0]:.3f},{pose6[1]:.3f},{pose6[2]:.3f}," \
               f"{pose6[3]:.2f},{pose6[4]:.2f},{pose6[5]:.2f}"

    # ================================================================
    #  RPS 游戏协调 (只读语音日志 + FIFO; 不 open 灵巧手/相机)
    # ================================================================

    def _auto_invite_trigger(self):
        """写 '\\n' 到 FIFO (realtime_voice stdin) 触发邀请; 失败提示人工按回车."""
        fifo = self._fifo_path
        try:
            fd = os.open(fifo, os.O_WRONLY | os.O_NONBLOCK)
        except OSError as e:
            if e.errno in (errno.ENXIO, errno.ENOENT):
                print(f"[RPS-PHASE-E] FIFO {fifo} 不存在/无读者 → 无法自动邀请, 请人工按回车")
                return False
            raise
        try:
            os.write(fd, b"\n")
        finally:
            os.close(fd)
        print(f"[RPS-PHASE-E] 已写回车到 FIFO {fifo} → 触发 RPS 邀请")
        return True

    def _wait_round_done(self, timeout, start_pos=None):
        """Tail 语音日志, 等游戏结束. 返回 bool (是否等到).

        rounds>=1: 数 [RPS][RESULT] 到 N 局 (臂保持 P1, 玩家每局说"开始"续局);
        rounds=0: 无限局直到玩家说退出关键词 (日志裸行 "退出", 见
        /data/Speech/rps_voice_game.py handle_user_text quit 分支 print("退出"))
        或超时. 超时/退出均视为"结束" → 发 DONE 回 Home.
        start_pos: 日志起始字节位置 (连接建立时捕获) — 只数它之后新出现的行.
        """
        if self._mock:
            print("[RPS-PHASE-E] --mock: 模拟游戏 3s 后完成")
            time.sleep(3.0)
            return True
        if self._no_rps:
            print("[RPS-PHASE-E] --no-rps: 跳过游戏等待, 立即完成")
            return True
        target = self._rounds if self._rounds >= 1 else None   # None = 等人退出
        log = self._voice_log
        pos = 0
        if start_pos is not None:
            pos = start_pos
        elif os.path.exists(log):
            pos = os.path.getsize(log)  # 只数新出现的行
        else:
            print(f"[RPS-PHASE-E] 语音日志 {log} 不存在 (RPS 服务没起?) → 等 {timeout:.0f}s 兜底")
        mode = f"目标 {target} 局" if target else "等人退出 (说 '退出游戏/不玩了' 结束)"
        print(f"[RPS-PHASE-E] 等待游戏结束: {mode}, 超时 {timeout:.0f}s (tail {log}) ...")
        seen = 0
        deadline = time.time() + timeout
        while time.time() < deadline and not self._stop_event.is_set():
            if os.path.exists(log):
                try:
                    with open(log, "r", encoding="utf-8", errors="replace") as fh:
                        if pos > os.path.getsize(log):  # 文件被截断/轮转
                            pos = 0
                        fh.seek(pos)
                        new = fh.read()
                        pos = fh.tell()
                except OSError:
                    new = ""
                if new:
                    for line in new.splitlines():
                        if "[RPS][RESULT]" in line:
                            seen += 1
                            print(f"[RPS-PHASE-E] 检测到第 {seen} 局完成: {line.strip()}")
                            if target is not None and seen >= target:
                                print("[RPS-PHASE-E] 达到目标局数, 即将发送完成标志")
                                return True
                        elif line.strip() == self._quit_keyword:
                            print("[RPS-PHASE-E] 玩家说退出关键词 → 结束游戏, 即将发送完成标志")
                            return True
            time.sleep(0.5)
        print(f"[RPS-PHASE-E] 等待游戏结束超时 ({timeout:.0f}s) → 兜底发完成标志回 Home")
        return False

    # ================================================================
    #  TCP 协议 (与用户 CRP: open→P1→While→Home→close 匹配)
    # ================================================================

    def _handle_session(self, conn):
        """X1→P1 / (游戏) →DONE / X2→Home."""
        print("[RPS-PHASE-E] 客户端已连接, 开始协议")
        try:
            # ---- 0. 记录语音日志起始位置 (跳过本连接之前的旧 RESULT) ----
            tail_start = 0
            if os.path.exists(self._voice_log):
                tail_start = os.path.getsize(self._voice_log)

            # ---- 1. 收 X1 (机器人当前位姿) → 回 P1 ----
            conn.settimeout(30.0)
            data = conn.recv(4096)
            if not data:
                print("[RPS-PHASE-E] 未收到 X1 (当前位姿)")
                return
            print(f"[RPS-PHASE-E] Step X1 recv: {data.decode('utf-8', errors='replace').strip()}")
            conn.sendall(self._fmt_pose6(self._play_pose).encode() + b"\n")
            print(f"[RPS-PHASE-E] 回 P1: {self._fmt_pose6(self._play_pose)} → 机器人 movl P1")

            # ---- 2. 等臂到位 → 触发邀请 → 等游戏完成 → 发 DONE ----
            print(f"[RPS-PHASE-E] 等机械臂到位 {self._settle_delay:.0f}s ...")
            time.sleep(self._settle_delay)
            if self._auto_invite:
                self._auto_invite_trigger()
            else:
                print("[RPS-PHASE-E] 请人工在 RPS 语音终端按回车触发邀请 (或加 --auto-invite)")
            self._wait_round_done(timeout=self._max_wait, start_pos=tail_start)
            conn.sendall(self._done_flag.encode() + b"\n")
            print(f"[RPS-PHASE-E] 发完成标志: {self._done_flag} → CRP While 退出")

            # ---- 3. 机器人 movl Home (存储点直走, py 不回位姿) → 收 EOF 优雅退出 ----
            conn.settimeout(15.0)
            try:
                data = conn.recv(4096)
            except socket.timeout:
                data = None
            if data:
                print(f"[RPS-PHASE-E] 收到额外数据 (忽略): {data.decode('utf-8', errors='replace').strip()}")
            print("[RPS-PHASE-E] 等待 socketclose (机器人已回 Home 存储点) → 优雅退出")
        except socket.timeout:
            print("[RPS-PHASE-E] Step 超时")
        except Exception as e:
            print(f"[RPS-PHASE-E] 异常: {e}")
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[RPS-PHASE-E] Phase E 完成")

    # ================================================================
    #  生命周期
    # ================================================================

    def serve(self):
        """非阻塞 accept + 单连接; 终端 'q' 跳过."""
        host = self._cfg.get("tcp_server", {}).get("host", "0.0.0.0")
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((host, self._port))
        server.listen(1)
        server.setblocking(False)
        self._server = server
        print(f"[RPS-PHASE-E] TCP 服务器: {host}:{self._port}")
        print("[RPS-PHASE-E] 等待机器人连接... (终端 'q' 跳过)")
        conn = None
        tty_stdin = sys.stdin.isatty()
        try:
            while conn is None and not self._stop_event.is_set():
                rlist = [server]
                if tty_stdin:
                    rlist.append(sys.stdin)
                try:
                    ready, _, _ = select.select(rlist, [], [], 0.1)
                except (OSError, ValueError):
                    ready = []
                if server in ready:
                    try:
                        conn, addr = server.accept()
                        print(f"[RPS-PHASE-E] 机器人已连接: {addr}")
                        break
                    except BlockingIOError:
                        pass
                if sys.stdin in ready:
                    line = sys.stdin.readline()
                    if line.strip().lower() == "q":
                        print("[RPS-PHASE-E] 用户跳过")
                        return
        finally:
            server.close()
            self._server = None
        if conn is not None:
            self._handle_session(conn)

    def stop(self):
        self._stop_event.set()
        if self._server is not None:
            try:
                self._server.close()
            except Exception:
                pass
            self._server = None
        print("[RPS-PHASE-E] 已退出")


def main():
    ap = argparse.ArgumentParser(
        description="Phase E RPS 机械臂接入 (配套 CRP: open→P1→While→Home→close, 端口 8700)")
    ap.add_argument("--port", type=int, default=None,
                    help="TCP 端口 (默认 config rps.port=8700)")
    ap.add_argument("--mock", action="store_true",
                    help="协议自测: 不发真游戏等待, 模拟完成")
    ap.add_argument("--no-rps", action="store_true",
                    help="跳过日志等待, 立即发完成标志")
    ap.add_argument("--auto-invite", default=None, choices=["0", "1", "true", "false"],
                    help="覆盖 config auto_invite.enabled")
    args = ap.parse_args()

    estun_cfg = _load_estun_config()
    r = estun_cfg.get("rps", {})
    port = args.port if args.port is not None else int(r.get("port", 8700))

    auto_invite = None
    if args.auto_invite is not None:
        auto_invite = args.auto_invite in ("1", "true")

    svc = PhaseERps(estun_cfg, port=port, mock=args.mock, no_rps=args.no_rps,
                    auto_invite=auto_invite)
    try:
        svc.serve()
    except KeyboardInterrupt:
        print("[RPS-PHASE-E] 用户中断")
    finally:
        svc.stop()


if __name__ == "__main__":
    main()
