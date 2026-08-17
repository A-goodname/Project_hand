# -*- coding: utf-8 -*-
"""
D2 遥测服务端 (端口 8082) — ESTUN 阶段实际位姿事件记录器
============================================================
对应对方方案 `d2-estun-event-pose-logging-operation-plan-2026-08-17.md` §3/§5:

  - CRP 在关键阶段到位后 GetCurCPos → CPosToStr → SocketSendStr (Socket0/8082),
    消息格式: <EVENT>|<x_m>,<y_m>,<z_m>,<a_deg>,<b_deg>,<c_deg>;  (分号 ";" 唯一结束符)
  - 本服务端按 ";" 分帧 (半帧/粘连/一次 recv 多帧/断链残帧), 白名单 9 事件,
    校验 6 个有限十进制数 + m/deg 单位纪律 + 事件顺序;
  - 每帧 append 一行 `signals/robot_state.jsonl` (schema estun-cpos-event-v1) 并 flush;
  - 同步写 state_transition.jsonl / action_receipt.jsonl;
  - 会话正常结束 → fsync + 生成 artifact_manifest.json / episode_manifest.json;
  - 断链残留字节 → PROTOCOL_TRUNCATED; 事件缺失/重复/乱序 → episode 标记无效
    (expert_valid=false 并附原因), 不阻塞 CRP 安全收尾.

本服务端只落盘与记录, 不下发运动命令, 不替代 ESTUN 运动权限 (方案 §5).
P2/P1 帧体 (去 ";") 通过 on_p2_p1 回调透传给既有 SessionBarrier (默认 no-op).

用法:
  python Orchestrator/estun_d2_telemetry_sm.py --port 8082 \
      --episode-id tea-frontal-canonical-a-001-20260817T1500 \
      --scene-batch-id canonical-a --out logs/d2
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import socket
import sys
import threading
import time

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Orchestrator.tcp_framing import FramedLineReader  # noqa: E402

# ---- D2 9 事件白名单与严格顺序 (方案 §3.2) ----
D2_EVENT_SEQUENCE = [
    "P0_ACTUAL",
    "P2_ARRIVED",
    "P1_ARRIVED",
    "P4_AFTER_LIFT",
    "P3_ARRIVED",
    "P4_AFTER_POUR",
    "P1_RETURNED",
    "P2_FINAL_ARRIVED",
    "HOME_ARRIVED",
]
D2_EVENT_SET = set(D2_EVENT_SEQUENCE)

# 单位纪律: 位置 m / 角度 deg (与预算判定器同阈值)
_UNIT_POS_MAX_M = 3.0
_UNIT_ANG_MAX_DEG = 360.0

SCHEMA_VERSION = "estun-cpos-event-v1"
CLOCK_ID = "pc-monotonic-ingress-v1"
IDLE_TIMEOUT_S = 300.0   # 无数据超过该时长视为死链 (CRP 事件间隔为秒级运动)


def parse_event_frame(line: str):
    """解析 'EVENT|x,y,z,a,b,c' → (event, xyz_m, abc_deg).

    校验: 恰好一个 '|' / 事件在白名单 / 恰好 6 个有限十进制数 /
          m-deg 单位纪律 (|pos|≤3m, |ang|≤360°). 失败抛 ValueError(原因).
    """
    if "|" not in line:
        raise ValueError("缺少 '|' 分隔符")
    event, _, payload = line.partition("|")
    if event not in D2_EVENT_SET:
        raise ValueError(f"未知事件 {event!r} (白名单: {D2_EVENT_SEQUENCE})")
    parts = payload.split(",")
    if len(parts) != 6:
        raise ValueError(f"字段数 {len(parts)} != 6")
    try:
        vals = [float(p) for p in parts]
    except ValueError as e:
        raise ValueError(f"非数值字段: {e}") from e
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("含 NaN/Infinity")
    xyz, abc = vals[:3], vals[3:]
    if any(abs(v) > _UNIT_POS_MAX_M for v in xyz):
        raise ValueError(f"位置疑似非 m 单位: {xyz}")
    if any(abs(v) > _UNIT_ANG_MAX_DEG for v in abc):
        raise ValueError(f"角度疑似非 deg 单位: {abc}")
    return event, xyz, abc


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _config_host(default="0.0.0.0"):
    """与 estun_* 一致: host 从 estun_config.yaml tcp_server.host 读取 (具体 IP)."""
    try:
        import yaml  # noqa: PLC0415
        path = os.path.join(_project_root, "Estun", "estun_config.yaml")
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return cfg["estun"]["tcp_server"]["host"]
    except Exception:  # noqa: BLE001
        return default


class D2TelemetryServer:
    """8082 遥测服务端 — 监听模式与 estun_handle_data_sm 完全一致:

    host 取自 estun_config.yaml tcp_server.host (本机具体 IP, 如 192.168.101.96),
    bind + listen(5) + setblocking(False), 主循环 _accept_tcp (BlockingIOError 跳过),
    每连接一个 daemon 线程. 无条件接受连接, 无对端过滤.
    """

    def __init__(self, host=None, port=8082, ports=None, out_dir=None,
                 episode_id=None, scene_batch_id=None,
                 on_p2_p1=None, crp_ip="192.168.101.100"):
        # host=None → 从 estun_config.yaml 读取 (与 estun_* 同源); 读不到回 0.0.0.0
        # ports: 同时监听多个端口 (CRP 可能走 Socket0=8080 或新 Socket=8082)
        self._host = host if host is not None else _config_host()
        self._ports = tuple(ports) if ports else (int(port),)
        self._out_dir = os.path.abspath(out_dir) if out_dir else os.path.join(
            _project_root, "logs", "d2")
        self._episode_id = episode_id
        self._scene_batch_id = scene_batch_id or "canonical-a"
        self._on_p2_p1 = on_p2_p1 or (lambda event, xyz, abc: None)
        self._crp_ip = crp_ip
        self._servers = {}     # port -> socket
        self._stop_event = threading.Event()
        self._session_active = False
        self._session_lock = threading.Lock()

    # ---------------- 生命周期 (镜像 estun_handle_data_sm) ----------------

    def _init_tcp(self):
        """每端口 bind + listen + 非阻塞 (estun 同款)."""
        os.makedirs(self._out_dir, exist_ok=True)
        for p in self._ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setblocking(False)
            s.bind((self._host, p))
            s.listen(5)
            self._servers[p] = s
            print(f"[D2-TELEMETRY] 监听 {self._host}:{p} → {self._out_dir}")

    def _accept_tcp(self, port):
        """非阻塞 accept (estun 同款: BlockingIOError 跳过)."""
        srv = self._servers.get(port)
        if srv is None:
            return
        try:
            conn, addr = srv.accept()
        except BlockingIOError:
            return
        with self._session_lock:
            if self._session_active:
                print(f"[D2-TELEMETRY] 拒绝并发遥测会话: {addr}")
                conn.close()
                return
            self._session_active = True
        print(f"[D2-TELEMETRY] CRP 已连接 ({port}): {addr}")
        t = threading.Thread(target=self.handle_connection,
                             args=(conn, addr), daemon=True)
        t.start()

    def _accept_loop(self, port):
        while not self._stop_event.is_set():
            self._accept_tcp(port)
            time.sleep(0.02)

    def start(self):
        """每端口一个 accept 线程 + 主线程等 stop."""
        self._init_tcp()
        threads = [threading.Thread(target=self._accept_loop, args=(p,),
                                    daemon=True) for p in self._ports]
        for t in threads:
            t.start()
        try:
            while not self._stop_event.is_set():
                time.sleep(0.1)
        finally:
            for s in self._servers.values():
                try:
                    s.close()
                except OSError:
                    pass

    def stop(self):
        self._stop_event.set()

    # ---------------- 会话处理 (可离线单测) ----------------

    def handle_connection(self, conn, addr="loopback"):
        """单会话: 分号分帧 → 校验 → 落盘 → 会话结束生成 manifest."""
        episode_id = self._episode_id or (
            f"tea-{self._scene_batch_id}-"
            f"{time.strftime('%Y%m%dT%H%M%S')}")
        ep_dir = os.path.join(self._out_dir, episode_id)
        raw_dir = os.path.join(ep_dir, "raw")
        sig_dir = os.path.join(ep_dir, "signals")
        os.makedirs(sig_dir, exist_ok=True)
        os.makedirs(raw_dir, exist_ok=True)   # raw/ 由相机记录器填充, 先建目录

        rs_path = os.path.join(sig_dir, "robot_state.jsonl")
        st_path = os.path.join(sig_dir, "state_transition.jsonl")
        ar_path = os.path.join(sig_dir, "action_receipt.jsonl")
        rs_fh = open(rs_path, "a", encoding="utf-8")
        st_fh = open(st_path, "a", encoding="utf-8")
        ar_fh = open(ar_path, "a", encoding="utf-8")

        def w(fh, rec):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()

        def ts_rec(rec_type, **fields):
            return {"type": rec_type,
                    "episode_id": episode_id,
                    "pc_monotonic_ns": time.monotonic_ns(),
                    "pc_wall_time_utc": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(),
                    "clock_id": CLOCK_ID, **fields}

        w(st_fh, ts_rec("session_start", remote=repr(addr)))
        w(ar_fh, ts_rec("socket_connected", remote=repr(addr)))

        # 事件顺序跟踪 (方案 §3.2: 缺项/重复/乱序 → episode 无效)
        seq_index = 0
        seen = []
        invalid_reasons = []
        reader = FramedLineReader(conn, delimiter=b";")
        protocol_truncated = False
        idle_deadline = time.monotonic() + IDLE_TIMEOUT_S   # 死链保护

        try:
            while True:
                line = reader.read_line(timeout=2.0)
                if line is None:
                    if reader.eof:
                        # 对端关闭: 残帧 → PROTOCOL_TRUNCATED, 会话结束
                        if reader.pending_bytes > 0:
                            protocol_truncated = True
                            reason = ("PROTOCOL_TRUNCATED: 断链残留 "
                                      f"{reader.pending_bytes} 字节未分帧")
                            invalid_reasons.append(reason)
                            w(st_fh, ts_rec("protocol_error", reason=reason))
                        break
                    # 超时无数据: 会话存活, 继续等 (CRP 事件间隔 = 秒级运动)
                    if time.monotonic() > idle_deadline:
                        reason = f"IDLE_TIMEOUT: {IDLE_TIMEOUT_S}s 无数据, 视为死链"
                        invalid_reasons.append(reason)
                        w(st_fh, ts_rec("protocol_error", reason=reason))
                        break
                    continue
                idle_deadline = time.monotonic() + IDLE_TIMEOUT_S
                if not line:
                    continue
                try:
                    event, xyz, abc = parse_event_frame(line)
                except ValueError as e:
                    reason = f"帧解析失败: {e} | raw={line!r}"
                    invalid_reasons.append(reason)
                    w(st_fh, ts_rec("protocol_error", reason=reason))
                    continue
                # 事件顺序纪律
                if seq_index < len(D2_EVENT_SEQUENCE) and \
                        event == D2_EVENT_SEQUENCE[seq_index]:
                    seq_index += 1
                else:
                    invalid_reasons.append(
                        f"事件顺序异常: 期望 {D2_EVENT_SEQUENCE[seq_index] if seq_index < len(D2_EVENT_SEQUENCE) else '(已完成)'}, "
                        f"实际 {event}")
                seen.append(event)
                # robot_state.jsonl (方案 §3.3 schema)
                w(rs_fh, {
                    "schema_version": SCHEMA_VERSION,
                    "episode_id": episode_id,
                    "event": event,
                    "pc_monotonic_ns": time.monotonic_ns(),
                    "pc_wall_time_utc": datetime.datetime.now(
                        datetime.timezone.utc).isoformat(),
                    "clock_id": CLOCK_ID,
                    "pose_source": "estun-crp-getcurcpos",
                    "controller_timestamp_available": False,
                    "state_kind": "ESTUN_TCP_EVENT_SNAPSHOT",
                    "tcp_frame": "base",
                    "xyz_m": xyz,
                    "abc_deg": abc,
                    "joint_state_available": False,
                    "raw_payload": line,
                })
                w(st_fh, ts_rec("event_received", event=event, xyz_m=xyz,
                                abc_deg=abc))
                # P2/P1 帧体透传给既有 SessionBarrier (方案 §5.4, 默认 no-op)
                if event in ("P2_ARRIVED", "P1_ARRIVED"):
                    self._on_p2_p1(event, xyz, abc)
                    w(ar_fh, ts_rec("forwarded_to_barrier", event=event))
        finally:
            conn.close()
            seq_complete = seq_index == len(D2_EVENT_SEQUENCE)
            w(st_fh, ts_rec("session_end",
                            events_seen=seen,
                            sequence_complete=seq_complete,
                            invalid_reasons=invalid_reasons))
            rs_fh.flush()
            os.fsync(rs_fh.fileno())
            os.fsync(st_fh.fileno())
            os.fsync(ar_fh.fileno())
            rs_fh.close()
            st_fh.close()
            ar_fh.close()
            self._finalize_manifests(ep_dir, episode_id, seen,
                                     seq_complete, invalid_reasons,
                                     protocol_truncated)
            with self._session_lock:
                self._session_active = False

    # ---------------- manifest (方案 §5.6/§7) ----------------

    def _finalize_manifests(self, ep_dir, episode_id, seen, seq_complete,
                            invalid_reasons, protocol_truncated):
        """会话结束: artifact_manifest.json + episode_manifest.json (SHA-256)."""
        files = []
        for root, _dirs, names in os.walk(ep_dir):
            for n in sorted(names):
                p = os.path.join(root, n)
                rel = os.path.relpath(p, ep_dir)
                st = os.stat(p)
                files.append({
                    "artifact": "ROBOT_STATE" if n == "robot_state.jsonl"
                                else "STATE_TRANSITION" if n == "state_transition.jsonl"
                                else "ACTION_RECEIPT" if n == "action_receipt.jsonl"
                                else "RAW_PLACEHOLDER",
                    "relpath": rel,
                    "media_type": "application/jsonl" if n.endswith(".jsonl")
                                  else "inode/directory",
                    "sha256": _sha256_file(p),
                    "bytes": st.st_size,
                    "clock_id": CLOCK_ID,
                })
        with open(os.path.join(ep_dir, "artifact_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"episode_id": episode_id, "files": files},
                      f, ensure_ascii=False, indent=2)

        expert_valid = seq_complete and not invalid_reasons \
            and not protocol_truncated
        with open(os.path.join(ep_dir, "episode_manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump({
                "episode_id": episode_id,
                "scene_batch_id": self._scene_batch_id,
                "crp_peer_ip": self._crp_ip,
                "outcome": ("SUCCESS" if seq_complete and not invalid_reasons
                            else "ABORTED"),
                "automatic_motion_started": seq_complete,
                "phase_flags": {e: (e in seen) for e in D2_EVENT_SEQUENCE},
                "events_seen": seen,
                "sequence_complete": seq_complete,
                "invalid_reasons": invalid_reasons,
                "protocol_truncated": protocol_truncated,
                "expert_valid": False,   # 保守: 默认 false, 待来源/标定审计
                "expert_valid_reason": ("待来源/标定/回执审计" if expert_valid
                                        else ("; ".join(invalid_reasons) or
                                              "事件序列不完整")),
                "source_evidence": {},
            }, f, ensure_ascii=False, indent=2)
        print(f"[D2-TELEMETRY] 会话结束: {episode_id} "
              f"seq_complete={seq_complete} invalid={len(invalid_reasons)} "
              f"expert_valid=false(待审计)")


def main():
    ap = argparse.ArgumentParser(description="D2 遥测服务端 (8082)")
    ap.add_argument("--host", default=None,
                    help="监听地址 (默认读 estun_config.yaml tcp_server.host)")
    ap.add_argument("--ports", default="8082,8080",
                    help="监听端口列表 (逗号分隔; CRP Socket0=8080 / 新 Socket=8082)")
    ap.add_argument("--port", type=int, default=None,
                    help="单端口简写 (等价 --ports)")
    ap.add_argument("--out", default=None, help="episode 输出根目录 (默认 logs/d2)")
    ap.add_argument("--episode-id", default=None)
    ap.add_argument("--scene-batch-id", default="canonical-a")
    ap.add_argument("--crp-ip", default="192.168.101.100",
                    help="CRP 地址 (manifest 记录用)")
    args = ap.parse_args()
    ports = (args.port,) if args.port else tuple(
        int(p) for p in args.ports.split(",") if p.strip())
    srv = D2TelemetryServer(host=args.host, ports=ports, out_dir=args.out,
                            episode_id=args.episode_id,
                            scene_batch_id=args.scene_batch_id,
                            crp_ip=args.crp_ip)
    try:
        srv.start()
    except KeyboardInterrupt:
        srv.stop()


if __name__ == "__main__":
    main()
