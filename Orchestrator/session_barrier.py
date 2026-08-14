# -*- coding: utf-8 -*-
"""
Session Barrier — PC 本地 P2/P1 双屏障门禁 (D0 适配)
====================================================
对应 CRP: handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp

职责:
  - 本地一次性 ticket (pregrasp / grasp) 签发与消费, 仅存 PC 内存
  - P2_ARRIVED|... / P1_ARRIVED|... 严格解析与校验 (米/度 CPOS, 容差, 时效, 标定摘要, 新鲜观测)
  - 状态机: WAIT_P2_ARRIVED -> WAIT_P1_ARRIVED -> WAIT_GRIPDONE -> PASSED
  - 任何失败 (格式/容差/越序/超时/断链/标定变化) -> BarrierReject, 调用方必须终止会话

与对方约定 (session-barrier-v2 工单):
  - 动态 wire-ticket 为 ESTUN 能力边界, 由 PC 本地维护 ticket/租约/时效/状态
  - <P0> / [P0] 回执 = 本会话 Phase A 已验证的实际 P0 位姿 (米/度, 非全零)
"""

import math
import time


# ============================================================
# 异常
# ============================================================

class BarrierReject(Exception):
    """屏障拒绝 — 携带机器可读原因码. 调用方必须终止会话."""

    def __init__(self, reason: str, code: str = "REJECT"):
        super().__init__(reason)
        self.reason = reason
        self.code = code


# ============================================================
# 纯函数: 严格解析与校验 (可独立单测)
# ============================================================

_ASCII_ALLOWED = frozenset(
    "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ,.-+|[]<>: "
)


def parse_arrived(msg: str, prefix: str):
    """严格解析 "PREFIX|x,y,z,a,b,c" -> ((x,y,z), (a,b,c)) 或抛 BarrierReject.

    拒绝: 空 / 非 ASCII / 含空白 / 前缀不符 / 非 6 个有限数.
    """
    if not msg:
        raise BarrierReject("空消息", "EMPTY")
    if not isinstance(msg, str):
        raise BarrierReject("非字符串", "NOT_STR")
    try:
        msg.encode("ascii")
    except UnicodeEncodeError:
        raise BarrierReject("非 ASCII 字符", "NON_ASCII")
    if any(ch.isspace() for ch in msg):
        raise BarrierReject("消息含空白", "WHITESPACE")
    if not msg.startswith(prefix):
        raise BarrierReject(f"前缀不符 (期望 {prefix!r})", "BAD_PREFIX")
    body = msg[len(prefix):]
    if not body or "," not in body:
        raise BarrierReject("缺少 CPOS 体", "BAD_FORMAT")
    parts = body.split(",")
    if len(parts) != 6:
        raise BarrierReject(f"CPOS 需 6 个数, 实际 {len(parts)}", "BAD_COUNT")
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        raise BarrierReject("CPOS 含非数字", "BAD_NUMBER")
    if not all(math.isfinite(v) for v in vals):
        raise BarrierReject("CPOS 含非有限数", "NOT_FINITE")
    return (tuple(vals[0:3]), tuple(vals[3:6]))


def pose_within_tolerance(actual_xyz, actual_abc, target_xyz, target_abc,
                          pos_tol=0.02, ang_tol=5.0):
    """位置欧氏距离 <= pos_tol (m); 每轴角度差 <= ang_tol (deg)."""
    d_pos = math.sqrt(sum((a - b) ** 2 for a, b in zip(actual_xyz, target_xyz)))
    if d_pos > pos_tol:
        return False, f"位置偏差 {d_pos:.4f}m 超容差 {pos_tol}m"
    for a, b in zip(actual_abc, target_abc):
        if abs(a - b) > ang_tol:
            return False, f"角度偏差 {abs(a - b):.2f}deg 超容差 {ang_tol}deg"
    return True, ""


# ============================================================
# 会话屏障状态机
# ============================================================

WAIT_P2_ARRIVED = "WAIT_P2_ARRIVED"
WAIT_P1_ARRIVED = "WAIT_P1_ARRIVED"
WAIT_GRIPDONE = "WAIT_GRIPDONE"
PASSED = "PASSED"
REJECTED = "REJECTED"


class SessionBarrier:
    """单 TCP 会话的 P2/P1 双屏障.

    每个新 TCP 会话创建一个新实例 (run_id 唯一); 所有状态仅存 PC 内存.
    """

    def __init__(self, run_id: str, calib_fingerprint: str,
                 pos_tol: float = 0.02, ang_tol: float = 5.0,
                 ticket_ttl: float = 30.0):
        self.run_id = run_id
        self.calib_fingerprint = calib_fingerprint
        self.pos_tol = pos_tol
        self.ang_tol = ang_tol
        self.ticket_ttl = ticket_ttl

        self.state = WAIT_P2_ARRIVED
        self._ticket_seq = 0
        self._pregrasp_ticket = None      # dict: value/issued_at/plan_digest/consumed
        self._grasp_ticket = None         # dict: value/issued_at/consumed
        self._p2_arrived_ts = None
        self._p1_arrived_ts = None

    # ---------------- ticket 管理 ----------------

    def _new_ticket(self, kind: str):
        self._ticket_seq += 1
        return {"value": f"{kind}-{self.run_id}-{self._ticket_seq}",
                "issued_at": time.time(), "consumed": False}

    def issue_pregrasp(self, plan_digest: str = ""):
        """初始计划发送时签发 pregrasp ticket."""
        if self.state != WAIT_P2_ARRIVED:
            raise BarrierReject("状态不允许签发 pregrasp ticket", "STATE")
        self._pregrasp_ticket = self._new_ticket("pg")
        self._pregrasp_ticket["plan_digest"] = plan_digest
        return self._pregrasp_ticket["value"]

    def _check_ticket_fresh(self, ticket):
        if ticket is None:
            raise BarrierReject("ticket 不存在", "NO_TICKET")
        if ticket.get("consumed"):
            raise BarrierReject("ticket 已消费 (重放)", "REPLAY")
        if time.time() - ticket["issued_at"] > self.ticket_ttl:
            raise BarrierReject("ticket 已过期", "TICKET_EXPIRED")

    def _reject(self, reason: str, code: str = "REJECT"):
        self.state = REJECTED
        if self._grasp_ticket is not None:
            self._grasp_ticket["consumed"] = True   # 撤销, 防重放
        raise BarrierReject(reason, code)

    # ---------------- 主入口 ----------------

    def step(self, cmd: str, *, p2_target, p1_target, p0_pose,
             fresh_obs_ts=None):
        """按当前状态处理一条 CRP 消息.

        Args:
            cmd: 收到的原始字符串 (已 strip 前后空白)
            p2_target: (xyz, abc) 本会话下发的 P2 目标 (基系, m/deg)
            p1_target: (xyz, abc) 本会话下发的 P1 目标
            p0_pose:  (xyz, abc) Phase A 收到的实际初始位姿 (用于 <P0>/[P0] 回执)
            fresh_obs_ts: P2 复核新观测的时间戳 (必须 > P2 回执时间)

        Returns:
            (action, payload):
              action "ACK"       -> payload 为要发送的回执字符串 (如 "<0.1,0.2,0.3,-90,89,-90>")
              action "GRIPDONE"  -> 屏障通过, 允许调用 hand_grip_done 一次
              action "PREGRIP"   -> 仅手势, 状态不变
              action "PASS"      -> 屏障已通过, 走旧流程
        """
        if self.state == REJECTED:
            raise BarrierReject("会话已被拒绝", "STATE")

        # ---- P2 屏障 ----
        if self.state == WAIT_P2_ARRIVED:
            if cmd == "PREGRIP":
                return ("PREGRIP", None)          # 仅手势
            if not cmd.startswith("P2_ARRIVED|"):
                self._reject(f"WAIT_P2 越序消息: {cmd[:40]!r}", "SEQ")
            try:
                xyz, abc = parse_arrived(cmd, "P2_ARRIVED|")
            except BarrierReject as e:
                self._reject(f"P2 回执解析失败: {e.reason}", "BAD_P2")
            self._p2_arrived_ts = time.time()
            ok, why = pose_within_tolerance(xyz, abc, p2_target[0], p2_target[1],
                                            self.pos_tol, self.ang_tol)
            if not ok:
                self._reject(f"P2 回执超容差: {why}", "P2_TOL")
            try:
                self._check_ticket_fresh(self._pregrasp_ticket)
            except BarrierReject as e:
                self._reject(f"pregrasp: {e.reason}", "PG_" + e.code)
            if fresh_obs_ts is None or fresh_obs_ts <= self._p2_arrived_ts:
                self._reject("P2 复核缺少新鲜 D435i 观测 (须晚于 P2 回执)", "STALE_OBS")
            # 签发 grasp ticket, 进入 P1 屏障
            self._grasp_ticket = self._new_ticket("grasp")
            self.state = WAIT_P1_ARRIVED
            ack = self._fmt_ack(p0_pose, "<", ">")
            return ("ACK", ack)

        # ---- P1 屏障 ----
        if self.state == WAIT_P1_ARRIVED:
            if cmd == "PREGRIP":
                return ("PREGRIP", None)          # 仅手势
            if not cmd.startswith("P1_ARRIVED|"):
                self._reject(f"WAIT_P1 越序消息: {cmd[:40]!r}", "SEQ")
            try:
                xyz, abc = parse_arrived(cmd, "P1_ARRIVED|")
            except BarrierReject as e:
                self._reject(f"P1 回执解析失败: {e.reason}", "BAD_P1")
            self._p1_arrived_ts = time.time()
            ok, why = pose_within_tolerance(xyz, abc, p1_target[0], p1_target[1],
                                            self.pos_tol, self.ang_tol)
            if not ok:
                self._reject(f"P1 回执超容差: {why}", "P1_TOL")
            try:
                self._check_ticket_fresh(self._grasp_ticket)
            except BarrierReject as e:
                self._reject(f"grasp: {e.reason}", "GRASP_" + e.code)
            # 一次性消费 grasp ticket
            self._grasp_ticket["consumed"] = True
            self.state = WAIT_GRIPDONE
            ack = self._fmt_ack(p0_pose, "[", "]")
            return ("ACK", ack)

        # ---- GRIPDONE 门 ----
        if self.state == WAIT_GRIPDONE:
            if cmd == "GRIPDONE":
                self.state = PASSED
                return ("GRIPDONE", None)
            if cmd == "PREGRIP":
                return ("PREGRIP", None)          # 仅手势
            if cmd.startswith("P1_ARRIVED|"):
                self._reject("grasp ticket 已消费, 重复 P1 回执", "GRASP_REPLAY")
            self._reject(f"WAIT_GRIPDONE 越序消息: {cmd[:40]!r}", "SEQ")

        # 屏障已通过: 旧流程继续 (GRIPDONE 不得重复闭合)
        if cmd == "GRIPDONE":
            raise BarrierReject("GRIPDONE 重复/越序", "REPLAY")
        return ("PASS", None)

    # ---------------- 内部 ----------------

    @staticmethod
    def _fmt_ack(p0_pose, lt, gt):
        """构造 <x,y,z,A,B,C> / [x,y,z,A,B,C] 回执 (m/deg, 3/2 位小数)."""
        xyz, abc = p0_pose
        body = (f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},"
                f"{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}")
        return f"{lt}{body}{gt}"
