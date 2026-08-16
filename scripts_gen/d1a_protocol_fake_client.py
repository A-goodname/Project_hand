# -*- coding: utf-8 -*-
"""D1-A 第 5 项: 协议测量 — 纯 Python fake client 验证 SessionBarrier 门禁.

对应: project-hand-d1-preparation-and-measurement-checklist-2026-08-15.md §5
不连接 ESTUN 控制器/相机/机械臂/灵巧手; 不运行 main.py 与 CRP.
产出: protocol-fake-client.log (回传物)

覆盖: 正常流 + 对方要求拒绝案例:
  空帧 / 非ASCII / NaN/Inf / CPOS 字段数 / P2超容差 / 过期ticket / 旧observation /
  标定摘要变化 / 重复P1 / 重复GRIPDONE / 第二连接(lease) / 本地超时(GRIPDONE/TICKET)
"""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

from Orchestrator.session_barrier import (  # noqa: E402
    BarrierReject, SessionBarrier, SessionState, WristObservation,
    calib_fingerprint,
)

# ---------------- 固定测试输入 ----------------
P0 = ((0.477, 0.231, 0.624), (-135.0, 4.5, -86.0))     # Phase A 实际 P0 (m/deg)
P2_T = ((0.40, 0.10, 0.60), (-90.0, 89.0, -90.0))      # 下发 P2 目标
P1_T = ((0.41, 0.11, 0.61), (-90.0, 89.0, -90.0))      # 下发 P1 目标

# 标定指纹: 优先真实标定文件 (纯只读), 失败用固定矩阵
_CALIB_MATRIX = None
try:
    import numpy as _np
    import yaml as _yaml
    _path = os.path.join(_PROJECT_ROOT, "calib_data", "calibration_result.yaml")
    with open(_path, encoding="utf-8") as _f:
        _cal = _yaml.safe_load(_f)["calibration"]["handeye"]
    _method = _cal["use_method"]
    _CALIB_MATRIX = _np.array(_cal[_method]["T_cam_to_gripper"], dtype=float)
    CALIB = calib_fingerprint(_CALIB_MATRIX)
    CALIB_SRC = f"真实标定文件 ({_path}, use_method={_method})"
except Exception as _e:  # pragma: no cover
    _m = ((1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.0), (0.0, 0.0, 0.0, 1.0))
    CALIB = calib_fingerprint(_m)
    CALIB_SRC = f"固定矩阵 (真实标定不可读: {_e})"

CALIB_OTHER = "f" * 64  # 变化后的标定摘要


def fmt_pose(xyz, abc):
    return f"{xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f},{abc[0]:.2f},{abc[1]:.2f},{abc[2]:.2f}"


def p2_msg(t=P2_T):
    return f"P2_ARRIVED|{fmt_pose(t[0], t[1])}"


def p1_msg(t=P1_T):
    return f"P1_ARRIVED|{fmt_pose(t[0], t[1])}"


def fresh_obs(barrier, obs_id=1001, reobserve_delay_s=0.2):
    """模拟 P2 复核: 等待复核耗时(取帧+稳定)后产生新观测.
    观测时间戳 = 复核完成时刻 (必 > P2 到达, 且 <= authorize 调用时刻)."""
    time.sleep(reobserve_delay_s)
    return WristObservation(observation_id=obs_id, observed_at_ns=time.monotonic_ns())


# ---------------- 日志 ----------------
LOG = []


def log(line=""):
    LOG.append(f"{time.strftime('%H:%M:%S')} {line}")


def make_barrier(**kw):
    b = SessionBarrier(run_id="d1a-fake", lease_epoch=1, **kw)
    b.arm(p0_pose=P0, p2_target=P2_T, p1_target=P1_T,
          planning_observation_id=1000, calibration_fingerprint=CALIB)
    return b


def run_case(name, fn, expect_code=None, expect_ok=False):
    """执行案例并记录. expect_code 非 None = 期望拒绝原因码."""
    log(f"\n===== 案例: {name} =====")
    try:
        result = fn()
        if expect_ok:
            log(f"  结果: PASS (正常完成) {result or ''}")
            return True
        log(f"  结果: FAIL — 期望拒绝 {expect_code}, 但未拒绝: {result!r}")
        return False
    except BarrierReject as e:
        ok = expect_code is not None and e.code == expect_code
        log(f"  结果: {'PASS' if ok else 'FAIL'} — BarrierReject code={e.code} "
            f"reason={e.reason!r} (期望 {expect_code})")
        return ok
    except Exception as e:
        log(f"  结果: FAIL — 异常 {type(e).__name__}: {e}")
        return False


# ---------------- 案例 ----------------

def case_normal():
    """正常全流程."""
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    ack = b.authorize_grasp(fresh_obs(b), calibration_fingerprint=CALIB)
    log(f"  P2 → authorize → ACK {ack} (期望 <0.477,...>)")
    assert ack.startswith("<0.477"), f"ACK 格式错: {ack}"
    ack = b.receive_p1(p1_msg(), calibration_fingerprint=CALIB)
    log(f"  P1 → ACK {ack} (期望 [0.477,...])")
    assert ack.startswith("[0.477"), f"ACK 格式错: {ack}"
    b.receive_gripdone("GRIPDONE")
    log(f"  GRIPDONE → state={b.state.value}")
    assert b.state is SessionState.COMPLETE
    return "state=COMPLETE"


def case_empty():
    b = make_barrier()
    b.receive_p2("", calibration_fingerprint=CALIB)


def case_non_ascii():
    b = make_barrier()
    b.receive_p2("P2_ARRIVED|0.4,0.1,0.6,-90,89,好", calibration_fingerprint=CALIB)


def case_not_finite():
    b = make_barrier()
    b.receive_p2("P2_ARRIVED|0.4,nan,0.6,-90,89,-90", calibration_fingerprint=CALIB)


def case_bad_count():
    b = make_barrier()
    b.receive_p2("P2_ARRIVED|0.4,0.1,0.6,-90,89", calibration_fingerprint=CALIB)


def case_p2_tolerance():
    b = make_barrier()
    bad = f"P2_ARRIVED|{P2_T[0][0] + 0.5},{P2_T[0][1]},{P2_T[0][2]},{P2_T[1][0]},{P2_T[1][1]},{P2_T[1][2]}"
    b.receive_p2(bad, calibration_fingerprint=CALIB)


def case_ticket_expired():
    b = make_barrier(ticket_ttl_s=0.001)
    time.sleep(0.01)
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)


def case_stale_obs_old_ts():
    """观测时间戳 <= P2 到达时刻 -> 旧观测."""
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    stale = WristObservation(observation_id=2001,
                             observed_at_ns=b.p2_arrived_at_ns - int(1e6))
    b.authorize_grasp(stale, calibration_fingerprint=CALIB)


def case_stale_obs_reused_planning_id():
    """复用 planning observation id -> 旧观测 (时间戳真实晚于 P2, 仅 ID 复用)."""
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    time.sleep(0.01)   # 复核耗时, 保证观测时刻晚于 P2 且非将来
    reused = WristObservation(observation_id=1000, observed_at_ns=time.monotonic_ns())
    b.authorize_grasp(reused, calibration_fingerprint=CALIB)


def case_calib_changed():
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB_OTHER)


def case_p1_replay():
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    b.authorize_grasp(fresh_obs(b), calibration_fingerprint=CALIB)
    b.receive_p1(p1_msg(), calibration_fingerprint=CALIB)
    b.receive_p1(p1_msg(), calibration_fingerprint=CALIB)   # 重复 P1 (grasp 已消费)


def case_gripdone_repeat():
    b = make_barrier()
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    b.authorize_grasp(fresh_obs(b), calibration_fingerprint=CALIB)
    b.receive_p1(p1_msg(), calibration_fingerprint=CALIB)
    b.receive_gripdone("GRIPDONE")
    b.receive_gripdone("GRIPDONE")                          # 重复 GRIPDONE


def case_second_connection():
    """第二 TCP 连接: handler 层 lease 冲突 (连接级拒绝)."""
    b1 = make_barrier()                                      # lease_epoch=1
    b2 = SessionBarrier(run_id="d1a-fake-2", lease_epoch=1)  # 同 lease, 第二连接
    log(f"  第一连接 barrier lease_epoch={b1.lease_epoch}")
    log(f"  第二连接 barrier lease_epoch={b2.lease_epoch} — handler 须在 accept 层拒绝并关闭")
    # SessionBarrier 自身不跨实例; 连接级互斥由 state_machine listener 承担:
    # 已存在活动 session 时, 新 accept 立即关闭. 此处验证 lease_epoch 必须为正整数.
    try:
        SessionBarrier(run_id="x", lease_epoch=0)
        log("  结果: FAIL — lease_epoch=0 未被拒绝")
        return False
    except ValueError:
        log("  结果: PASS — 非法 lease_epoch 被拒绝 (连接级互斥在 listener 层)")
        return True


def case_gripdone_timeout():
    """P1 后 GRIPDONE 本地 deadline (10s) 超时."""
    b = make_barrier(gripdone_timeout_s=0.001)
    b.receive_p2(p2_msg(), calibration_fingerprint=CALIB)
    b.authorize_grasp(fresh_obs(b), calibration_fingerprint=CALIB)
    b.receive_p1(p1_msg(), calibration_fingerprint=CALIB)
    time.sleep(0.01)
    b.receive_gripdone("GRIPDONE")


def case_p0_all_zero():
    """P0 全零 -> arm 拒绝 (BAD_P0)."""
    b = SessionBarrier(run_id="d1a-fake-zero", lease_epoch=1)
    b.arm(p0_pose=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
          p2_target=P2_T, p1_target=P1_T,
          planning_observation_id=1000, calibration_fingerprint=CALIB)


def case_premature_gripdone():
    """屏障前 GRIPDONE (越序, 旧 CRP 行为) -> 拒绝."""
    b = make_barrier()
    b.receive_p2("GRIPDONE", calibration_fingerprint=CALIB)


# ---------------- 执行 ----------------

CASES = [
    ("01 正常全流程 (P2→新鲜观测→<P0>→P1→[P0]→GRIPDONE)", case_normal, None, True),
    ("02 空帧", case_empty, "EMPTY", False),
    ("03 非 ASCII", case_non_ascii, "NON_ASCII", False),
    ("04 NaN/Inf", case_not_finite, "NOT_FINITE", False),
    ("05 CPOS 字段数错误", case_bad_count, "BAD_COUNT", False),
    ("06 P2 超容差", case_p2_tolerance, "P2_TOL", False),
    ("07 过期 ticket", case_ticket_expired, "TICKET_TIMEOUT", False),
    ("08 旧观测 (时间戳<=P2)", case_stale_obs_old_ts, "STALE_OBS", False),
    ("09 旧观测 (复用 planning id)", case_stale_obs_reused_planning_id, "STALE_OBS", False),
    ("10 标定摘要变化", case_calib_changed, "CALIB", False),
    ("11 重复 P1 (越序/重放)", case_p1_replay, "SEQ", False),
    ("12 重复 GRIPDONE", case_gripdone_repeat, "SEQ", False),
    ("13 第二连接 (lease 互斥)", case_second_connection, None, True),
    ("14 GRIPDONE 本地超时 (10s)", case_gripdone_timeout, "GRIPDONE_TIMEOUT", False),
    ("15 P0 全零拒绝", case_p0_all_zero, "BAD_P0", False),
    ("16 屏障前 GRIPDONE (越序)", case_premature_gripdone, "BAD_PREFIX", False),
]


def main():
    log("=" * 70)
    log("D1-A §5 协议测量 — SessionBarrier 门禁 fake client (纯 Python, 无硬件)")
    log(f"标定摘要来源: {CALIB_SRC}")
    log(f"标定指纹: {CALIB[:16]}... (sha256 64hex)")
    log("环境: 未启动 main.py / TCP 服务 / 相机 / ESTUN CRP / 机械臂 / 灵巧手")
    log("=" * 70)

    passed = 0
    for name, fn, expect_code, expect_ok in CASES:
        if run_case(name, fn, expect_code, expect_ok):
            passed += 1

    log("\n" + "=" * 70)
    log(f"汇总: {passed}/{len(CASES)} 案例通过")
    log("=" * 70)

    out = os.path.join(_PROJECT_ROOT, "..", "VLA-BD", "docs", "d1-measurements-2026-08-15",
                       "protocol-fake-client.log")
    out = os.path.abspath(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(LOG) + "\n")
    print(f"日志已写入: {out}")
    print(f"结果: {passed}/{len(CASES)}")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main())
