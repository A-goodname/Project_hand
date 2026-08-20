# -*- coding: utf-8 -*-
"""18 回合旋转判档抓取证据集 — 清单 + 单回合驱动 + 证据校验.

对方 2026-08-20 `runtime-evidence-addendum` §3: 中心档 ±30/±15/0 每档 2 回合 +
换档边界 ±22/±8 两侧 ±2° 各 1 回合 = 18 回合. +45 不在矩阵; 每回合记录
物理限位就位; 每个有效检测帧记录 ratio/d/rd/handle_side/bbox/贴纸中心.

用法 (conda Proj):
  python scripts_gen/rotation_evidence_rounds.py            # 打印 18 回合清单
  python scripts_gen/rotation_evidence_rounds.py --run R01  # 启动 R01 回合 (subprocess data_sm),
                                                             #   等待 socket 断开自动保存证据后停止
  python scripts_gen/rotation_evidence_rounds.py --check    # 校验已采集回合的证据字段齐全

data_sm 会话由机械臂侧连接触发 (ESTUN CRP SocketConnect → Phase A/B/C);
证据 JSONL 在 socket 断开 (会话 finally) 时自动落盘, 不依赖手动退出.
"""
import argparse
import json
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_EVIDENCE_DIR = os.path.join(_PROJECT_ROOT, "logs", "rotation_evidence")
_DATA_SM = os.path.join(_PROJECT_ROOT, "Orchestrator", "estun_handle_data_sm.py")
_D2_OUT = os.path.join(_PROJECT_ROOT, "logs", "d2")

# 18 回合清单: (round_id, target_angle_deg, 说明)
ROUNDS = [
    # 中心档 5 档 × 2 回合
    ("R01", -30, "中心档 -30° 回合 1"),
    ("R02", -30, "中心档 -30° 回合 2"),
    ("R03", -15, "中心档 -15° 回合 1"),
    ("R04", -15, "中心档 -15° 回合 2"),
    ("R05", 0,   "中心档 0° 回合 1"),
    ("R06", 0,   "中心档 0° 回合 2"),
    ("R07", 15,  "中心档 +15° 回合 1"),
    ("R08", 15,  "中心档 +15° 回合 2"),
    ("R09", 30,  "中心档 +30° 回合 1"),
    ("R10", 30,  "中心档 +30° 回合 2"),
    # 换档边界 ±22/±8 两侧 ±2° × 1 回合
    ("R11", -24, "边界 -22 侧 (-24°)"),
    ("R12", -20, "边界 -22 侧 (-20°)"),
    ("R13", -10, "边界 -8 侧 (-10°)"),
    ("R14", -6,  "边界 -8 侧 (-6°)"),
    ("R15", 6,   "边界 +8 侧 (+6°)"),
    ("R16", 10,  "边界 +8 侧 (+10°)"),
    ("R17", 20,  "边界 +22 侧 (+20°)"),
    ("R18", 24,  "边界 +22 侧 (+24°)"),
]
ROUND_MAP = {rid: (ang, desc) for rid, ang, desc in ROUNDS}

REQUIRED_ROUND_FIELDS = [
    "status", "grasp_result", "theta_samples", "theta_med", "band",
    "p1", "p2", "abc", "p1_z_pre_clamp", "p1_z_post_clamp",
    "target_angle_deg", "config_sha256",
]
REQUIRED_SAMPLE_FIELDS = ["ratio", "d", "rd", "handle_side",
                          "kettle_bbox", "tag_u", "tag_v"]


def cmd_annotate(rid, note):
    """人工标注失败/异常原因 — 追加 type=annotation 记录到证据 JSONL.
    用于机械臂侧拒绝 (如 P2 逆解不可达) 等 data_sm 拿不到的原因, 不影响 done 判定."""
    path = _evidence_path(rid)
    if not os.path.exists(path):
        print(f"[ERROR] 无证据文件: {path}")
        return 1
    rec = {"type": "annotation", "round_id": rid,
           "note": note, "ts": time.time()}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[OK] {rid} 已标注: {note}")
    return 0


def _data_sm_cmd(rid, angle):
    return [sys.executable, _DATA_SM,
            "--out", _D2_OUT, "--scene-batch-id", "rotation-evidence",
            "--round-id", rid, "--target-angle", str(angle)]


def _evidence_path(rid):
    return os.path.join(_EVIDENCE_DIR, f"{rid}.jsonl")


def _read_lines(path):
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def _round_done(path):
    """证据文件中是否已有 type=round 且 status=done 的记录."""
    for rec in _read_lines(path):
        if rec.get("type") == "round" and rec.get("status") == "done":
            return rec
    return None


def cmd_list():
    print(f"18 回合证据集清单 (证据目录: {_EVIDENCE_DIR})")
    print(f"{'回合':<6}{'目标角':>8}  说明")
    print("-" * 46)
    for rid, ang, desc in ROUNDS:
        done = "✓" if os.path.exists(_evidence_path(rid)) else " "
        print(f"{done} {rid:<5}{ang:>8}  {desc}")


def cmd_run(rid):
    if rid not in ROUND_MAP:
        print(f"[ERROR] 未知回合 {rid}; 可用: {sorted(ROUND_MAP)}")
        return 1
    ang, desc = ROUND_MAP[rid]
    path = _evidence_path(rid)
    if _round_done(path):
        print(f"[WARN] {rid} 已有 done 记录: {path}")
        return 1

    print(f"===== {rid}: {desc} =====")
    print(f"1) 请操作员把壶把手摆到 {ang}° (量角器/参考线), 物理限位就位")
    print(f"2) 机械臂侧触发抓取 (CRP SocketConnect → Phase A/B/C)")
    print(f"   启动 data_sm: {' '.join(_data_sm_cmd(rid, ang))}")
    print(f"3) socket 断开后证据自动保存到: {path}")
    if input("回车开始启动 data_sm (Ctrl+C 取消): ").strip():
        pass

    proc = subprocess.Popen(_data_sm_cmd(rid, ang))
    try:
        # 轮询等待回合 done 记录 (socket 断开 → finally 自动落盘)
        deadline = time.time() + 180
        while time.time() < deadline:
            done = _round_done(path)
            if done:
                print(f"\n[OK] {rid} 完成: {done.get('grasp_result')} "
                      f"(band={done.get('band')}, θ_med={done.get('theta_med')})")
                return 0
            time.sleep(2)
        print(f"[ERROR] 180s 内未出现 done 记录: {path}")
        return 1
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def cmd_check():
    bad = 0
    for rid, ang, desc in ROUNDS:
        path = _evidence_path(rid)
        done = _round_done(path)
        if done is None:
            print(f"[  ] {rid}: 无 done 记录")
            continue
        missing_r = [f for f in REQUIRED_ROUND_FIELDS if f not in done]
        samples = [r for r in _read_lines(path) if r.get("type") == "sample"]
        missing_s = []
        if samples:
            first = samples[0]
            missing_s = [f for f in REQUIRED_SAMPLE_FIELDS if f not in first]
        ok = not missing_r and not missing_s and done.get("grasp_result") in (
            "SUCCESS", "REJECT", "NO_GRIPDONE")
        if ok:
            print(f"[✓] {rid} ({ang}°): {done['grasp_result']} "
                  f"band={done.get('band')} θ_med={done.get('theta_med')} "
                  f"samples={len(samples)}")
        else:
            bad += 1
            print(f"[✗] {rid} ({ang}°): grasp={done.get('grasp_result')} "
                  f"缺round字段={missing_r} 缺sample字段={missing_s}")
    print(f"\n校验完成: 18 回合中 {18 - bad} 合格, {bad} 异常")
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description="18 回合证据集驱动")
    ap.add_argument("--run", metavar="R01", help="启动并驱动单回合 (subprocess data_sm)")
    ap.add_argument("--check", action="store_true", help="校验已采集回合证据字段齐全")
    ap.add_argument("--annotate", nargs=2, metavar=("R01", "原因"),
                    help="人工标注失败/异常原因 (追加 type=annotation 记录)")
    args = ap.parse_args()
    if args.annotate:
        return cmd_annotate(args.annotate[0], args.annotate[1])
    if args.run:
        return cmd_run(args.run)
    if args.check:
        return cmd_check()
    cmd_list()
    return 0


if __name__ == "__main__":
    sys.exit(main())
