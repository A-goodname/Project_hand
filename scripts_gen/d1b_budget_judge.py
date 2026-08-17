# -*- coding: utf-8 -*-
"""D1-B R2/R3/R6 离线预算判定器 (2026-08-16).

审计整改需求 `d1b-b1-coordinate-chain-repair-requirements-2026-08-16.md`:
  R2: 三维总误差预算 — max_norm_m (三维范数硬上限) + 分位数 + 样本覆盖度
      + 误差分配表 (视觉/深度、手眼、机器人重复定位、TCP/灵巧手偏置、
      目标测量、余量); 保守组合须 < 实际最小碰撞余量 (P1 间隙 10mm)
  R3: 姿态测地误差必须实测; 姿态真值/预测缺失 → FAIL (禁止跳过)
  R6: 离线测试矩阵 8 用例 (--self-test)

纯离线: 仅 numpy/yaml/json/stdlib, 不导入 pyrealsense2/相机/控制器 (R1.3).

样本 schema (与现场采集记录一致):
  {err_m, err_axis[x,y,z], geodesic_deg|None, stale|False,
   pose_id, region, point, base_pred_m, gt_m, monotonic_ns}
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

# ---- 预算 v2 (R2): 三维硬上限由误差分配表推导 (保守组合 8mm < P1 间隙 10mm) ----
ERROR_BUDGET_V2 = {
    "rms_m": 0.006,          # 位置 RMS
    "max_norm_m": 0.008,     # 三维误差范数硬上限 (R2 新增)
    "max_axis_m": 0.008,     # 单轴最大
    "p95_m": 0.010,          # 95 分位数 (R2 新增)
    "geodesic_deg": 3.0,     # 姿态测地 (R3: 缺失 → FAIL, 不跳过)
    "min_samples": 5,        # 最小有效样本数 (覆盖度)
}

# 误差分配表 (R2.2, 保守组合 = 8mm < 10mm P1 间隙, 余量 2mm)
ERROR_ALLOCATION = {
    "vision_depth": {"label": "视觉/深度(替代内参+彩深对齐)", "mm": 2.5, "status": "待独立验证"},
    "hand_eye": {"label": "手眼外参 (Daniilidis 26样本 0.228px)", "mm": 2.5, "status": "已标定待复核"},
    "robot_repeat": {"label": "机器人重复定位", "mm": 1.0, "status": "待测"},
    "tcp_hand_offset": {"label": "TCP/灵巧手偏置 (示教 6D)", "mm": 1.0, "status": "示教已含"},
    "target_measure": {"label": "目标测量 (基准点真值)", "mm": 1.0, "status": "卷尺 cm 级, 待机器人自量测"},
    "margin": {"label": "安全余量", "mm": 2.0, "status": "P1 间隙 10mm - 保守组合 8mm"},
}

# 单位纪律: 位置数值超过该范围视为 mm 误作 m (R6 用例)
_UNIT_SANITY_MAX_M = 3.0


def allocation_summary():
    """误差分配表: 分项 + 保守组合 + vs 实际最小碰撞余量判定."""
    items = []
    total = 0.0
    for key, v in ERROR_ALLOCATION.items():
        items.append({"item": key, **v})
        total += v["mm"]
    gap_p1 = 0.010
    ok = total < gap_p1
    return {
        "items": items,
        "conservative_sum_mm": total,
        "min_collision_gap_mm": gap_p1 * 1000.0,
        "pass": ok,
        "note": ("保守组合 < P1 间隙 10mm" if ok
                 else "保守组合 ≥ P1 间隙 — 需增大场景余量而非降阈值 (R2.3)"),
    }


def validate_rigid_matrix(T):
    """纯 numpy 刚体检查: 非刚体 / 单位矩阵 → 失败原因 (与 D0 同规则, 独立实现)."""
    if T.shape != (4, 4):
        return f"形状 {T.shape} 非 4x4"
    if not np.all(np.isfinite(T)):
        return "含非有限值"
    R = T[:3, :3]
    det = float(np.linalg.det(R))
    if not (0.9 <= det <= 1.1):
        return f"旋转行列式异常 {det:.4f}"
    if np.max(np.abs(R @ R.T - np.eye(3))) > 1e-3:
        return "旋转矩阵非正交 (非刚体)"
    if np.max(np.abs(T[3, :] - np.array([0.0, 0.0, 0.0, 1.0]))) > 1e-6:
        return "齐次变换最后一行异常"
    if (np.max(np.abs(T[:3, 3])) < 1e-6 and
            np.max(np.abs(R - np.eye(3))) < 1e-6):
        return "单位变换标定, 拒绝使用"
    return None


def _sample_norm(record):
    return float(np.linalg.norm(np.asarray(record["err_axis"], dtype=float)))


def judge(records, budget=None, *, attitude_required=True):
    """R2/R3 预算判定: 三维范数 + 分位数 + 覆盖度 + 姿态 FAIL 语义.

    records: 样本列表 (schema 见模块头). 返回 (summary, reasons: list[str]).
    reasons 非空 = 不通过 (含缺失姿态/作废样本/单位异常的具体原因).
    """
    budget = budget or ERROR_BUDGET_V2
    reasons: list[str] = []

    # 样本质量/单位纪律 (R6: 过期或运动中的法兰位姿 → 拒绝; mm 误作 m → 拒绝)
    valid = []
    for r in records:
        if r.get("stale"):
            reasons.append(f"样本 {r.get('pose_id', '?')} 作废: 位姿非拍摄时刻 (stale)")
            continue
        pred = np.asarray(r["base_pred_m"], dtype=float)
        gt = np.asarray(r["gt_m"], dtype=float)
        if (np.max(np.abs(pred)) > _UNIT_SANITY_MAX_M
                or np.max(np.abs(gt)) > _UNIT_SANITY_MAX_M):
            reasons.append(f"样本 {r.get('pose_id', '?')} 数值疑似 mm 误作 m: "
                           f"pred={pred.tolist()} gt={gt.tolist()}")
            continue
        valid.append(r)

    if len(valid) < budget["min_samples"]:
        reasons.append(f"有效样本 {len(valid)} < 最小覆盖 {budget['min_samples']} (R2 覆盖度)")
        if not valid:
            return _summary_empty(valid, budget, reasons), reasons

    errs = np.array([r["err_m"] for r in valid])
    norms = np.array([_sample_norm(r) for r in valid])
    axes = np.array([r["err_axis"] for r in valid])

    rms = float(np.sqrt(np.mean(errs ** 2)))
    max_norm = float(norms.max())
    max_axis = float(np.max(np.abs(axes)))
    p95 = float(np.percentile(errs, 95))
    # 姿态项 (R3): 每样本必须有 geodesic_deg; 缺失 → FAIL 并注明样本
    geodesics = []
    for r in valid:
        g = r.get("geodesic_deg")
        if g is None:
            reasons.append(f"样本 {r.get('pose_id', '?')} 姿态真值/预测缺失 "
                           f"(geodesic_deg=None) — R3 禁止跳过")
        else:
            geodesics.append(float(g))
    max_geo = max(geodesics) if geodesics else None
    if attitude_required and not geodesics:
        reasons.append("无任何有效姿态测地样本 (R3)")

    # 判定: 全部通过才 pass (R2: RMS + 最大三维范数 + 单轴 + 分位数 + 覆盖度)
    checks = {
        "rms_m": (rms <= budget["rms_m"], f"RMS {rms*1000:.1f}mm ≤ {budget['rms_m']*1000:.0f}mm"),
        "max_norm_m": (max_norm <= budget["max_norm_m"],
                       f"三维范数最大 {max_norm*1000:.1f}mm ≤ {budget['max_norm_m']*1000:.0f}mm"),
        "max_axis_m": (max_axis <= budget["max_axis_m"],
                       f"单轴最大 {max_axis*1000:.1f}mm ≤ {budget['max_axis_m']*1000:.0f}mm"),
        "p95_m": (p95 <= budget["p95_m"],
                  f"95 分位 {p95*1000:.1f}mm ≤ {budget['p95_m']*1000:.0f}mm"),
    }
    if max_geo is not None:
        checks["geodesic_deg"] = (max_geo <= budget["geodesic_deg"],
                                  f"姿态测地最大 {max_geo:.2f}° ≤ {budget['geodesic_deg']:.0f}°")
    for key, (ok, desc) in checks.items():
        if not ok:
            reasons.append(f"{key}: {desc}")

    coverage = {}
    for r in valid:
        coverage.setdefault(r.get("pose_id", "?"), 0)
        coverage[r.get("pose_id", "?")] += 1

    summary = {
        "n_valid": len(valid), "n_total": len(records),
        "rms_m": rms, "max_norm_m": max_norm, "max_axis_m": max_axis,
        "p95_m": p95, "max_geodesic_deg": max_geo,
        "budget": dict(budget), "checks": {k: v[0] for k, v in checks.items()},
        "coverage": coverage,
        "pass": not reasons,
        "reasons": reasons,
        "allocation": allocation_summary(),
        "monotonic_ns": time.monotonic_ns(),
    }
    return summary, reasons


def _summary_empty(valid, budget, reasons):
    return {
        "n_valid": 0, "n_total": len(valid) + 0, "rms_m": None,
        "max_norm_m": None, "max_axis_m": None, "p95_m": None,
        "max_geodesic_deg": None, "budget": dict(budget),
        "checks": {}, "coverage": {}, "pass": False, "reasons": reasons,
        "allocation": allocation_summary(),
        "monotonic_ns": time.monotonic_ns(),
    }


def report_jsonl(records, out_path):
    """同基准跨视场 JSONL 报告 (R6 用例 8): 原样落盘 + 覆盖统计."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out_path


# ============================================================
#  R6 离线测试矩阵 (8 用例)
# ============================================================

def _rec(err_m=0.002, axis=None, geo=1.0, stale=False, pred=None, gt=None, pid="P1"):
    return {
        "err_m": err_m, "err_axis": list(axis or [err_m / math.sqrt(3)] * 3),
        "geodesic_deg": geo, "stale": stale,
        "pose_id": pid, "region": "center", "point": "desk_ref",
        "base_pred_m": list(pred or [-0.370, 0.800, 0.295]),
        "gt_m": list(gt or [-0.370, 0.800, 0.295]),
        "monotonic_ns": 0,
    }


def _make_records(errs_mm, axis_pattern="spread", geos_deg=None, stale_flags=None):
    """按误差毫米构造样本集; axis_pattern: spread|one_axis."""
    records = []
    for i, e in enumerate(errs_mm):
        e = e / 1000.0
        if axis_pattern == "one_axis":
            axis = [e, 0.0, 0.0]
        else:
            axis = [e / math.sqrt(3)] * 3
        rec = _rec(err_m=e, axis=axis,
                   geo=None if geos_deg is None else geos_deg[i],
                   stale=bool(stale_flags and stale_flags[i]),
                   pid=f"P{i}")
        records.append(rec)
    return records


def self_test():
    """R6 八用例离线矩阵 (退出码 0/1)."""
    print(f"[R6 self-test] python={sys.version.split()[0]} "
          f"numpy={np.__version__} 纯离线 (无硬件导入)")
    results = []

    def case(name, expect_pass, records, **kw):
        summary, reasons = judge(records, **kw)
        ok = summary["pass"] == expect_pass
        results.append((name, ok, summary, reasons))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} "
              f"(期望 {'通过' if expect_pass else '拒绝'}, "
              f"实际 {'通过' if summary['pass'] else '拒绝'})")
        for rsn in reasons:
            print(f"      - {rsn}")

    # 1. 小三维误差且完整姿态真值 → 通过
    case("小三维误差且完整姿态真值", True,
         _make_records([2, 3, 2, 3, 2], geos_deg=[1, 1, 1, 1, 1]))
    # 2. 单轴未超限但三维范数超限 → 拒绝 (R2: max_norm 8mm, 单轴 [7,7,7]mm 未超)
    recs2 = [_rec(err_m=12.1 / 1000, axis=[7e-3, 7e-3, 7e-3], geo=1.0, pid=f"P{i}")
             for i in range(5)]
    case("单轴未超但三维范数超限", False, recs2)
    # 3. RMS 通过但存在单个大误差 → 拒绝 (20 小 + 1 大: RMS≈5.7mm 仍通过, 范数/分位拒绝)
    recs3 = _make_records([1] * 20 + [25], geos_deg=[1] * 21)
    case("RMS 通过但单点大误差", False, recs3)
    # 4. 姿态误差 4° (预算 3°) → 拒绝 (R3)
    case("姿态误差 4°(预算 3°)", False,
         _make_records([2] * 5, geos_deg=[4] * 5))
    # 5. 姿态真值/预测缺失 → 拒绝 (R3 禁止跳过)
    case("姿态真值/预测缺失", False,
         _make_records([2] * 5, geos_deg=[None] * 5))
    # 6. mm 误作 m / 非刚体 / 单位矩阵 → 拒绝 (单位纪律 + D0 刚体规则)
    recs = _make_records([2] * 5, geos_deg=[1] * 5)
    recs[0]["base_pred_m"] = [-370.0, 800.0, 295.0]   # mm 误作 m
    case("mm 误作 m", False, recs)
    for label, T, expect in (
            ("非刚体外参", np.array([[1, 1, 0, 0], [0, 1, 0, 0],
                                     [0, 0, 1, 0], [0, 0, 0, 1.]], dtype=float), True),
            ("单位矩阵", np.eye(4), True),
    ):
        reason = validate_rigid_matrix(T)
        ok = (reason is not None) == expect
        results.append((f"{label}拒绝", ok, {}, []))
        print(f"  [{'PASS' if ok else 'FAIL'}] {label} "
              f"({'拒绝' if reason is not None else '未拒绝'}: {reason})")
    # 7. 过期/运动中的法兰位姿 → 拒绝
    case("过期/运动中法兰位姿", False,
         _make_records([2] * 5, geos_deg=[1] * 5, stale_flags=[0, 1, 0, 0, 0]))
    # 8. 同基准跨视场 JSONL 报告
    recs8 = _make_records([2, 3, 2, 3, 2, 4], geos_deg=[1] * 6)
    for i, r in enumerate(recs8):
        r["pose_id"] = f"P_center" if i == 0 else (f"P_ring{i}" if i < 3 else f"P_edge{i}")
        r["region"] = "center" if i == 0 else ("ring" if i < 3 else "edge")
    out = os.path.join(_PROJECT_ROOT := os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "d1b", "budget-judge-self-test.jsonl")
    report_jsonl(recs8, out)
    summary8, _ = judge(recs8)
    ok8 = summary8["pass"] is True and len(summary8["coverage"]) >= 3
    print(f"  [{'PASS' if ok8 else 'FAIL'}] 同基准跨视场 JSONL 报告 "
          f"(覆盖 {len(summary8['coverage'])} 视场, 输出 {out})")
    results.append(("跨视场 JSONL", ok8, {}, []))

    npass = sum(1 for _, ok, _, _ in results if ok)
    print(f"\n[R6 self-test] {npass}/{len(results)} 用例通过 → "
          f"{'✅ 全绿' if npass == len(results) else '❌ 存在失败'}")
    return 0 if npass == len(results) else 1


def main():
    ap = argparse.ArgumentParser(description="D1-B R2/R3/R6 离线预算判定器")
    ap.add_argument("--self-test", action="store_true", help="R6 测试矩阵 (默认)")
    ap.add_argument("--judge", type=str, default=None, help="判定 JSONL 样本文件")
    ap.add_argument("--allocation", action="store_true", help="输出误差分配表")
    args = ap.parse_args()

    if args.allocation:
        print(json.dumps(allocation_summary(), ensure_ascii=False, indent=2))
        return 0
    if args.judge:
        with open(args.judge, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]
        summary, reasons = judge(records)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if summary["pass"] else 1
    return self_test()


if __name__ == "__main__":
    sys.exit(main())
