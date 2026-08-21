#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""三 CRP 融合: Phase 0(8600) + Phase A2D(8080) + Phase E(8700) → doc/estun_fused_phase_all.crp.

用户定案 (2026-08-21):
  - 段间 wait 3s (三 py 均预启动并已监听, wait 仅作 socket 交接缓冲)
  - Home 去重为一个点 (Phase0/A2D 共享 ID ptmquqgw91h2026u; PhaseE 的 Home 重指到此)
  - A2D Phase D socketreadstr 接受延时 20s→60s (py 最坏 ~55s 判决; 不动框架)
  - 变量全保留: projectcfg.vars.data 三文件逐字节相同(用 A2D); data.vars.project = A2D(15)+E(5) 并集
  - 位姿单位各段保留: 段1 mm, 段2/段3 m
"""
import copy
import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, ".claude", "skills", "estun-omnihand-robotics", "scripts"))

from estun_crp_helpers import (  # noqa: E402
    clone_block, find_child, find_task, load_json, write_json,
)

P0 = os.path.join(_ROOT, "doc", "estun_greeting_phase0.crp")
A2D = os.path.join(_ROOT, "doc", "estun_handle_cup_sm.crp")
E = os.path.join(_ROOT, "doc", "estun_rps_phaseE.crp")
OUT = os.path.join(_ROOT, "doc", "estun_fused_phase_all.crp")

HOME_ID = "ptmquqgw91h2026u"   # Phase0/A2D 共享 Home 点 ID (保留)
E_HOME_ID = "ptmt2eglfiqt8c3e"  # PhaseE 的 Home 点 ID (重指到 HOME_ID 后丢弃)


def _get_data(w):
    d = w.get("data", {})
    return json.loads(d) if isinstance(d, str) else d


def _clone_wait(wait_template, ms):
    """克隆一个 wait 块并设时长 (ms)."""
    blk = clone_block(wait_template)
    d = _get_data(blk)
    d["value"]["time"] = ms
    blk["data"] = d
    return blk


def _walk_widgets(wlist, fn):
    for w in wlist:
        fn(w)
        for c in w.get("children", []):
            _walk_widgets([c], fn)


def main():
    p0 = load_json(P0)
    a2d = load_json(A2D)
    e = load_json(E)

    t_p0, t_a2d, t_e = find_task(p0), find_task(a2d), find_task(e)
    w_p0 = find_child(t_p0, "widgets")
    w_a2d = find_child(t_a2d, "widgets")
    w_e = find_child(t_e, "widgets")
    pts_p0 = find_child(t_p0, "points")
    pts_e = find_child(t_e, "points")

    # ---- 以 A2D 为基 (widgets/points/vars 最全, projectcfg 三文件逐字节相同) ----
    fused = copy.deepcopy(a2d)
    t_f = find_task(fused)
    w_f = find_child(t_f, "widgets")
    pts_f = find_child(t_f, "points")

    # ---- 找一个 wait 模板 (A2D 内) ----
    wait_tpl = next((w for w in w_f["children"] if w.get("type") == "wait"), None)
    if wait_tpl is None:
        raise RuntimeError("A2D 内无 wait 模板")

    # ---- 组装 widget 顺序: P0 + wait3s + A2D + wait3s + E ----
    final = []
    final.extend(copy.deepcopy(w_p0["children"]))   # 段1 Phase 0
    final.append(_clone_wait(wait_tpl, 3000))        # 段1→段2
    final.extend(copy.deepcopy(w_f["children"]))     # 段2 Phase A2D
    final.append(_clone_wait(wait_tpl, 3000))        # 段2→段3
    final.extend(copy.deepcopy(w_e["children"]))     # 段3 Phase E
    w_f["children"] = final

    # ---- 点位合并: 去重 Home (统一到 A2D), 其余全保留 ----
    for pts in (pts_p0, pts_e):
        for p in pts["children"]:
            if p.get("label") == "Home":
                continue  # 统一到 A2D 的 Home (P0 同 ID 已存在; E 的 Home 重指)
            if any(q.get("id") == p.get("id") for q in pts_f["children"]):
                continue
            pts_f["children"].append(copy.deepcopy(p))

    # ---- data.vars.project 并集: A2D 15 + E 5 ----
    pv_f = fused.get("vars", {}).get("project", [])
    have = {v.get("key") for v in pv_f}
    for v in e.get("vars", {}).get("project", []):
        if v.get("key") not in have:
            pv_f.append(copy.deepcopy(v))
            have.add(v.get("key"))

    # ---- 重指 PhaseE 的 Home 引用 (E_HOME_ID → HOME_ID) ----
    replaced = {"n": 0}

    def _remap_refs(w):
        d = _get_data(w)
        if isinstance(d, dict):
            def _walk(o):
                if isinstance(o, dict):
                    for k, val in o.items():
                        if isinstance(val, str) and val == E_HOME_ID:
                            o[k] = HOME_ID
                            replaced["n"] += 1
                        else:
                            _walk(val)
                elif isinstance(o, list):
                    for item in o:
                        _walk(item)
            _walk(d)
        w["data"] = d if isinstance(d, dict) else w.get("data")

    _walk_widgets(w_f["children"], _remap_refs)

    # ---- A2D Phase D socketreadstr: mtesttime 20000→60000 ----
    fixed = {"n": 0}

    def _fix_phase_d(w):
        if w.get("type") != "socketreadstr":
            return
        d = _get_data(w)
        val = d.get("value", {})
        if val.get("mdataval", {}).get("datavar", {}).get("value") == "socketstr5":
            old = val.get("mtesttime")
            val["mtesttime"] = 60000
            w["data"] = d
            fixed["n"] += 1
            print(f"[FUSE] Phase D socketreadstr mtesttime {old} → 60000")

    _walk_widgets(w_f["children"], _fix_phase_d)

    # ---- 写盘 ----
    write_json(OUT, fused)
    print(f"[FUSE] 已写出 {OUT}")
    print(f"[FUSE] Home 重指 {replaced['n']} 处; Phase D 超时改 {fixed['n']} 处")
    print(f"[FUSE] widgets 顶层 {len(w_f['children'])}; points {len(pts_f['children'])}; project vars {len(pv_f)}")


if __name__ == "__main__":
    main()
