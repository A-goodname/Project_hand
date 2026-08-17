# -*- coding: utf-8 -*-
"""D2 CRP 静态审计脚本 (2026-08-17 v2) — 对应训练侧最终裁决 §4.1.

自动断言 (d2-controller-only-execution-authority-2026-08-17.md §4.1):
  A. GetCurCPos.pointa 与其后 CPosToStr.pointa 一一对应
  B. 所有 Actual/回读容器不在任何 运动目标 引用集合
     (MovL/MovJ/CposOffset/TranStrToCpos 的目标点)
  C. P0 消息 (strstring="P0[" 或 "[" 的首条运动前回读) 在第一条 MovL 之前
  E. 新旧 CRP 的 8 个 MovL point ID 序列完全一致
  F. 仅 Socket0 (无 Socket1/8082)
  H. 9 事件次数式顺序完整: [ P2[ P1[ P4[ P3[ P4[ P1[ P2[ Home[
     (最终裁决 §2: P0_ACTUAL→P2_ARRIVED→P1_ARRIVED→P4_AFTER_LIFT→P3_ARRIVED
      →P4_AFTER_POUR→P1_RETURNED→P2_FINAL_ARRIVED→HOME_ARRIVED)

注: 按最终裁决, "PREGRIP 必须发送" 与 "9 前缀唯一" 两条断言已移除
(接受省略 PREGRIP 与次数式映射).

用法:
  python scripts_gen/d2_crp_static_audit.py <new.crp> [<reference.crp>]
退出码 0 = 全部通过; 1 = 存在失败项.
"""
import json
import sys

# 次数式顺序 (最终裁决 §2): 每条消息的 CPosToStr 前缀
EXPECTED_SEQUENCE = ["", "P2", "P1", "P4", "P3", "P4", "P1", "P2", "Home"]


def load_blocks(path):
    """按执行顺序提取块: [(type, data_dict, label), ...]."""
    data = json.load(open(path, encoding="utf-8"))
    blocks = []

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") in ("movl", "movj", "getcurcpos", "cpostostr",
                                    "socketsendstr", "socketreadstr",
                                    "socketcreate", "cposoffset",
                                    "transtrtocpos", "wait"):
                blocks.append((node.get("type"), node.get("data", {}),
                               node.get("label", "")))
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)

    walk(data)
    return blocks


def _point_ref(data):
    """取块的目标点引用 (pointa/pointb/pointvar 等) → id 集合."""
    refs = set()
    if not isinstance(data, dict):
        return refs
    v = data.get("value", {})
    if isinstance(v, dict):
        for key in ("pointa", "pointb"):
            p = v.get(key)
            if isinstance(p, dict):
                pv = p.get("pointvar", {})
                if isinstance(pv, dict) and pv.get("value"):
                    refs.add(pv["value"])
        pv = v.get("pointvar")
        if isinstance(pv, dict) and pv.get("value"):
            refs.add(pv["value"])
    return refs


def _str_strings(data):
    """提取 CPosToStr 的 strstring/endstring/单位."""
    v = data.get("value", {}) if isinstance(data, dict) else {}
    if not isinstance(v, dict):
        return None
    ss = v.get("strstring")
    es = v.get("endstring")
    if isinstance(ss, dict):
        ss = ss.get("value")
    if isinstance(es, dict):
        es = es.get("value")
    return (str(ss) if ss is not None else None,
            str(es) if es is not None else None)


def _send_var(data):
    v = data.get("value", {}) if isinstance(data, dict) else {}
    if not isinstance(v, dict):
        return None
    mv = v.get("msendvar", {})
    if isinstance(mv, dict):
        dv = mv.get("datavar", {})
        if isinstance(dv, dict):
            return dv.get("value")
    return None


def _string_vars(path):
    """STRING 变量定义: {变量名: 值}.

    实机 CRP schema: {"key": "str1", "type": "STRING",
                      "value": {"STRING": {"value": "PREGRIP"}}}
    """
    data = json.load(open(path, encoding="utf-8"))
    out = {}

    def walk(node):
        if isinstance(node, dict):
            if node.get("type") == "STRING" and node.get("key"):
                v = node.get("value", {})
                if isinstance(v, dict):
                    inner = v.get("STRING", {})
                    if isinstance(inner, dict):
                        out[node["key"]] = inner.get("value")
            for x in node.values():
                walk(x)
        elif isinstance(node, list):
            for x in node:
                walk(x)

    walk(data)
    return out


def audit(new_path, ref_path=None):
    ok = True
    fails = []

    def check(name, cond, detail=""):
        nonlocal ok
        status = "PASS" if cond else "FAIL"
        if not cond:
            ok = False
            fails.append(name)
        print(f"  [{status}] {name} {detail}")

    blocks = load_blocks(new_path)
    print(f"=== D2 CRP 静态审计: {new_path} ({len(blocks)} 块) ===")

    # ---- 运动目标集合 ----
    motion_refs = set()
    for t, d, _l in blocks:
        if t in ("movl", "movj", "cposoffset", "transtrtocpos"):
            motion_refs |= _point_ref(d)
    print(f"  运动目标引用: {len(motion_refs)} 个点")

    # ---- A/B: GetCurCPos → CPosToStr 对应 + 容器非运动目标 ----
    getcur_targets = []
    bad_pair = []
    for i, (t, d, _l) in enumerate(blocks):
        if t == "getcurcpos":
            refs = _point_ref(d)
            target = next(iter(refs), None)
            # 紧随其后的 CPosToStr
            nxt = blocks[i + 1] if i + 1 < len(blocks) else None
            cstr = None
            if nxt and nxt[0] == "cpostostr":
                cstr = next(iter(_point_ref(nxt[1])), None)
            getcur_targets.append((target, cstr, _l))
            if target is None or target != cstr:
                bad_pair.append((target, cstr, _l))
    check("A: GetCurCPos→CPosToStr 点一一对应", not bad_pair,
          f"({len(getcur_targets)} 组, 异常: {bad_pair[:2]})")
    in_motion = [(t, c, l) for t, c, l in getcur_targets
                 if t in motion_refs]
    check("B: 回读容器不在运动目标集合", not in_motion,
          f"(违规: {in_motion[:2]})")

    # ---- C: P0 消息在第一条 MovL 前 ----
    first_movl = next((i for i, (t, _d, _l) in enumerate(blocks)
                       if t == "movl"), None)
    p0_idx = None
    for i, (t, d, _l) in enumerate(blocks):
        if t == "cpostostr":
            ss, es = _str_strings(d)
            if ss in ("P0[", "[") and es == "]":
                p0_idx = i
                break
    check("C: P0 回读在第一条 MovL 前",
          first_movl is not None and p0_idx is not None
          and p0_idx < first_movl,
          f"(P0 在块 {p0_idx}, 首 MovL 在块 {first_movl})")

    # ---- E: 8 MovL point ID 序列与参考一致 ----
    if ref_path:
        old_movl = [next(iter(_point_ref(d)), "?")
                    for t, d, _l in load_blocks(ref_path) if t == "movl"]
        new_movl = [next(iter(_point_ref(d)), "?")
                    for t, d, _l in blocks if t == "movl"]
        check("E: 新旧 8 MovL point ID 序列一致",
              len(old_movl) == len(new_movl) and old_movl == new_movl,
              f"(旧 {len(old_movl)} 个: {old_movl})")

    # ---- F: 仅 Socket0 ----
    sockets = [d.get("value", {}).get("mport")
               for t, d, _l in blocks if t == "socketcreate"]
    check("F: 仅 Socket0 (无 8082)", sockets == [8080] or sockets == [],
          f"(端口: {sockets})")

    # ---- H: 9 事件次数式顺序完整 (最终裁决 §2) ----
    seq = []
    for t, d, _l in blocks:
        if t == "cpostostr":
            ss, es = _str_strings(d)
            if ss and ss.endswith("[") and es == "]":
                seq.append(ss[:-1])
    check("H: 9 事件次数式顺序完整",
          seq == EXPECTED_SEQUENCE,
          f"(实际: {seq})")

    print(f"\n=== 结论: {'✅ 全部通过' if ok else '❌ 失败项: ' + ', '.join(fails)} ===")
    return 0 if ok else 1


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    new = sys.argv[1]
    ref = sys.argv[2] if len(sys.argv) > 2 else None
    return audit(new, ref)


if __name__ == "__main__":
    sys.exit(main())
