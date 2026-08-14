#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成 handlepluscup-orchestrator-session-barrier-2026-08-14.crp

对方工单: project-hand-estun-session-bound-barrier-work-order-2026-08-14.md
方法: 从 handlepluscup-orchestrator.crp 克隆同类型 block (真实 schema), 不改发明新 block,
      只修改参数/变量/点引用并生成新 id. 新增 block 全部来自源工程已有类型.

P2 屏障 (插在 MovL H_P2 之后、MovL H_P1 之前):
  GetCurCPos -> H_P2_ACTUAL
  CPosToStr(H_P2_ACTUAL -> p2_arrived, 前缀 "P2_ARRIVED|")
  SocketSendStr(Socket0, p2_arrived)
  SocketReadStr(Socket0 -> grasp_signal, 15000ms, flag=socketflag)
  If socketflag != 0 -> SocketClose(Socket0) + Return
  TranStrToCpos(grasp_signal -> DummyGraspAck, <...>, flag=grasp_parse_flag)
  If grasp_parse_flag != 0 -> SocketClose(Socket0) + Return

P1 屏障 (插在 MovL H_P1 之后、SocketSendStr(str2=GRIPDONE) 之前): 同上, 用 P1_ARRIVED|/[..]/DummyGripAck
"""
import copy
import json
import random
import string
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "doc" / "handlepluscup-orchestrator.crp"
OUT = ROOT / "doc" / "handlepluscup-orchestrator-session-barrier-v2-2026-08-14.crp"

# 源工程中的关键点 id (抓壶段)
H_P2 = "ptmquqnbcippj6ns"
H_P1 = "ptmquqnaxr62mo6p"


def new_id(prefix: str) -> str:
    alphabet = string.digits + string.ascii_lowercase
    stamp = ""
    v = int(time.time() * 1000)
    while v:
        v, rem = divmod(v, 36)
        stamp = alphabet[rem] + stamp
    suffix = "".join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    return f"{prefix}{stamp}{suffix}"


def load(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(data):
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def find_widgets(data):
    for top in data.get("children", []):
        if top.get("type") == "task":
            for c in top.get("children", []):
                if c.get("type") == "widgets":
                    return c
    raise RuntimeError("widgets 未找到")


def find_point(data, pid):
    for top in data.get("children", []):
        if top.get("type") == "task":
            for c in top.get("children", []):
                if c.get("type") == "points":
                    for p in c.get("children", []):
                        if p.get("id") == pid:
                            return p
    raise RuntimeError(f"point {pid} 未找到")


def find_block(widgets, block_type, point_ref=None, var_ref=None):
    """在 widgets 顶层 children 找特定 block (可选按点引用/变量引用过滤)."""
    for b in widgets.get("children", []):
        if b.get("type") != block_type:
            continue
        if point_ref is not None:
            if b.get("data", {}).get("value", {}).get("pointa", {}).get("pointvar", {}).get("value") != point_ref:
                continue
        if var_ref is not None:
            if b.get("data", {}).get("value", {}).get("msendvar", {}).get("datavar", {}).get("value") != var_ref:
                continue
        return b
    raise RuntimeError(f"block 未找到: {block_type} point={point_ref} var={var_ref}")


def relabel(node, id_map):
    """给克隆块换新 id, 并修复 parents 引用."""
    if not isinstance(node, dict):
        return
    if node.get("id"):
        node["id"] = id_map[node["id"]]
    if node.get("parents"):
        node["parents"] = [id_map.get(p, p) for p in node["parents"]]
    for c in node.get("children", []) or []:
        relabel(c, id_map)


def main():
    data = load(SRC)
    data = copy.deepcopy(data)
    widgets = find_widgets(data)

    # ---------- 1. 新增工程变量 ----------
    vars_list = data["vars"]["project"]
    tpl_int = copy.deepcopy(vars_list[0])   # socketflag: INT
    tpl_str = copy.deepcopy(vars_list[4])   # socketstr1: STRING
    for key, typ, init in [
        ("p2_arrived", "STRING", ""),
        ("p1_arrived", "STRING", ""),
        ("grasp_signal", "STRING", ""),
        ("grip_signal", "STRING", ""),
        ("grasp_parse_flag", "INT", 0),
        ("grip_parse_flag", "INT", 0),
    ]:
        tpl = tpl_str if typ == "STRING" else tpl_int
        v = copy.deepcopy(tpl)
        v["key"] = key
        v["type"] = typ
        v["value"] = {typ: {"value": init}}
        vars_list.append(v)
    print(f"变量表: {len(vars_list)} 个 (新增 6)")

    # ---------- 2. 新增位姿点 (全零, 动态写入) ----------
    points_node = None
    for top in data.get("children", []):
        if top.get("type") == "task":
            for c in top.get("children", []):
                if c.get("type") == "points":
                    points_node = c
    zero_pt = find_point(data, "ptmrd3lond16tg1jdo")  # GH_CUR 全零点模板
    for label in ["H_P2_ACTUAL", "H_P1_ACTUAL", "DummyGraspAck", "DummyGripAck"]:
        p = copy.deepcopy(zero_pt)
        old = p["id"]
        p["id"] = new_id("ptm")
        p["label"] = label
        p["parents"] = [points_node["id"]]
        # 清坐标, 保留 DEFAULT 引用结构
        p["data"]["value"]["posvalue"]["cpos"] = copy.deepcopy(zero_pt["data"]["value"]["posvalue"]["cpos"])
        points_node["children"].append(p)
    print(f"位姿点: {len(points_node['children'])} 个 (新增 4)")

    # ---------- 3. 抓取模板 block (从源工程克隆) ----------
    tpl_getcurcpos = find_block(widgets, "getcurcpos")            # [29] -> P0
    tpl_cpostostr = find_block(widgets, "cpostostr")              # [30] '[' 前缀
    tpl_socketsendstr = find_block(widgets, "socketsendstr")      # [31] Socket0 socketstr1
    tpl_socketreadstr = find_block(widgets, "socketreadstr", )    # [75] Phase D 20s Socket0
    # 用 while 内的 SocketReadStr(5000ms, socketflag) 更贴近屏障语义
    tpl_socketreadstr = find_block(widgets, "socketreadstr")
    tpl_transtrtocpos_lt = find_block(widgets, "transtrtocpos")   # [76] <>
    tpl_if_deny = find_block(widgets, "if")                        # [28] socketflag != 0 -> Return
    tpl_socketclose = find_block(widgets, "socketclose")          # Phase D Socket0 版
    # SocketClose 模板: 递归搜索 (Phase D 分支内的 Socket0 版)
    def find_all(node, out):
        if isinstance(node, dict):
            if node.get("type") == "socketclose":
                out.append(node)
            for v in node.values():
                find_all(v, out)
        elif isinstance(node, list):
            for v in node:
                find_all(v, out)
        return out
    closes = find_all(data, [])
    sc0 = None
    for c in closes:
        if c.get("data", {}).get("value", {}).get("mscname", {}).get("datavar", {}).get("value") == "Socket0":
            sc0 = c
            break
    assert sc0 is not None, "Socket0 的 SocketClose 未找到"
    # 确认 [28] if 条件为 socketflag != 0
    ops = tpl_if_deny["data"]["value"]["operatordata"]
    assert ops[0]["data"]["data"]["var"][0] == "socketflag" and ops[1]["operator"] == "!=", "if 模板条件不符"

    def make_block(tpl, new_type_id, patch):
        b = copy.deepcopy(tpl)
        old_id = b["id"]
        b["id"] = new_type_id
        b["parents"] = []
        patch(b)
        return b

    # ---------- 4. 构造屏障块组 ----------
    def barrier_group(point_actual, msg_var, msg_prefix, recv_var, parse_flag,
                      dummy_point, bracket_lt, bracket_gt):
        blocks = []
        # a. GetCurCPos -> H_Px_ACTUAL
        g = make_block(tpl_getcurcpos, new_id("gcm"), lambda b: b["data"]["value"]["pointa"]["pointvar"].update(value=point_actual))
        blocks.append(g)
        # b. CPosToStr(H_Px_ACTUAL -> msg_var, prefix, lengthunit=m)
        c = make_block(tpl_cpostostr, new_id("cpm"), lambda b: (
            b["data"]["value"]["pointa"]["pointvar"].update(value=point_actual),
            b["data"]["value"]["memorystr"]["datavar"].update(value=msg_var),
            b["data"]["value"].update(strstring=msg_prefix, endstring="", lengthunit="m"),
        ))
        blocks.append(c)
        # c. SocketSendStr(Socket0, msg_var)   [必须显式 Socket0: 模板可能是 Phase 0 的 Socket1]
        s = make_block(tpl_socketsendstr, new_id("ssm"), lambda b: (
            b["data"]["value"]["msendvar"]["datavar"].update(value=msg_var),
            b["data"]["value"]["mscname"]["datavar"].update(value="Socket0"),
        ))
        blocks.append(s)
        # d. SocketReadStr(Socket0 -> recv_var, 15000ms, flag=socketflag)
        r = make_block(tpl_socketreadstr, new_id("srm"), lambda b: (
            b["data"]["value"]["mdataval"]["datavar"].update(value=recv_var),
            b["data"]["value"].update(mtesttime=15000),
            b["data"]["value"]["operateval"]["datavar"].update(value="socketflag"),
            b["data"]["value"]["mscname"]["datavar"].update(value="Socket0"),
        ))
        blocks.append(r)
        # e. If socketflag != 0 -> [SocketClose(Socket0), Return]
        i1 = make_block(tpl_if_deny, new_id("ifm"), lambda b: None)
        i1["children"] = [copy.deepcopy(sc0), copy.deepcopy(tpl_if_deny["children"][0])]
        i1["children"][0]["parents"] = [i1["id"]]
        i1["children"][1]["parents"] = [i1["id"]]
        blocks.append(i1)
        # f. TranStrToCpos(recv_var -> DummyXxxAck, brackets, parse_flag, lengthunit=m)
        t = make_block(tpl_transtrtocpos_lt, new_id("tsm"), lambda b: (
            b["data"]["value"]["currentstr"]["datavar"].update(value=recv_var),
            b["data"]["value"]["pointa"]["pointvar"].update(value=dummy_point),
            b["data"]["value"]["operateval"]["datavar"].update(value=parse_flag),
            b["data"]["value"].update(strstring=bracket_lt, endstring=bracket_gt, lengthunit="m"),
        ))
        blocks.append(t)
        # g. If parse_flag != 0 -> [SocketClose(Socket0), Return]
        i2 = make_block(tpl_if_deny, new_id("ifm"), lambda b: (
            b["data"]["value"]["operatordata"][0]["data"]["data"]["var"].__setitem__(0, parse_flag),
        ))
        i2["children"] = [copy.deepcopy(sc0), copy.deepcopy(tpl_if_deny["children"][0])]
        i2["children"][0]["parents"] = [i2["id"]]
        i2["children"][1]["parents"] = [i2["id"]]
        blocks.append(i2)
        return blocks

    p2_group = barrier_group(
        point_actual=find_point(data, "ptmrd3lond16tg1jdo")["id"] or None,  # placeholder
        msg_var="p2_arrived", msg_prefix="P2_ARRIVED|",
        recv_var="grasp_signal", parse_flag="grasp_parse_flag",
        dummy_point=None, bracket_lt="<", bracket_gt=">",
    )
    # 修正: point_actual/dummy_point 用新生成的 point id
    new_points = [p for p in points_node["children"] if p["label"] in
                  ("H_P2_ACTUAL", "H_P1_ACTUAL", "DummyGraspAck", "DummyGripAck")]
    ids = {p["label"]: p["id"] for p in new_points}
    p2_group[0]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["H_P2_ACTUAL"]
    p2_group[1]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["H_P2_ACTUAL"]
    p2_group[5]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["DummyGraspAck"]

    p1_group = barrier_group(
        point_actual=None, msg_var="p1_arrived", msg_prefix="P1_ARRIVED|",
        recv_var="grip_signal", parse_flag="grip_parse_flag",
        dummy_point=None, bracket_lt="[", bracket_gt="]",
    )
    p1_group[0]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["H_P1_ACTUAL"]
    p1_group[1]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["H_P1_ACTUAL"]
    p1_group[5]["data"]["value"]["pointa"]["pointvar"]["value"] = ids["DummyGripAck"]

    # ---------- 5. 定位抓壶段并插入 (取第一次匹配: 抓壶段在倒水段之前) ----------
    idx_hp2 = idx_hp1 = idx_gripdone = None
    for i, b in enumerate(widgets["children"]):
        if b.get("type") == "movl" and idx_hp2 is None:
            pv = b.get("data", {}).get("value", {}).get("pointa", {}).get("pointvar", {}).get("value")
            if pv == H_P2:
                idx_hp2 = i
        if b.get("type") == "movl" and idx_hp1 is None:
            pv = b.get("data", {}).get("value", {}).get("pointa", {}).get("pointvar", {}).get("value")
            if pv == H_P1:
                idx_hp1 = i
        if b.get("type") == "socketsendstr" and idx_gripdone is None:
            mv = b.get("data", {}).get("value", {}).get("msendvar", {}).get("datavar", {}).get("value")
            if mv == "str2":  # GRIPDONE
                idx_gripdone = i
    assert idx_hp2 is not None and idx_hp1 is not None and idx_gripdone is not None
    assert idx_hp2 < idx_hp1 < idx_gripdone, "抓壶段顺序异常"
    print(f"抓壶段: MovL H_P2 @{idx_hp2}, MovL H_P1 @{idx_hp1}, GRIPDONE @{idx_gripdone}")

    children = widgets["children"]
    # 在 idx_hp1 前插入 P2 屏障 (P2 屏障应在 Wait 之后, MovL H_P1 之前)
    children[idx_hp1:idx_hp1] = p2_group
    idx_hp1 += len(p2_group)
    idx_gripdone += len(p2_group)
    # 在 idx_gripdone 前插入 P1 屏障
    children[idx_gripdone:idx_gripdone] = p1_group

    # ---------- 6. 输出 ----------
    OUT.write_text(dump(data), encoding="utf-8", newline="\n")
    print(f"输出: {OUT} ({OUT.stat().st_size} 字节)")


if __name__ == "__main__":
    main()
