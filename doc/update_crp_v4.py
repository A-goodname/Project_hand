# -*- coding: utf-8 -*-
"""
CRP 更新脚本: v3 → v4 (Orchestrator 双相机协议 + HandInCup 判定)

使用方法:
    cd doc
    python update_crp_v4.py

输入:  handlepluscup-Down-41.crp     (原始 v3)
输出:  handlepluscup-v4.crp          (更新后 v4)
备份:  handlepluscup-Down-41.crp.bak (自动备份)

v4 变更:
  1. 新增变量: str7="READY", socketstr4, socketstr5, strflag2, strflag_cont, strflag_abort
  2. 新增位姿点: DCp2 (DCPOS), C_P4 (CPOS), DummyCpos (CPOS, 用于解析响应)
  3. Cup P3 后新增: READY → Wait 2s → RELEASE → 等待 CONTINUE/ABORT
  4. CONTINUE 分支: C_P4 退避 → RESET → Home → SocketClose
  5. ABORT 分支:   C_P1 放回 → RELEASE → C_P2 退避 → RESET → Home → SocketClose
"""

import copy
import json
import os
import time
from datetime import datetime


# ==================================================================
# 工具函数
# ==================================================================

_id_seq = 0

def new_id(prefix: str) -> str:
    """生成唯一节点 ID (保留原始前缀风格 + 自增序号)."""
    global _id_seq
    _id_seq += 1
    ts = int(time.time() * 1000) % 100000
    return f"{prefix}v4{ts:05d}{_id_seq:05d}"


def backup_file(path: str):
    """备份原始文件."""
    bak = path + ".bak"
    if not os.path.exists(bak):
        with open(path, "rb") as src:
            with open(bak, "wb") as dst:
                dst.write(src.read())
        print(f"[BACKUP] {path} → {bak}")
    else:
        print(f"[SKIP] 备份已存在: {bak}")


# ==================================================================
# 节点模板工厂
# ==================================================================

def make_wait(milliseconds: int) -> dict:
    return {
        "type": "wait", "icon": "wait", "qt": "a",
        "id": new_id("wt"), "index": 0,
        "label": "Wait",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {"time": milliseconds}},
        "style": {"wa": "400px", "ha": "235px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#8470e5"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_socketsendstr(var_name: str, flag_var: str = "socketflag") -> dict:
    return {
        "type": "socketsendstr", "icon": "socketsendstr", "qt": "a",
        "id": new_id("ss"), "index": 0,
        "label": "SocketSendStr",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "mscname": {"datavar": {"type": "Socket", "value": "Socket0"}},
            "msendvar": {"datavar": {"type": "STRING", "value": var_name}},
            "operateval": {"datavar": {"type": "INT", "value": flag_var}},
            "separateend": False,
        }},
        "style": {"wa": "400px", "ha": "375px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#e8774b"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_socketreadstr(str_var: str, flag_var: str = "socketflag",
                       timeout: int = 5000, start_str: str = "",
                       end_str: str = "") -> dict:
    return {
        "type": "socketreadstr", "icon": "socketreadstr", "qt": "a",
        "id": new_id("sr"), "index": 0,
        "label": "SocketReadStr",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "mscname": {"datavar": {"type": "Socket", "value": "Socket0"}},
            "mdataval": {"datavar": {"type": "STRING", "value": str_var}},
            "mtesttime": timeout,
            "operateval": {"datavar": {"type": "INT", "value": flag_var}},
            "strstring": start_str,
            "endstring": end_str,
        }},
        "style": {"wa": "400px", "ha": "375px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#e8774b"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_transtrtocpos(src_str_var: str, flag_var: str, point_var: str,
                       start_str: str, end_str: str) -> dict:
    return {
        "type": "transtrtocpos", "icon": "transtrtocpos", "qt": "b",
        "id": new_id("ts"), "index": 0,
        "label": "TranStrToCpos",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "sortby": 1,
            "currentstr": {"datavar": {"type": "STRING", "value": src_str_var}},
            "separate": ",",
            "pointa": {"langs": "memoryvar",
                        "pointvar": {"type": ["cpos"], "value": point_var}},
            "operateval": {"datavar": {"type": "INT", "value": flag_var}},
            "strstring": start_str,
            "endstring": end_str,
            "angleunit": "deg", "lengthunit": "m",
        }},
        "style": {"wa": "480px", "ha": "450px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#7acc89"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_movl(point_var: str) -> dict:
    return {
        "type": "movl", "icon": "movl", "qt": "b",
        "id": new_id("ml"), "index": 0,
        "label": "MovL",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "pointa": {"pointvar": {"type": ["apos", "cpos"], "value": point_var}},
            "mspeed": {"datavar": {"type": "SPEED", "value": "V100", "valuekeep": 1}},
            "mblendtype": "FINE",
            "mblendvalue": {"datavar": {"type": "ZONE", "value": "ZONE0"}},
            "macc": {"datavar": {"type": "ACC", "value": "ACC100", "valuekeep": 1}},
        }},
        "style": {"wa": "480px", "ha": "500px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#cc66cc"},
        "ex": {"ps": "", "isclose": False}, "parents": [], "children": [],
        "flag": {"showplay": 1}, "showchildren": True, "attrshow": True,
    }


def make_socketclose(flag_var: str = "socketflag") -> dict:
    return {
        "type": "socketclose", "icon": "socketclose", "qt": "a",
        "id": new_id("sc"), "index": 0,
        "label": "SocketClose",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "mscname": {"datavar": {"type": "Socket", "value": "Socket0"}},
            "operateval": {"datavar": {"type": "INT", "value": flag_var}},
        }},
        "style": {"wa": "400px", "ha": "320px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#e8774b"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_equal_assign(var_name: str, var_path: list, value: str,
                       point_path: list = None) -> dict:
    """创建 Equal 赋值节点.

    支持两种模式:
      point_path!=None:  pointRef = expr (用于计算偏移量)
      point_path==None:  var = value (用于 strflag=1)
    """
    if point_path:
        operatordata = [
            {"type": "data", "addlevel": 1, "dellevel": 1, "editlevel": 3,
             "operator": "",
             "data": {"datatype": "point",
                       "data": {"var": [], "path": "", "value": {"value": ""},
                                "point": point_path}}},
            {"type": "operator", "addlevel": 1, "dellevel": 1, "editlevel": 4,
             "operator": "=",
             "data": {"datatype": "value",
                       "data": {"var": "", "path": "", "value": {"value": ""}}}},
            {"type": "data", "dellevel": 1, "operator": "",
             "data": {"datatype": "var",
                       "data": {"var": var_path, "path": "",
                                "value": {"value": value}}}},
        ]
    else:
        operatordata = [
            {"type": "data", "addlevel": 1, "dellevel": 1, "editlevel": 3,
             "operator": "",
             "data": {"datatype": "var",
                       "data": {"var": var_path, "path": "",
                                "value": {"value": ""}}}},
            {"type": "operator", "addlevel": 1, "dellevel": 1, "editlevel": 4,
             "operator": "=",
             "data": {"datatype": "value",
                       "data": {"var": "", "path": "", "value": {"value": ""}}}},
            {"type": "data", "dellevel": 1, "operator": "",
             "data": {"datatype": "value",
                       "data": {"var": "", "path": "",
                                "value": {"value": value}}}},
        ]
    return {
        "type": "equal", "icon": "equal", "qt": "a",
        "id": new_id("eq"), "index": 0,
        "label": "...=...",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {"operatordata": operatordata}},
        "style": {"wa": "450px", "ha": "480px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#6bd6c4"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_equal_dz(dcpos_name: str, dcpos_id: str,
                  p3_name: str, p3_id: str,
                  p1_name: str, p1_id: str) -> dict:
    """创建 DCp.dz = P3.z - P1.z 计算节点."""
    operatordata = [
        {"type": "data", "addlevel": 1, "dellevel": 1, "editlevel": 3,
         "operator": "",
         "data": {"datatype": "point",
                   "data": {"var": ["INT0", "value"], "path": "",
                            "value": {"value": ""},
                            "point": [f"{dcpos_name}({dcpos_id})", "dcpos", "dz"]}}},
        {"type": "operator", "addlevel": 1, "dellevel": 1, "editlevel": 4,
         "operator": "=",
         "data": {"datatype": "value",
                   "data": {"var": "", "path": "", "value": {"value": ""}}}},
        {"type": "data", "dellevel": 1, "operator": "",
         "data": {"datatype": "point",
                   "data": {"var": "", "path": "", "value": {"value": "true"},
                            "point": [f"{p3_name}({p3_id})", "cpos", "z"]}},
         "fun": {"funtype": ["math", "sin"], "funparams": [
             {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
             {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
             {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}}],
             "ptypes": [], "ptypesvar": []}},
        {"type": "operator", "operator": "-",
         "data": {"datatype": "value",
                   "data": {"point": "", "var": "", "path": "", "value": {"value": ""}}},
         "fun": {"funtype": ["Math", "sin"], "ptypes": [], "ptypesvar": [], "plen": 1,
                 "funparams": [
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}}]}},
        {"type": "data", "operator": "",
         "data": {"datatype": "point",
                   "data": {"point": [f"{p1_name}({p1_id})", "cpos", "z"],
                            "var": "", "path": "", "value": {"value": ""}}},
         "fun": {"funtype": ["Math", "sin"], "ptypes": [], "ptypesvar": [], "plen": 1,
                 "funparams": [
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}}]}},
    ]
    return {
        "type": "equal", "icon": "equal", "qt": "a",
        "id": new_id("eq"), "index": 0,
        "label": "...=...",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {"operatordata": operatordata}},
        "style": {"wa": "450px", "ha": "480px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#6bd6c4"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_cposoffset(src_point_id: str, dcpos_id: str, result_point_id: str) -> dict:
    return {
        "type": "cposoffset", "icon": "cposoffset", "qt": "b",
        "id": new_id("ca"), "index": 0,
        "label": "CposOffset",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "pointb": {"langs": "originalpoint",
                        "pointvar": {"type": ["cpos"], "value": src_point_id}},
            "pointa": {"langs": "offsetquantity",
                        "pointvar": {"type": ["dcpos"], "value": dcpos_id}},
            "tarpos": {"pointvar": {"type": ["cpos"], "value": result_point_id}},
        }},
        "style": {"wa": "480px", "ha": "340px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#69e0e0"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
    }


def make_if_var_equals(var_name: str, value: str) -> dict:
    """创建 If 条件节点 (var == value)."""
    operatordata = [
        {"type": "data", "operator": "",
         "fun": {"funtype": ["math", "sin"], "funparams": [
             {"datatype": "value", "data": {"point": [], "var": [], "path": "", "value": {"value": 0}}},
             {"datatype": "value", "data": {"point": [], "var": [], "path": "", "value": {"value": 0}}},
             {"datatype": "value", "data": {"point": [], "var": [], "path": "", "value": {"value": 0}}}]},
         "data": {"datatype": "var",
                   "data": {"point": [], "var": [var_name, "value"], "path": "",
                            "value": {"value": "true"}}}},
        {"type": "operator", "operator": "==",
         "data": {"datatype": "value",
                   "data": {"point": "", "var": "", "path": "", "value": {"value": ""}}},
         "fun": {"funtype": ["Math", "sin"], "ptypes": [], "ptypesvar": [], "plen": 1,
                 "funparams": [
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}}]}},
        {"type": "data", "operator": "",
         "data": {"datatype": "value",
                   "data": {"point": "", "var": "", "path": "", "value": {"value": value}}},
         "fun": {"funtype": ["Math", "sin"], "ptypes": [], "ptypesvar": [], "plen": 1,
                 "funparams": [
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}},
                     {"datatype": "value", "data": {"var": [], "path": "", "value": {"value": 0}}}]}},
    ]
    return {
        "type": "if", "icon": "if", "qt": "d",
        "id": new_id("if"), "index": 0,
        "label": "If",
        "datatype": "value",
        "style": {"wa": "450px", "ha": "480px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#6bd6c4"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
        "data": {"var": "", "path": "", "value": {"operatordata": operatordata}},
    }


def make_point_cpos(label: str, pid: str, x=0.0, y=0.0, z=0.0,
                    a=0.0, b=0.0, c=0.0, is_default=True) -> dict:
    """创建位姿点 (CPOS)."""
    return {
        "type": "point", "icon": "point",
        "id": pid, "label": label, "status": 0,
        "datatype": "value",
        "data": {"var": "", "path": "/a/b/c", "value": {
            "posstat": 1 if is_default else 0,
            "postype": "cpos",
            "posvar": {"datavar": {"default": "DEFAULT" if is_default else "",
                                    "type": "CPOS", "value": "DEFAULT" if is_default else ""}},
            "posvalue": {"cpos": {
                "x": x, "y": y, "z": z, "a": a, "b": b, "c": c, "e": 0,
                "poscfg": {"mode": 0, "cf1": -1, "cf2": -1, "cf3": -1,
                           "cf4": -1, "cf5": -1, "cf6": -1, "cf7": -1},
                "exjntpos1": 0, "exjntpos2": 0, "exjntpos3": 0,
                "exjntpos4": 0, "exjntpos5": 0, "exjntpos6": 0,
                "exjntpos7": 0, "exjntpos8": 0, "exjntpos9": 0, "exjntpos10": 0,
            }},
        }},
        "style": {"wa": "520px", "ha": "448px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#7d94b5"},
        "ex": {"ps": ""}, "path": "/a/b/c",
        "parents": [], "children": [],
        "candrop": True, "showchildren": True, "attrshow": True,
    }


def make_point_dcpos(label: str, pid: str) -> dict:
    """创建偏移位姿点 (DCPOS)."""
    return {
        "type": "point", "icon": "point",
        "id": pid, "label": label, "status": 0,
        "datatype": "value",
        "data": {"var": "", "path": "/a/b/c", "value": {
            "posstat": 0, "postype": "dcpos",
            "posvar": {"datavar": {"default": "DEFAULT", "type": "DCPOS", "value": "DEFAULT"}},
            "posvalue": {"dcpos": {
                "dx": 0, "dy": 0, "dz": 0, "da": 0, "db": 0, "dc": 0, "de": 0,
                "dexjntpos1": 0, "dexjntpos2": 0, "dexjntpos3": 0,
                "dexjntpos4": 0, "dexjntpos5": 0, "dexjntpos6": 0,
                "dexjntpos7": 0, "dexjntpos8": 0, "dexjntpos9": 0, "dexjntpos10": 0,
            }},
        }},
        "style": {"wa": "520px", "ha": "448px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#7d94b5"},
        "ex": {"ps": ""}, "path": "/a/b/c",
        "parents": [], "children": [],
        "candrop": True, "showchildren": True, "attrshow": True,
    }


# ==================================================================
# 主修改逻辑
# ==================================================================

def get_point_ids(data: dict) -> dict:
    """获取所有位姿点的 ID 映射."""
    points_nodes = None
    for child in data["children"]:
        if child.get("type") == "task":
            for w in child.get("children", []):
                if w.get("type") == "points":
                    points_nodes = w
    if points_nodes is None:
        raise ValueError("未找到 points 节点")

    ids = {}
    for pt in points_nodes.get("children", []):
        ids[pt["label"]] = pt["id"]
    return ids, points_nodes


def get_widgets(data: dict) -> dict:
    """获取主程序 widgets 节点."""
    for child in data["children"]:
        if child.get("type") == "task":
            for w in child.get("children", []):
                if w.get("type") == "widgets":
                    return w
    raise ValueError("未找到 widgets 节点")


def modify_crp(input_path: str, output_path: str):
    """主入口: 加载 → 修改 → 保存."""

    # ---- 加载 ----
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[LOAD] {input_path}")

    # ---- 获取关键 ID ----
    pt_ids, points_node = get_point_ids(data)
    widgets = get_widgets(data)
    nodes = widgets["children"]  # 主程序节点列表

    print(f"[INFO] 位姿点: {list(pt_ids.keys())}")

    # ---- 1. 添加新变量 ----
    vars_list = data["vars"]["project"]
    existing_keys = {v["key"] for v in vars_list}

    new_vars = [
        {"key": "str7", "type": "STRING", "value": {"STRING": {"value": "READY"}},
         "saveflag": False, "open": False},
        {"key": "socketstr4", "type": "STRING", "value": {"STRING": {"value": ""}},
         "saveflag": False, "open": False},
        {"key": "socketstr5", "type": "STRING", "value": {"STRING": {"value": ""}},
         "saveflag": False, "open": False},
        {"key": "strflag2", "type": "INT", "value": {"INT": {"value": 1}},
         "saveflag": False, "open": False},
        {"key": "strflag_cont", "type": "INT", "value": {"INT": {"value": 1}},
         "saveflag": False, "open": False},
        {"key": "strflag_abort", "type": "INT", "value": {"INT": {"value": 1}},
         "saveflag": False, "open": False},
    ]
    for v in new_vars:
        if v["key"] not in existing_keys:
            vars_list.append(v)
            print(f"[VAR] + {v['key']}")
        else:
            print(f"[VAR] SKIP {v['key']} (已存在)")

    # ---- 2. 添加新位姿点 ----
    new_point_ids = {}
    new_points = []
    point_labels = {pt["label"] for pt in points_node["children"]}

    # DCp2 (DCPOS 偏移量)
    dcpid = new_id("pt")
    if "DCp2" not in point_labels:
        new_points.append(make_point_dcpos("DCp2", dcpid))
        new_point_ids["DCp2"] = dcpid
        print(f"[POINT] + DCp2 (dcpos) id={dcpid}")

    # C_P4 (CPOS 安全退避点)
    cp4id = new_id("pt")
    if "C_P4" not in point_labels:
        new_points.append(make_point_cpos("C_P4", cp4id))
        new_point_ids["C_P4"] = cp4id
        print(f"[POINT] + C_P4 (cpos) id={cp4id}")

    # DummyCpos (用于解析 CONTINUE/ABORT 响应)
    dmyid = new_id("pt")
    if "DummyCpos" not in point_labels:
        new_points.append(make_point_cpos("DummyCpos", dmyid))
        new_point_ids["DummyCpos"] = dmyid
        print(f"[POINT] + DummyCpos (cpos) id={dmyid}")

    points_node["children"].extend(new_points)

    # ---- 3. 定位旧 Cup 段节点 (需要删除) ----
    # 从 C_P3 MovL 之后到 SocketClose 之前的所有节点
    c_p3_id = pt_ids["C_P3"]
    home_id = pt_ids["Home"]
    c_p1_id = pt_ids["C_P1"]
    c_p2_id = pt_ids["C_P2"]
    dc_p1_id = pt_ids.get("DCp1", "")

    # 找到 MovL C_P3 的索引
    c_p3_movl_idx = None
    socketclose_idx = None
    for i, node in enumerate(nodes):
        if node["type"] == "movl":
            pv = node.get("data", {}).get("value", {}).get("pointa", {}).get("pointvar", {}).get("value", "")
            if pv == c_p3_id:
                c_p3_movl_idx = i
        if node["type"] == "socketclose":
            socketclose_idx = i

    if c_p3_movl_idx is None:
        raise ValueError(f"未找到 MovL C_P3 (id={c_p3_id})")
    if socketclose_idx is None:
        raise ValueError("未找到 SocketClose 节点")

    print(f"[INFO] MovL C_P3 在索引 {c_p3_movl_idx}")
    print(f"[INFO] SocketClose 在索引 {socketclose_idx}")

    # 要删除的旧节点: C_P3 之后到 SocketClose (不含)
    old_nodes = nodes[c_p3_movl_idx + 1:socketclose_idx]
    old_labels = [(n["type"], n["label"]) for n in old_nodes]
    print(f"[INFO] 删除 {len(old_nodes)} 个旧节点: {old_labels}")

    # ---- 4. 构建新的 v4 节点序列 ----
    new_nodes = []

    # (a) Wait 1s (保留)
    new_nodes.append(make_wait(1000))

    # (b) SocketSendStr('READY') → PC 启动 HandInCup
    new_nodes.append(make_socketsendstr("str7"))

    # (c) Wait 2s (留给 HandInCup 检测)
    new_nodes.append(make_wait(2000))

    # (d) SocketSendStr('RELEASE') → PC 判定
    new_nodes.append(make_socketsendstr("str5"))

    # (e) SocketReadStr → 读取 CONTINUE 或 ABORT 响应
    new_nodes.append(make_socketreadstr("socketstr5", "socketflag", 5000, "", ""))

    # (f) 尝试 TranStrToCpos: 用 "<...>" 检测 CONTINUE 响应
    #     PC 发送:  CONTINUE → "<0,0,0,0,0,0>"  (TranStrToCpos 成功 → flag_cont=0)
    #            ABORT → "[0,0,0,0,0,0]"      (TranStrToCpos 失败 → flag_cont≠0)
    new_nodes.append(
        make_transtrtocpos("socketstr5", "strflag_cont",
                           new_point_ids["DummyCpos"], "<", ">"))

    # (g) 尝试 TranStrToCpos: 用 "[...]" 检测 ABORT 响应
    new_nodes.append(
        make_transtrtocpos("socketstr5", "strflag_abort",
                           new_point_ids["DummyCpos"], "[", "]"))

    # (h) 计算 Cup P4 安全退避点: DCp2.dz = C_P3.z - C_P1.z
    new_nodes.append(
        make_equal_dz("DCp2", new_point_ids["DCp2"],
                       "C_P3", c_p3_id,
                       "C_P1", c_p1_id))

    # (i) C_P4 = CposOffset(C_P1, DCp2)
    new_nodes.append(
        make_cposoffset(c_p1_id, new_point_ids["DCp2"], new_point_ids["C_P4"]))

    # (j) If strflag_cont == 0 → CONTINUE 分支
    if_continue = make_if_var_equals("strflag_cont", "0")

    # CONTINUE 分支子节点
    continue_branch = [
        make_movl(new_point_ids["C_P4"]),       # MovL C_P4 (安全退避)
        make_wait(1000),
        make_socketsendstr("str3"),              # RESET
        make_wait(1000),
        make_movl(home_id),                     # MovL Home
        make_socketclose(),                      # SocketClose
    ]
    if_continue["children"] = continue_branch
    new_nodes.append(if_continue)

    # (k) If strflag_abort == 0 → ABORT 分支
    if_abort = make_if_var_equals("strflag_abort", "0")

    # ABORT 分支子节点
    abort_branch = [
        make_movl(c_p1_id),                     # MovL C_P1 (放回)
        make_wait(1000),
        make_socketsendstr("str5"),              # RELEASE (松杯)
        make_wait(1000),
        make_movl(c_p2_id),                     # MovL C_P2 (退避)
        make_wait(1000),
        make_socketsendstr("str3"),              # RESET
        make_wait(1000),
        make_movl(home_id),                     # MovL Home
        make_socketclose(),                      # SocketClose
    ]
    if_abort["children"] = abort_branch
    new_nodes.append(if_abort)

    # ---- 5. 替换节点 ----
    # 删除: C_P3 后的所有节点 (包括旧 SocketClose)
    del nodes[c_p3_movl_idx + 1:]

    # 插入新节点 (紧跟 C_P3 MovL 之后)
    nodes.extend(new_nodes)

    print(f"[INFO] 插入 {len(new_nodes)} 个新节点")

    # ---- 6. 更新元数据 ----
    data["mark"] = int(time.time() * 1000)

    # ---- 7. 保存 ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[SAVE] {output_path}")
    print(f"[DONE] CRP v3 → v4 转换完成!")

    # ---- 8. 输出变更摘要 ----
    print("""
============================================================
CRP v4 变更摘要
============================================================
新增变量:
  str7 = "READY"          HandInCup 启动指令
  socketstr4, socketstr5  响应缓冲区
  strflag2, strflag_cont,
  strflag_abort           分支标志变量

新增位姿点:
  DCp2 (DCPOS)            Cup P4 偏移量
  C_P4 (CPOS)             安全退避点
  DummyCpos (CPOS)        解析 CONTINUE/ABORT 用

修改的程序段:
  旧: C_P3 → Wait 2s → C_P1 → RELEASE → C_P2 → RESET → Home → SocketClose
  新: C_P3 → Wait 1s → READY → Wait 2s → RELEASE → ReadResponse
       → 计算 C_P4
       → CONTINUE? → C_P4 → RESET → Home → SocketClose
       → ABORT?    → C_P1 → RELEASE → C_P2 → RESET → Home → SocketClose

PC 端需同步修改:
  CONTINUE 响应格式: "<0,0,0,0,0,0>"  (用 < > 标记)
  ABORT   响应格式: "[0,0,0,0,0,0]"  (用 [ ] 标记)
  见 doc/estun_handle_cup_code_v4.txt 附注
============================================================
""")


# ==================================================================
# 主程序
# ==================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "handlepluscup-Down-41.crp")
    output_path = os.path.join(script_dir, "handlepluscup-v4.crp")

    if not os.path.exists(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}")
        exit(1)

    backup_file(input_path)
    modify_crp(input_path, output_path)
