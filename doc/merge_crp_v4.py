# -*- coding: utf-8 -*-
"""
CRP 合并脚本: 598 (P4 Y后退) + v4 Phase D (HandInCup 判定) → handlepluscup-v4.crp

输入:  handlepluscup-Down-598.crp  (P4 Y后退 v3)
输出:  handlepluscup-v4.crp        (P4 Y后退 + HandInCup 全流程)
备份:  handlepluscup-v4.crp.bak    (如已存在)
"""

import copy
import json
import os
import shutil
import time


# ==================================================================
# 工具函数
# ==================================================================

_id_seq = 0

def new_id(prefix: str) -> str:
    """生成唯一节点 ID."""
    global _id_seq
    _id_seq += 1
    ts = int(time.time() * 1000) % 100000
    return f"{prefix}v8{ts:05d}{_id_seq:05d}"


def backup_file(path: str):
    """备份目标文件."""
    if os.path.exists(path):
        bak = path + ".bak"
        shutil.copy2(path, bak)
        print(f"[BACKUP] {path} → {bak}")


# ==================================================================
# 节点模板工厂 (与 update_crp_v4.py 共享)
# ==================================================================

def make_wait(milliseconds: int) -> dict:
    return {
        "type": "wait", "icon": "wait", "qt": "a",
        "id": new_id("wt"), "index": 0, "label": "Wait",
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
        "id": new_id("ss"), "index": 0, "label": "SocketSendStr",
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
                       timeout: int = 5000) -> dict:
    return {
        "type": "socketreadstr", "icon": "socketreadstr", "qt": "a",
        "id": new_id("sr"), "index": 0, "label": "SocketReadStr",
        "datatype": "value",
        "data": {"var": "", "path": "", "value": {
            "mscname": {"datavar": {"type": "Socket", "value": "Socket0"}},
            "mdataval": {"datavar": {"type": "STRING", "value": str_var}},
            "mtesttime": timeout,
            "operateval": {"datavar": {"type": "INT", "value": flag_var}},
            "strstring": "",
            "endstring": "",
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
        "id": new_id("ts"), "index": 0, "label": "TranStrToCpos",
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
        "id": new_id("ml"), "index": 0, "label": "MovL",
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
        "id": new_id("sc"), "index": 0, "label": "SocketClose",
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
        "id": new_id("eq"), "index": 0, "label": "...=...",
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
        "id": new_id("ca"), "index": 0, "label": "CposOffset",
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
        "id": new_id("if"), "index": 0, "label": "If",
        "datatype": "value",
        "style": {"wa": "450px", "ha": "480px", "color": "var(--ccf)",
                   "icolor": "#ffffff", "backgroundColor": "#6bd6c4"},
        "ex": {"ps": ""}, "parents": [], "children": [],
        "flag": {}, "showchildren": True, "attrshow": True,
        "data": {"var": "", "path": "", "value": {"operatordata": operatordata}},
    }


def make_point_cpos(label: str, pid: str, is_default=True) -> dict:
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
                "x": 0, "y": 0, "z": 0, "a": 0, "b": 0, "c": 0, "e": 0,
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
    for child in data["children"]:
        if child.get("type") == "task":
            for w in child.get("children", []):
                if w.get("type") == "points":
                    ids = {}
                    for pt in w.get("children", []):
                        ids[pt["label"]] = pt["id"]
                    return ids, w
    raise ValueError("未找到 points 节点")


def get_widgets(data: dict) -> dict:
    """获取主程序 widgets 节点."""
    for child in data["children"]:
        if child.get("type") == "task":
            for w in child.get("children", []):
                if w.get("type") == "widgets":
                    return w
    raise ValueError("未找到 widgets 节点")


def modify_crp(input_path: str, output_path: str):
    """主入口: 加载 598 → 注入 Phase D → 保存."""

    # ---- 加载 ----
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"[LOAD] {input_path}")

    # ---- 获取关键 ID ----
    pt_ids, points_node = get_point_ids(data)
    widgets = get_widgets(data)
    nodes = widgets["children"]

    print(f"[INFO] 位姿点: {list(pt_ids.keys())}")
    print(f"[INFO] 节点总数: {len(nodes)}")

    c_p1_id = pt_ids["C_P1"]
    c_p2_id = pt_ids["C_P2"]
    c_p3_id = pt_ids["C_P3"]
    home_id = pt_ids["Home"]

    # ---- 1. 添加 v4 新变量 ----
    vars_list = data["vars"]["project"]
    existing_keys = {v["key"] for v in vars_list}

    new_vars = [
        {"key": "str7", "type": "STRING", "value": {"STRING": {"value": "READY"}},
         "saveflag": False, "open": False},
        {"key": "socketstr4", "type": "STRING", "value": {"STRING": {"value": ""}},
         "saveflag": False, "open": False},
        {"key": "socketstr5", "type": "STRING", "value": {"STRING": {"value": ""}},
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

    # ---- 2. 添加 v4 Phase D 新位姿点 ----
    new_point_ids = {}
    new_points = []
    point_labels = {pt["label"] for pt in points_node["children"]}

    for lbl, maker in [
        ("DCp2", lambda pid: make_point_dcpos("DCp2", pid)),
        ("C_P4", lambda pid: make_point_cpos("C_P4", pid)),
        ("DummyCpos", lambda pid: make_point_cpos("DummyCpos", pid)),
    ]:
        if lbl not in point_labels:
            pid = new_id("pt")
            new_points.append(maker(pid))
            new_point_ids[lbl] = pid
            print(f"[POINT] + {lbl} id={pid}")

    points_node["children"].extend(new_points)

    # ---- 3. 定位 Cup 段 ----
    # 598 中: [43] MovL C_P3 → [44] Wait 1s → ... → [52] SocketClose
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

    # ---- 4. 删除旧 Cup 后续节点 [c_p3_movl_idx+1 : socketclose_idx+1] ----
    del_count = socketclose_idx - c_p3_movl_idx
    del nodes[c_p3_movl_idx + 1 : socketclose_idx + 1]
    print(f"[INFO] 删除 {del_count} 个旧节点 (C_P3 Wait → SocketClose)")

    # ---- 5. 构建 v4 Phase D 节点序列 ----
    new_nodes = []

    # (a) Wait 1s (C_P3 到达后的短暂停留)
    new_nodes.append(make_wait(1000))

    # (b) SocketSendStr('READY') → PC 启动 HandInCup
    new_nodes.append(make_socketsendstr("str7"))

    # (c) Wait 2s (留给 HandInCup 检测)
    new_nodes.append(make_wait(2000))

    # (d) SocketSendStr('RELEASE') → PC 判定
    new_nodes.append(make_socketsendstr("str5"))

    # (e) SocketReadStr → 读取 CONTINUE 或 ABORT 响应
    new_nodes.append(make_socketreadstr("socketstr5", "socketflag", 5000))

    # (f) TranStrToCpos: "<...>" 检测 CONTINUE
    new_nodes.append(
        make_transtrtocpos("socketstr5", "strflag_cont",
                           new_point_ids["DummyCpos"], "<", ">"))

    # (g) TranStrToCpos: "[...]" 检测 ABORT
    new_nodes.append(
        make_transtrtocpos("socketstr5", "strflag_abort",
                           new_point_ids["DummyCpos"], "[", "]"))

    # (h) DCp2.dz = C_P3.z - C_P1.z  (Cup P4 安全退避点用)
    new_nodes.append(
        make_equal_dz("DCp2", new_point_ids["DCp2"],
                       "C_P3", c_p3_id,
                       "C_P1", c_p1_id))

    # (i) C_P4 = CposOffset(C_P1, DCp2)
    new_nodes.append(
        make_cposoffset(c_p1_id, new_point_ids["DCp2"], new_point_ids["C_P4"]))

    # (j) If strflag_cont == 0 → CONTINUE 分支
    if_continue = make_if_var_equals("strflag_cont", "0")
    if_continue["children"] = [
        make_movl(new_point_ids["C_P4"]),       # MovL C_P4 (安全退避)
        make_wait(1000),
        make_socketsendstr("str3"),              # RESET
        make_wait(1000),
        make_movl(home_id),                     # MovL Home
        make_socketclose(),                      # SocketClose
    ]
    new_nodes.append(if_continue)

    # (k) If strflag_abort == 0 → ABORT 分支
    if_abort = make_if_var_equals("strflag_abort", "0")
    if_abort["children"] = [
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
    new_nodes.append(if_abort)

    # ---- 6. 插入新节点 ----
    nodes.extend(new_nodes)
    print(f"[INFO] 插入 {len(new_nodes)} 个新节点 (Phase D)")

    # ---- 7. 更新元数据 ----
    data["mark"] = int(time.time() * 1000)

    # ---- 8. 保存 ----
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"[SAVE] {output_path}")

    # ---- 9. 输出变更摘要 ----
    print("""
============================================================
CRP 合并完成: 598 (P4 Y后退) + v4 Phase D (HandInCup)
============================================================
基版:  handlepluscup-Down-598.crp (P4 Y后退)
输出:  handlepluscup-v4.crp

新增变量:
  str7 = "READY"          HandInCup 启动指令
  socketstr4, socketstr5  响应缓冲区
  strflag_cont, strflag_abort  分支标志

新增位姿点 (Phase D):
  DCp2 (DCPOS)            Cup P4 偏移量
  C_P4 (CPOS)             安全退避点
  DummyCpos (CPOS)        解析 CONTINUE/ABORT 用

保留 598 P4 Y后退逻辑:
  Retreat1 (DCPOS)        dx=50mm(默认)+da/db/dc=阈值
  Retreat2 (DCPOS)        dx/dy/dz=各角度Y后退量
  DCp1.dx = P3.x - P1.x
  DCp1.dz = P3.z - P1.z
  Retreat1.dy = Retreat1.dx (默认) → If cascading 覆写
  DCp1.dy = P3.y - P1.y - Retreat1.dy
  H_P4 = CposOffset(H_P1, DCp1)  → [P3.x, P3.y-Y_RETREAT, P3.z, P1.abc]

PC 端响应格式:
  CONTINUE → "<0,0,0,0,0,0>"
  ABORT    → "[0,0,0,0,0,0]"
============================================================
""")


# ==================================================================
# 主程序
# ==================================================================

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, "handlepluscup-Down-598.crp")
    output_path = os.path.join(script_dir, "handlepluscup-v4.crp")

    if not os.path.exists(input_path):
        print(f"[ERROR] 输入文件不存在: {input_path}")
        exit(1)

    backup_file(output_path)
    modify_crp(input_path, output_path)
