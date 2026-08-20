#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 state_machine_reference.crp (状态机/main 全流程) 抽取 Phase 0 打招呼段,
生成独立 doc/estun_greeting_phase0.crp.

方法 (对齐 gen_session_barrier_crp.py): deepcopy 源工程, 只保留 Phase 0 块序列
(首个 Socket1:8600 四步协议, 到 socketclose 为止), 位姿点只保留 GH_*/Home,
`vars.project` 中 Phase 0 未用的 handle 流程变量一并剔除.

Phase 0 协议 (与 estun_greeting_phase0.py 匹配):
  1. SendCurCPos -> recv -> ZHAOSHOU1 (MovL)
  2. SendCurCPos -> recv -> ZHAOSHOU2 (MovL)
  3. SendCurCPos -> recv -> WOSHOU   (MovL)
  4. SendCurCPos -> recv -> Home    (MovL) -> SocketClose
"""
import copy
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "doc" / "state_machine_reference.crp"
OUT = ROOT / "doc" / "estun_greeting_phase0.crp"

# Phase 0 保留的位姿点 label (除块实际引用的 Home 外, 保留 GH_* 家族)
KEEP_POINT_LABELS = ("GH_CUR", "GH_ZHAOSHOU1", "GH_ZHAOSHOU2",
                     "GH_WOSHOU", "GH_HOME2", "Home")


def load(path: Path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def dump(data) -> str:
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


def find_task(data):
    for top in data.get("children", []):
        if top.get("type") == "task":
            return top
    raise RuntimeError("task 未找到")


def find_child(task, child_type):
    for c in task.get("children", []):
        if c.get("type") == child_type:
            return c
    raise RuntimeError(f"child {child_type} 未找到")


def block_point_refs(block) -> set:
    """块 data 中引用的 pointvar 点 id 集合."""
    s = json.dumps(block.get("data", {}), ensure_ascii=False)
    return set(re.findall(r'"pointvar"\s*:\s*\{[^}]*"value"\s*:\s*"([^"]+)"', s))


def main():
    data = load(SRC)
    data = copy.deepcopy(data)
    task = find_task(data)
    widgets = find_child(task, "widgets")
    points = find_child(task, "points")
    blocks = widgets.get("children", [])
    all_points = points.get("children", [])

    # ---- 1. Phase 0 块序列: 从首块到第二个 socketcreate 之前的 socketclose ----
    #     首个 socketcreate = Socket1:8600 (打招呼); 其后紧跟 handle 流程的 Socket0:8080
    first_sc = next((i for i, b in enumerate(blocks) if b["type"] == "socketcreate"), None)
    assert first_sc is not None, "源工程无 socketcreate"
    port0 = json.dumps(blocks[first_sc]["data"]).find("8600")
    assert port0 >= 0, f"首 socketcreate 非 8600: {blocks[first_sc]['label']}"
    second_sc = next((i for i in range(first_sc + 1, len(blocks))
                      if blocks[i]["type"] == "socketcreate"), None)
    assert second_sc is not None, "源工程无第二个 socketcreate (handle 流程)"
    # Phase 0 结束 = 第二个 socketcreate 之前的最后一个 socketclose (Socket1 关闭);
    # 其后的 wait 是转 handle 流程的停顿, 不纳入
    last_close = max(i for i in range(second_sc) if blocks[i]["type"] == "socketclose")
    phase0_blocks = blocks[:last_close + 1]
    print(f"[INFO] Phase 0 块: 0..{last_close} ({len(phase0_blocks)} 块, "
          f"末尾 socketclose)")

    # ---- 2. 保留点位: 块实际引用点 + GH_* 家族 ----
    ref_ids = set()
    for b in phase0_blocks:
        ref_ids |= block_point_refs(b)
    keep_points = [p for p in all_points
                   if p.get("id") in ref_ids or p.get("label") in KEEP_POINT_LABELS]
    dropped_points = len(all_points) - len(keep_points)
    print(f"[INFO] 保留点: {[p.get('label') for p in keep_points]}  (丢弃 {dropped_points})")
    for p in keep_points:
        assert p.get("id") in ref_ids or p.get("label") in KEEP_POINT_LABELS

    # ---- 3. 裁剪 ----
    widgets["children"] = phase0_blocks
    points["children"] = keep_points

    # ---- 4. vars.project: 剔除 Phase 0 未用的 handle 流程变量 ----
    #     Phase 0 只引用系统变量 (Socket1/XYZData/tra), vars.project 全为 handle 用,
    #     全部移除 (留空数组, 编辑器可接受).
    data["vars"]["project"] = []

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(dump(data) + "\n", encoding="utf-8")
    print(f"[OK] 已生成: {OUT}")

    # ---- 5. 校验: 保留块的所有点引用均能解析 ----
    kept_ids = {p["id"] for p in keep_points}
    for i, b in enumerate(phase0_blocks):
        for pid in block_point_refs(b):
            assert pid in kept_ids, f"块 {i} 引用缺失点 {pid}"
    print("[OK] 所有 Phase 0 块点引用可解析")
    print(f"[INFO] 输出 sha256: {__import__('hashlib').sha256(OUT.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    sys.exit(main())
