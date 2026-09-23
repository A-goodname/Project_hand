"""Import controller exports and derive a Phase0+A-D ALL project without RPS.

The files in D:/Download are read-only inputs. Run from the project root.
"""
from __future__ import annotations

import json
import shutil
import copy
import secrets
import string
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("D:/Download")
OUTPUT = ROOT / "doc" / "controller_exports"
EXPORTS = (
    "Phase0-Down-965.crp",
    "PhaseA2D-Down-444.crp",
    "PhaseAll-Down-617.crp",
)
def clone_block(template: dict) -> dict:
    """Clone a controller widget with a fresh ID (same rule as CRP helper)."""
    prefixes = {
        "transtrtocpos": "ts", "if": "nd", "movl": "ml",
        "socketsendstr": "ss", "socketclose": "sc", "wait": "wt",
    }
    block = copy.deepcopy(template)
    prefix = prefixes.get(block.get("type"), "nd")
    stamp = int(time.time() * 1000)
    base36 = ""
    while stamp:
        stamp, digit = divmod(stamp, 36)
        base36 = string.digits[digit] + base36 if digit < 10 else string.ascii_lowercase[digit - 10] + base36
    suffix = "".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8))
    block["id"] = f"{prefix}{base36}{suffix}"
    block["index"] = 0
    block["parents"] = []
    return block


def main_task(project: dict) -> dict:
    return next(
        child for child in project["children"]
        if child.get("type") == "task" and child.get("typetask") == "main"
    )


def container(task: dict, kind: str) -> dict:
    return next(child for child in task["children"] if child.get("type") == kind)


def block_data(block: dict) -> dict:
    value = block.get("data", {})
    return json.loads(value) if isinstance(value, str) else value


def add_phase_d_branch(project: dict, ref_widgets: list[dict], home_movl: dict) -> int:
    task = main_task(project)
    widgets = container(task, "widgets")["children"]
    points = container(task, "points")["children"]
    assert [w["type"] for w in ref_widgets[76:80]] == [
        "transtrtocpos", "transtrtocpos", "if", "if"
    ]
    assert widgets[-2]["type"] == "socketreadstr" and widgets[-1]["type"] == "socketclose"
    source_widgets = widgets[:]
    point_ids = {point["label"]: point["id"] for point in points}
    def source_movl(label: str) -> dict:
        return next(block for block in source_widgets
                    if block["type"] == "movl" and
                    block_data(block)["value"]["pointa"]["pointvar"]["value"] == point_ids[label])

    def source_send(value: str) -> dict:
        return next(block for block in source_widgets
                    if block["type"] == "socketsendstr" and
                    block_data(block)["value"]["msendvar"]["datavar"]["value"] == value)

    wait_1s = next(block for block in source_widgets
                   if block["type"] == "wait" and block_data(block)["value"]["time"] == 1000)
    assert block_data(home_movl)["value"]["pointa"]["pointvar"]["value"] == point_ids["Home"]
    assert next(v["value"]["STRING"]["value"] for v in project["vars"]["project"] if v["key"] == "str5") == "RELEASE"
    assert next(v["value"]["STRING"]["value"] for v in project["vars"]["project"] if v["key"] == "str3") == "RESET"
    old_close = widgets.pop()
    c_p3_id = point_ids["C_P3"]
    for source in ref_widgets[76:78]:
        parsed = clone_block(source)
        parsed["data"]["value"]["pointa"]["pointvar"]["value"] = c_p3_id
        widgets.append(parsed)
    branch_templates = {
        "CONTINUE": [wait_1s, home_movl, old_close],
        "ABORT": [source_movl("C_P1"), source_send("str5"), wait_1s,
                  source_movl("C_P2"), source_send("str3"), wait_1s,
                  home_movl, old_close],
    }
    for source, label in zip(ref_widgets[78:80], ("CONTINUE", "ABORT")):
        branch = clone_block(source)
        branch["label"] = f"PhaseD {label} (keyboard)"
        branch["children"] = [clone_block(template) for template in branch_templates[label]]
        for child in branch["children"]:
            child["parents"] = [branch["id"]]
            if child["type"] == "socketclose":
                child["data"] = block_data(child)
        if label == "ABORT":
            # Retreat with the cup still held: use the existing cup-carry speed.
            branch["children"][0]["data"]["value"]["mspeed"]["datavar"]["value"] = "sudu2"
        widgets.append(branch)
    return len(widgets)


def write_derived(path: Path, project: dict) -> None:
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(path, backup)
    path.write_text(json.dumps(project, ensure_ascii=False, separators=(",", ":")) + "\n",
                    encoding="utf-8", newline="\n")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for name in EXPORTS:
        src = SOURCE / name
        dst = OUTPUT / name
        if not src.is_file():
            raise FileNotFoundError(src)
        if dst.exists() and src.read_bytes() != dst.read_bytes():
            raise RuntimeError(f"Existing imported CRP differs; refusing overwrite: {dst}")
        if not dst.exists():
            shutil.copy2(src, dst)

    original = OUTPUT / "PhaseAll-Down-617.crp"
    derived = OUTPUT / "PhaseAll-NoRPS.crp"
    project = json.loads(original.read_text(encoding="utf-8"))
    task = main_task(project)
    widgets = container(task, "widgets")["children"]
    points = container(task, "points")["children"]

    assert len(widgets) == 94, f"Unexpected ALL widget count: {len(widgets)}"
    assert widgets[80]["type"] == "socketclose"
    assert widgets[81]["type"] == "wait"
    assert widgets[82]["type"] == "socketcreate"
    assert widgets[-1]["type"] == "socketclose"
    assert {p["label"] for p in points if p["label"] in {"CurPos", "RPSpos"}} == {"CurPos", "RPSpos"}

    # The final wait and subsequent socket blocks are the complete RPS phase.
    del widgets[81:]
    points[:] = [p for p in points if p["label"] not in {"CurPos", "RPSpos"}]
    retained_ids = {p["id"] for p in points}
    removed_ids = {p["id"] for p in container(main_task(json.loads(original.read_text(encoding="utf-8"))), "points")["children"] if p["label"] in {"CurPos", "RPSpos"}}
    for block in widgets:
        if any(point_id in json.dumps(block, ensure_ascii=False) for point_id in removed_ids):
            raise RuntimeError(f"Remaining widget references RPS point: {block.get('label')}")
    assert len(widgets) == 81 and len(points) == 16
    assert len(retained_ids) == len(points)
    rps_vars = {"rpsstr1", "rpsstr2", "rpsstr3", "endflag", "whileflag"}
    for category in project.get("projectcfg", {}).get("vars", {}).get("data", {}).values():
        if isinstance(category, list):
            category[:] = [item for item in category if item.get("key") not in rps_vars]
    project.get("vars", {}).get("project", [])[:] = [
        item for item in project.get("vars", {}).get("project", [])
        if item.get("key") not in rps_vars
    ]

    # Reuse the tested parse/If schema while preserving controller motion paths.
    reference = json.loads((ROOT / "doc" / "handlepluscup-orchestrator.crp").read_text(encoding="utf-8"))
    ref_widgets = container(main_task(reference), "widgets")["children"]
    phase0_export = json.loads((OUTPUT / "Phase0-Down-965.crp").read_text(encoding="utf-8"))
    phase0_widgets = container(main_task(phase0_export), "widgets")["children"]
    assert phase0_widgets[24]["type"] == "movl"
    home_movl = phase0_widgets[24]
    assert add_phase_d_branch(project, ref_widgets, home_movl) == 84

    text = json.dumps(project, ensure_ascii=False, separators=(",", ":")) + "\n"
    if "192.168.101.96" not in text or "8700" in text or "RPSpos" in text or "rpsstr" in text:
        raise RuntimeError("Port or RPS validation failed")
    write_derived(derived, project)
    # Round-trip and verify all three protocol sockets remain.
    loaded = json.loads(derived.read_text(encoding="utf-8"))
    after = container(main_task(loaded), "widgets")["children"]
    assert len(after) == 84
    assert sum(block["type"] == "socketcreate" for block in after) == 2
    a2d_project = json.loads((OUTPUT / "PhaseA2D-Down-444.crp").read_text(encoding="utf-8"))
    a2d_count = add_phase_d_branch(a2d_project, ref_widgets, home_movl)
    assert a2d_count == 57
    a2d_derived = OUTPUT / "PhaseA2D-KeyboardGate.crp"
    write_derived(a2d_derived, a2d_project)
    a2d_loaded = json.loads(a2d_derived.read_text(encoding="utf-8"))
    assert len(container(main_task(a2d_loaded), "widgets")["children"]) == 57
    print(f"Imported {len(EXPORTS)} controller exports to {OUTPUT}")
    print(f"Derived {derived.name}: {len(after)} widgets, {len(points)} points, no RPS, PhaseD CONTINUE/ABORT")
    print(f"Derived {a2d_derived.name}: {a2d_count} widgets, PhaseD CONTINUE/ABORT")


if __name__ == "__main__":
    main()
