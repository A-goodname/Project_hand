"""Keep the kettle-pouring section of the exported Windows A2D controller project."""
from __future__ import annotations

import json

from prepare_controller_crps import OUTPUT, block_data, clone_block, container, main_task

NAME = "PhaseA2C-HandleOnly"


def derive(project):
    task = main_task(project)
    widgets = container(task, "widgets")
    points = container(task, "points")
    source = widgets["children"]
    point_names = {p["id"]: p["label"] for p in points["children"]}
    motions = []
    for block in source[:34]:
        if block["type"] in {"movj", "movl"}:
            point = block_data(block)["value"]["pointa"]["pointvar"]["value"]
            motions.append(point_names[point])
    assert motions == ["H_P2", "H_P1", "H_P4", "H_P3", "H_P4", "H_P1", "H_P2", "Home"], motions
    assert source[20]["type"] == "cposoffset"
    assert source[34]["type"] == "socketsendstr"
    assert block_data(source[34])["value"]["msendvar"]["datavar"]["value"] == "str4"
    continuation = source[55]
    close_template = next(b for b in continuation["children"] if b["type"] == "socketclose")
    close = clone_block(close_template)
    close["data"] = block_data(close)
    widgets["children"] = source[:34] + [close]
    if isinstance(widgets.get("doing"), dict):
        widgets["doing"]["restorepc"] = {"id": "", "parents": []}
        widgets["doing"]["current"] = {"id": "", "parents": [], "play": {}}
    points["children"] = [p for p in points["children"] if p["label"] not in {"C_P1", "C_P2", "C_P3"}]
    project["vars"]["project"][:] = [
        v for v in project["vars"]["project"]
        if v["key"] in {"socketflag", "strflag", "socketstr1", "socketstr2", "str2", "str3"}
    ]
    project["label"] = NAME
    return project


def main():
    path = OUTPUT / "PhaseA2D-KeyboardGate.crp"
    project = derive(json.loads(path.read_text(encoding="utf-8-sig")))
    target = OUTPUT / (NAME + ".crp")
    target.write_text(json.dumps(project, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8", newline="\n")
    print(target)


if __name__ == "__main__":
    main()
