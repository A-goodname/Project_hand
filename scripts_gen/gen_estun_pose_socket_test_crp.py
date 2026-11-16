"""Generate a motion-free ESTUN project with a directly runnable pose socket test."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time

from prepare_controller_crps import (
    OUTPUT, ROOT, block_data, clone_block, container, main_task,
)

sys.path.insert(0, str(ROOT))
from Orchestrator.estun_pose_socket_test import PORT, SAMPLE_COUNT

NAME = "Estun-Pose-Socket-Test-v4"
JOINT_NAME = "Estun-Pose-Joint-Socket-Test"
SUB_EXPORT = "Project20260629170652354-Down-928.crp"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def derive(project, sub_export, with_joints=False):
    main = main_task(project)
    original = container(main, "widgets")["children"]
    connect, on_connect_failure, read, encode, send = original[:5]
    wait = original[8]
    branch = original[55]
    close = next(b for b in branch["children"] if b["type"] == "socketclose")
    assert [b["type"] for b in (connect, read, encode, send, wait)] == [
        "socketcreate", "getcurcpos", "cpostostr", "socketsendstr", "wait",
    ]
    point_id = block_data(read)["value"]["pointa"]["pointvar"]["value"]
    point_template = next(p for p in container(main, "points")["children"] if p["id"] == point_id)

    def clone(template):
        widget = clone_block(template)
        widget["data"] = block_data(widget)
        return widget

    def blocks(target_point, joint_point=None):
        socket_open = clone(connect)
        socket_open["data"]["value"]["mport"] = PORT
        guard = clone(on_connect_failure)
        guard["children"] = [clone(b) for b in on_connect_failure["children"]]
        for child in guard["children"]:
            child["parents"] = [guard["id"]]
        result = [socket_open, guard]
        for sequence in range(1, SAMPLE_COUNT + 1):
            capture, format_pose, transmit = (clone(t) for t in (read, encode, send))
            for widget in (capture, format_pose):
                widget["data"]["value"]["pointa"]["pointvar"]["value"] = target_point
            format_pose["data"]["value"].update(strstring=f"P|{sequence:04d}|[", endstring="];")
            result.extend((capture, format_pose, transmit))
            if joint_point is not None:
                # Candidate APOS widgets based on the corresponding CPOS export
                # and the documented APOS parameters. Verify in the pendant.
                capture_joint, format_joint, transmit_joint = (clone(t) for t in (read, encode, send))
                for widget, kind, label, location in (
                    (capture_joint, "getcurapos", "GetCurAPos", "storagelocation"),
                    (format_joint, "apostostr", "APosToStr", "unconvertedapos"),
                ):
                    widget.update(type=kind, icon=kind, label=label)
                    widget["data"]["value"]["pointa"] = {
                        "langs": location, "pointvar": {"type": ["apos"], "value": joint_point},
                    }
                v = format_joint["data"]["value"]
                v.pop("lengthunit", None)
                v.update(strstring=f"J|{sequence:04d}|[", endstring="];")
                result.extend((capture_joint, format_joint, transmit_joint))
            if sequence < SAMPLE_COUNT:
                pause = clone(wait)
                pause["data"]["value"]["time"] = 250
                result.append(pause)
        result.append(clone(close))
        return result

    def fresh_id(node):
        # Preserve the actual prefix used by the controller for this node type.
        return node["id"][:2] + clone_block(read)["id"][2:]

    def populate(task, points):
        task["id"] = fresh_id(task)
        task["parents"] = [project["id"]]
        for child in task["children"]:
            child["id"] = fresh_id(child)
            child["parents"] = [project["id"], task["id"]]
        container(task, "points")["children"] = points
        container(task, "widgets")["children"] = blocks(point["id"], joint_id)

    project["id"] = fresh_id(project)
    point = copy.deepcopy(point_template)
    point["id"] = fresh_id(point)
    point["label"] = "PoseTestMain"
    points = [point]
    joint_id = None
    if with_joints:
        joint = copy.deepcopy(point_template)
        joint_id = fresh_id(joint)
        joint.update(id=joint_id, label="PoseTestJoints")
        v = joint["data"]["value"]
        v["postype"] = "apos"
        v["posvar"]["datavar"]["type"] = "APOS"
        v["posvalue"] = {"apos": {**{f"j{i}": 0 for i in range(1, 7)}, "e": 0}}
        points.append(joint)
    populate(main, points)

    # Use the subroutine task structure from a real controller export. There is
    # no RUN block template in the supplied exports, so main remains standalone.
    sub = copy.deepcopy(next(t for t in sub_export["children"] if t.get("typetask") == "subroutine"))
    sub["label"] = "PoseSocketSample"
    # With merged project points, the pendant exposes the main point table
    # in this program too. Both paths use the same project scratch point.
    populate(sub, [])
    project["children"] = [main, sub]
    project["label"] = JOINT_NAME if with_joints else NAME
    project["mark"] = int(time.time() * 1000)
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sub-template", type=Path,
                        help="Controller export containing an empty subroutine task")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--with-joints", action="store_true", help="Include APOS candidate widgets")
    args = parser.parse_args()
    sub_path = args.sub_template or next((ROOT / "reference").rglob(SUB_EXPORT), None)
    if sub_path is None:
        parser.error("Provide --sub-template with a controller-exported subroutine")
    project = derive(load(OUTPUT / "PhaseA2D-KeyboardGate.crp"), load(sub_path), args.with_joints)
    output = args.output or OUTPUT / f"{project['label']}.crp"
    output.write_text(json.dumps(project, ensure_ascii=False, separators=(",", ":")) + "\n",
                      encoding="utf-8", newline="\n")
    print(output.resolve())


if __name__ == "__main__":
    main()
