"""Build the standalone Windows Phase A–C ZIP from the project checkout."""
from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = "PhaseA2C-Windows"
FILES = (
    "Orchestrator/estun_phase_a2c_handle.py",
    "Orchestrator/estun_handle_cup_sm.py",
    "Orchestrator/handle_cup_detector.py",
    "Orchestrator/rot_math.py",
    "Orchestrator/run_log.py",
    "Orchestrator/config.yaml",
    "Estun/estun_config.yaml",
    "Estun/utils_shared.py",
    "EyeToHand/camera.py",
    "EyeToHand/hand_in_cup.py",
    "RealSense_R/realsense_shared.py",
    "RealSense_R/D435i_info.yaml",
    "Omnihand/omnihand_gesture.py",
    "Omnihand/omnihand_usb_serial.py",
    "calib_data/calibration_result.yaml",
    "doc/controller_exports/PhaseA2C-HandleOnly.crp",
    "model/kettle_yolov8n_finetune.pt",
    "model/yolov8m.pt",
)
SKILL = ROOT / ".claude/skills/estun-omnihand-robotics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "reference" / f"{BUNDLE}.zip")
    args = parser.parse_args()
    sources = {name: ROOT / name for name in FILES}
    for package in ("Orchestrator", "Estun", "EyeToHand", "RealSense_R", "Omnihand"):
        name = f"{package}/__init__.py"
        if (ROOT / name).is_file():
            sources[name] = ROOT / name
    for name in ("README.md", "requirements.txt"):
        sources[name] = ROOT / "packaging" / "phase_a2c_windows" / name
    for pattern in ("SKILL.md", "references/*.md", "scripts/*.py"):
        for path in sorted(SKILL.glob(pattern)):
            sources[path.relative_to(ROOT).as_posix()] = path
    required_skill = (SKILL / "SKILL.md").relative_to(ROOT).as_posix()
    missing = [name for name, path in sources.items() if not path.is_file()]
    if required_skill not in sources:
        missing.append(required_skill)
    if missing:
        parser.error("Missing bundle inputs: " + ", ".join(missing))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, "x", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in sources.items():
            archive.write(path, f"{BUNDLE}/{name}")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
