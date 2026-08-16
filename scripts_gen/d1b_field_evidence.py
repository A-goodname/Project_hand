# -*- coding: utf-8 -*-
"""D1-B B3 现场取证脚本 (2026-08-16).

审计 B3 要求:
  1. Guest/Scene 相机身份闭环 — UGREEN 假序列号 SN0001 + USB 物理口 ID_PATH
  2. 固定曝光/增益/焦距/照明参数记录; D435i 连续彩色/深度 + 深度空洞 >10s 原始证据
  3. 安全证据照片 (workspace_clear / arm_power_state / e_stop) — 操作员现场拍摄
  4. 机械臂有线网链路核验 (NO-CARRIER 记录)

模式:
  --dry-run (默认): 只打印取证清单与每项动作, 不访问任何硬件
  --execute       : 真实采集 (相机采集/系统信息为只读, 可执行;
                    安全照片需操作员拍摄, 脚本在清单中标注待办)

输出: logs/d1b/field-evidence-<时间戳>/  (manifest.md + 证据文件 + sha256sum.txt)
纪律: 禁硬编码 /dev/videoN (按 sysfs 名称枚举); 只读不动 laser_power;
      每项证据记录 monotonic_ns 时间戳; 输出目录入 .sha256 校验.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from EyeToHand.camera import _list_v4l2_devices  # noqa: E402

D435I_SERIAL = "342222071364"
CALIB_YAML = os.path.join(_PROJECT_ROOT, "calib_data", "calibration_result.yaml")
D435I_YAML = os.path.join(_PROJECT_ROOT, "RealSense_R", "D435i_info.yaml")
OUT_ROOT = os.path.join(_PROJECT_ROOT, "logs", "d1b")

# 相机物理口约定 (Camera B = UGREEN usb-0:4.2; D435i 独立控制器)
CAMERA_B_PATH = "usb-0:4.2"
ETH_IFACE = "enp36s0f1"   # 机械臂有线网


def _now_ns():
    return time.monotonic_ns()


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _run(cmd, timeout=5):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout)
        return (out.stdout or out.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"[失败] {type(e).__name__}: {e}"


def _write_file(out_dir, name, content):
    path = os.path.join(out_dir, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return path


# ============================================================
#  证据步骤 (dry-run 只打印; execute 执行)
# ============================================================

def step_system(out_dir):
    """系统与机械臂有线网链路状态."""
    if not out_dir:
        return ["uname -a", f"ip -o link show {ETH_IFACE} (NO-CARRIER 记录)",
                "SSH 训练侧可达性 (ConnectTimeout=3, 只读)"]
    lines = [
        f"## 1. 系统与链路 (monotonic_ns={_now_ns()})",
        f"- uname: {_run(['uname', '-a'])}",
        f"- {ETH_IFACE} 链路: {_run(['ip', '-o', 'link', 'show', ETH_IFACE]) or '无接口'}",
        f"- SSH 训练侧: {_run(['ssh', '-o', 'ConnectTimeout=3', '-o', 'BatchMode=yes', 'nickwang@192.168.112.82', 'echo OK']) or '不通'}",
        f"- 时间戳纪律: 上述命令执行于 monotonic_ns={_now_ns()}",
    ]
    path = _write_file(out_dir, "01-system-link.md", "\n".join(lines) + "\n")
    return [path]


def step_camera_identity(out_dir):
    """相机身份闭环: sysfs 名称 + USB 物理口 ID_PATH (禁 /dev/videoN 硬编码)."""
    if not out_dir:
        return ["枚举 /sys/class/video4linux/videoN/name + udevadm ID_PATH",
                "D435i serial (pyrealsense2 只读枚举)",
                "UGREEN 假序列号 SN0001 标注 + 物理口绑定 (Camera B = usb-0:4.2)"]
    lines = [f"## 2. 相机身份 (monotonic_ns={_now_ns()})",
             "- V4L2 设备枚举 (index / name / ID_PATH):"]
    for idx, name, id_path in _list_v4l2_devices():
        tag = ""
        if "UGREEN" in name:
            tag = " [UGREEN 假序列号 SN0001, 仅物理口可区分]"
            if id_path and CAMERA_B_PATH in id_path:
                tag += " ← Camera B (usb-0:4.2)"
        lines.append(f"  - [{idx}] {name}  ({id_path}){tag}")
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
        ctx = rs.context()
        for dev in ctx.query_devices():
            sn = dev.get_info(rs.camera_info.serial_number)
            lines.append(f"- D435i serial: {sn} "
                         f"({'✅ 匹配' if sn == D435I_SERIAL else '⚠️ 不匹配!'})")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- D435i 枚举失败: {type(e).__name__}: {e}")
    path = _write_file(out_dir, "02-camera-identity.md", "\n".join(lines) + "\n")
    return [path]


def step_d435i_params_and_evidence(out_dir):
    """D435i 参数 (只读) + >10s 连续彩色/深度 + 深度空洞统计."""
    if not out_dir:
        return ["读 D435i exposure/gain/laser_power (只读, 不动 laser_power)",
                "连续采集 ~10s (≥30 帧): 每帧深度有效点/空洞比",
                "保存首/中/末 彩色+深度 PNG (原始证据)"]
    lines = [f"## 3. D435i 参数与 10s 连续证据 (monotonic_ns={_now_ns()})"]
    import pyrealsense2 as rs  # noqa: PLC0415
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(D435I_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    dev = pipe.get_active_profile().get_device()
    try:
        sensor = dev.first_color_sensor()
        for opt, name in ((rs.option.exposure, "exposure"),
                          (rs.option.gain, "gain")):
            try:
                lines.append(f"- color {name}: {sensor.get_option(opt)}")
            except Exception:  # noqa: BLE001
                pass
        try:
            ds = dev.first_depth_sensor()
            lines.append(f"- depth laser_power (只读): {ds.get_option(rs.option.laser_power)}")
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        lines.append(f"- 参数读取失败: {type(e).__name__}: {e}")
    # 10s 连续: 每帧深度有效点统计
    import cv2  # noqa: PLC0415
    for _ in range(15):
        pipe.wait_for_frames()
    t0 = time.monotonic()
    stats = []
    saved = []
    frame_i = 0
    while time.monotonic() - t0 < 10.0:
        frames = pipe.wait_for_frames()
        color = np.asanyarray(frames.get_color_frame().get_data())
        depth = np.asanyarray(frames.get_depth_frame().get_data())
        valid = int((depth > 0).sum())
        total = depth.size
        stats.append({"frame": frame_i, "monotonic_ns": _now_ns(),
                      "valid_depth_px": valid,
                      "hole_ratio": round(1.0 - valid / total, 4)})
        if frame_i in (0, 15, 29):
            cv2.imwrite(os.path.join(out_dir, f"color_{frame_i:02d}.png"), color)
            np.save(os.path.join(out_dir, f"depth_{frame_i:02d}.npy"), depth)
            saved.append(frame_i)
        frame_i += 1
    pipe.stop()
    lines.append(f"- 连续采样: {len(stats)} 帧 / 10s; 保存帧: {saved}")
    lines.append(f"- 深度空洞统计: 有效点均值 "
                 f"{np.mean([s['valid_depth_px'] for s in stats]):.0f}px, "
                 f"空洞比均值 {np.mean([s['hole_ratio'] for s in stats]):.4f}, "
                 f"最大空洞比 {max(s['hole_ratio'] for s in stats):.4f}")
    path = _write_file(out_dir, "03-d435i-params-evidence.md",
                       "\n".join(lines) + "\n")
    json_path = _write_file(out_dir, "d435i-frame-stats.json",
                            json.dumps(stats, indent=2))
    return [path, json_path] + [os.path.join(out_dir, f"color_{i:02d}.png")
                                for i in saved]


def step_calibration_digest(out_dir):
    """标定摘要: SHA256 + 方法/样本数/重投影误差 + T_cam2ee + 内参文件摘要."""
    if not out_dir:
        return [f"sha256 {os.path.basename(CALIB_YAML)}",
                "use_method / num_samples / reprojection_error_px / T_cam_to_gripper",
                f"sha256 {os.path.basename(D435I_YAML)} (替代内参来源)"]
    import yaml  # noqa: PLC0415
    lines = [f"## 4. 标定摘要 (monotonic_ns={_now_ns()})"]
    for p in (CALIB_YAML, D435I_YAML):
        if os.path.exists(p):
            lines.append(f"- {os.path.basename(p)} sha256: {_sha256(p)}")
    try:
        with open(CALIB_YAML, encoding="utf-8") as f:
            cal = yaml.safe_load(f)["calibration"]["handeye"]
        m = cal["use_method"]
        lines.append(f"- use_method: {m}, num_samples: {cal['num_samples']}, "
                     f"reprojection_error_px: {cal['reprojection_error_px']}")
        lines.append(f"- T_cam_to_gripper 平移: "
                     f"{cal[m]['T_cam_to_gripper'][0][3]:.6f}, "
                     f"{cal[m]['T_cam_to_gripper'][1][3]:.6f}, "
                     f"{cal[m]['T_cam_to_gripper'][2][3]:.6f}")
    except Exception as e:  # noqa: BLE001
        lines.append(f"- 标定读取失败: {type(e).__name__}: {e}")
    path = _write_file(out_dir, "04-calibration-digest.md", "\n".join(lines) + "\n")
    return [path]


def step_safety_photos(out_dir):
    """安全证据照片清单 (操作员现场拍摄, 脚本只登记待办)."""
    items = ["workspace_clear.jpg", "arm_power_state.jpg", "e_stop.jpg"]
    if not out_dir:
        return [f"操作员现场拍摄: {', '.join(items)} (脚本不涉及机械臂动作)"]
    lines = [f"## 5. 安全证据照片 (monotonic_ns={_now_ns()})",
             "以下照片需操作员现场拍摄后放入本目录 (D1-A 传输清单缺失项):"]
    for it in items:
        p = os.path.join(out_dir, it)
        if os.path.exists(p):
            lines.append(f"- [x] {it} (sha256: {_sha256(p)})")
        else:
            lines.append(f"- [ ] {it} (待操作员拍摄)")
    lines.append("- 现场灯光: 待操作员填写 (照明参数要求)")
    path = _write_file(out_dir, "05-safety-photos-checklist.md",
                       "\n".join(lines) + "\n")
    return [path]


STEPS = [
    ("系统与机械臂网线链路", step_system),
    ("相机身份闭环 (物理口)", step_camera_identity),
    ("D435i 参数 + 10s 连续证据", step_d435i_params_and_evidence),
    ("标定摘要", step_calibration_digest),
    ("安全证据照片清单", step_safety_photos),
]


def main():
    ap = argparse.ArgumentParser(description="D1-B B3 现场取证 (默认 --dry-run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="只打印取证清单, 不访问硬件 (默认)")
    ap.add_argument("--execute", action="store_true",
                    help="真实采集 (相机/系统只读; 照片清单待操作员)")
    args = ap.parse_args()
    execute = args.execute

    if not execute:
        print("=== D1-B B3 现场取证清单 (--dry-run, 未访问任何硬件) ===")
        print(f"输出目录: {OUT_ROOT}/field-evidence-<时间戳>/")
        for name, fn in STEPS:
            print(f"\n[{name}]")
            for item in fn(None):
                print(f"  - {item}")
        print("\n[dry-run] 执行取证请加 --execute (相机/系统采集为只读; "
              "安全照片需操作员拍摄)")
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = os.path.join(OUT_ROOT, f"field-evidence-{stamp}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"=== B3 现场取证执行中 → {out_dir} ===")
    all_files = []
    manifest = [f"# B3 现场取证 (2026-08-16 {stamp})"]
    for name, fn in STEPS:
        print(f"[{name}]")
        try:
            files = fn(out_dir)
            manifest.append(f"\n## {name}")
            for p in files:
                rel = os.path.relpath(p, out_dir)
                manifest.append(f"- {rel} (sha256: {_sha256(p)})")
                all_files.append(p)
            print(f"  → {len(files)} 个文件")
        except Exception as e:  # noqa: BLE001 — 单步容错: 一项失败不中断整体
            msg = f"{type(e).__name__}: {e}"
            manifest.append(f"\n## {name}\n- [失败] {msg}")
            print(f"  → [失败] {msg} (已记录, 继续后续步骤)")
    manifest_path = _write_file(out_dir, "manifest.md", "\n".join(manifest) + "\n")
    all_files.append(manifest_path)
    sha_lines = []
    for p in sorted(all_files):
        rel = os.path.relpath(p, out_dir)
        sha_lines.append(f"{_sha256(p)}  {rel}")
    _write_file(out_dir, "sha256sum.txt", "\n".join(sha_lines) + "\n")
    print(f"\n完成: {len(all_files)} 个文件, 校验清单 {out_dir}/sha256sum.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
