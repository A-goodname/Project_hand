# -*- coding: utf-8 -*-
"""现场门禁 site_gate (audit 修复 1, 2026-08-16).

在 load_cam2ee (D0 标定门禁) 之后、TCP server bind 之前执行, 全部只读检查:

  1. 现场准备确认 (site_gate.preparation_ok) — 操作员清场后置位, 防未清场启动
  2. 相机身份 — Camera A (D435i) 实际 USB 设备序列号 == 配置期望
                   Camera B (UGREEN) 物理口 ID_PATH == usb-0:4.2 (假序列号 SN0001
                   只能靠物理口区分, 禁硬编码 /dev/videoN)
  3. carrier 活体 — 灵巧手串口稳定路径存在 (by-id, 禁 ttyACM 编号)
  4. 机械臂有线网链路 — enp36s0f1 载波状态 (NO-CARRIER 记录; 审计要求联机前
     现场核验, 此处仅记录警告不阻断本地调试)

任一硬性检查失败 → 阻止启动 (与 D0 同纪律: 不启动比带病启动好).
"""
import os
import subprocess
import sys

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from EyeToHand.camera import _list_v4l2_devices, find_camera_id  # noqa: E402

_IS_WINDOWS = sys.platform == "win32"


def _eth_carrier(iface: str) -> str | None:
    """读取网口载波状态 (NO-CARRIER / UP 等); 无该接口 → None."""
    try:
        if _IS_WINDOWS:
            safe_iface = iface.replace("'", "''")
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command",
                 f"(Get-NetAdapter -Name '{safe_iface}').Status"],
                capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=5)
            return (out.stdout or out.stderr).strip() or None
        out = subprocess.run(["ip", "-o", "link", "show", iface],
                             capture_output=True, text=True, timeout=5)
        return (out.stdout or out.stderr).strip() or None
    except Exception:  # noqa: BLE001
        return None


def check_camera_a_serial(expected: str) -> str | None:
    """Camera A (D435i) 序列号匹配. 返回失败原因或 None."""
    try:
        import pyrealsense2 as rs  # noqa: PLC0415
        ctx = rs.context()
        actual = [d.get_info(rs.camera_info.serial_number)
                  for d in ctx.query_devices()]
    except Exception as e:  # noqa: BLE001
        return f"Camera A 枚举失败: {type(e).__name__}: {e}"
    if expected not in actual:
        return (f"Camera A 序列号不匹配: 期望 {expected}, 实际 {actual or '无设备'}")
    return None


def check_camera_b_physical_port(path_keyword: str = "usb-0:4.2") -> str | None:
    """Camera B (UGREEN) USB 物理口匹配 (假序列号 SN0001, 仅物理口可区分)."""
    devices = _list_v4l2_devices()
    ugreen = [d for d in devices if "UGREEN" in d[1]]
    if not ugreen:
        return "Camera B (UGREEN) 未枚举到"
    if not any(path_keyword in d[2] for d in ugreen):
        ports = ", ".join(d[2] for d in ugreen)
        return f"Camera B 物理口不匹配: 期望含 {path_keyword}, 实际 {ports or '无 ID_PATH'}"
    return None


def check_camera_b_device(camera_id: int = -1, camera_name: str = "",
                          camera_path: str = "") -> str | None:
    """跨平台核验 Camera B 身份，不打开视频流。

    Windows 的两个 UGREEN 有不同 DSHOW 完整名称，按名称核验；Linux 仍按
    USB 物理口核验，避免依赖会漂移的 /dev/videoN。
    """
    if not _IS_WINDOWS:
        return check_camera_b_physical_port(camera_path)
    try:
        resolved = find_camera_id([camera_name] if camera_name else ["UGREEN"])
    except Exception as exc:  # noqa: BLE001
        return f"Camera B 枚举失败: {type(exc).__name__}: {exc}"
    if resolved is None:
        return f"Camera B 未找到: name={camera_name!r}"
    if int(camera_id) >= 0 and resolved != int(camera_id):
        return f"Camera B 索引不匹配: 期望 {camera_id}, 按名称解析为 {resolved}"
    return None


def check_hand_carrier_alive(serial_path: str) -> str | None:
    """灵巧手 carrier 活体: 串口稳定路径存在 (by-id)."""
    if _IS_WINDOWS:
        try:
            from serial.tools import list_ports  # noqa: PLC0415
            ports = {p.device.upper() for p in list_ports.comports()}
        except Exception as exc:  # noqa: BLE001
            return f"灵巧手串口枚举失败: {type(exc).__name__}: {exc}"
        if serial_path.upper() not in ports:
            return f"灵巧手串口不存在: {serial_path}; 实际 {sorted(ports)}"
        return None
    if not os.path.exists(serial_path):
        return f"灵巧手串口不存在: {serial_path}"
    return None


def check_ether_link(iface: str = "enp36s0f1") -> tuple[str | None, str | None]:
    """机械臂有线网链路 (记录型, 不阻断). 返回 (警告, 状态描述)."""
    state = _eth_carrier(iface)
    if state is None:
        return None, "无该网络接口"
    if "NO-CARRIER" in state or state.lower() in {"disconnected", "down"}:
        return (f"机械臂有线网 {iface} NO-CARRIER — 联机测试前需现场物理链路核验",
                state)
    return None, state


def run_site_gate(*, preparation_ok: bool, camera_a_serial: str,
                  camera_b_path: str, hand_serial_path: str,
                  camera_b_name: str = "", camera_b_id: int = -1,
                  eth_iface: str = "enp36s0f1",
                  block_link_down: bool = False) -> list[str]:
    """执行现场门禁. 返回失败原因列表 (空 = 通过).

    block_link_down=False: 网线链路只记录警告 (审计: 联机前核验), 不阻断.
    """
    failures: list[str] = []
    warnings: list[str] = []

    if not preparation_ok:
        failures.append("现场准备未确认 (site_gate.preparation_ok=false, "
                        "操作员清场后置位)")

    fail = check_camera_a_serial(camera_a_serial)
    if fail:
        failures.append(fail)

    fail = (check_camera_b_device(camera_b_id, camera_b_name, camera_b_path)
            if camera_b_name else check_camera_b_physical_port(camera_b_path))
    if fail:
        failures.append(fail)

    fail = check_hand_carrier_alive(hand_serial_path)
    if fail:
        failures.append(fail)

    warn, state = check_ether_link(eth_iface)
    if warn:
        if block_link_down:
            failures.append(warn)
        else:
            warnings.append(warn)

    for w in warnings:
        print(f"[SITE-GATE] 警告: {w}")
    for f in failures:
        print(f"[SITE-GATE] 失败: {f}")
    if not failures and not warnings:
        print("[SITE-GATE] 全部通过")
    return failures
