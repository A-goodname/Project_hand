# -*- coding: utf-8 -*-
"""D1-B B1 现场复测 (单次运行, 2026-08-16, v2).

流程 (用户方向): 实时画面点选基准点(可连点, u 撤销) → 输入示教器法兰位姿 →
      输出 点 (px,py) -> 相机系 3D -> 基系预测位姿 (+ 误差 vs 基准点真值)
操作员自行移动机械臂, 让基准点落在画面不同位置 (中心/中环/边缘/角落),
每轮点选+位姿构成一个样本, 直接积累"误差-图像位置"数据 (B1 诊断).

变换链 (与生产同函数):
  相机系 ->(T_cam2ee)-> 末端系 ->(R_ee_base, t_ee_base)-> 基系
显示: matplotlib TkAgg 实时画面 (规避 cv2 Qt 本机黑屏).
位姿输入在选点之后: 机器人停稳时, 输入位姿与点击时刻的画面一致
(深度/像素在点击时刻取帧); 位姿支持 空格/逗号/括号 格式.
"""
import os
import sys
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _PROJECT_ROOT)

import numpy as np  # noqa: E402

from Orchestrator.rot_math import sanity_check_pose  # noqa: E402
from Orchestrator.state_machine import CalibrationError, load_cam2ee  # noqa: E402
from Estun.utils_shared import transform_pose_to_base  # noqa: E402

SERIAL = "342222071364"
D435I_YAML = os.path.join(_PROJECT_ROOT, "RealSense_R", "D435i_info.yaml")
CALIB_YAML = os.path.join(_PROJECT_ROOT, "calib_data", "calibration_result.yaml")
POINT_TIMEOUT_S = 300.0   # 单轮实时点选超时 (秒), 超时进入位姿输入

# 被测点 (D1-A calibration-validation.md, y 修正后; 换点改这里)
TEST_POINT = {"name": "desk_ref (基准点)", "xyz_m": [-0.370, 0.800, 0.295]}


def _parse_pose_input(text):
    """'x y z A B C' (mm/deg) → (xyz_m, abc_deg).

    兼容: 空格分隔 / 逗号分隔 / 方括号列表 (如 [-111.896,378.271,...,4.801]).
    非法返回 None.
    """
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    text = text.replace(",", " ")
    parts = text.split()
    if len(parts) != 6:
        return None
    try:
        vals = [float(p) for p in parts]
    except ValueError:
        return None
    xyz_mm, abc = vals[:3], vals[3:]
    xyz_m = [v / 1000.0 for v in xyz_mm]
    sanity_check_pose(xyz_m + abc, "m")   # 单位纪律: 显式 m 校验
    return xyz_m, abc


def main():
    import pyrealsense2 as rs  # noqa: PLC0415
    from RealSense_R.realsense_shared import (  # noqa: PLC0415
        color_to_depth_pixel, depth_median_around, load_camera_params,
    )
    import matplotlib  # noqa: PLC0415
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    cv2 = __import__("cv2")

    params = load_camera_params(D435I_YAML)
    color_intr, depth_intr = params["color_intr"], params["depth_intr"]
    c2d, d2c = params["c2d_extr"], params["d2c_extr"]
    try:
        T_cam2ee = load_cam2ee(CALIB_YAML)
    except CalibrationError as e:
        print(f"[D0 门禁] 标定加载失败, 拒绝继续: {e}")
        return 1

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe.start(cfg)
    depth_sensor = pipe.get_active_profile().get_device().first_depth_sensor()
    scale = depth_sensor.get_depth_scale()
    for _ in range(15):
        pipe.wait_for_frames()

    gt = np.asarray(TEST_POINT["xyz_m"], dtype=float)
    R_cam2ee, t_cam2ee = T_cam2ee[:3, :3], T_cam2ee[:3, 3]
    print(f"[D0] 标定已加载: {CALIB_YAML}")
    print(f"被测点: {TEST_POINT['name']} 基系真值 {TEST_POINT['xyz_m']} m")
    print("流程: 实时画面点选基准点(可连点, u 撤销) → q/Esc 结束 → "
          "输入法兰位姿(mm/deg, 空=退出)")
    print("示例位姿 (D1-A 实测): -184.743 525.382 720.443 -132.511 -2.001 -1.939")

    while True:
        # ---- 实时显示 + 点选 (深度/像素在点击时刻取帧) ----
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        fig.canvas.manager.set_window_title(
            "D1B live - click reference (q = input pose, u = undo)")
        frames = pipe.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        rgb = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()),
                           cv2.COLOR_BGR2RGB)
        img = ax.imshow(rgb)
        clicks = []    # (cx, cy) 像素
        cam3ds = []    # 相机系 (x, y, z), 与 clicks 对应
        dots, = ax.plot([], [], "r+", ms=12, mew=2)
        done = [False]
        t0 = time.monotonic()

        def redraw_markers():
            if clicks:
                xs, ys = zip(*clicks)
                dots.set_data(xs, ys)
            else:
                dots.set_data([], [])
            fig.canvas.draw_idle()

        def on_click(ev):
            if ev.inaxes != ax or ev.xdata is None:
                return
            cx, cy = int(round(ev.xdata)), int(round(ev.ydata))
            res = color_to_depth_pixel(cx, cy, depth_frame, scale,
                                       depth_intr, color_intr, c2d, d2c)
            if res is None:
                print(f"  点 ({cx},{cy}) -> 深度映射失败 (无有效深度)")
                return
            dx, dy = res
            depth = depth_median_around(dx, dy, depth_frame)
            if depth is None:
                print(f"  点 ({cx},{cy}) -> 深度无效 (无有效点)")
                return
            z = float(depth)
            x = (cx - color_intr.ppx) * z / color_intr.fx
            y = (cy - color_intr.ppy) * z / color_intr.fy
            clicks.append((cx, cy))
            cam3ds.append((x, y, z))
            print(f"  记录点 ({cx},{cy}) -> 相机系 [{x:.4f}, {y:.4f}, {z:.4f}] m")
            redraw_markers()

        def on_key(ev):
            if ev.key in ("q", "escape", "enter"):
                done[0] = True
            elif ev.key == "u" and clicks:
                clicks.pop()
                cam3ds.pop()
                print(f"  撤销, 剩余 {len(clicks)} 点")
                redraw_markers()

        def on_close(ev):
            done[0] = True

        fig.canvas.mpl_connect("button_press_event", on_click)
        fig.canvas.mpl_connect("key_press_event", on_key)
        fig.canvas.mpl_connect("close_event", on_close)
        while not done[0] and time.monotonic() - t0 < POINT_TIMEOUT_S:
            frames = pipe.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            rgb = cv2.cvtColor(np.asanyarray(frames.get_color_frame().get_data()),
                               cv2.COLOR_BGR2RGB)
            img.set_data(rgb)
            fig.canvas.draw_idle()
            plt.pause(0.02)
        plt.close(fig)

        if not clicks:
            print("本轮无有效点击, 跳过")
            continue

        # ---- 选点后输入位姿 (机器人停稳, 位姿对应点击时刻画面) ----
        text = input(f"\n已选点 {len(clicks)} 个. 示教器法兰位姿 "
                     f"(xyz mm + ABC deg, 6 数, 空=退出): ").strip()
        if not text:
            print("退出")
            break
        parsed = _parse_pose_input(text)
        if parsed is None:
            print("[作废] 位姿输入格式错误 (需 6 个数: x y z A B C, mm/deg)")
            continue
        xyz_m, abc = parsed

        for (cx, cy), cam3d in zip(clicks, cam3ds):
            pos_ee = R_cam2ee @ np.array(cam3d) + t_cam2ee
            p_base = transform_pose_to_base(pos_ee, xyz_m, abc)
            err = float(np.linalg.norm(p_base - gt))
            rn = float(np.hypot(cx - color_intr.ppx, cy - color_intr.ppy)
                       / np.hypot(320.0, 240.0))
            print(f"  点 ({cx},{cy}) -> 相机系 [{cam3d[0]:.4f}, {cam3d[1]:.4f}, "
                  f"{cam3d[2]:.4f}] m -> 基系预测 [{p_base[0]:.4f}, "
                  f"{p_base[1]:.4f}, {p_base[2]:.4f}] m "
                  f"误差 {err*1000:.1f}mm 径向 {rn:.2f}")

    pipe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
