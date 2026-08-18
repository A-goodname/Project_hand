# -*- coding: utf-8 -*-
"""
Cup P1 抓杯点示教工具 (Cup 检测 + 人工示教)
==========================================
基于 HandleCupDetector 水杯检测, 人工示教 Cup P1 抓取点位,
自动计算"检测水杯基系位置 → Cup P1 末端基系位置"的**基系示教偏置**.

原理 (基系直接加, 与 data_sm 运行时链对齐):
    t_offset = Cup_P1_teach_xyz - detect_cup_base_xyz
    p1_abc   = Cup_P1_teach_ABC   (cup 无动态 euler, 姿态直接用示教值)
    运行时: Cup_P1_base = detect_cup_base + t_offset

detect_cup_base 链 (与 data_sm 完全一致):
    c_pos_ee             = T_cam2ee[:3,:3] @ pos_cam + T_cam2ee[:3,3]
    c_pos_ee_compensated = c_pos_ee - t_cup_effective   (灵巧手有效偏移)
    detect_cup_base      = transform_pose_to_base(c_pos_ee_compensated, R0_xyz, R0_ABC)

流程:
  [Step 1] 加载配置 / 标定矩阵 / 灵巧手偏移
  [Step 2] 初始化 HandleCupDetector (启动 pipeline)
  [Step 3] 控制台输入机械臂当前位姿 R0 (机器人静止, 数值仅在差分中抵消)
  [Step 4] 稳定检测水杯 (位置) → 基系预测 detect_cup_base
  [Step 5] 人工示教 Cup P1 (掌心正对杯身, 手指穿过) → 输入示教器读数 → 计算偏置
  [Step 6] 重复多次 → 平均 + 方差 (位置 <5mm)
  [Step 7] 输出 yaml 片段 / --apply 写入 estun_config.yaml

安全: 只读相机 + 控制台输入, 程序绝不发送任何运动指令.

输入:
  - Orchestrator/config.yaml            : 检测器配置
  - Estun/estun_config.yaml             : 标定路径 / cup_grasp 偏移
  - calib_data/calibration_result.yaml  : 手眼标定矩阵
输出:
  - logs/teach_cup_*.log                : 示教记录
"""

import os
import re
import sys
import time
import argparse
import yaml
import numpy as np
from datetime import datetime

# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入工具 ----
from Estun.utils_shared import (
    transform_pose_to_base,
    compute_gripper_offset_transform,
)
from RealSense_R.realsense_shared import load_camera_serial


# =============================================================
# 配置加载
# =============================================================

_orch_cfg_path = os.path.join(_project_root, "Orchestrator", "config.yaml")
with open(_orch_cfg_path, "r", encoding="utf-8") as f:
    _orch_cfg = yaml.safe_load(f)

_estun_cfg_path = os.path.join(_project_root, "Estun", "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

CALIB_RESULT_PATH = os.path.join(_project_root, _cfg_estun["calibration"]["result_path"])

# ---- 水杯灵巧手偏移 (用于计算 t_cup_effective, 与 data_sm 对齐) ----
CUP_HANDLE_SIDE = _orch_cfg.get("cup_handle_side", "back")
if CUP_HANDLE_SIDE == "left":
    CUP_T_OFFSET = np.array(
        _cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset_left"], dtype=float)
else:
    CUP_T_OFFSET = np.array(
        _cfg_estun["cup_grasp"]["gripper_offset"]["t_cup_offset"], dtype=float)
CUP_R_OFFSET = np.array(
    _cfg_estun["cup_grasp"]["gripper_offset"]["r_cup_offset"], dtype=float)

DETECT_SKIP_FRAMES = 1     # 检测间隔帧 (每帧尝试 — 提高成功帧累积率)
STABLE_WINDOW = 3          # 稳定窗口 (成功帧累积, 允许失败帧穿插)
POS_THRESH = 0.005         # 位置稳定阈值 (m) — teach 场景更严
MIN_TEACH_COUNT = 3        # 最少示教次数 (方差验收)
BUF_MAX_AGE = 5.0          # 相邻成功帧最大间隔 (s) — 超时重置 (防位置漂移)


# =============================================================
# 标定矩阵加载 (lenient 版, 不 import state_machine)
# =============================================================

def load_cam2ee(yaml_path: str) -> np.ndarray:
    """从标定结果加载 相机 -> 末端法兰 变换矩阵 (4x4 齐次矩阵)."""
    I = np.eye(4)
    if not os.path.exists(yaml_path):
        print(f"[WARN] 手眼标定结果未找到: {yaml_path}, 使用单位矩阵 (偏置将耦合标定误差)")
        return I
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    calib = data.get("calibration", {})
    hand_eye = calib.get("handeye", {})
    if not hand_eye:
        print("[WARN] 标定文件中未找到 calibration/handeye 章节, 使用单位矩阵")
        return I
    use_method = hand_eye.get("use_method", "Tsai")
    method_data = hand_eye.get(use_method, {})
    if not method_data:
        print(f"[WARN] 方法 {use_method} 无数据, 使用单位矩阵")
        return I
    T_raw = method_data.get("T_cam_to_gripper", None)
    if T_raw is None or len(T_raw) != 4 or len(T_raw[0]) != 4:
        print("[WARN] T_cam_to_gripper 格式错误, 使用单位矩阵")
        return I
    T = np.array(T_raw, dtype=float)
    if T.shape != (4, 4):
        print(f"[WARN] T_cam_to_gripper 形状 {T.shape}, 使用单位矩阵")
        return I
    t = T[:3, 3]
    print(f"[INFO] 已加载标定矩阵: 平移=({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}) m")
    return T


def parse_pose_input(prompt: str, allow_empty: bool = False):
    """解析示教器位姿输入 [X,Y,Z,A,B,C] (mm/deg) → (xyz_m np.ndarray, abc_deg list).

    allow_empty=True 时空回车返回 (None, None) (用于 R0 复用上次值).
    """
    raw = input(prompt).strip().strip("[]")
    # 剥离 ANSI 转义序列 (终端方向键误按会混入 ^[[D, 导致 float 解析失败丢示教)
    raw = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", raw)
    if not raw:
        if allow_empty:
            return None, None
        raise ValueError("输入为空 (需要 6 个数值)")
    parts = raw.split(",")
    if len(parts) != 6:
        raise ValueError(f"需要 6 个数值, 收到 {len(parts)} 个")
    nums = [float(p) for p in parts]
    xyz_m = np.array([nums[0] / 1000.0, nums[1] / 1000.0, nums[2] / 1000.0])
    abc = nums[3:6]
    return xyz_m, abc


def main():
    parser = argparse.ArgumentParser(description="Cup P1 抓杯点示教工具")
    parser.add_argument("--apply", action="store_true",
                        help="示教结束后将 cup_teach 段写入 estun_config.yaml (默认仅打印)")
    args = parser.parse_args()

    print("=" * 60)
    print("  Cup P1 抓杯点示教工具 (基系偏置)")
    print("  检测水杯基系位置 + 人工示教 Cup P1 → t_offset (基系直接加)")
    print("  cup 无动态 euler → p1_abc 直接用示教值 (固定抓取姿态)")
    print("  只读相机, 程序不发任何运动指令")
    print("=" * 60, flush=True)

    # ---- 加载手眼标定 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)

    # ---- 预计算水杯灵巧手有效偏移 (与 data_sm 一致) ----
    _, t_cup_effective = compute_gripper_offset_transform(CUP_T_OFFSET, CUP_R_OFFSET)
    print(f"[INFO] t_cup_effective = ({t_cup_effective[0]:.4f}, {t_cup_effective[1]:.4f}, "
          f"{t_cup_effective[2]:.4f}) m (mode={CUP_HANDLE_SIDE})")

    # ---- 初始化 HandleCupDetector (仅用 Cup 部分) ----
    from Orchestrator.handle_cup_detector import HandleCupDetector
    serial = load_camera_serial(
        os.path.join(_project_root, "RealSense_R/D435i_info.yaml"))
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 初始化 HandleCupDetector (serial={serial or 'default'}, device={dev})...")
    try:
        detector = HandleCupDetector(orch_config=_orch_cfg, serial=serial, device=dev)
    except Exception as e:
        print(f"[FATAL] 检测器初始化失败: {e}")
        return

    # ---- 日志 (R0 在每次稳定检测后输入, 记录见循环内) ----
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"teach_cup_{ts_now}.log")
    log_lines = []
    log_lines.append(f"检测间隔: 每 {DETECT_SKIP_FRAMES} 帧 | 稳定窗口: {STABLE_WINDOW} | "
                     f"位置阈值: {POS_THRESH} m\n")
    log_lines.append(f"模式: {CUP_HANDLE_SIDE}\n")

    import cv2
    cv2.namedWindow("Cup Teach - Cup Detection", cv2.WINDOW_NORMAL)

    teach_results = []   # [(detect_base, p1_xyz, t_offset, p1_abc)]
    p2_results = []      # [p2_offset] or None
    teach_count = 0
    running = True
    r0_xyz = None        # 机器人当前位姿 (首次示教输入, 之后回车复用)
    r0_abc = None

    while running:
        # ---- 稳定检测水杯 ----
        print("\n[INFO] 正在检测水杯... (按 q 退出)", flush=True)
        stable = False
        frame_counter = 0
        buf = []              # [(pos_cam, timestamp)]
        mean_cam = None       # 稳定窗口均值 (相机系)
        stable_pos_range = 0.0

        while not stable:
            color, depth_img, depth_cmap, depth_frame = detector.get_frames()
            if color is None or depth_frame is None:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    print("[INFO] 用户按 q 退出")
                    running = False
                    break
                continue

            display = color.copy()
            frame_counter += 1

            if frame_counter % DETECT_SKIP_FRAMES == 0:
                result = detector.detect(color, depth_frame)
                c = result["cup"]

                if c["pos_cam"] is not None:
                    pos_cam = np.array(c["pos_cam"])
                    now = time.time()
                    # 非连续帧适配: 示教场景机械臂静止+水杯不动, 稳定窗口
                    # 只需成功帧累积 — 检测偶发失败帧不再清空缓冲; 但相邻
                    # 成功帧间隔过长 (位置可能已变) 时重置
                    if buf and now - buf[-1][1] > BUF_MAX_AGE:
                        print(f"[INFO] 相邻成功帧间隔 {now - buf[-1][1]:.1f}s > "
                              f"{BUF_MAX_AGE}s, 重置稳定窗口 (请确认水杯位置未变)",
                              flush=True)
                        buf = []
                    buf.append((pos_cam, now))
                    if len(buf) > STABLE_WINDOW:
                        buf.pop(0)
                    if len(buf) == STABLE_WINDOW:
                        pos_arr = np.array([b[0] for b in buf])
                        pos_range = np.ptp(pos_arr, axis=0).max()
                        if pos_range < POS_THRESH:
                            stable_pos_range = pos_range
                            mean_cam = pos_arr.mean(axis=0)
                            stable = True
                            print(f"\n[TEACH] Cup 稳定! pos_cam=({mean_cam[0]:.3f},"
                                  f"{mean_cam[1]:.3f},{mean_cam[2]:.3f}) m "
                                  f"(极差 {stable_pos_range*1000:.1f}mm)", flush=True)

            display = detector.visualize(color)
            cv2.putText(display, f"Cup stabilizing... ({len(buf)}/{STABLE_WINDOW})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if c["pos_cam"] is None:
                cv2.putText(display, "no cup detected",
                            (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.imshow("Cup Teach - Cup Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 用户按 q 退出")
                running = False
                break

        if not running or mean_cam is None:
            break

        # ---- 关闭显示窗口 (示教阶段无需画面, 下次迭代检测时重新打开) ----
        cv2.destroyWindow("Cup Teach - Cup Detection")

        # ---- 输入机器人当前位姿 R0 (稳定检测后; 首次必输, 之后回车复用) ----
        if r0_xyz is not None:
            _last_r0 = (f"({r0_xyz[0]*1000:.1f},{r0_xyz[1]*1000:.1f},{r0_xyz[2]*1000:.1f},"
                        f"{r0_abc[0]:.1f},{r0_abc[1]:.1f},{r0_abc[2]:.1f})")
            _r0_prompt = f"输入机器人当前位姿 [X,Y,Z,A,B,C] (mm/deg), 回车复用 {_last_r0}: "
        else:
            _r0_prompt = "输入机器人当前位姿 [X,Y,Z,A,B,C] (mm/deg): "
        try:
            _xyz, _abc = parse_pose_input(_r0_prompt, allow_empty=True)
        except EOFError:
            print("[INFO] 用户中断, 退出")
            running = False
            break
        except ValueError as e:
            print(f"[ERROR] R0 输入失败: {e}, 重新检测")
            continue
        if _xyz is None:
            if r0_xyz is None:
                print("[ERROR] 首次示教必须输入 R0, 重新检测")
                continue
            print(f"[INFO] 复用 R0: {_last_r0}")
        else:
            r0_xyz, r0_abc = _xyz, _abc
            print(f"[INFO] R0: xyz=({r0_xyz[0]:.4f},{r0_xyz[1]:.4f},{r0_xyz[2]:.4f}) m "
                  f"ABC=({r0_abc[0]:.1f},{r0_abc[1]:.1f},{r0_abc[2]:.1f}) deg")

        # ---- 计算 Cup P1 基系位置 (与 data_sm 完全对齐) ----
        # c_pos_ee = T_cam2ee[:3,:3] @ pos_cam + T_cam2ee[:3,3]
        c_pos_ee = T_cam2ee[:3, :3] @ mean_cam + T_cam2ee[:3, 3]
        # 减去灵巧手有效偏移 (与 data_sm c_pos_ee_compensated = c_pos_ee - t_cup_effective)
        c_pos_ee_compensated = c_pos_ee - t_cup_effective
        # ee → 基系 (位置, 用 R0 ABC)
        detect_base = transform_pose_to_base(c_pos_ee_compensated, r0_xyz, r0_abc)
        print(f"[TEACH] detect_cup_base = ({detect_base[0]:.4f}, {detect_base[1]:.4f}, "
              f"{detect_base[2]:.4f}) m", flush=True)

        # 日志: 本次使用的 R0
        log_lines.append(f"[{teach_count+1}] R0: xyz=({r0_xyz[0]:.4f},{r0_xyz[1]:.4f},"
                         f"{r0_xyz[2]:.4f}) ABC=({r0_abc[0]:.1f},{r0_abc[1]:.1f},"
                         f"{r0_abc[2]:.1f})\n")

        # ---- 人工示教 Cup P1 ----
        teach_count += 1
        print(f"\n{'='*50}")
        print(f"[TEACH] 第 {teach_count} 次示教")
        print(f"请用示教器把机械臂移动到\"好抓\"位姿 Cup P1 (掌心正对杯身, 手指穿过)")
        print(f"读取示教器上的末端位姿读数")
        print(f"{'='*50}")
        try:
            p1_xyz, p1_abc = parse_pose_input("输入 Cup P1 位姿 [X,Y,Z,A,B,C] (mm/deg): ")
        except (ValueError, EOFError) as e:
            print(f"[ERROR] 位姿解析失败: {e}")
            continue

        # ---- 计算基系偏置 (位置) ----
        # 语义: 运行时 Cup_P1_base = detect_cup_base + t_offset
        # p1_abc: cup 无动态 euler, 姿态直接用示教值 (固定抓取姿态)
        t_offset = p1_xyz - detect_base

        # ---- 可选示教 P2 (接近点, 回车跳过) ----
        p2_offset = None
        p2_in = input("可选: 输入 Cup P2 预抓位姿 [X,Y,Z,A,B,C] (mm/deg), 直接回车跳过: ").strip()
        if p2_in:
            try:
                p2_xyz, p2_abc = parse_pose_input("输入 Cup P2 位姿: ")
                p2_offset = p2_xyz - p1_xyz  # P2 = P1 + p2_offset (基系)
            except (ValueError, EOFError) as e:
                print(f"[ERROR] P2 解析失败, 本次 P2 跳过: {e}")

        print(f"\n[TEACH] 第 {teach_count} 次结果:")
        print(f"[TEACH]   detect_cup_base = ({detect_base[0]:.4f}, {detect_base[1]:.4f}, "
              f"{detect_base[2]:.4f}) m")
        print(f"[TEACH]   示教 Cup P1     = ({p1_xyz[0]:.4f}, {p1_xyz[1]:.4f}, {p1_xyz[2]:.4f}) "
              f"ABC=({p1_abc[0]:.1f},{p1_abc[1]:.1f},{p1_abc[2]:.1f}) deg")
        print(f"[TEACH]   t_offset        = ({t_offset[0]:.4f}, {t_offset[1]:.4f}, "
              f"{t_offset[2]:.4f}) m")
        if p2_offset is not None:
            print(f"[TEACH]   p2_offset       = ({p2_offset[0]:.4f}, {p2_offset[1]:.4f}, "
                  f"{p2_offset[2]:.4f}) m")

        # ---- 写入日志 (自包含: 时间戳 + 相机系稳定均值 + 窗口极差 + 基系偏置) ----
        _ts_now = datetime.now().strftime("%H:%M:%S")
        log_line = (
            f"[{teach_count}] {_ts_now} cam_mean=({mean_cam[0]:.4f},{mean_cam[1]:.4f},"
            f"{mean_cam[2]:.4f}) | range pos={stable_pos_range*1000:.1f}mm\n"
            f"[{teach_count}] detect_cup_base=({detect_base[0]:.4f},{detect_base[1]:.4f},"
            f"{detect_base[2]:.4f}) | "
            f"Cup P1=({p1_xyz[0]:.4f},{p1_xyz[1]:.4f},{p1_xyz[2]:.4f}) "
            f"ABC=({p1_abc[0]:.1f},{p1_abc[1]:.1f},{p1_abc[2]:.1f}) | "
            f"t_offset=({t_offset[0]:.4f},{t_offset[1]:.4f},{t_offset[2]:.4f})\n"
        )
        if p2_offset is not None:
            log_line += (f"[{teach_count}] P2 offset=({p2_offset[0]:.4f},{p2_offset[1]:.4f},"
                         f"{p2_offset[2]:.4f}) m\n")
        log_lines.append(log_line)
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)
        print(f"[TEACH] 已保存到日志: {log_path}")

        teach_results.append((detect_base, p1_xyz, t_offset, p1_abc))
        p2_results.append(p2_offset)

        # ---- 询问是否继续 ----
        print()
        again = input("是否继续下一次示教 (换个位置再示教一次)? [y/N]: ").strip().lower()
        if again != "y":
            running = False

    # ---- 汇总 + 平均 ----
    cv2.destroyAllWindows()
    detector.close()
    print(f"\n{'='*60}")
    print(f"  示教结束, 共采集 {teach_count} 次")
    print(f"  日志: {log_path}")

    if teach_count >= 1:
        offsets = np.array([r[2] for r in teach_results])
        mean_off = offsets.mean(axis=0)
        std_off = offsets.std(axis=0)
        # p1_abc 取最后一次示教值 (姿态通常不变; 若变化大人工取舍)
        p1_abc_final = teach_results[-1][3]
        print(f"\n  t_offset 平均: ({mean_off[0]:.4f}, {mean_off[1]:.4f}, {mean_off[2]:.4f}) m")
        print(f"  t_offset 方差: ({std_off[0]:.4f}, {std_off[1]:.4f}, {std_off[2]:.4f}) m "
              f"(验收: <0.005 m)")
        print(f"  p1_abc (末次): ({p1_abc_final[0]:.1f}, {p1_abc_final[1]:.1f}, "
              f"{p1_abc_final[2]:.1f}) deg")

        p2_offsets = [r for r in p2_results if r is not None]
        p2_mean = np.array(p2_offsets).mean(axis=0) if p2_offsets else None
        if p2_mean is not None:
            print(f"  p2_offset 平均: ({p2_mean[0]:.4f}, {p2_mean[1]:.4f}, {p2_mean[2]:.4f}) m")

        # ---- 汇总写入日志 (验收统计自包含, 不依赖终端输出) ----
        log_lines.append(f"== 汇总 ({teach_count} 次) ==\n")
        log_lines.append(
            f"t_offset 平均: ({mean_off[0]:.4f}, {mean_off[1]:.4f}, {mean_off[2]:.4f}) m | "
            f"方差: ({std_off[0]:.4f}, {std_off[1]:.4f}, {std_off[2]:.4f}) m (验收 <0.005)\n")
        log_lines.append(
            f"p1_abc (末次): ({p1_abc_final[0]:.1f}, {p1_abc_final[1]:.1f}, "
            f"{p1_abc_final[2]:.1f}) deg\n")
        if p2_mean is not None:
            log_lines.append(
                f"p2_offset 平均: ({p2_mean[0]:.4f}, {p2_mean[1]:.4f}, {p2_mean[2]:.4f}) m\n")
        with open(log_path, "w", encoding="utf-8") as f:
            f.writelines(log_lines)

        # ---- 输出 yaml 片段 ----
        print(f"\n  [提示] 将以下片段写入 Estun/estun_config.yaml 的 estun: 段:")
        yaml_block = [
            "  cup_teach:                     # 水杯抓取示教偏置 (基系直接加)",
            "    enabled: true",
            f"    p1_offset: [{mean_off[0]:.4f}, {mean_off[1]:.4f}, {mean_off[2]:.4f}]  "
            f"# 基系: Cup P1 = detect_cup_base + p1_offset (m)",
            f"    p1_abc: [{p1_abc_final[0]:.1f}, {p1_abc_final[1]:.1f}, {p1_abc_final[2]:.1f}]  "
            f"# 示教 ABC (deg, 固定抓取姿态)",
        ]
        if p2_mean is not None:
            yaml_block.append(
                f"    p2_offset: [{p2_mean[0]:.4f}, {p2_mean[1]:.4f}, {p2_mean[2]:.4f}]  "
                f"# P2 = Cup P1 + p2_offset (m)")
        else:
            yaml_block.append("    p2_offset: [0.0, 0.0, 0.0]  # P2 = Cup P1 + p2_offset (m)")
        print("\n".join(f"    {l}" for l in yaml_block))

        # ---- --apply 写 config ----
        if args.apply:
            try:
                with open(_estun_cfg_path, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                cfg_all.setdefault("estun", {})["cup_teach"] = {
                    "enabled": True,
                    "p1_offset": [round(float(x), 4) for x in mean_off],
                    "p1_abc": [round(float(x), 1) for x in p1_abc_final],
                    "p2_offset": ([round(float(x), 4) for x in p2_mean]
                                  if p2_mean is not None else [0.0, 0.0, 0.0]),
                }
                with open(_estun_cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg_all, f, allow_unicode=True, sort_keys=False)
                print(f"[INFO] 已写入 {_estun_cfg_path} (注意: yaml 重写会丢失注释)")
            except Exception as e:
                print(f"[ERROR] 写入配置失败: {e}")
    print(f"{'='*60}")
    print("[INFO] 示教工具退出")


if __name__ == "__main__":
    main()
