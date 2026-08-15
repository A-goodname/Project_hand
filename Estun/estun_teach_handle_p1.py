# -*- coding: utf-8 -*-
"""
P1 把手抓取点示教工具 (Handle 检测 + 人工示教)
=============================================
基于 yolo+sam 把手检测 (HandleCupDetector), 人工示教 P1 抓取点位,
自动计算"检测把手基系位置 → P1 末端基系位置"的**基系示教偏置**.

原理 (基系直接加, 含 TCP 偏移语义, 无 t_handle_effective 补偿链):
    t_offset = P1_teach_base - detect_handle_base
    运行时: P1_base = detect_handle_base + t_offset

流程:
  [Step 1] 加载配置 / 标定矩阵
  [Step 2] 初始化 HandleCupDetector (启动 pipeline)
  [Step 3] 控制台输入机械臂当前位姿 R0 (机器人静止, 数值仅在差分中抵消)
  [Step 4] 稳定检测把手 → detect_handle_base
  [Step 5] 人工示教 P1 (可选 P2) → 输入示教器读数 → 计算偏移
  [Step 6] 重复多次 → 平均 + 方差
  [Step 7] 输出 yaml 片段 / --apply 写入 estun_config.yaml

安全: 只读相机 + 控制台输入, 程序绝不发送任何运动指令.

输入:
  - Orchestrator/config.yaml : 检测器配置
  - Estun/estun_config.yaml   : 标定路径
  - calib_data/calibration_result.yaml : 手眼标定矩阵
输出:
  - logs/teach_handle_p1_*.log : 示教记录
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime

# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入工具 ----
from Estun.utils_shared import transform_pose_to_base


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

DETECT_SKIP_FRAMES = 3     # 检测间隔帧
STABLE_WINDOW = 3          # 稳定窗口
POS_THRESH = 0.005         # 位置稳定阈值 (m) — teach 场景更严
MIN_TEACH_COUNT = 3        # 最少示教次数 (方差验收)


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


def load_serial():
    """从 D435i_info.yaml 读取 Camera A serial."""
    _info_path = os.path.join(_project_root, "RealSense_R", "D435i_info.yaml")
    try:
        with open(_info_path, "r", encoding="utf-8") as f:
            _info = yaml.safe_load(f)
        return str(_info["camera"]["serial"])
    except Exception:
        print("[WARN] 无法读取 serial, 使用默认设备")
        return ""


def parse_pose_input(prompt: str):
    """解析示教器位姿输入 [X,Y,Z,A,B,C] (mm/deg) → (xyz_m np.ndarray, abc_deg list)."""
    raw = input(prompt).strip().strip("[]")
    parts = raw.split(",")
    if len(parts) != 6:
        raise ValueError(f"需要 6 个数值, 收到 {len(parts)} 个")
    nums = [float(p) for p in parts]
    xyz_m = np.array([nums[0] / 1000.0, nums[1] / 1000.0, nums[2] / 1000.0])
    abc = nums[3:6]
    return xyz_m, abc


def main():
    parser = argparse.ArgumentParser(description="P1 把手抓取点示教工具")
    parser.add_argument("--apply", action="store_true",
                        help="示教结束后将 handle_teach 段写入 estun_config.yaml (默认仅打印)")
    args = parser.parse_args()

    print("=" * 60)
    print("  P1 把手抓取点示教工具 (基系偏置)")
    print("  检测把手基系位置 + 人工示教 P1 → t_offset (基系直接加)")
    print("  只读相机, 程序不发任何运动指令")
    print("=" * 60, flush=True)

    # ---- 加载手眼标定 ----
    T_cam2ee = load_cam2ee(CALIB_RESULT_PATH)

    # ---- 初始化 HandleCupDetector ----
    from Orchestrator.handle_cup_detector import HandleCupDetector
    serial = load_serial()
    import torch
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] 初始化 HandleCupDetector (serial={serial or 'default'}, device={dev})...")
    try:
        detector = HandleCupDetector(orch_config=_orch_cfg, serial=serial, device=dev)
    except Exception as e:
        print(f"[FATAL] 检测器初始化失败: {e}")
        return

    # ---- 控制台输入机器人当前位姿 R0 ----
    print("\n请把机械臂停在接近产线工作姿态的位置 (静止不动)")
    print("从示教器读取当前末端位姿")
    try:
        r0_xyz, r0_abc = parse_pose_input("输入机器人当前位姿 [X,Y,Z,A,B,C] (mm/deg): ")
    except (ValueError, EOFError) as e:
        print(f"[ERROR] 位姿解析失败: {e}")
        detector.close()
        return
    print(f"[INFO] R0: xyz=({r0_xyz[0]:.4f},{r0_xyz[1]:.4f},{r0_xyz[2]:.4f}) m "
          f"ABC=({r0_abc[0]:.1f},{r0_abc[1]:.1f},{r0_abc[2]:.1f}) deg")

    # ---- 日志 ----
    log_dir = os.path.join(_project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    ts_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"teach_handle_p1_{ts_now}.log")
    log_lines = []
    log_lines.append(f"R0: xyz=({r0_xyz[0]:.4f},{r0_xyz[1]:.4f},{r0_xyz[2]:.4f}) "
                     f"ABC=({r0_abc[0]:.1f},{r0_abc[1]:.1f},{r0_abc[2]:.1f})\n")
    log_lines.append(f"检测间隔: 每 {DETECT_SKIP_FRAMES} 帧 | 稳定窗口: {STABLE_WINDOW} | "
                     f"位置阈值: {POS_THRESH} m\n")

    import cv2
    cv2.namedWindow("P1 Teach - Handle Detection", cv2.WINDOW_NORMAL)

    teach_results = []   # (detect_base, p1_xyz, t_offset, p1_abc)
    p2_results = []      # (p2_offset) or None
    teach_count = 0
    running = True

    while running:
        # ---- 稳定检测把手 ----
        print("\n[INFO] 正在检测水壶把手... (按 q 退出)", flush=True)
        stable = False
        frame_counter = 0
        buf = []
        detect_base = None

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

            if frame_counter % DETECT_SKIP_FRAMES == 1:
                result = detector.detect(color, depth_frame)
                h = result["handle"]

                if h["pos_cam"] is not None:
                    pos_cam = np.array(h["pos_cam"])
                    buf.append(pos_cam)
                    if len(buf) > STABLE_WINDOW:
                        buf.pop(0)
                    if len(buf) == STABLE_WINDOW:
                        pos_range = np.ptp(np.array(buf), axis=0).max()
                        if pos_range < POS_THRESH:
                            mean_cam = np.array(buf).mean(axis=0)
                            # 检测把手位置 → 末端系 → 基系 (不减 t_handle_effective,
                            # 新方案语义: 偏置已含 TCP 偏移)
                            pos_ee = T_cam2ee[:3, :3] @ mean_cam + T_cam2ee[:3, 3]
                            detect_base = transform_pose_to_base(
                                pos_ee, r0_xyz, r0_abc)
                            stable = True
                            print(f"\n[TEACH] 把手稳定! pos_cam=({mean_cam[0]:.3f},{mean_cam[1]:.3f},"
                                  f"{mean_cam[2]:.3f}) m\n"
                                  f"[TEACH] detect_handle_base=({detect_base[0]:.4f},{detect_base[1]:.4f},"
                                  f"{detect_base[2]:.4f}) m", flush=True)
                else:
                    buf = []

            display = detector.visualize(color)
            cv2.putText(display, f"把手稳定中... ({len(buf)}/{STABLE_WINDOW})",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("P1 Teach - Handle Detection", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] 用户按 q 退出")
                running = False
                break

        if not running or detect_base is None:
            break

        # ---- 人工示教 P1 ----
        teach_count += 1
        print(f"\n{'='*50}")
        print(f"[TEACH] 第 {teach_count} 次示教")
        print(f"请用示教器把机械臂移动到\"好抓\"位姿 P1 (掌心正对把手, 手指穿过孔洞)")
        print(f"读取示教器上的末端位姿读数")
        print(f"{'='*50}")
        try:
            p1_xyz, p1_abc = parse_pose_input("输入 P1 位姿 [X,Y,Z,A,B,C] (mm/deg): ")
        except (ValueError, EOFError) as e:
            print(f"[ERROR] 位姿解析失败: {e}")
            continue

        # ---- 计算基系偏置 ----
        t_offset = p1_xyz - detect_base

        # ---- 可选示教 P2 (接近点, 回车跳过) ----
        p2_offset = None
        p2_in = input("可选: 输入 P2 预抓位姿 [X,Y,Z,A,B,C] (mm/deg), 直接回车跳过: ").strip()
        if p2_in:
            try:
                p2_xyz, p2_abc = parse_pose_input("输入 P2 位姿: ")
                p2_offset = p2_xyz - p1_xyz  # P2 = P1 + p2_offset (基系)
            except (ValueError, EOFError) as e:
                print(f"[ERROR] P2 解析失败, 本次 P2 跳过: {e}")

        print(f"\n[TEACH] 第 {teach_count} 次结果:")
        print(f"[TEACH]   p1_offset = ({t_offset[0]:.4f}, {t_offset[1]:.4f}, {t_offset[2]:.4f}) m")
        print(f"[TEACH]   p1_abc    = ({p1_abc[0]:.1f}, {p1_abc[1]:.1f}, {p1_abc[2]:.1f}) deg")
        if p2_offset is not None:
            print(f"[TEACH]   p2_offset = ({p2_offset[0]:.4f}, {p2_offset[1]:.4f}, {p2_offset[2]:.4f}) m")

        # ---- 写入日志 ----
        log_line = (
            f"[{teach_count}] detect_base=({detect_base[0]:.4f},{detect_base[1]:.4f},{detect_base[2]:.4f}) | "
            f"P1=({p1_xyz[0]:.4f},{p1_xyz[1]:.4f},{p1_xyz[2]:.4f}) "
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
        print(f"\n  p1_offset 平均: ({mean_off[0]:.4f}, {mean_off[1]:.4f}, {mean_off[2]:.4f}) m")
        print(f"  p1_offset 方差: ({std_off[0]:.4f}, {std_off[1]:.4f}, {std_off[2]:.4f}) m "
              f"(验收: <0.005 m)")
        # p1_abc 取最后一次示教值 (姿态通常不变; 若变化大人工取舍)
        p1_abc_final = teach_results[-1][3]
        print(f"  p1_abc (末次): ({p1_abc_final[0]:.1f}, {p1_abc_final[1]:.1f}, "
              f"{p1_abc_final[2]:.1f}) deg")

        p2_offsets = [r for r in p2_results if r is not None]
        p2_mean = np.array(p2_offsets).mean(axis=0) if p2_offsets else None
        if p2_mean is not None:
            print(f"  p2_offset 平均: ({p2_mean[0]:.4f}, {p2_mean[1]:.4f}, {p2_mean[2]:.4f}) m")

        # ---- 输出 yaml 片段 ----
        print(f"\n  [提示] 将以下片段写入 Estun/estun_config.yaml 的 estun: 段:")
        yaml_block = [
            "  handle_teach:                     # 把手抓取示教偏置 (基系直接加)",
            "    enabled: true",
            f"    p1_offset: [{mean_off[0]:.4f}, {mean_off[1]:.4f}, {mean_off[2]:.4f}]  # 基系: P1 = detect_base + p1_offset (m)",
            f"    p1_abc: [{p1_abc_final[0]:.1f}, {p1_abc_final[1]:.1f}, {p1_abc_final[2]:.1f}]  # 示教 ABC (deg)",
        ]
        if p2_mean is not None:
            yaml_block.append(
                f"    p2_offset: [{p2_mean[0]:.4f}, {p2_mean[1]:.4f}, {p2_mean[2]:.4f}]  # P2 = P1 + p2_offset (m)")
        else:
            yaml_block.append("    p2_offset: [0.0, 0.0, 0.0]  # P2 = P1 + p2_offset (m)")
        yaml_block.append("    p2_abc: [0.0, 0.0, 0.0]  # 通常 = p1_abc")
        yaml_block.append("    grasp_offset: [0.0, 0.0, 0.0]  # P1 可选微调")
        print("\n".join(f"    {l}" for l in yaml_block))

        # ---- --apply 写 config ----
        if args.apply:
            try:
                with open(_estun_cfg_path, "r", encoding="utf-8") as f:
                    cfg_all = yaml.safe_load(f)
                cfg_all.setdefault("estun", {})["handle_teach"] = {
                    "enabled": True,
                    "p1_offset": [round(float(x), 4) for x in mean_off],
                    "p1_abc": [round(float(x), 1) for x in p1_abc_final],
                    "p2_offset": [round(float(x), 4) for x in p2_mean] if p2_mean is not None else [0.0, 0.0, 0.0],
                    "p2_abc": [round(float(x), 1) for x in p1_abc_final],
                    "grasp_offset": [0.0, 0.0, 0.0],
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
