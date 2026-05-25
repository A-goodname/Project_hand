# -*- coding: utf-8 -*-
"""
RealSense 手眼标定 — 数据采集
================================
eye-in-hand 模式

流程：
  1. 启动 RealSense 相机，显示彩色画面（叠加棋盘格角点检测）
  2. 操作机械臂到不同姿态，使标定板处于视野中
  3. 按 s 键保存当前帧 + 在终端输入 [x,y,z,a,b,c]
  4. 按 q 键退出

终端输入格式: [x,y,z,a,b,c]
  - x,y,z: 位置, 单位 mm（保存时自动转换为 m）
  - a,b,c: X-Y-Z 固定角, 单位度
  - 坐标系: 机器人基座 → 机械臂末端 (base→ee)

文件结构（输出）：
  calib_data/
    images/
      img_000.png
      img_001.png
      ...
    poses.csv      # index, x[m], y[m], z[m], rx[°], ry[°], rz[°]
"""

import os
import sys
import cv2
import numpy as np
import yaml

# ---- 脚本所在目录 (使路径不依赖 CWD) ----
_script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- 将项目根目录加入 sys.path (以便导入根目录下的 pyrealsense2) ----
_root_dir = os.path.dirname(_script_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

import pyrealsense2 as rs

# ── 加载配置 ──────────────────────────────────────────────────
config_path = os.path.join(_script_dir, "calib_config.yaml")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

cb_cfg   = cfg["calibration"]["chessboard"]
pattern_size = tuple(cb_cfg["pattern_size"])   # (cols, rows) 内角点
square_size  = cb_cfg["square_size"]           # 米

data_cfg     = cfg["calibration"]["data"]
save_dir     = os.path.join(_script_dir, data_cfg["save_dir"])
img_dir      = os.path.join(save_dir, "images")
pose_csv     = os.path.join(save_dir, "poses.csv")
img_format   = data_cfg.get("image_format", "png")
required     = data_cfg.get("required_samples", 15)  # 最低采集张数（可多于该值）

os.makedirs(img_dir, exist_ok=True)

# ── 棋盘格检测终止条件 ───────────────────────────────────────
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ── 辅助函数 ──────────────────────────────────────────────────

def draw_chessboard_corners(image, corners):
    # 在图上绘制检测到的棋盘格角点
    return cv2.drawChessboardCorners(image.copy(), pattern_size, corners, True)


def get_next_index(img_dir):
    """
    扫描图片目录，返回下一个可用的 index
    例如目录中有 img_000.png, img_002.png（img_001 被删），返回 3。
    """
    max_idx = -1
    for f in os.listdir(img_dir):
        if f.startswith("img_"):
            try:
                idx = int(os.path.splitext(f)[0].split("_")[1])
                if idx > max_idx:
                    max_idx = idx
            except (ValueError, IndexError):
                pass
    return max_idx + 1


# ── 主函数 ──────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("RealSense 手眼标定 — 数据采集")
    print("eye-in-hand 模式")
    print("=" * 60)
    print(f"棋盘格内角点: {pattern_size}, 方格边长: {square_size} m")
    print(f"最低采集张数: {required}（按 q 随时退出）")
    print(f"图片保存目录: {img_dir}")
    print(f"位姿文件: {pose_csv}")
    print()
    print("操作说明:")
    print("  s — 保存当前帧（检测到棋盘格后）")
    print("  q — 退出")
    print()
    print("位姿格式: [x,y,z,a,b,c]")
    print("  x,y,z 单位为 mm（保存时自动转换为 m）")
    print("  a,b,c 单位为度（X-Y-Z 固定角）")
    print("=" * 60)

    # ── 初始化 RealSense（仅彩色流）──
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # 注意：不使用 depth 流，降低带宽占用

    profile = pipeline.start(config)

    # ── CSV header（若文件不存在） ──
    if not os.path.exists(pose_csv):
        os.makedirs(save_dir, exist_ok=True)
        with open(pose_csv, "w") as f:
            f.write("index,x,y,z,rx,ry,rz\n")

    # ── 统计：扫描已有图片，计算下一个 index ──
    total_saved = sum(1 for f in os.listdir(img_dir) if f.startswith("img_"))
    next_index = get_next_index(img_dir)
    print(f"[INFO] 已有图片: {total_saved} 张, 下一个 index: {next_index}")
    print(f"[INFO] 目标张数:  {required} 张")

    # 已保存的位姿列表，用于检测多样性
    saved_poses = []  # list of (x, y, z, rx, ry, rz)

    # ── 主循环 ──
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            # ── 棋盘格检测 ──
            found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

            # 叠加显示（棋盘格角点叠加）
            if found:
                # 亚像素细化
                corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                display = draw_chessboard_corners(color_img, corners_refined)
            else:
                display = color_img.copy()

            cv2.imshow("Hand-Eye Calibration Data Collection", display)

            key = cv2.waitKey(25) & 0xFF

            if key == ord("q"):
                print("[INFO] 用户退出")
                break

            elif key == ord("s"):
                if not found:
                    print("未检测到棋盘格，无法保存。请调整标定板位置。")
                    continue

                # ── 保存图片，使用安全递增的 index ──
                index = next_index
                fname = f"img_{index:03d}.{img_format}"
                cv2.imwrite(os.path.join(img_dir, fname), color_img)
                print(f"\n图片已保存: {fname}")
                next_index += 1

                # ── 手动输入末端位姿 ──
                print("请在终端输入对应的机械臂末端位姿（基座坐标系）")
                print("  格式: [x,y,z,a,b,c]  位置单位:mm, 角度单位:度")
                while True:
                    inp = input("  pose [x,y,z,a,b,c]: ").strip()
                    if not inp:
                        print("输入为空，请重新输入")
                        continue
                    # 去掉首尾方括号
                    inp_clean = inp.strip("[]")
                    tokens = inp_clean.split(",")
                    if len(tokens) != 6:
                        print(f"需要 6 个数值（逗号分隔），收到 {len(tokens)} 个，请重新输入")
                        continue
                    try:
                        pose_mm_deg = [float(v) for v in tokens]
                    except ValueError:
                        print("数值格式错误，请重新输入")
                        continue
                    break

                # 转换: 位置 mm→m, 角度不变仍为度
                pose = [
                    pose_mm_deg[0] / 1000.0,  # x: mm → m
                    pose_mm_deg[1] / 1000.0,  # y: mm → m
                    pose_mm_deg[2] / 1000.0,  # z: mm → m
                    pose_mm_deg[3],            # a: 度（不变）
                    pose_mm_deg[4],            # b: 度（不变）
                    pose_mm_deg[5],            # c: 度（不变）
                ]

                # ── 写入 CSV ──
                with open(pose_csv, "a") as f:
                    line = f"{index},{pose[0]:.6f},{pose[1]:.6f},{pose[2]:.6f}," \
                           f"{pose[3]:.6f},{pose[4]:.6f},{pose[5]:.6f}\n"
                    f.write(line)
                print(f"位姿已记录到 {pose_csv}")
                print(f"位姿: x={pose[0]:.6f}m y={pose[1]:.6f}m z={pose[2]:.6f}m "
                      f"a={pose[3]:.2f}° b={pose[4]:.2f}° c={pose[5]:.2f}°")

                # ── 检查位姿多样性 ──
                pos = np.array(pose[:3])
                rot = np.array(pose[3:])
                if saved_poses:
                    last_pos = np.array(saved_poses[-1][:3])
                    last_rot = np.array(saved_poses[-1][3:])
                    pos_dist = np.linalg.norm(pos - last_pos)
                    rot_dist = np.linalg.norm(rot - last_rot)
                    if pos_dist < 0.05 and rot_dist < 10.0:
                        print(f"  [WARN] 当前位姿与上一个太近 (位置差={pos_dist:.3f}m, 角度差={rot_dist:.1f}°)")
                        print(f"         建议改变机器人姿态后再采集")
                    # 检查与所有已保存位姿的最小差异
                    min_pos_dist = min(np.linalg.norm(pos - np.array(p[:3])) for p in saved_poses)
                    min_rot_dist = min(np.linalg.norm(rot - np.array(p[3:])) for p in saved_poses)
                    if min_pos_dist < 0.02 and min_rot_dist < 5.0:
                        print(f"  [WARN] 当前位姿与某已存位姿几乎重复 (最小位置差={min_pos_dist:.3f}m, 最小角度差={min_rot_dist:.1f}°)")
                        print(f"         手眼标定需要多样化的位姿，建议大幅改变姿态")
                saved_poses.append(pose)

                print(f"[INFO] 已保存 {next_index} 张（最低 {required}）")
                if next_index >= required:
                    print(f"[INFO] 已达到最低要求，可继续采集或按 q 退出")

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("\n[INFO] 数据采集结束")


if __name__ == "__main__":
    main()