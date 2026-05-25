# -*- coding: utf-8 -*-
"""
棋盘格角点可视化 --- 辅助人工检查角点起点一致性
=============================================
读取标定图片，检测棋盘格角点，在每张图片上标注：
  - 所有角点用彩色连线绘制 (drawChessboardCorners)
  - 黄色圆点 (O) 标记第一个角点 (index=0) 作为起始点
  - 红色箭头 (col) 从原点沿角点 0->1 方向（棋盘格短边/列方向）
  - 蓝色箭头 (row) 从原点沿角点 0->[pattern_cols] 方向（棋盘格长边/行方向）
  - 底部信息条

输出到 calib_data/images_annotated/ 目录。

使用方法：
  1. 运行本脚本生成标注图片
  2. 打开 calib_data/images_annotated/ 目录
  3. 人工浏览所有图片，确认黄色圆点 (O) 在所有图片中
     是否位于棋盘格的同一个物理角点上
"""

import os
import sys
import cv2
import numpy as np
import yaml


# ---- 脚本所在目录 (使路径不依赖 CWD) ----
_script_dir = os.path.dirname(os.path.abspath(__file__))

# ---- 将项目根目录加入 sys.path (以便导入根目录下的包, 如 pyrealsense2) ----
_root_dir = os.path.dirname(_script_dir)
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)


def main():
    # 加载配置
    config_path = os.path.join(_script_dir, "calib_config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cb_cfg = cfg["calibration"]["chessboard"]
    pattern_size = tuple(cb_cfg["pattern_size"])  # (cols, rows)
    pattern_cols, pattern_rows = pattern_size

    data_cfg = cfg["calibration"]["data"]
    save_dir = os.path.join(_script_dir, data_cfg["save_dir"])
    img_dir = os.path.join(save_dir, "images")
    out_dir = os.path.join(save_dir, "images_annotated")

    os.makedirs(out_dir, exist_ok=True)

    # 获取图片列表
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith(valid_exts)
    ])

    if not image_files:
        print(f"[ERROR] 在 {img_dir} 中未找到图片文件")
        return

    total_corners = pattern_cols * pattern_rows

    print(f"找到 {len(image_files)} 张图片")
    print(f"棋盘格内角点: {pattern_size} (cols={pattern_cols}, rows={pattern_rows})")
    print(f"总角点数: {total_corners}")
    print(f"输出目录: {out_dir}")
    print()

    # 棋盘格检测终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    issues = []  # 记录有问题的图片

    for fname in image_files:
        filepath = os.path.join(img_dir, fname)
        img = cv2.imread(filepath)
        if img is None:
            print(f"  [SKIP] 无法读取: {fname}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        annotated = img.copy()

        if not found:
            cv2.putText(annotated, "NO CORNERS DETECTED",
                        (50, 240), cv2.FONT_HERSHEY_SIMPLEX,
                        1.0, (0, 0, 255), 2)
            issues.append(f"{fname}: 未检测到棋盘格")
            status = "NO_DETECT"
        else:
            # 亚像素细化
            corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # 1. 绘制所有角点的标准连线
            cv2.drawChessboardCorners(annotated, pattern_size, corners_refined, found)

            # 2. 提取关键角点
            pt_origin = tuple(corners_refined[0, 0].astype(int))          # 原点
            pt_col_dir = tuple(corners_refined[1, 0].astype(int))        # 列方向 (短边)
            pt_row_dir = tuple(corners_refined[pattern_cols, 0].astype(int))  # 行方向 (长边)

            # 3. 黄色大圆 (O) 标记原点
            cv2.circle(annotated, pt_origin, 10, (0, 255, 255), 3)

            # 4. 红色箭头 (col 方向: 0 -> 1, 短边/列方向)
            cv2.arrowedLine(annotated, pt_origin, pt_col_dir,
                            (0, 0, 255), 2, tipLength=0.15)
            cv2.putText(annotated, "col", (pt_col_dir[0] + 5, pt_col_dir[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 5. 蓝色箭头 (row 方向: 0 -> pattern_cols, 长边/行方向)
            cv2.arrowedLine(annotated, pt_origin, pt_row_dir,
                            (255, 0, 0), 2, tipLength=0.15)
            cv2.putText(annotated, "row", (pt_row_dir[0] + 5, pt_row_dir[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            status = "OK"

        # 底部信息条
        info_line = f"{fname} | corners: {status} | pattern: {pattern_size} | grid: {total_corners}"
        cv2.putText(annotated, info_line, (10, annotated.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # 保存
        out_path = os.path.join(out_dir, fname)
        cv2.imwrite(out_path, annotated)
        print(f"  [{'OK' if status == 'OK' else 'WARN'}] {fname}")

    # 汇总
    print()
    print("=" * 60)
    print("检查结果汇总")
    print("=" * 60)
    print(f"  总图片数: {len(image_files)}")
    print(f"  检测成功: {len(image_files) - len(issues)}")
    print(f"  检测失败: {len(issues)}")

    if issues:
        print()
        print("  有问题的图片:")
        for issue in issues:
            print(f"    [WARN] {issue}")

    # 人工检查提示
    print()
    print("人工检查步骤:")
    print("  1. 打开 images_annotated/ 目录")
    print("  2. 浏览所有图片，关注黄色圆点 (O) 和红色 col/蓝色 row 箭头")
    print("  3. 确认黄色圆点 (O) 在所有图片中均位于棋盘格的同一物理角点")
    print("  4. 确认红色箭头 (col) 和蓝色箭头 (row) 方向在各图片中一致")
    print("  5. 如果发现原点位置或箭头方向不一致（例如 180 度翻转），")
    print("     说明角点检测存在起点歧义，需要重新采集数据或手动修正")
    print()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()