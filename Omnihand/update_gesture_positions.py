# -*- coding: utf-8 -*-
"""
OmniHand 手势位置更新工具
========================
读取 actlist.json (上位机导出的编码器位置), 
更新 omnihand_gesture.py 中 DIRECT_POSITIONS 字典。

用法:
    python update_gesture_positions.py

JSON 格式 (actlist.json):
    [
        {
            "name": "GRIP",
            "pos_axis_1": 4095,
            "pos_axis_2": 4015,
            ...
            "pos_axis_10": 1622,
            "pos_axis_11": 0,    # 忽略 (手只有10轴)
            "pos_axis_12": 0     # 忽略
        }
    ]

关节映射:
    axis_1  -> J0: Thumb Roll
    axis_2  -> J1: Thumb Ab/Ad
    axis_3  -> J2: Thumb MCP
    axis_4  -> J3: Index Ab/Ad
    axis_5  -> J4: Index PIP
    axis_6  -> J5: Middle PIP
    axis_7  -> J6: Ring Ab/Ad
    axis_8  -> J7: Ring PIP
    axis_9  -> J8: Pinky Ab/Ad
    axis_10 -> J9: Pinky PIP
"""

import json
import re
from pathlib import Path


# 关节注释 (用于生成代码中的注释行)
JOINT_COMMENTS = [
    "Thumb Roll",
    "Thumb Ab/Ad",
    "Thumb MCP",
    "Index Ab/Ad",
    "Index PIP",
    "Middle PIP",
    "Ring Ab/Ad",
    "Ring PIP",
    "Pinky Ab/Ad",
    "Pinky PIP",
]


def read_actlist(json_path: Path) -> dict:
    """读取 actlist.json, 返回 {name: [pos_0..pos_9]}"""
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    entries = {}
    for item in data:
        name = item.get("name", "").strip().upper()
        if not name:
            continue
        # 提取 pos_axis_1 ~ pos_axis_10 (axis_11, axis_12 忽略)
        positions = []
        for i in range(1, 11):
            key = f"pos_axis_{i}"
            if key not in item:
                print(f"  [警告] {name} 缺少 {key}, 跳过")
                break
            positions.append(item[key])
        else:
            entries[name] = positions
            print(f"  读取: {name} -> {positions}")

    return entries


def find_block_boundaries(lines: list, marker: str) -> tuple:
    """
    查找代码块的起止行号。
    marker: 块起始行的特征字符串 (如 'DIRECT_POSITIONS:')
    返回: (start_line, end_line), 找不到返回 (-1, -1)
    """
    start = -1
    for i, line in enumerate(lines):
        if marker in line:
            start = i
            break

    if start == -1:
        return -1, -1

    # 从 start 开始追踪大括号深度
    depth = 0
    end = start
    for i in range(start, len(lines)):
        line = lines[i]
        # 移除字符串内的括号影响
        stripped = re.sub(r'"[^"]*"', '', line)
        stripped = re.sub(r"'[^']*'", '', stripped)
        depth += stripped.count('{') - stripped.count('}')
        if depth == 0 and i > start:
            end = i
            break

    return start, end


def parse_existing_entries(block_lines: list) -> dict:
    """
    解析现有 DIRECT_POSITIONS 块中的条目。
    返回: {GestureName: [pos_0..pos_9]}
    """
    entries = {}
    block_text = "\n".join(block_lines)

    # 匹配 Gesture.XXX.value: [ 数字列表 ]
    pattern = r'Gesture\.(\w+)\.value:\s*\[([^\]]+(?:\[[^\]]*\][^\]]*)*)\]'
    matches = re.finditer(pattern, block_text, re.DOTALL)

    for m in matches:
        name = m.group(1)
        pos_str = m.group(2)
        # 提取所有整数
        positions = [int(x) for x in re.findall(r'\b\d+\b', pos_str)]
        if len(positions) >= 10:
            entries[name] = positions[:10]

    return entries


def build_new_block(entries: dict, old_comment_lines: list) -> str:
    """
    构建新的 DIRECT_POSITIONS 代码块。
    保留原有注释行，更新/新增手势条目。
    """
    lines = list(old_comment_lines)  # 保留原始注释
    lines.append("    DIRECT_POSITIONS: Dict[int, List[int]] = {")

    # 手势名称 -> 枚举值映射 (用于排序)
    GESTURE_VALUE_MAP = {
        "RESET": 0, "PAPER": 1, "FIST": 2, "OK": 3, "HEART": 4,
        "LIKE": 5, "ILY": 6, "NUM1": 7, "NUM2": 8, "NUM3": 9,
        "NUM4": 10, "NUM6": 11, "NUM8": 12, "CLASPING": 13,
        "MIDDLE": 14, "POINT": 15, "GRIP": 16, "EXIT": 99,
    }
    # 按 Gesture 枚举值排序
    sorted_names = sorted(entries.keys(),
                          key=lambda n: GESTURE_VALUE_MAP.get(n, 99))

    for name in sorted_names:
        positions = entries[name]
        lines.append(f"        Gesture.{name}.value: [")
        for i, pos in enumerate(positions):
            comment = JOINT_COMMENTS[i]
            lines.append(f"            {pos},  # J{i} {comment} (axis_{i+1})")
        lines.append("        ],")

    lines.append("    }")
    return "\n".join(lines)


def update_gesture_file(py_path: Path, new_entries: dict):
    """更新 omnihand_gesture.py 中的 DIRECT_POSITIONS"""
    with open(py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1. 找到 DIRECT_POSITIONS 块
    start, end = find_block_boundaries(lines, 'DIRECT_POSITIONS:')
    if start == -1:
        print("  [错误] 未找到 DIRECT_POSITIONS 定义")
        return False

    print(f"\n  找到 DIRECT_POSITIONS: 行 {start + 1} ~ {end + 1}")

    # 2. 获取块前的注释行 (以 # 开头的行, 在 start 之前)
    comment_start = start
    while comment_start > 0 and lines[comment_start - 1].strip().startswith('#'):
        comment_start -= 1

    old_comment_lines = [l.rstrip('\n') for l in lines[comment_start:start]]

    # 3. 解析现有条目
    old_block_lines = [l.rstrip('\n') for l in lines[start:end + 1]]
    old_entries = parse_existing_entries(old_block_lines)
    print(f"  现有手势: {list(old_entries.keys())}")

    # 4. 合并: 新条目覆盖旧条目
    combined = dict(old_entries)
    combined.update(new_entries)
    print(f"  更新手势: {list(new_entries.keys())}")
    print(f"  合并后:   {list(combined.keys())}")

    # 5. 构建新块
    new_block = build_new_block(combined, old_comment_lines)

    # 6. 替换文件内容
    new_lines = lines[:comment_start] + [new_block + "\n"] + lines[end + 1:]
    with open(py_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print(f"\n  => 已更新 {len(combined)} 个手势的编码器位置")
    return True


def main():
    base_dir = Path(__file__).parent
    json_path = base_dir / "actlist.json"
    py_path = base_dir / "omnihand_gesture.py"

    if not json_path.exists():
        print(f"[错误] 未找到 JSON 文件: {json_path}")
        print("请将上位机导出的编码器位置保存为 actlist.json")
        return

    if not py_path.exists():
        print(f"[错误] 未找到目标文件: {py_path}")
        return

    print("=" * 50)
    print("  OmniHand 手势位置更新工具")
    print("=" * 50)
    print(f"  JSON: {json_path}")
    print(f"  目标: {py_path}")
    print()

    # 读取 JSON
    new_entries = read_actlist(json_path)

    if not new_entries:
        print("\n  [错误] JSON 中未找到有效手势数据")
        return

    # 更新文件
    update_gesture_file(py_path, new_entries)

    print("\n  完成!")


if __name__ == "__main__":
    main()