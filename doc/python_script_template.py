# -*- coding: utf-8 -*-
"""
模板项目名称 (中文注释 + 英文标点规范)
======================================
功能描述: 本次任务的具体功能

标点声明: 本文件遵循英文标点规范, 注释中使用的句号为 `.` 而非 `.`,
          逗号为 `,` 而非 `,`, 冒号为 `:` 而非 `:`, 不得出现中文全角标点.

输入:
  - input_file_1   : 说明
  - input_file_2   : 说明

输出:
  - output_file_1  : 说明
  - output_file_2  : 说明

流程:
  1. Step 1 说明
  2. Step 2 说明
  3. Step 3 说明

架构:
  - 辅助函数: 纯函数, 无副作用
  - 功能函数: 封装主要业务逻辑
  - 主函数:   按步骤串联完整流程
"""

import os
import sys
import csv
import json
from datetime import datetime


import cv2
import numpy as np
import yaml


# ---- 加载配置 ----
_config_path = "config.yaml"
with open(_config_path, "r", encoding="utf-8") as f:
    _cfg = yaml.safe_load(f)

# 提取配置项 (使用大写变量或 _cfg 前缀)
SAVE_DIR = _cfg.get("save_dir", "output")
INPUT_DIR = _cfg.get("input_dir", "data")


# =============================================================
# 辅助函数
# =============================================================

def helper_function_1(param1, param2):
    """
    辅助函数 1 的功能描述

    参数:
        param1 : 类型, 说明
        param2 : 类型, 说明

    返回:
        类型, 说明
    """
    result = param1 + param2
    return result


def helper_function_2(param):
    """
    辅助函数 2 的功能描述
    参数:
        param : 类型, 说明

    返回:
        类型, 说明
    """
    processed = param * 2
    return processed


# =============================================================
# 功能模块: 数据加载
# =============================================================

def load_data(file_path):
    """
    从指定路径加载数据

    参数:
        file_path : str, 文件路径

    返回:
        dict, 加载的数据
    """
    print("[INFO] 加载数据: %s" % file_path)
    data = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(",")
                key = int(parts[0])
                values = [float(v) for v in parts[1:]]
                data[key] = values
        print("[INFO] 加载了 %d 条数据" % len(data))
    except Exception as e:
        print("[ERROR] 加载失败: %s" % str(e))
    return data


# =============================================================
# 功能模块: 核心处理
# =============================================================

def process_data(data, config):
    """
    核心数据处理

    参数:
        data   : dict, 输入数据
        config : dict, 配置参数

    返回:
        list, 处理结果列表
    """
    results = []
    for key, values in data.items():
        print("  [INFO] 处理 #%d ..." % key)
        processed = helper_function_1(values[0], values[1])
        results.append({
            "index": key,
            "processed": processed,
        })
    return results


# =============================================================
# 功能模块: 结果保存
# =============================================================

def save_results(save_dir, results, metadata):
    """
    保存结果到 YAML 文件

    参数:
        save_dir : str, 保存目录
        results  : list, 结果列表
        metadata : dict, 附加元数据
    """
    os.makedirs(save_dir, exist_ok=True)

    output = {
        "metadata": metadata,
        "results": results,
    }

    output_path = os.path.join(save_dir, "result.yaml")
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(output, f, default_flow_style=None, allow_unicode=True, sort_keys=False)
    print("[DONE] 结果已保存: %s" % output_path)


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  模板项目名称")
    print("  配置: config_path=%s" % _config_path)
    print("=" * 60)

    # ---- [Step 1] 加载输入数据 ----
    print("[INFO] Step 1: 加载数据...")
    input_file = os.path.join(INPUT_DIR, "input.csv")
    if not os.path.exists(input_file):
        print("[ERROR] 输入文件不存在: %s" % input_file)
        return

    data = load_data(input_file)
    if not data:
        print("[ERROR] 数据为空")
        return

    # ---- [Step 2] 执行核心处理 ----
    print("[INFO] Step 2: 执行数据处理...")
    results = process_data(data, _cfg)
    print("[INFO] 处理完成, 共 %d 条结果" % len(results))

    # ---- [Step 3] 保存结果 ----
    print("[INFO] Step 3: 保存结果...")
    metadata = {
        "num_samples": len(results),
        "timestamp": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_results(SAVE_DIR, results, metadata)

    # ---- 打印摘要 ----
    print()
    print("=" * 60)
    print("  处理完成")
    print("  输入条数: %d" % len(data))
    print("  输出条数: %d" % len(results))
    print("=" * 60)


if __name__ == "__main__":
    main()