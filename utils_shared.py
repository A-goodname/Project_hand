# -*- coding: utf-8 -*-
"""
ESTUN 共享工具库
================
提供 estun_run.py 和 estun_epiceye_run.py 共用的坐标变换函数与灵巧手控制.

架构:
  - 第1部分: 坐标变换工具 (纯 numpy, 无硬件依赖)
  - 第2部分: 灵巧手控制 (使用 try/except 保护 OmniHand 依赖)

使用:
    from utils_shared import (
        euler_to_rotation_matrix,
        transform_pose_to_base,
        compute_gripper_offset_transform,
        compute_grasp_rotation,
        post_correct_angles,
        init_hand, grip_object, release_object, disable_hand,
    )
"""

import time

import numpy as np


# =============================================================
# 第1部分: 坐标变换工具 (纯 numpy, 无硬件依赖)
# =============================================================

def euler_to_rotation_matrix(A, B, C, order="zyx"):
    """
    将欧拉角 (A, B, C) 转换为 3x3 旋转矩阵

    参数:
        A, B, C : float, 欧拉角 (度)
        order   : str, 旋转顺序, 默认 "zyx"
                  R = Rz(C) @ Ry(B) @ Rx(A)

    返回:
        np.ndarray, 3x3 旋转矩阵
    """
    a = np.deg2rad(A)
    b = np.deg2rad(B)
    c = np.deg2rad(C)

    ca, sa = np.cos(a), np.sin(a)
    cb, sb = np.cos(b), np.sin(b)
    cc, sc = np.cos(c), np.sin(c)

    Rx = np.array([[1, 0, 0],
                   [0, ca, -sa],
                   [0, sa,  ca]])
    Ry = np.array([[cb, 0, sb],
                   [0,  1, 0],
                   [-sb, 0, cb]])
    Rz = np.array([[cc, -sc, 0],
                   [sc,  cc, 0],
                   [0,  0,  1]])

    if order == "zyx":
        return Rz @ Ry @ Rx
    elif order == "xyz":
        return Rx @ Ry @ Rz
    elif order == "zxz":
        return Rz @ Rx @ Rz
    elif order == "yzy":
        return Ry @ Rz @ Ry
    else:
        raise ValueError(f"不支持的旋转顺序: {order}")


def transform_pose_to_base(pose_ee, robot_xyz, robot_ABC):
    """
    将末端坐标系下的位置转换到机器人基坐标系

    参数:
        pose_ee   : (float, float, float), 末端坐标系下 3D 位置 (x, y, z), 米
        robot_xyz : (float, float, float), 机器人末端在基坐标系下位置 (x, y, z), 米
        robot_ABC : (float, float, float), 机器人末端欧拉角 (A, B, C), 度

    返回:
        np.ndarray, 基坐标系下 3D 位置 (x, y, z), 米
    """
    t_ee_base = np.array(robot_xyz, dtype=float)
    R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
    pos_ee = np.array(pose_ee, dtype=float)
    return R_ee_base @ pos_ee + t_ee_base


def compute_gripper_offset_transform(t_offset_std, r_offset):
    """
    计算末端法兰 -> 夹爪 TCP 的变换矩阵

    t_offset_std 是 C=0 (右手大拇指向上) 时的标准平移偏移
    根据实际 r_offset.C 将标准偏移旋转到当前方向:
        t_effective = R_z(C) @ t_offset_std
        T_flange2gripper = Trans(t_effective) @ Rot_z(C)
                        = [ R_offset | t_effective ]
                          [ 0        | 1             ]
        p_gripper = R_offset^T @ (p_flange - t_effective)

    参数:
        t_offset_std : np.ndarray, C=0 时标准平移偏移 [x, y, z], 米
        r_offset     : np.ndarray, 旋转偏移 [A, B, C], 度 (A=B=0)

    返回:
        R_fl2gr : np.ndarray, 3x3 旋转矩阵 (法兰 -> 夹爪)
        t_fl2gr : np.ndarray, 3x1 平移向量 (法兰 -> 夹爪, 已旋转到实际方向)
    """
    R = euler_to_rotation_matrix(*r_offset, order="zyx")  # = R_z(C)
    t_effective = R @ np.array(t_offset_std, dtype=float)
    return R, t_effective


def compute_grasp_rotation(R_obj_ee, mode, r_offset, R_ee_base=None):
    """
    计算末端系下的目标抓取旋转矩阵 R_grasp_ee

    坐标约定 (YOLO 6D输出):
        R_obj(:,0) = X轴 (Y x Z)
        R_obj(:,1) = Y轴 (PCA主方向, 瓶身轴线)
        R_obj(:,2) = Z轴 (指向相机)

    模式说明:
        free      : 仅叠加固定夹爪安装偏移 Rz(C),
                    将水瓶Y轴统一为向下以消除PCA符号歧义
        vertical  : 末端XY平面与基坐标系Z轴平行,
                    ee-X || 水瓶Y轴(向下), ee-Y || 水瓶X轴, ee-Z || -水瓶Z轴
        horizontal: 暂未实现

    参数:
        R_obj_ee : 3x3, 物体在末端系下的旋转矩阵
        mode     : str, "vertical" | "horizontal" | "free"
        r_offset : np.ndarray, 旋转偏移 [A, B, C], 度
        R_ee_base : 3x3 or None, 末端在基系下的旋转矩阵, free模式下用于Y轴符号修正

    返回:
        R_grasp_ee : 3x3
    """
    # 第一步: 统一 Y 轴指向为竖直向下 (消除 PCA 符号歧义)
    R = R_obj_ee.copy()
    if R_ee_base is not None:
        y_base = R_ee_base @ R[:, 1]
        if y_base[2] > 0:  # Y轴指向上方, 需要翻转
            R[:, 1] *= -1
            R[:, 0] *= -1

    # 第二步: 按模式计算
    if mode == "free":
        # 仅叠加固定夹爪安装偏移,不做方向修正
        R_offset = euler_to_rotation_matrix(*r_offset, order="zyx")
        return R @ R_offset.T

    if mode == "vertical":
        assert R_ee_base is not None, "vertical 模式需要 R_ee_base"
        # 基系约束: 末端 XY 平面平行于基坐标 Z 轴 (即末端 Z 轴水平)
        # 1. 水瓶在基系下的朝向
        R_obj_base = R_ee_base @ R
        y_obj_base = R_obj_base[:, 1]  # 水瓶Y轴(已统一向下, 瓶身轴线)
        z_obj_base = R_obj_base[:, 2]  # 水瓶Z轴(指向相机)

        # 2. 末端 Z 轴: 水平指向水瓶 (z_obj_base 在基系XY平面的投影)
        ee_z_base = np.array([z_obj_base[0], z_obj_base[1], 0.0])
        ee_z_base = ee_z_base / np.linalg.norm(ee_z_base)

        # 3. 末端 X 轴: 沿瓶身轴线竖直向下
        ee_x_base = y_obj_base

        # 4. 末端 Y 轴: 右手系, 由 Z × X 确定
        ee_y_base = np.cross(ee_z_base, ee_x_base)

        # 5. 重新正交化: X = Y × Z, 确保严格右手系
        ee_x_base = np.cross(ee_y_base, ee_z_base)

        # 6. 基系下目标旋转矩阵
        R_grasp_base = np.column_stack([ee_x_base, ee_y_base, ee_z_base])

        # 7. 反推回末端系
        return R_ee_base.T @ R_grasp_base

    elif mode == "horizontal":
        raise NotImplementedError("horizontal 抓取模式暂未实现")

    else:
        raise ValueError(f"不支持的抓取模式: {mode}")


def post_correct_angles(A, B, C, robot_ABC):
    """
    对基坐标系欧拉角进行可达性修正

    将 A (rx) 和 C (rz) 加减 180 度, 使其偏离当前末端位姿最小

    参数:
        A, B, C  : float, 基坐标系欧拉角 (度)
        robot_ABC : (float, float, float), 机器人末端欧拉角 (度)

    返回:
        A, B, C : float, 修正后的欧拉角
    """
    while A - robot_ABC[0] > 90:
        A -= 180
    while robot_ABC[0] - A > 90:
        A += 180
    while C - robot_ABC[2] > 90:
        C -= 180
    while robot_ABC[2] - C > 90:
        C += 180

    return A, B, C


# =============================================================
# 第2部分: 灵巧手控制 (使用 try/except 保护 OmniHand 依赖)
# =============================================================

_hand = None

try:
    from Omnihand.omnihand_gesture import OmniHandGesture, Gesture
    OMNIHAND_AVAILABLE = True
except ImportError:
    OMNIHAND_AVAILABLE = False


def init_hand(hand_port=None):
    """
    初始化并启用灵巧手

    参数:
        hand_port : str or None, 串口号
    """
    global _hand
    if not OMNIHAND_AVAILABLE:
        print("[HAND] OmniHand 不可用, 将模拟所有灵巧手操作")
        _hand = None
        return
    try:
        _hand = OmniHandGesture(port=hand_port or "")
        _hand.open()
        _hand.enable()
        _hand.do_gesture(Gesture.PAPER, delay=0.3)
        print("[INFO] 灵巧手已初始化并启用")
    except Exception as e:
        print(f"[WARN] 灵巧手初始化失败: {e}")
        print("[WARN] 将模拟灵巧手操作")
        if _hand is not None:
            try:
                _hand.close()
            except Exception:
                pass
        _hand = None


def grip_object():
    """执行抓取手势 (GRIP)"""
    if _hand is None:
        print("[HAND] 模拟抓取")
        time.sleep(0.5)
        return
    try:
        _hand.do_gesture(Gesture.GRIP, delay=1.0)
        print("[HAND] 抓取完成")
    except Exception as e:
        print(f"[WARN] 抓取失败: {e}")


def release_object():
    """执行释放手势 (PAPER)"""
    if _hand is None:
        print("[HAND] 模拟释放")
        time.sleep(0.5)
        return
    try:
        _hand.do_gesture(Gesture.PAPER, delay=1.0)
        print("[HAND] 释放完成")
    except Exception as e:
        print(f"[WARN] 释放失败: {e}")


def disable_hand():
    """关闭灵巧手 (RESET -> disable -> close)"""
    global _hand
    if _hand is not None:
        try:
            _hand.do_gesture(Gesture.RESET, delay=0.3)
            time.sleep(0.5)
            _hand.disable()
            _hand.close()
            print("[INFO] 灵巧手已禁用并关闭串口")
        except Exception as e:
            print(f"[WARN] 灵巧手禁用失败: {e}")
        _hand = None