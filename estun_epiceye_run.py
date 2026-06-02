# -*- coding: utf-8 -*-
"""
ESTUN EpicEye 6D 抓取系统
=========================
整合 EpicEye 相机 YOLO 灰度 6D 检测,手眼矩阵, TCP 服务器与灵巧手,实现
单次触发水瓶检测,位姿转换与机器人抓取控制.

与 estun_run.py (RealSense 实时版) 的区别:
  - 帧获取: EpicEye trigger_frame (单次触发), 而非实时流
  - 位姿稳定: 单帧直接使用, 无 PoseStabilizer
  - 检测模型: 灰度 YOLO (yolo11n-grayscale), 而非彩色 YOLO
  - 手眼矩阵: 从 epiceye_config.yaml 手动输入, 而非标定文件
  - 显示: 单帧灰度图 + 6D 坐标轴, 而非实时彩色+深度拼接
  - 检测失败: 窗口提示 + 空格重拍, 而非跳过帧

使用说明:
  1. 连接 EpicEye 相机
  2. 连接灵巧手 (USB 串口)
  3. 确认 EpicEye/epiceye_config.yaml 中 handeye_matrix 已正确填入
  4. 运行 python estun_epiceye_run.py
  5. 机器人作为 TCP 客户端连接本服务器

输入:
  - estun_config.yaml               : TCP 服务器 / 灵巧手 / 夹爪偏移 / 抓取模式
  - EpicEye/epiceye_config.yaml     : 相机 IP / 灰度模型 / 深度 / 手眼矩阵

输出:
  - 目标位姿 [x,y,z,A,B,C] 发送至机器人 (TCP)
  - 显示窗口: 灰度图 + 检测框 + 6D 坐标轴 + 位姿文字

流程:
  主线程:
    [Init] 加载配置 / 手眼矩阵 / 相机 / YOLO / 灵巧手 / TCP -> 主循环
    [Step1] 需要时触发 EpicEye 拍照 (客户端连接后 / 空格重拍)
    [Step2] YOLO 灰度检测 -> PCA 6D 位姿 -> 更新 last_known_pose
    [Step3] 显示窗口 (灰度图 + 检测结果)
    [Step4] 非阻塞 accept -> 子线程 handle_client
    [Step5] 键盘: q=退出, 空格=重拍

  子线程 handle_client (daemon=True):
    [Phase A] 接收机器人初始位姿 (5s 超时)
    [Phase B] 轮询 last_known_pose -> 手眼矩阵 -> 夹爪补偿 -> 基坐标 -> 发送
    [Phase C] CRP 指令循环: GRIP / RELEASE

架构:
  - 辅助函数: 从 estun_run.py 复制的纯函数, 无外部依赖
  - 主循环: 有限状态机 (IDLE / CAPTURE / SUCCESS / FAILED)
  - 检测器: 直接使用 EpicEye/test_6D.py 的 EpicEyeYOLO6DPoseDetector
  - 窗口显示: cv2 单帧显示, 检测失败时提示空格重拍
  - 灵巧手控制: 同 estun_run (OmniHand / 模拟)
"""

import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import yaml
import torch
from ultralytics import YOLO


# ---- 将项目根目录加入 sys.path ----
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ---- 导入自定义包 ----
from EpicEye.epiceye_camera import EpicEyeCamera
try:
    from Omnihand.omnihand_gesture import OmniHandGesture, Gesture
    OMNIHAND_AVAILABLE = True
except ImportError:
    OMNIHAND_AVAILABLE = False
    print("[WARN] OmniHand 模块导入失败,将模拟灵巧手操作")


# ---- 全局退出事件 ----
stop_event = threading.Event()


# =============================================================
# 配置加载
# =============================================================

# ---- 加载 ESTUN 配置 ----
_estun_cfg_path = os.path.join(_project_root, "estun_config.yaml")
if not os.path.exists(_estun_cfg_path):
    print(f"[FATAL] ESTUN 配置文件未找到: {_estun_cfg_path}")
    sys.exit(1)
with open(_estun_cfg_path, "r", encoding="utf-8") as f:
    _cfg_estun = yaml.safe_load(f)["estun"]

HOST = _cfg_estun["tcp_server"]["host"]
PORT = _cfg_estun["tcp_server"]["port"]
HAND_PORT = _cfg_estun["hand"]["port"]
GRIPPER_T_OFFSET = np.array(_cfg_estun["gripper_offset"]["t_offset"], dtype=float)
GRIPPER_R_OFFSET = np.array(_cfg_estun["gripper_offset"]["r_offset"], dtype=float)
ORIENTATION_MODE = _cfg_estun["orientation_mode"]

# ---- 防御性检查:确保 orientation_mode 合法 ----
VALID_MODES = {"vertical", "horizontal", "free"}
if ORIENTATION_MODE not in VALID_MODES:
    print(f"[WARN] 未知的 orientation_mode: {ORIENTATION_MODE},将使用 free 模式")
    ORIENTATION_MODE = "free"
elif ORIENTATION_MODE == "horizontal":
    print("[WARN] horizontal 模式暂未实现,将使用 free 模式")
    ORIENTATION_MODE = "free"

# ---- 加载 EpicEye 配置 ----
_epiceye_cfg_path = os.path.join(_project_root, "EpicEye", "epiceye_config.yaml")
if not os.path.exists(_epiceye_cfg_path):
    print(f"[FATAL] EpicEye 配置文件未找到: {_epiceye_cfg_path}")
    sys.exit(1)
with open(_epiceye_cfg_path, "r", encoding="utf-8") as f:
    _cfg_eye = yaml.safe_load(f)

CAMERA_IP = _cfg_eye.get("camera", {}).get("ip", "")
AUTO_SEARCH = _cfg_eye.get("camera", {}).get("auto_search", True)
GRAY_MODEL_NAME = _cfg_eye.get("model", {}).get("gray_path", "../model/yolo11n-grayscale")
CLASS_ID = _cfg_eye.get("model", {}).get("class_id", 39)
CONF_THRESHOLD = _cfg_eye.get("model", {}).get("conf_threshold", 0.5)
DEVICE_CFG = _cfg_eye.get("model", {}).get("device", "gpu")
DEPTH_MIN = _cfg_eye.get("depth", {}).get("min", 0.15)
DEPTH_MAX = _cfg_eye.get("depth", {}).get("max", 1.5)
DEPTH_Z_THRESHOLD = _cfg_eye.get("depth", {}).get("z_threshold", 0.05)
HANDEYE_MATRIX_RAW = _cfg_eye.get("handeye_matrix", None)

# ---- 设备选择 ----
if DEVICE_CFG == "gpu" and torch.cuda.is_available():
    DEVICE = "cuda:0"
else:
    DEVICE = "cpu"
print(f"[INFO] Running on device: {DEVICE} (config: {DEVICE_CFG})")


# =============================================================
# 辅助函数 (从 estun_run.py 复制, 纯函数无副作用)
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


def load_epiceye_cam2ee():
    """
    从 EpicEye/epiceye_config.yaml 加载手眼矩阵 (相机 -> 末端法兰)

    返回:
        np.ndarray, 4x4 齐次矩阵; 格式错误时返回单位矩阵
    """
    I = np.eye(4)
    if HANDEYE_MATRIX_RAW is None:
        print("[WARN] epiceye_config.yaml 中未找到 handeye_matrix, 使用单位矩阵")
        return I

    T = np.array(HANDEYE_MATRIX_RAW, dtype=float)
    if T.shape != (4, 4):
        print(f"[WARN] handeye_matrix 形状 {T.shape}, 应为 4x4, 使用单位矩阵")
        return I

    t = T[:3, 3]
    t_str = f"({t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f})"
    print(f"[INFO] 已加载手眼矩阵 (cam->ee), 平移: {t_str} m")
    print("[INFO] 注意: 还需叠加 gripper_offset 变换才得到最终夹爪 TCP 位姿")
    return T


# =============================================================
# 功能模块: 灵巧手控制 (OmniHand)
# =============================================================

_hand = None


def init_hand(hand_port=None):
    """
    初始化并启用灵巧手

    参数:
        hand_port : str or None, 串口号; 默认使用 HAND_PORT 配置
    """
    global _hand
    port = hand_port or HAND_PORT
    if not OMNIHAND_AVAILABLE:
        print("[HAND] OmniHand 不可用, 将模拟所有灵巧手操作")
        _hand = None
        return
    try:
        _hand = OmniHandGesture(port=port)
        _hand.open()
        _hand.enable()
        _hand.do_gesture(Gesture.PAPER, delay=0.3)  # 确保初始手势
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


# =============================================================
# 功能模块: EpicEye 单次检测封装
# =============================================================

class EpicEyeSingleShotDetector:
    """
    EpicEye 单次检测封装

    整合 YOLO 灰度检测 + PCA 6D 位姿估计, 提供统一接口
    """

    def __init__(self):
        """初始化 YOLO 灰度模型"""
        # ---- 加载 YOLO 模型 (固定灰度模型) ----
        model_path = f"{GRAY_MODEL_NAME}.pt"
        if not os.path.exists(model_path):
            model_path = GRAY_MODEL_NAME
        self.model = YOLO(model_path).to(DEVICE)
        print(f"[INFO] YOLO 灰度模型已加载: {model_path} | 类别: {CLASS_ID} | 阈值: {CONF_THRESHOLD}")

        # ---- 相机内参 (online 模式从 SDK 获取, 稍后设置) ----
        self.camera_matrix = None
        self.fx = self.fy = self.cx = self.cy = None

    def set_camera_matrix(self, K):
        """从相机 SDK 获取后设置内参矩阵"""
        self.camera_matrix = K.reshape(3, 3)
        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

    def detect_targets(self, gray_image):
        """
        YOLO 检测水瓶目标 (灰度输入)

        参数:
            gray_image : numpy.ndarray, 灰度图像 (H x W)

        返回:
            list, 每个元素为 [x1, y1, x2, y2, conf, class_id]
        """
        results = self.model(gray_image, conf=CONF_THRESHOLD, classes=[CLASS_ID])

        detections = []
        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, class_ids):
                if cls_id == CLASS_ID:
                    detections.append([*box, conf, cls_id])

        return detections

    def compute_6d_pose(self, bbox, point_map):
        """
        计算水瓶的 6D 位姿 (PCA 方法)

        point_map 为 H×W×3 的点云图, 直接 bbox 区域采样

        参数:
            bbox      : [x1, y1, x2, y2, conf, cls_id]
            point_map : (H×W×3) 点云数据

        返回:
            position : (x, y, z) 重心坐标 (m)
            rotation : 3x3 旋转矩阵
            points   : N×3 有效点云
        """
        if point_map is None or self.camera_matrix is None:
            return None, None, None

        H, W = point_map.shape[:2]
        x1 = max(0, min(int(bbox[0]), W - 1))
        y1 = max(0, min(int(bbox[1]), H - 1))
        x2 = max(0, min(int(bbox[2]), W - 1))
        y2 = max(0, min(int(bbox[3]), H - 1))

        # 锚点深度
        cx_bbox = int((x1 + x2) / 2)
        cy_bbox = int((y1 + y2) / 2)
        half = 3
        patch = point_map[
            max(0, cy_bbox - half):min(H, cy_bbox + half + 1),
            max(0, cx_bbox - half):min(W, cx_bbox + half + 1),
            :
        ]
        z_vals = patch[:, :, 2].flatten()
        valid_z = z_vals[(z_vals > DEPTH_MIN) & (z_vals < DEPTH_MAX) & np.isfinite(z_vals)]
        if len(valid_z) < 3:
            return None, None, None
        center_depth = float(np.median(valid_z))

        # 采样 bbox 区域
        region = point_map[y1:y2 + 1:2, x1:x2 + 1:2, :]
        points_flat = region.reshape(-1, 3)
        valid_mask = (
            np.isfinite(points_flat[:, 2]) &
            (points_flat[:, 2] > DEPTH_MIN) &
            (points_flat[:, 2] < DEPTH_MAX) &
            (np.abs(points_flat[:, 2] - center_depth) < DEPTH_Z_THRESHOLD)
        )
        points = points_flat[valid_mask]

        if len(points) < 10:
            valid_mask_wide = (
                np.isfinite(points_flat[:, 2]) &
                (points_flat[:, 2] > DEPTH_MIN) &
                (points_flat[:, 2] < DEPTH_MAX)
            )
            points = points_flat[valid_mask_wide]
            if len(points) < 10:
                return None, None, None
            print(f"[INFO] anchor depth: {center_depth:.3f}m, no-filter points: {len(points)}")
        else:
            print(f"[INFO] anchor depth: {center_depth:.3f}m, filtered points: {len(points)}")

        # ---- PCA 位姿 ----
        center = np.mean(points, axis=0)

        cov = np.cov(points, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主方向 (最大特征值)-> Y轴 (水瓶纵向)
        target_y_main = np.real(eig_vec[:, 0])
        # 最小方差方向 -> Z轴 (指向相机)
        target_z = np.real(eig_vec[:, 2])

        if target_z[2] > 0:
            target_z = -target_z

        # X轴 = Y x Z,确保右手系
        target_x = np.cross(target_y_main, target_z)
        target_x = target_x / np.linalg.norm(target_x)
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)

        return center, rot, points

    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """
        从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw)

        返回:
            tuple: (A, B, C) 角度值 (deg)
        """
        if abs(abs(R[2, 0]) - 1.0) < 1e-6:
            if R[2, 0] < 0:
                pitch = np.pi / 2
                roll = 0.0
                yaw = np.arctan2(R[0, 1], R[0, 2])
            else:
                pitch = -np.pi / 2
                roll = 0.0
                yaw = np.arctan2(-R[0, 1], -R[0, 2])
        else:
            pitch = -np.arcsin(R[2, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])
            yaw = np.arctan2(R[1, 0], R[0, 0])

        return np.degrees(roll), np.degrees(pitch), np.degrees(yaw)

    def visualize_pose(self, image, center_3d, rot, K, scale=0.08):
        """
        在图像上绘制 6D 位姿坐标轴

        红=X轴, 绿=Y轴 (主方向,水瓶纵向), 蓝=Z轴
        """
        center_2d = K @ np.array(center_3d)
        center_2d = center_2d / center_2d[2]
        cx_2d = int(center_2d[0])
        cy_2d = int(center_2d[1])

        axes = [
            (rot[:, 0], (0, 0, 255)),   # X -> 红色
            (rot[:, 1], (0, 255, 0)),   # Y -> 绿色
            (rot[:, 2], (255, 0, 0)),   # Z -> 蓝色
        ]

        cv2.circle(image, (cx_2d, cy_2d), 5, (124, 0, 255), 3)

        for axis, color in axes:
            end_3d = np.array(center_3d) + scale * axis
            end_2d = K @ end_3d
            end_2d = end_2d / end_2d[2]
            ex_2d = int(end_2d[0])
            ey_2d = int(end_2d[1])
            cv2.arrowedLine(image, (cx_2d, cy_2d), (ex_2d, ey_2d), color, 3, tipLength=0.2)


# =============================================================
# 功能模块: TCP 客户端处理
# =============================================================

# ---- 全局数据: 主循环和子线程共享 ----
_last_known_pose = None       # (pos_cam, rot_cam) or None
_pose_lock = threading.Lock()

def get_last_known_pose():
    """线程安全地读取 last_known_pose"""
    global _last_known_pose
    with _pose_lock:
        return _last_known_pose

def set_last_known_pose(pose):
    """线程安全地写入 last_known_pose"""
    global _last_known_pose
    with _pose_lock:
        _last_known_pose = pose


def handle_client(conn, addr, T_cam2ee, gripper_t_offset, gripper_r_offset, orientation_mode):
    """
    TCP 客户端处理

    参数:
        conn              : socket, 客户端连接
        addr              : (str, int), 客户端地址
        T_cam2ee          : np.ndarray, 4x4 相机 -> 末端法兰变换矩阵
        gripper_t_offset  : np.ndarray, 夹爪平移偏移
        gripper_r_offset  : np.ndarray, 夹爪旋转偏移
        orientation_mode  : str, 抓取模式
    """
    print(f"[INFO] 客户端已连接: {addr}", flush=True)

    # ---- [Phase A] 接收机器人初始末端位姿 (5s 超时) ----
    robot_xyz = None
    robot_ABC = None
    try:
        conn.settimeout(5.0)
        init_data = conn.recv(1024)
        if not init_data:
            print("[WARN] 机器人未发送初始位姿, 断开连接", flush=True)
            return
        pose_str = init_data.decode("utf-8").strip().strip("[]")
        parts = pose_str.split(",")
        if len(parts) == 6:
            robot_xyz = (float(parts[0]), float(parts[1]), float(parts[2]))
            robot_ABC = (float(parts[3]), float(parts[4]), float(parts[5]))
            print(f"[INFO] 机器人当前末端位姿: xyz=({robot_xyz[0]:.3f},{robot_xyz[1]:.3f},{robot_xyz[2]:.3f}) "
                  f"ABC=({robot_ABC[0]:.2f},{robot_ABC[1]:.2f},{robot_ABC[2]:.2f})", flush=True)
        else:
            print(f"[WARN] 初始位姿格式错误: {init_data.decode('utf-8').strip()}", flush=True)
            return
    except socket.timeout:
        print("[WARN] 等待机器人初始位姿超时, 断开连接", flush=True)
        return
    except Exception as e:
        print(f"[WARN] 接收初始位姿异常: {e}", flush=True)
        return

    # ---- [Phase B] 轮询 last_known_pose -> 基坐标转换 -> 补偿偏移 (30s 超时) ----
    # 预计算夹爪偏移变换 (常量, 在循环外执行一次)
    _, t_fl2gr = compute_gripper_offset_transform(gripper_t_offset, gripper_r_offset)
    print("[INFO] 等待检测位姿...", flush=True)
    target_sent = False
    wait_start = time.time()
    while not target_sent and not stop_event.is_set() and (time.time() - wait_start) < 30.0:
        last_pose = get_last_known_pose()
        if last_pose is not None:
            pos_cam, rot_cam = last_pose

            # 手眼矩阵: cam -> ee
            pos_ee = T_cam2ee[:3, :3] @ np.array(pos_cam) + T_cam2ee[:3, 3]
            rot_ee = T_cam2ee[:3, :3] @ rot_cam
            A_ee, B_ee, C_ee = EpicEyeSingleShotDetector.rotation_matrix_to_euler_angles(rot_ee)

            # 位置补偿: t_grasp_ee = t_obj_ee - t_tool'_ee
            pos_ee_compensated = np.array(pos_ee) - t_fl2gr

            # 方向对齐: 根据物体朝向和抓取模式计算 R_grasp_ee
            R_obj_ee = euler_to_rotation_matrix(A_ee, B_ee, C_ee, order="zyx")
            R_ee_base = euler_to_rotation_matrix(*robot_ABC, order="zyx")
            R_grasp_ee = compute_grasp_rotation(R_obj_ee, orientation_mode, gripper_r_offset, R_ee_base)

            # 转换到基坐标系
            rot_base = R_ee_base @ R_grasp_ee
            pos_base = transform_pose_to_base(pos_ee_compensated, robot_xyz, robot_ABC)
            A_base, B_base, C_base = EpicEyeSingleShotDetector.rotation_matrix_to_euler_angles(
                rot_base
            )

            # 可达性修正: 将 A/C 加减 180 度, 使其偏离当前末端位姿最小
            A_base, B_base, C_base = post_correct_angles(
                A_base, B_base, C_base, robot_ABC
            )

            # 发送目标位姿 (仅一次)
            msg = (
                f"[{pos_base[0]:.3f},{pos_base[1]:.3f},{pos_base[2]:.3f},"
                f"{A_base:.2f},{B_base:.2f},{C_base:.2f}]"
            )
            try:
                conn.sendall(msg.encode("utf-8"))
                print(f"[SEND] 目标基座标: {msg.strip()}", flush=True)
                target_sent = True
            except (BrokenPipeError, ConnectionResetError) as e:
                print(f"[WARN] 发送目标位姿失败: {e}", flush=True)
                return
        else:
            time.sleep(0.1)

    if not target_sent:
        print("[WARN] 超时: 未获得检测位姿, 断开连接", flush=True)
        return

    # ---- [Phase C] CRP 指令循环: GRIP / RELEASE / 断连 ----
    conn.settimeout(2.0)
    try:
        while not stop_event.is_set():
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue

            if not data:
                print("[INFO] 连接已关闭 (机器人执行了 SocketClose)", flush=True)
                break

            cmd = data.decode("utf-8").strip()

            # 忽略机器人发来的循环位姿更新
            if cmd.startswith("[") and cmd.endswith("]"):
                if cmd.count(",") == 5:
                    continue

            print(f"[RECV] {cmd}", flush=True)

            if "GRIP" in cmd:
                grip_object()
                conn.sendall(b"111")
                print("[INFO] 已回复 111 (抓取完成)", flush=True)

            elif "RELEASE" in cmd:
                release_object()
                conn.sendall(b"222")
                print("[INFO] 已回复 222 (释放完成)", flush=True)
                break

            else:
                print(f"[INFO] 忽略非指令数据: {cmd}", flush=True)

    except ConnectionResetError:
        print(f"[INFO] 连接重置: {addr}", flush=True)
    except Exception as e:
        print(f"[ERROR] 客户端处理异常: {e}", flush=True)
    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[INFO] 客户端已断开: {addr}", flush=True)


# =============================================================
# 主函数
# =============================================================

def main():
    print("=" * 60)
    print("  ESTUN EpicEye 6D 抓取系统")
    print("  模型: " + GRAY_MODEL_NAME)
    print("=" * 60, flush=True)

    # ---- [Init 1] 加载手眼矩阵 (cam -> ee) ----
    T_cam2ee = load_epiceye_cam2ee()
    if np.allclose(T_cam2ee, np.eye(4)):
        print("[INFO] 手眼矩阵为单位矩阵 (直接发送相机坐标系位姿)")
    else:
        print("[INFO] 手眼矩阵已加载, 位姿将转换到末端坐标系")
        print("[INFO] 后续还将叠加 gripper_offset 得到夹爪 TCP 位姿")

    # ---- [Init 2] 初始化灵巧手 (OmniHand / 模拟) ----
    init_hand()

    # ---- [Init 3] 初始化 EpicEye 相机 ----
    print(f"\n[INFO] 正在连接 EpicEye 相机...")
    camera = None
    try:
        camera = EpicEyeCamera(ip=CAMERA_IP, auto_search=AUTO_SEARCH)
        print("[INFO] EpicEye 相机已连接")
    except Exception as e:
        print(f"[FATAL] EpicEye 相机连接失败: {e}")
        print("[FATAL] 请确认已连接 EpicEye 相机并检查 IP 配置")
        disable_hand()
        return

    # ---- [Init 4] 初始化 YOLO 灰度检测器 ----
    print(f"[INFO] 正在初始化 YOLO 灰度检测器...")
    try:
        detector = EpicEyeSingleShotDetector()
    except Exception as e:
        print(f"[FATAL] YOLO 模型加载失败: {e}")
        camera.close()
        disable_hand()
        return

    # 从 SDK 获取相机内参
    K = camera.get_camera_matrix()
    if K is not None:
        detector.set_camera_matrix(K)
        print(f"[INFO] 相机内参已从 SDK 获取: fx={detector.fx:.2f}, fy={detector.fy:.2f}, "
              f"cx={detector.cx:.2f}, cy={detector.cy:.2f}")
    else:
        print("[FATAL] 无法从 SDK 获取相机内参")
        camera.close()
        disable_hand()
        return

    # ---- [Init 5] 启动 TCP 服务器 (非阻塞) ----
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setblocking(False)
    server.bind((HOST, PORT))
    server.listen(5)
    print(f"\n[INFO] 服务器已启动: {HOST}:{PORT}")
    print("[INFO] 等待机器人连接...", flush=True)

    cv2.namedWindow("ESTUN EpicEye 6D Pose", cv2.WINDOW_NORMAL)

    # ---- 主循环状态 ----
    client_connected = False
    need_trigger = False       # 是否需要触发拍照
    last_gray_image = None     # 用于显示的灰度图
    last_display_ready = False # 是否有结果显示

    running = True
    try:
        while running:
            # ---- [Step 1] 需要时触发 EpicEye 拍照 ----
            if need_trigger and camera is not None:
                print("[INFO] 触发拍照...", flush=True)
                data = camera.capture_all(pointcloud=True)
                if data["image"] is None:
                    print("[WARN] 拍照返回图像为空", flush=True)
                else:
                    last_gray_image = data["image"]  # 3通道BGR (用于显示)
                    point_map = data["point_cloud"]

                    # ---- [Step 2] YOLO 检测 -> PCA 6D 位姿 ----
                    print("[INFO] 执行检测...", flush=True)
                    # YOLO 灰度模型需要灰度输入
                    gray_input = cv2.cvtColor(last_gray_image, cv2.COLOR_BGR2GRAY)
                    detections = detector.detect_targets(gray_input)
                    print(f"  YOLO 检测到 {len(detections)} 个目标")

                    if detections:
                        best_det = max(detections, key=lambda d: d[4])
                        bbox = best_det[:4]
                        conf = best_det[4]

                        position, rotation, points = detector.compute_6d_pose(bbox, point_map)

                        if position is not None and rotation is not None:
                            A, B, C = detector.rotation_matrix_to_euler_angles(rotation)
                            print(f"  6D位姿: (X={position[0]:.3f}, Y={position[1]:.3f}, "
                                  f"Z={position[2]:.3f}, A={A:.1f}, B={B:.1f}, C={C:.1f})", flush=True)

                            # 更新 last_known_pose (相机坐标系)
                            set_last_known_pose((position, rotation))
                            last_display_ready = True
                            need_trigger = False
                        else:
                            print("  [WARN] 点云不足,无法计算位姿", flush=True)
                            last_display_ready = False
                    else:
                        print("  [WARN] 未检测到目标", flush=True)
                        last_display_ready = False

                need_trigger = False  # 无论成功失败, 标记清零

            # ---- [Step 3] 显示窗口 ----
            display_img = None
            if last_gray_image is not None:
                display_img = last_gray_image.copy()
                h, w = display_img.shape[:2]
                display_w = 960
                display_h = int(h * display_w / w)

                if last_display_ready:
                    # 绘制检测结果
                    last_pose = get_last_known_pose()
                    if last_pose is not None:
                        position, rotation = last_pose
                        # 在图像上绘制 6D 坐标轴
                        detector.visualize_pose(display_img, position, rotation, K, scale=0.08)
                        A, B, C = detector.rotation_matrix_to_euler_angles(rotation)
                        text_pos = f"Pos: ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})"
                        cv2.putText(display_img, text_pos, (30, 40),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        text_euler = f"Euler: (A={A:.1f}, B={B:.1f}, C={C:.1f})"
                        cv2.putText(display_img, text_euler, (30, 70),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # 状态: 等待机器人
                    status_text = "Waiting for robot..." if not client_connected else "Target acquired, waiting for robot..."
                    cv2.putText(display_img, status_text, (30, display_img.shape[0] - 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                elif need_trigger is False:
                    # 检测失败提示
                    cv2.putText(display_img, "No target detected", (30, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(display_img, "Press SPACE to retry capture", (30, 70),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                display_img = cv2.resize(display_img, (display_w, display_h))
            else:
                # 无图像时的占位
                display_img = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(display_img, "Waiting for robot connection...", (50, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

            cv2.imshow("ESTUN EpicEye 6D Pose", display_img)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] 用户按 q 退出", flush=True)
                stop_event.set()
                running = False
                break

            if key == ord(" "):  # 空格重拍
                if not last_display_ready:
                    print("[INFO] 用户按空格重新触发拍照", flush=True)
                    need_trigger = True

            # ---- [Step 4] 非阻塞 accept -> 子线程 handle_client ----
            try:
                conn, addr = server.accept()
            except BlockingIOError:
                # 无客户端连接, 继续循环
                pass
            else:
                # 有新客户端连接
                client_connected = True
                set_last_known_pose(None)  # 清除旧数据
                need_trigger = True        # 触发拍照
                last_display_ready = False

                t = threading.Thread(
                    target=handle_client,
                    args=(
                        conn, addr,
                        T_cam2ee,
                        GRIPPER_T_OFFSET, GRIPPER_R_OFFSET,
                        ORIENTATION_MODE,
                    ),
                    daemon=True
                )
                t.start()
                print("\n[INFO] 等待下一次连接...", flush=True)

            # 检查客户端是否已断开 (子线程结束后 client_connected 保持 True, 但子线程已结束)
            # 通过检查是否有新连接来更新状态 - 当 accept 成功时重置

    except KeyboardInterrupt:
        print("\n[INFO] 键盘中断 (Ctrl+C)")
    except Exception as e:
        print(f"[ERROR] 服务器异常: {e}")
    finally:
        cv2.destroyAllWindows()
        print("\n[INFO] 正在清理资源...")
        server.close()
        if camera is not None:
            camera.close()
            print("[INFO] EpicEye 相机已关闭")
        disable_hand()
        print("[INFO] 程序已退出", flush=True)


if __name__ == "__main__":
    main()