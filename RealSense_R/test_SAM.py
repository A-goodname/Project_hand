# -*- coding: utf-8 -*-
"""
RealSense YOLO+SAM 水瓶6D位姿估计 (RealSense_R 版)
=================================================
使用 YOLO(GPU) 检测 + SAM(CPU) 精分割,计算 6D 位姿.

策略:
  - YOLO → GPU (cuda:0) 快速检测 bbox
  - SAM  → CPU (sam_b.pt) 精确分割 mask
  - 在 mask 区域内手动映射到深度坐标,反投影生成点云
  - PCA 计算 6D 位姿 (与 test_6D.py 一致,仅修正 Z 轴指向)

使用方法:
  1. 连接 Intel RealSense D435i 相机
  2. 运行: python RealSense_R/test_SAM.py
  3. 按 'q' 键退出

输出格式: (X, Y, Z, A, B, C)
  X, Y, Z = 位置 (m)
  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw
"""

import sys
import os
import yaml

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import time
import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from ultralytics import YOLO, SAM, FastSAM
from typing import Optional


# ============================================================
#  0. 参数加载工具函数 (与 test_6D.py 相同)
# ============================================================

def _load_camera_params(yaml_path):
    """
    从标准 YAML 文件加载相机参数(深度+彩色+外参).
    返回 dict:
      {
          'depth_intr': rs.intrinsics,
          'color_intr': rs.intrinsics,
          'd2c_extr': rs.extrinsics,
          'c2d_extr': rs.extrinsics,
      }
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    def _intr(src):
        i = rs.intrinsics()
        i.width, i.height = data['camera']['resolution']
        i.ppx, i.ppy = src['ppx'], src['ppy']
        i.fx, i.fy = src['fx'], src['fy']
        i.model = getattr(rs.distortion, src['model'])
        i.coeffs = src['distortion']
        return i

    def _extr(src):
        e = rs.extrinsics()
        e.rotation = src['rotation']
        e.translation = src['translation']
        return e

    return {
        'depth_intr': _intr(data['depth']['intrinsics']),
        'color_intr': _intr(data['color']['intrinsics']),
        'd2c_extr': _extr(data['extrinsics']['depth_to_color']),
        'c2d_extr': _extr(data['extrinsics']['color_to_depth']),
    }


# ---- 加载配置 ----
_config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(_config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

model_name = os.path.join(_project_root, cfg["model"]["name"])
class_id = cfg["model"]["target_class_id"]
conf_threshold = cfg["model"]["conf_threshold"]
depth_min = cfg["depth"]["min"]
depth_max = cfg["depth"]["max"]
SMOOTH_ALPHA = cfg["smoothing"]["alpha"]
FRAME_W = cfg["resolution"]["width"]
FRAME_H = cfg["resolution"]["height"]
FRAME_FPS = cfg["resolution"]["fps"]

_cam_params = _load_camera_params(
    os.path.join(os.path.dirname(__file__), 'D435i_info.yaml')
)

# ---- 设备分配 ----
_device_pref = cfg["model"].get("device", "cpu")
if _device_pref == "gpu" and torch.cuda.is_available():
    DEVICE = 'cuda:0'
else:
    DEVICE = 'cpu'
print("[INFO] YOLO 运行设备: %s" % DEVICE)

# ---- SAM 配置 (从 config.yaml 读取) ----
_sam_cfg = cfg.get("sam", {})
SAM_MODEL_NAME = _sam_cfg.get("model", "mobile_sam.pt")  # 模型文件名
SAM_CONF = _sam_cfg.get("conf", 0.3)                     # 检测置信度
_sam_device_pref = _sam_cfg.get("device", "gpu")
SAM_DEVICE = 'cuda:0' if (_sam_device_pref == "gpu" and torch.cuda.is_available()) else 'cpu'
print("[INFO] SAM  运行设备: %s" % SAM_DEVICE)

# ---- SAM 模型路径 ----
SAM_MODEL_PATH = os.path.join(_project_root, "model", SAM_MODEL_NAME)
IS_FAST_SAM = SAM_MODEL_NAME.lower().startswith("fastsam")
if IS_FAST_SAM:
    print("[INFO] SAM 模型类型: FastSAM (使用 FastSAM 类加载)")
else:
    print("[INFO] SAM 模型类型: Ultralytics SAM (使用 SAM 框架加载)")


# ============================================================
#  1. 平滑工具类 (与 test_6D.py 相同)
# ============================================================

class PositionSmoother:
    """EMA 时域平滑器 (用于3D位置)"""
    def __init__(self, alpha=SMOOTH_ALPHA):
        self.alpha = alpha
        self.filtered = None

    def update(self, pos):
        if pos is None:
            return None
        if self.filtered is None:
            self.filtered = list(pos)
        else:
            for i in range(3):
                self.filtered[i] = (1 - self.alpha) * self.filtered[i] + self.alpha * pos[i]
        return tuple(self.filtered)

    def reset(self):
        self.filtered = None


class PoseSmoother:
    """6D 位姿平滑器: 位置 EMA + 旋转矩阵指数移动平均 + SVD 正交化"""
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.filtered_pos = None
        self.filtered_rot = None

    def update(self, pos, rot):
        if pos is None or rot is None:
            return None, None
        if self.filtered_pos is None or self.filtered_rot is None:
            self.filtered_pos = list(pos)
            self.filtered_rot = rot.copy()
            return tuple(self.filtered_pos), self.filtered_rot
        for i in range(3):
            self.filtered_pos[i] = (1 - self.alpha) * self.filtered_pos[i] + self.alpha * pos[i]
        self.filtered_rot = (1 - self.alpha) * self.filtered_rot + self.alpha * rot
        # SVD 正交化, 确保是有效的旋转矩阵
        U, _, Vt = np.linalg.svd(self.filtered_rot)
        self.filtered_rot = U @ Vt
        return tuple(self.filtered_pos), self.filtered_rot

    def reset(self):
        self.filtered_pos = None
        self.filtered_rot = None


# ============================================================
#  2. 主检测器 (YOLO + SAM)
# ============================================================

class RealSenseYOLOSAM6DPoseDetector:
    def __init__(self, model_name=model_name, conf_threshold=conf_threshold):
        self.conf_threshold = conf_threshold
        self.target_class_id = class_id

        # --- RealSense 初始化 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, FRAME_FPS)
        config.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, FRAME_FPS)

        self.profile = self.pipeline.start(config)

        # --- 深度缩放系数 ---
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[INFO] depth_scale = {self.depth_scale}")

        # 激光功率设适中, 减少近距光斑过曝
        try:
            depth_sensor.set_option(rs.option.laser_power, 150)
            print("[INFO] laser_power set to 150")
        except Exception as e:
            print(f"[WARN] laser_power set failed: {e}")

        # 不使用 rs.align,改为手动映射

        # --- 相机参数: 全部从 D435i_info.yaml 加载 ---
        self.depth_intrinsics = _cam_params['depth_intr']
        self.color_intrinsics = _cam_params['color_intr']
        self.color_to_depth_extrinsics = _cam_params['c2d_extr']
        self.depth_to_color_extrinsics = _cam_params['d2c_extr']

        print(f"[INFO] 深度内参 (YAML): fx={self.depth_intrinsics.fx:.2f}, "
              f"fy={self.depth_intrinsics.fy:.2f}, "
              f"ppx={self.depth_intrinsics.ppx:.2f}, ppy={self.depth_intrinsics.ppy:.2f}")
        print(f"[INFO] 彩色内参 (YAML): fx={self.color_intrinsics.fx:.2f}, "
              f"fy={self.color_intrinsics.fy:.2f}, "
              f"ppx={self.color_intrinsics.ppx:.2f}, ppy={self.color_intrinsics.ppy:.2f}")

        # --- YOLO ---
        yolo_path = f"{model_name}.pt"
        if not os.path.exists(yolo_path):
            yolo_path = model_name
        self.yolo_model = YOLO(yolo_path).to(DEVICE)
        print(f"[INFO] YOLO 模型已加载: {yolo_path}")

        # --- SAM (FastSAM 使用 YOLO 框架, 其他使用 SAM 框架) ---
        if not os.path.exists(SAM_MODEL_PATH):
            print(f"[WARN] SAM 模型未找到: {SAM_MODEL_PATH}")
            print("[WARN] 将跳过 SAM 推理, 使用 bbox 直接采样")
            self.sam_model = None
        elif IS_FAST_SAM:
            # FastSAM 使用 FastSAM 类加载
            self.sam_model = FastSAM(SAM_MODEL_PATH)
            self.sam_model.to(SAM_DEVICE)
            print(f"[INFO] FastSAM 模型已加载: {SAM_MODEL_PATH}")
        else:
            # 普通 SAM / MobileSAM 使用 SAM 框架加载
            self.sam_model = SAM(SAM_MODEL_PATH)
            self.sam_model.to(SAM_DEVICE)
            print(f"[INFO] SAM 模型已加载: {SAM_MODEL_PATH}")

        self.colorizer = rs.colorizer()

        # --- 平滑 ---
        self._pose_smoother = PoseSmoother(alpha=0.3)

        # --- 耗时统计 ---
        self._yolo_time_ms = 0.0
        self._sam_time_ms = 0.0

        # --- estun_run.py 外部使用的属性 ---
        self.last_known_pose: Optional[tuple] = None
        self._cached_raw_depth = None

    # ------------------------------------------------------------
    #  工具: 将单个彩色像素映射到深度图坐标
    # ------------------------------------------------------------
    def _color_to_depth_pixel(self, cx, cy, depth_frame):
        """
        将彩色图上的像素 (cx, cy) 映射到深度图坐标系.
        返回 (dx, dy) 或 None (映射失败时)
        """
        try:
            depth_pixel = rs.rs2_project_color_pixel_to_depth_pixel(
                depth_frame.get_data(),
                self.depth_scale,
                0.1, 6.0,
                self.depth_intrinsics,
                self.color_intrinsics,
                self.color_to_depth_extrinsics,
                self.depth_to_color_extrinsics,
                [cx, cy]
            )
            if depth_pixel and np.isfinite(depth_pixel).all():
                dx, dy = int(depth_pixel[0]), int(depth_pixel[1])
                w = depth_frame.get_width()
                h = depth_frame.get_height()
                if 0 <= dx < w and 0 <= dy < h:
                    return dx, dy
        except Exception:
            pass
        return None

    # ------------------------------------------------------------
    def get_frames(self):
        """
        获取原始彩色帧和原始深度帧.
        不经过 rs.align
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return None, None, None, None

        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())

        return color_image, depth_image, depth_colormap, depth_frame

    # ------------------------------------------------------------
    def compute_6d_pose(self, color_image, depth_frame):
        """
        YOLO + SAM 联合 6D 位姿估计.
        任一模型失败或点数不足 → return (None, None, None, None)

        参数:
            color_image : numpy, H×W×3, BGR
            depth_frame : rs.frame

        返回:
            position    : (x, y, z) 重心 (m) 或 None
            rotation    : 3×3 旋转矩阵 或 None
            point_cloud : N×3 点云 或 None
            mask        : (H, W) bool, True=物体区域; 或 None
        """
        # === Step 1: YOLO 检测 ===
        _t_yolo_start = time.perf_counter()
        yolo_results = self.yolo_model(
            color_image,
            conf=self.conf_threshold,
            classes=[self.target_class_id],
            verbose=False
        )
        self._yolo_time_ms = (time.perf_counter() - _t_yolo_start) * 1000

        if len(yolo_results) == 0 or yolo_results[0].boxes is None:
            return None, None, None, None

        boxes = yolo_results[0].boxes
        if boxes.shape[0] == 0:
            return None, None, None, None

        # 取最高置信度 bbox
        confs = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confs))
        bbox = boxes.xyxy[best_idx].cpu().numpy()  # [x1, y1, x2, y2]

        # === Step 2: SAM 预测 mask  ===
        _t_sam_start = time.perf_counter()
        if self.sam_model is None:
            # SAM 模型未加载, 跳过
            return None, None, None, None

        # SAM 需要 RGB 输入
        rgb_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        sam_results = self.sam_model(
            rgb_image,
            bboxes=[bbox.tolist()],
            conf=SAM_CONF,
            device=SAM_DEVICE,
            verbose=False
        )
        self._sam_time_ms = (time.perf_counter() - _t_sam_start) * 1000

        if len(sam_results) == 0 or sam_results[0].masks is None:
            return None, None, None, None

        mask = sam_results[0].masks.data[0].cpu().numpy()  # float32, H×W
        mask = mask > 0.5  # float → bool

        # === Step 3: mask 区域逐像素映射 → 点云 ===
        ys, xs = np.where(mask)
        if len(xs) < 10:
            return None, None, None, None

        fx = self.depth_intrinsics.fx
        fy = self.depth_intrinsics.fy
        ppx = self.depth_intrinsics.ppx
        ppy = self.depth_intrinsics.ppy

        points = []
        for cx, cy in zip(xs, ys):
            # 彩色像素 → 映射到深度图坐标
            depth_pixel = self._color_to_depth_pixel(cx, cy, depth_frame)
            if depth_pixel is None:
                continue

            dx, dy = depth_pixel
            d = depth_frame.get_distance(dx, dy)
            if not (depth_min < d < depth_max):
                continue

            # 反投影 (使用深度内参)
            depth_x = (dx - ppx) * d / fx
            depth_y = (dy - ppy) * d / fy
            points.append([depth_x, depth_y, d])

        if len(points) < 10:
            return None, None, None, None

        point_cloud = np.array(points)

        # === Step 4: PCA 位姿 ===
        center = np.mean(point_cloud, axis=0)

        cov = np.cov(point_cloud, rowvar=False)
        eig_val, eig_vec = np.linalg.eig(cov)

        idx = np.argsort(eig_val)[::-1]
        eig_vec = eig_vec[:, idx]

        # 主方向 (最大特征值) → Y轴 (瓶身主轴)
        target_y = np.real(eig_vec[:, 0])
        # 最小方差方向 → Z轴 (指向相机)
        target_z = np.real(eig_vec[:, 2])

        # 方向修正: 仅 Z 轴 (与 test_6D.py 一致)
        if target_z[2] > 0:
            target_z = -target_z
        # Y 轴不修 (留给 estun_run.py 在基坐标系下修正)

        # 叉积重建右手系
        target_x = np.cross(target_y, target_z)
        target_x = target_x / np.linalg.norm(target_x)
        target_y = np.cross(target_z, target_x)
        target_y = target_y / np.linalg.norm(target_y)

        rot = np.stack((target_x, target_y, target_z), axis=1)

        return center, rot, point_cloud, mask

    # ------------------------------------------------------------
    @staticmethod
    def rotation_matrix_to_euler_angles(R):
        """
        从旋转矩阵提取 ZYX 顺序欧拉角 (A=roll, B=pitch, C=yaw).
        返回角度值 (deg).
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

    # ------------------------------------------------------------
    def visualize_pose(self, image, center_3d, rot, K, scale=0.08):
        """
        在图像上绘制 6D 位姿坐标轴.
        红=X轴, 绿=Y轴, 蓝=Z轴
        """
        center_2d = K @ np.array(center_3d)
        center_2d = center_2d / center_2d[2]
        cx_2d = int(center_2d[0])
        cy_2d = int(center_2d[1])

        axes = [
            (rot[:, 0], (0, 0, 255)),   # X → 红色
            (rot[:, 1], (0, 255, 0)),   # Y → 绿色
            (rot[:, 2], (255, 0, 0)),   # Z → 蓝色
        ]

        cv2.circle(image, (cx_2d, cy_2d), 5, (124, 0, 255), 3)

        for axis, color in axes:
            end_3d = np.array(center_3d) + scale * axis
            end_2d = K @ end_3d
            end_2d = end_2d / end_2d[2]
            ex_2d = int(end_2d[0])
            ey_2d = int(end_2d[1])

            cv2.arrowedLine(image, (cx_2d, cy_2d), (ex_2d, ey_2d), color, 3, tipLength=0.2)

    # ------------------------------------------------------------
    def run(self):
        """
        主循环: 单线程 YOLO→SAM→PCA→显示.
        每帧完整处理完后更新显示, 不要求实时性.
        """
        print("按 'q' 键退出程序...")
        print("\n6D位姿格式: (X, Y, Z, A, B, C)")
        print("  X, Y, Z = 位置 (m)")
        print("  A, B, C = ZYX 欧拉角 (deg), 即 Roll, Pitch, Yaw")
        print("坐标轴颜色: 红=X 绿=Y(主方向, 瓶身主轴) 蓝=Z\n")

        try:
            while True:
                # ---- [Step 1] 获取帧 ----
                color_image, depth_image, depth_colormap, depth_frame = self.get_frames()
                if color_image is None or depth_frame is None:
                    continue

                display_img = color_image.copy()

                # ---- [Step 2] YOLO + SAM + PCA ----
                position, rotation, point_cloud, mask = self.compute_6d_pose(
                    color_image, depth_frame
                )

                if position is not None and rotation is not None and point_cloud is not None:
                    # 平滑
                    smooth_pos, smooth_rot = self._pose_smoother.update(
                        position, rotation
                    )
                    if smooth_pos is not None:
                        assert smooth_rot is not None

                        # 旋转矩阵 → ZYX 欧拉角
                        A, B, C = self.rotation_matrix_to_euler_angles(smooth_rot)

                        # 输出 (包含耗时)
                        print(f"\n[RESULT] SAM Pose: "
                              f"({smooth_pos[0]:.3f}, {smooth_pos[1]:.3f}, {smooth_pos[2]:.3f}, "
                              f"{A:.1f}, {B:.1f}, {C:.1f})")
                        print(f"      YOLO: {self._yolo_time_ms:.0f}ms | "
                              f"SAM: {self._sam_time_ms:.0f}ms | "
                              f"Points: {point_cloud.shape[0]}")

                        # 绘制 SAM mask (半透明绿色覆盖)
                        if mask is not None:
                            overlay = display_img.copy()
                            overlay[mask] = (0, 255, 0)  # 绿色高亮 mask 区域
                            cv2.addWeighted(overlay, 0.3, display_img, 0.7, 0, display_img)
                            # 绘制 mask 轮廓
                            mask_uint8 = mask.astype(np.uint8) * 255
                            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL,
                                                           cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display_img, contours, -1, (0, 255, 0), 2)

                        # 可视化坐标轴 (用深度内参投影)
                        K = np.array([
                            [self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
                            [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
                            [0, 0, 1]
                        ], dtype=float)
                        self.visualize_pose(display_img, smooth_pos, smooth_rot,
                                            K, scale=0.08)

                        # 显示位姿文本
                        text_pos = (f'Pos: {smooth_pos[0]:.2f}, '
                                    f'{smooth_pos[1]:.2f}, {smooth_pos[2]:.2f}')
                        cv2.putText(display_img, text_pos, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        text_euler = (f'Euler: A={A:.1f}, B={B:.1f}, C={C:.1f}')
                        cv2.putText(display_img, text_euler, (10, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        text_points = (f'Points: {point_cloud.shape[0]}')
                        cv2.putText(display_img, text_points, (10, 80),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        text_time = (f'YOLO: {self._yolo_time_ms:.0f}ms  '
                                     f'SAM: {self._sam_time_ms:.0f}ms')
                        cv2.putText(display_img, text_time, (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                else:
                    self._pose_smoother.reset()
                    # 显示检测失败提示
                    cv2.putText(display_img, "Detection failed",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (0, 0, 255), 2)

                # ---- [Step 3] 显示 ----
                cv2.imshow('RealSense YOLO+SAM 6D Pose', display_img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        finally:
            self.pipeline.stop()
            cv2.destroyAllWindows()
            print("程序已退出")


# ============================================================
#  3. 主入口
# ============================================================

if __name__ == "__main__":
    detector = RealSenseYOLOSAM6DPoseDetector(
        model_name=model_name, conf_threshold=conf_threshold
    )
    detector.run()