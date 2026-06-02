# -*- coding: utf-8 -*-
"""
EpicEye Camera Wrapper
=====================
功能描述: 封装 EpicEye SDK 相机操作,提供统一的接口

输入:
  - EpicEye/epiceye_config.yaml     : 相机IP等配置参数

输出:
  - logs/epiceye_*.log              : 运行日志 (每次实例化一个文件)
  - captures/                       : 保存图像/点云/深度图的目录

架构:
  - 静态方法:     纯函数, 无副作用 (convert_raw10_to_8bit, depth_to_colormap)
  - 功能方法:     封装 EpicEye SDK 调用
  - 主流程:       在 __init__ 中自动完成初始化
"""

import logging
import os
import yaml
from datetime import datetime

import cv2
import numpy as np
import epiceye


# ---- 加载配置 (模块级,遵循 template 风格) ----
_config_path = os.path.join(os.path.dirname(__file__), "epiceye_config.yaml")
if os.path.exists(_config_path):
    with open(_config_path, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
else:
    _cfg = {}

# 提取配置项
CAMERA_IP = _cfg.get("camera", {}).get("ip", "")
AUTO_SEARCH = _cfg.get("camera", {}).get("auto_search", True)
SAVE_DIR = _cfg.get("save", {}).get("dir", "captures")
LOG_DIR = _cfg.get("log", {}).get("dir", "logs")
LOG_LEVEL = _cfg.get("log", {}).get("level", "INFO")

# 将日志级别字符串转换为 logging 常量
LOG_LEVEL_MAP = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
}
_log_level_num = LOG_LEVEL_MAP.get(LOG_LEVEL.upper(), logging.INFO)


# =============================================================
# 辅助函数:获取项目根目录
# =============================================================

def _get_project_root():
    """
    获取项目根目录 (Project_hand)

    返回:
        str, 项目根目录绝对路径
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# =============================================================
# 功能模块:EpicEye 相机封装
# =============================================================

class EpicEyeCamera:
    """
    EpicEye 相机封装类

    封装 SDK 的所有相机操作,提供统一的接口.
    实例化时自动连接相机 (指定IP或自动搜索),
    并缓存在 __init__ 期间即可获得的设备信息 (info,内参等).

    使用示例:
        camera = EpicEyeCamera("192.168.1.100")
        frame_id = camera.trigger_frame()
        image = camera.get_image(frame_id)
        camera.save_image(frame_id)
        camera.close()
    """

    def __init__(self, ip="", auto_search=True):
        """
        初始化相机连接

        参数:
            ip          : str, 相机IP地址.为空字符串时根据配置文件或 auto_search 决定
            auto_search : bool, 未指定IP时是否自动搜索网络中的相机
        """
        # ---- 解析最终 IP ----
        self.ip = ip or CAMERA_IP
        if not self.ip and (auto_search or AUTO_SEARCH):
            self.ip = self._search_camera()
        if not self.ip:
            raise RuntimeError(
                "未指定相机IP且自动搜索失败.请检查相机连接或手动指定IP."
            )

        # ---- 初始化日志 (仅写入文件,不输出到控制台) ----
        self.logger = self._setup_logger()

        # ---- 获取并缓存设备信息 ----
        self.info = None
        self.camera_matrix = None
        self.distortion = None
        self._width = None
        self._height = None

        self._cache_device_info()

        self.logger.info(
            "EpicEye 相机初始化完成: SN=%s | IP=%s | 分辨率=%dx%d",
            self.info.get("sn", "N/A") if self.info else "N/A",
            self.ip,
            self._width or 0,
            self._height or 0,
        )

    # ----------------------------------------------------------
    # 日志设置
    # ----------------------------------------------------------

    def _setup_logger(self):
        """
        设置日志记录器

        日志仅写入文件,不输出到控制台.
        每次实例化创建一个独立的日志文件 (带时间戳).

        Returns:
            logging.Logger
        """
        logger = logging.getLogger(f"EpicEyeCamera_{id(self)}")
        logger.setLevel(_log_level_num)
        logger.propagate = False  # 阻止传播到根日志,避免控制台输出

        # 清除已有 handler (确保每个实例干净)
        logger.handlers.clear()

        # 构建日志目录路径 (相对于项目根目录)
        project_root = _get_project_root()
        log_dir = os.path.join(project_root, LOG_DIR)
        os.makedirs(log_dir, exist_ok=True)

        # 日志文件名:epiceye_YYYYMMDD_HHMMSS.log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"epiceye_{timestamp}.log")

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(_log_level_num)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 将 handler 引用保存在实例上,方便 close 时关闭
        self._log_handler = file_handler
        self._log_path = log_path

        return logger

    # ----------------------------------------------------------
    # 相机搜索与连接
    # ----------------------------------------------------------

    @staticmethod
    def _search_camera():
        """
        搜索网络中的可用相机

        Returns:
            str, 第一个找到的相机IP;搜索失败返回空字符串
        """
        found = epiceye.search_camera()
        if found is not None and len(found) > 0:
            return found[0]["ip"]
        return ""

    # ----------------------------------------------------------
    # 设备信息缓存
    # ----------------------------------------------------------

    def _cache_device_info(self):
        """
        获取并缓存在初始化时可获得的设备信息

        包括 info (序列号/IP/型号/别名/分辨率),内参矩阵,畸变系数
        """
        # 获取设备基本信息
        info = epiceye.get_info(self.ip)
        if info is not None:
            self.info = info
            self._width = info.get("width")
            self._height = info.get("height")
        else:
            self.logger.warning("获取相机信息失败,部分功能可能受限")

        # 获取内参矩阵
        self.camera_matrix = epiceye.get_camera_matrix(self.ip)

        # 获取畸变系数
        self.distortion = epiceye.get_distortion(self.ip)

    # ----------------------------------------------------------
    # 信息查询
    # ----------------------------------------------------------

    def get_sdk_version(self):
        """
        获取 SDK 版本号

        返回:
            str, SDK 版本号
        """
        return epiceye.get_sdk_version()

    def get_info(self):
        """
        获取相机基本信息 (已缓存)

        返回:
            dict or None, 包含 sn/ip/model/alias/width/height
        """
        return self.info

    def get_camera_matrix(self):
        """
        获取相机内参矩阵 (已缓存)

        返回:
            numpy.ndarray or None, 3x3 相机内参矩阵,与 OpenCV 兼容
        """
        return self.camera_matrix

    def get_distortion(self):
        """
        获取相机畸变系数 (已缓存)

        返回:
            numpy.ndarray or None, 畸变参数,与 OpenCV 兼容
        """
        return self.distortion

    # ----------------------------------------------------------
    # 配置管理
    # ----------------------------------------------------------

    def get_config(self, frame_id=""):
        """
        获取相机当前或指定帧的配置参数

        参数:
            frame_id : str, 待获取数据的 frameID,为空字符串则返回最新配置

        返回:
            dict or None, 相机配置参数
        """
        config = epiceye.get_config(self.ip, frame_id=frame_id)
        if config is None:
            self.logger.warning("获取配置失败 (frame_id=%s)", frame_id or "latest")
        return config

    def set_config(self, config):
        """
        更新相机参数配置

        参数:
            config : dict, 配置参数,定义同 get_config

        返回:
            dict or None, 更新后的相机当前配置
        """
        result = epiceye.set_config(self.ip, config=config)
        if result is None:
            self.logger.warning("设置配置失败")
        else:
            self.logger.info("配置已更新")
        return result

    # ----------------------------------------------------------
    # 触发拍照
    # ----------------------------------------------------------

    def trigger_frame(self, pointcloud=True):
        """
        触发拍摄一个 Frame

        一个 Frame 可能同时包含 2D 图像和点云数据,通过 frameID 进行索引.

        参数:
            pointcloud : bool, 是否请求点云数据.设为 False 则仅包含 2D 图像

        返回:
            str or None, 此次触发拍照返回的 frame_id
        """
        frame_id = epiceye.trigger_frame(self.ip, pointcloud=pointcloud)
        if frame_id is None:
            self.logger.warning("触发拍照失败")
        else:
            self.logger.info("触发拍照成功: frame_id=%s", frame_id)
        return frame_id

    # ----------------------------------------------------------
    # 数据获取:图像
    # ----------------------------------------------------------

    def get_image(self, frame_id):
        """
        根据 frameID 获取 2D 图像

        自动将 10bit raw 图像转换为 8bit 图像.

        参数:
            frame_id : str, 待获取数据的 frameID

        返回:
            numpy.ndarray or None, 8bit 图像数据 (H x W x 3)
        """
        image = epiceye.get_image(self.ip, frame_id=frame_id)
        if image is None:
            self.logger.warning("获取图像失败 (frame_id=%s)", frame_id)
            return None
        # 将10bit raw图像转换为8bit
        image = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
        return image

    # ----------------------------------------------------------
    # 数据获取:点云
    # ----------------------------------------------------------

    def get_point_cloud(self, frame_id):
        """
        根据 frameID 获取点云

        参数:
            frame_id : str, 待获取数据的 frameID

        返回:
            numpy.ndarray or None, 点云数据 (H x W x 3),和 2D 图像逐像素对齐
        """
        point_map = epiceye.get_point_cloud(self.ip, frame_id=frame_id)
        if point_map is None:
            self.logger.warning("获取点云失败 (frame_id=%s)", frame_id)
        return point_map

    # ----------------------------------------------------------
    # 数据获取:图像 + 点云 (同时获取)
    # ----------------------------------------------------------

    def get_image_and_point_cloud(self, frame_id):
        """
        根据 frameID 获取 2D 图像和点云

        参数:
            frame_id : str, 待获取数据的 frameID

        返回:
            tuple: (image, point_map)
            image     : numpy.ndarray, 8bit 图像
            point_map : numpy.ndarray, 点云数据
            失败时返回 (None, None)
        """
        result = epiceye.get_image_and_point_cloud(self.ip, frame_id=frame_id)
        if result is None:
            self.logger.warning("获取图像和点云失败 (frame_id=%s)", frame_id)
            return None, None
        image, point_map = result
        if image is not None:
            image = cv2.convertScaleAbs(image, alpha=(255.0 / 1024.0))
        return image, point_map

    # ----------------------------------------------------------
    # 数据获取:深度图
    # ----------------------------------------------------------

    def get_depth(self, frame_id):
        """
        根据 frameID 获取深度图

        参数:
            frame_id : str, 待获取数据的 frameID

        返回:
            numpy.ndarray or None, 深度数据,和 2D 图像逐像素对齐
        """
        depth = epiceye.get_depth(self.ip, frame_id=frame_id)
        if depth is None:
            self.logger.warning("获取深度图失败 (frame_id=%s)", frame_id)
        return depth

    # ----------------------------------------------------------
    # 静态工具方法
    # ----------------------------------------------------------

    @staticmethod
    def convert_raw10_to_8bit(raw_image):
        """
        将 10bit raw 图像转换为 8bit 图像

        参数:
            raw_image : numpy.ndarray, 10bit 输入图像

        返回:
            numpy.ndarray, 8bit 输出图像
        """
        return cv2.convertScaleAbs(raw_image, alpha=(255.0 / 1024.0))

    @staticmethod
    def depth_to_colormap(depth, scale=10.0):
        """
        将深度图转换为伪彩色图

        参数:
            depth : numpy.ndarray, 深度数据
            scale : float, 深度值缩放比例,默认 10.0

        返回:
            numpy.ndarray, 伪彩色深度图
        """
        pseudo_color = cv2.applyColorMap(
            (depth / scale).astype("uint8"), cv2.COLORMAP_JET
        )
        return pseudo_color

    # ----------------------------------------------------------
    # 保存功能
    # ----------------------------------------------------------

    def _get_save_dir(self):
        """
        获取保存目录路径 (相对于 EpicEye/ 目录)

        Returns:
            str, 绝对路径
        """
        epiceye_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(epiceye_dir, SAVE_DIR)
        os.makedirs(save_dir, exist_ok=True)
        return save_dir

    def _timestamp_str(self):
        """生成用于文件命名的时间戳字符串"""
        return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]

    def save_image(self, frame_id, save_dir=None, ts=None):
        """
        保存 2D 图像到文件

        参数:
            frame_id : str, 待获取数据的 frameID
            save_dir : str or None, 保存目录,默认使用配置中的 saves 目录
            ts       : str or None, 文件命名时间戳,None 则自动生成

        返回:
            str or None, 保存的文件路径;失败返回 None
        """
        image = self.get_image(frame_id)
        if image is None:
            self.logger.error("保存图像失败:获取图像返回 None (frame_id=%s)", frame_id)
            return None

        ts = ts or self._timestamp_str()
        save_path = os.path.join(
            save_dir or self._get_save_dir(),
            f"image_{ts}.png",
        )
        cv2.imwrite(save_path, image)
        self.logger.info("图像已保存: %s", save_path)
        return save_path

    def save_point_cloud(self, frame_id, save_dir=None, ts=None):
        """
        保存点云为 ASCII PLY 文件 (不依赖 open3d)

        参数:
            frame_id : str, 待获取数据的 frameID
            save_dir : str or None, 保存目录,默认使用配置中的 saves 目录
            ts       : str or None, 文件命名时间戳,None 则自动生成

        返回:
            str or None, 保存的文件路径;失败返回 None
        """
        point_map = self.get_point_cloud(frame_id)
        if point_map is None:
            self.logger.error(
                "保存点云失败:获取点云返回 None (frame_id=%s)", frame_id
            )
            return None

        # 展平有效点 (z 值可能是米或毫米,自适应过滤)
        points = point_map.reshape(-1, 3)
        z_valid = points[np.isfinite(points[:, 2]), 2]
        if len(z_valid) == 0:
            self.logger.error("保存点云失败:z 全部为 NaN (frame_id=%s)", frame_id)
            return None
        z_median = float(np.median(z_valid))
        # 根据中值判断单位:若中值 > 100 则视为毫米
        if z_median > 100:
            valid_mask = (points[:, 2] > z_median * 0.1) & (points[:, 2] < z_median * 5.0)
        else:
            valid_mask = (points[:, 2] > 0.01) & (points[:, 2] < 10.0)
        valid_mask &= np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
        valid_points = points[valid_mask]
        if len(valid_points) == 0:
            self.logger.error("保存点云失败:无有效点 (frame_id=%s)", frame_id)
            return None

        ts = ts or self._timestamp_str()
        save_path = os.path.join(
            save_dir or self._get_save_dir(),
            f"cloud_{ts}.ply",
        )

        # 写入 ASCII PLY
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(valid_points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")
            for p in valid_points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")

        self.logger.info("点云已保存: %s (%d 点)", save_path, len(valid_points))
        return save_path

    def save_depth(self, frame_id, save_dir=None, scale=10.0, ts=None):
        """
        保存深度伪彩色图到文件

        参数:
            frame_id : str, 待获取数据的 frameID
            save_dir : str or None, 保存目录,默认使用配置中的 saves 目录
            scale    : float, 深度值缩放比例
            ts       : str or None, 文件命名时间戳,None 则自动生成

        返回:
            str or None, 保存的文件路径;失败返回 None
        """
        depth = self.get_depth(frame_id)
        if depth is None:
            self.logger.error(
                "保存深度图失败:获取深度返回 None (frame_id=%s)", frame_id
            )
            return None

        pseudo_color = self.depth_to_colormap(depth, scale=scale)
        ts = ts or self._timestamp_str()
        save_path = os.path.join(
            save_dir or self._get_save_dir(),
            f"depth_{ts}.png",
        )
        cv2.imwrite(save_path, pseudo_color)
        self.logger.info("深度伪彩色图已保存: %s", save_path)
        return save_path

    def save_all(self, frame_id, save_dir=None):
        """
        一次性保存图像,点云,深度图到文件

        所有文件使用同一时间戳命名,确保数据组一致性.

        参数:
            frame_id : str, 待获取数据的 frameID
            save_dir : str or None, 保存目录,默认使用配置中的 saves 目录

        返回:
            dict, 包含已保存文件的路径映射:
            {
                "image": str or None,
                "point_cloud": str or None,
                "depth": str or None,
            }
        """
        ts = self._timestamp_str()
        return {
            "image": self.save_image(frame_id, save_dir=save_dir, ts=ts),
            "point_cloud": self.save_point_cloud(frame_id, save_dir=save_dir, ts=ts),
            "depth": self.save_depth(frame_id, save_dir=save_dir, ts=ts),
        }

    # ----------------------------------------------------------
    # 一站式方法
    # ----------------------------------------------------------

    def capture_all(self, pointcloud=True):
        """
        触发拍照并获取所有数据 (图像,点云,深度图)

        一个便捷方法:触发 → 获取 (image, point_cloud, depth) 三件套.
        如果需要更多控制,可以分别调用 trigger_frame 和 get_xxx 方法.

        参数:
            pointcloud : bool, 是否请求点云数据

        返回:
            dict, 包含 frame_id 和数据字典:
            {
                "frame_id": str or None,
                "image": numpy.ndarray or None,
                "point_cloud": numpy.ndarray or None,
                "depth": numpy.ndarray or None,
            }
        """
        frame_id = self.trigger_frame(pointcloud=pointcloud)
        if frame_id is None:
            return {
                "frame_id": None,
                "image": None,
                "point_cloud": None,
                "depth": None,
            }

        image, point_cloud = self.get_image_and_point_cloud(frame_id)
        depth = self.get_depth(frame_id)

        return {
            "frame_id": frame_id,
            "image": image,
            "point_cloud": point_cloud,
            "depth": depth,
        }

    # ----------------------------------------------------------
    # 资源管理
    # ----------------------------------------------------------

    def close(self):
        """
        清理资源

        关闭日志文件句柄.SDK 本身是无状态的 HTTP 通信,无需释放连接.
        调用后实例仍可继续调用 SDK 方法,但日志写入已停止.
        """
        if hasattr(self, "_log_handler") and self._log_handler is not None:
            self._log_handler.close()
            if hasattr(self, "logger") and self.logger is not None:
                self.logger.removeHandler(self._log_handler)
            self._log_handler = None
    def __enter__(self):
        """支持 with 语句上下文管理"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 块时自动清理资源"""
        self.close()


# =============================================================
# 使用示例
# =============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  EpicEye 相机封装类")
    print("=" * 60)

    print("\n[INFO] 正在初始化相机...")
    print("[INFO] 使用说明: 可传入 IP 参数,如: python epiceye_camera.py 192.168.1.100\n")

    # ---- 初始化 (默认自动搜索相机) ----
    camera = EpicEyeCamera()

    # ---- 获取 SDK 版本 ----
    print("\n[INFO] SDK版本:", camera.get_sdk_version())

    # ---- 获取相机信息 ----
    info = camera.get_info()
    if info is not None:
        print(f"[INFO] 相机序列号: {info.get('sn', 'N/A')}")
        print(f"[INFO] 相机型号:   {info.get('model', 'N/A')}")
        print(f"[INFO] 分辨率:     {info.get('width', 0)} x {info.get('height', 0)}")

    # ---- 触发拍照 ----
    frame_id = camera.trigger_frame(pointcloud=True)
    if frame_id is not None:
        # 保存图像,点云,深度图
        paths = camera.save_all(frame_id)
        print(f"[INFO] 图像已保存:   {paths['image']}")
        print(f"[INFO] 点云已保存:   {paths['point_cloud']}")
        print(f"[INFO] 深度已保存:   {paths['depth']}")
    else:
        print("[ERROR] 触发拍照失败")

    # ---- 清理 ----
    camera.close()
    print("\n[INFO] 程序执行完毕")