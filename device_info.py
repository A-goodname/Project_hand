# -*- coding: utf-8 -*-
"""
查询 RealSense 设备信息
======================
包含两部分:
  1. list_devices(): 列出所有已连接 RealSense 设备序列号等信息
  2. 原有功能: 启动 pipeline 并打印相机内参/外参

使用方法:
    python device_info.py          # 同时执行两部分
"""

import pyrealsense2 as rs


def list_devices():
    """查询并打印所有已连接 RealSense 设备的信息."""
    ctx = rs.context()
    devices = ctx.query_devices()
    num = len(devices)

    if num == 0:
        print("[INFO] 未检测到 RealSense 设备")
        return

    print("=" * 60)
    print("  检测到 %d 个 RealSense 设备:" % num)
    print("=" * 60)

    for i, dev in enumerate(devices):
        name = dev.get_info(rs.camera_info.name)
        serial = dev.get_info(rs.camera_info.serial_number)
        usb_port = dev.get_info(rs.camera_info.usb_type_descriptor)
        fw_ver = dev.get_info(rs.camera_info.firmware_version)
        product_line = dev.get_info(rs.camera_info.product_line)

        print("  Device %d:" % i)
        print("    Name:          %s" % name)
        print("    Serial:        %s" % serial)
        print("    USB Port:      %s" % usb_port)
        print("    Firmware:      %s" % fw_ver)
        print("    Product Line:  %s" % product_line)
        print("-" * 60)

    print()
    print("配置方法: 在 RealSense_R/config.yaml 中添加")
    print("  device_serial: \"<序列号>\"  # 留空则使用第一台设备")
    print()


def print_camera_info():
    """原有功能: 启动 pipeline 并打印内参/外参."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    cfg = pipeline.start(config)

    device = cfg.get_device()
    name = device.get_info(rs.camera_info.name)
    print(name)

    # 与 test_3D 一致:获取 depth stream 和 color stream 的 profile
    depth_profile = cfg.get_stream(rs.stream.depth)
    color_profile = cfg.get_stream(rs.stream.color)

    # 深度内参
    depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
    print("深度传感器内参:", depth_intrinsics)

    # 彩色内参 (与 test_3D 相同的分辨率 640x480, bgr8)
    color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
    print("彩色相机内参:", color_intrinsics)

    # color to depth 外参 (与 test_3D 一致)
    color_to_depth_extrinsics = color_profile.get_extrinsics_to(depth_profile)
    print("color to depth extrinsics:")
    print("rotation:", color_to_depth_extrinsics.rotation)
    print("translation:", color_to_depth_extrinsics.translation)

    pipeline.stop()


if __name__ == "__main__":
    print("=" * 60)
    print("  [Part 1] 设备列表")
    print("=" * 60)
    list_devices()

    print("\n" + "=" * 60)
    print("  [Part 2] 相机内参 (第一台设备)")
    print("=" * 60)
    print_camera_info()