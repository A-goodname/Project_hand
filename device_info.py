# -*- coding: utf-8 -*-
import pyrealsense2 as rs

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
