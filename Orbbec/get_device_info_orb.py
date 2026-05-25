# -*- coding: utf-8 -*-
"""
Orbbec Gemini 215 设备信息查询工具 (YAML 输出版)
=================================================
基于 pyorbbecsdk2，实现以下功能：
1. 基本设备信息（名称、PID、UID、SN、固件版本等）
2. 所有传感器支持的 stream profile 枚举（分辨率、格式、FPS）
3. RGB 传感器流格式详细列表
4. Depth 传感器流格式详细列表
5. RGB 曝光相关属性详细查询
6. 设备可读写属性列表及当前值查询

输出：
  - 终端显示进度摘要
  - 详细结果保存到 Orbbec/device_info.yaml
"""

from pyorbbecsdk import (
    Context, Pipeline, OBPropertyType, OBPermissionType, OBSensorType, OBFormat
)
import sys
import os
import yaml


# ========== 辅助函数 ==========

def permission_type_to_string(permission):
    """权限类型转可读字符串"""
    if permission == OBPermissionType.PERMISSION_READ:
        return "R/_"
    elif permission == OBPermissionType.PERMISSION_WRITE:
        return "_/W"
    elif permission == OBPermissionType.PERMISSION_READ_WRITE:
        return "R/W"
    else:
        return "_/_"


def property_type_to_string(ptype):
    """属性类型转可读字符串"""
    if ptype == OBPropertyType.OB_BOOL_PROPERTY:
        return "BOOL"
    elif ptype == OBPropertyType.OB_INT_PROPERTY:
        return "INT"
    elif ptype == OBPropertyType.OB_FLOAT_PROPERTY:
        return "FLOAT"
    else:
        return f"OTHER({int(ptype)})"


def obformat_to_string(fmt):
    """OBFormat 枚举值 -> SDK 原生字符串"""
    # 直接使用 SDK 的 str() 转换，保留原生格式名称如 OBFormat.Y16
    return str(fmt)


def read_property_value(device, item):
    """读取单个属性的当前值"""
    try:
        if item.type == OBPropertyType.OB_BOOL_PROPERTY:
            val = device.get_bool_property(item.id)
            return int(val)
        elif item.type == OBPropertyType.OB_INT_PROPERTY:
            return device.get_int_property(item.id)
        elif item.type == OBPropertyType.OB_FLOAT_PROPERTY:
            return round(device.get_float_property(item.id), 4)
    except Exception:
        return None


# ========== 数据收集模块 ==========

def collect_device_info(device):
    """收集基本设备信息"""
    info = device.get_device_info()
    data = {
        "name":        info.get_name(),
        "pid":         f"0x{info.get_pid():04X}",
        "vid":         f"0x{info.get_vid():04X}",
        "uid":         f"0x{info.get_uid()}",
        "serial_number": info.get_serial_number(),
        "connection_type": info.get_connection_type(),
        "firmware_version": info.get_firmware_version(),
    }
    try:
        data["hardware_version"] = info.get_hardware_version()
    except Exception:
        data["hardware_version"] = None
    return data


def collect_sensor_profiles(device):
    """收集所有传感器的 stream profile 信息"""
    sensor_list = device.get_sensor_list()
    sensors = []

    for idx in range(sensor_list.get_count()):
        sensor = sensor_list.get_sensor_by_index(idx)
        sensor_type = str(sensor.get_type())
        sensor_entry = {"type": sensor_type, "profiles": []}

        try:
            profile_list = sensor.get_stream_profile_list()
        except Exception as e:
            sensor_entry["error"] = str(e)
            sensors.append(sensor_entry)
            continue

        count = profile_list.get_count()
        is_video = sensor.get_type() in [
            OBSensorType.COLOR_SENSOR,
            OBSensorType.DEPTH_SENSOR,
            OBSensorType.IR_SENSOR,
        ]

        for p_idx in range(count):
            try:
                profile = profile_list.get_stream_profile_by_index(p_idx)
                fmt_str = obformat_to_string(profile.get_format())
                entry = {"format": fmt_str}
                if is_video:
                    entry["width"]  = profile.get_width()
                    entry["height"] = profile.get_height()
                    entry["fps"]    = profile.get_fps()
                else:
                    entry["fps"] = str(profile.get_fps()) if hasattr(profile, "get_fps") else "N/A"
                sensor_entry["profiles"].append(entry)
            except Exception as e:
                sensor_entry["profiles"].append({"error": str(e)})

        sensors.append(sensor_entry)

    return {"sensor_count": sensor_list.get_count(), "sensors": sensors}


def collect_rgb_stream_info(device):
    """单独收集 RGB 传感器支持的流格式"""
    sensor_list = device.get_sensor_list()
    color_sensor = None
    for idx in range(sensor_list.get_count()):
        s = sensor_list.get_sensor_by_index(idx)
        if s.get_type() == OBSensorType.COLOR_SENSOR:
            color_sensor = s
            break

    if color_sensor is None:
        return {"error": "COLOR sensor not found"}

    try:
        profile_list = color_sensor.get_stream_profile_list()
    except Exception as e:
        return {"error": str(e)}

    profiles = []
    for p_idx in range(profile_list.get_count()):
        try:
            profile = profile_list.get_stream_profile_by_index(p_idx)
            profiles.append({
                "format": obformat_to_string(profile.get_format()),
                "width":  profile.get_width(),
                "height": profile.get_height(),
                "fps":    profile.get_fps(),
            })
        except Exception as e:
            profiles.append({"error": str(e)})

    return {"profile_count": len(profiles), "profiles": profiles}


def collect_depth_stream_info(device):
    """单独收集 Depth 传感器支持的流格式"""
    sensor_list = device.get_sensor_list()
    depth_sensor = None
    for idx in range(sensor_list.get_count()):
        s = sensor_list.get_sensor_by_index(idx)
        if s.get_type() == OBSensorType.DEPTH_SENSOR:
            depth_sensor = s
            break

    if depth_sensor is None:
        return {"error": "DEPTH sensor not found"}

    try:
        profile_list = depth_sensor.get_stream_profile_list()
    except Exception as e:
        return {"error": str(e)}

    profiles = []
    for p_idx in range(profile_list.get_count()):
        try:
            profile = profile_list.get_stream_profile_by_index(p_idx)
            profiles.append({
                "format": obformat_to_string(profile.get_format()),
                "width":  profile.get_width(),
                "height": profile.get_height(),
                "fps":    profile.get_fps(),
            })
        except Exception as e:
            profiles.append({"error": str(e)})

    return {"profile_count": len(profiles), "profiles": profiles}


def collect_rgb_exposure_info(device):
    """收集 RGB 曝光相关属性的详细信息"""
    rgb_exposure_keywords = ['color_auto_exposure', 'color_exposure',
                              'color_gain', 'color_auto_white_balance',
                              'color_white_balance', 'color_brightness',
                              'color_sharpness', 'color_saturation',
                              'color_contrast', 'color_power_line_frequency']

    count = device.get_support_property_count()
    properties = []

    for i in range(count):
        try:
            item = device.get_supported_property(i)
        except Exception:
            continue

        name_lower = item.name.lower()
        # 只保留包含 color_ 的属性 (RGB 相关)
        if not name_lower.startswith("ob_prop_color_"):
            continue
        if item.permission == OBPermissionType.PERMISSION_DENY:
            continue
        if item.type not in [
            OBPropertyType.OB_BOOL_PROPERTY,
            OBPropertyType.OB_INT_PROPERTY,
            OBPropertyType.OB_FLOAT_PROPERTY,
        ]:
            continue

        entry = {
            "id":   int(item.id),
            "name": item.name,
            "permission": permission_type_to_string(item.permission),
            "type": property_type_to_string(item.type),
        }

        # 获取范围
        try:
            if item.type == OBPropertyType.OB_BOOL_PROPERTY:
                entry["values"] = [0, 1]
            elif item.type == OBPropertyType.OB_INT_PROPERTY:
                r = device.get_int_property_range(item.id)
                entry["range"] = {"min": r.min, "max": r.max, "step": r.step}
            elif item.type == OBPropertyType.OB_FLOAT_PROPERTY:
                r = device.get_float_property_range(item.id)
                entry["range"] = {"min": round(r.min, 4), "max": round(r.max, 4), "step": round(r.step, 4)}
        except Exception:
            pass

        # 读取当前值
        entry["current"] = read_property_value(device, item)

        properties.append(entry)

    properties.sort(key=lambda x: x["id"])
    return {"rgb_property_count": len(properties), "rgb_properties": properties}


def collect_all_properties(device):
    """收集所有可读写属性"""
    count = device.get_support_property_count()
    properties = []

    for i in range(count):
        try:
            item = device.get_supported_property(i)
        except Exception:
            continue

        if item.permission == OBPermissionType.PERMISSION_DENY:
            continue
        if item.type not in [
            OBPropertyType.OB_BOOL_PROPERTY,
            OBPropertyType.OB_INT_PROPERTY,
            OBPropertyType.OB_FLOAT_PROPERTY,
        ]:
            continue

        entry = {
            "id":   int(item.id),
            "name": item.name,
            "permission": permission_type_to_string(item.permission),
            "type": property_type_to_string(item.type),
        }

        # 获取范围
        try:
            if item.type == OBPropertyType.OB_BOOL_PROPERTY:
                entry["values"] = [0, 1]
            elif item.type == OBPropertyType.OB_INT_PROPERTY:
                r = device.get_int_property_range(item.id)
                entry["range"] = {"min": r.min, "max": r.max, "step": r.step}
            elif item.type == OBPropertyType.OB_FLOAT_PROPERTY:
                r = device.get_float_property_range(item.id)
                entry["range"] = {"min": round(r.min, 4), "max": round(r.max, 4), "step": round(r.step, 4)}
        except Exception:
            pass

        entry["current"] = read_property_value(device, item)
        properties.append(entry)

    properties.sort(key=lambda x: x["id"])
    return {"property_count": len(properties), "properties": properties}


# ========== 主函数 ==========

def main():
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "device_info.yaml")

    print("Orbbec 设备信息查询工具 (pyorbbecsdk2) — YAML 输出")
    print("-" * 64)

    ctx = Context()
    device_list = ctx.query_devices()
    dev_count = device_list.get_count()

    if dev_count < 1:
        print("错误: 未检测到任何 Orbbec 设备！请确认设备已连接。")
        sys.exit(1)

    print(f"已检测到 {dev_count} 台 Orbbec 设备:\n")
    for idx in range(dev_count):
        dev_info = device_list[idx].get_device_info()
        print(f"  [{idx}] {dev_info.get_name()} | PID=0x{dev_info.get_pid():04X} "
              f"| SN={dev_info.get_serial_number()} | FW={dev_info.get_firmware_version()}")

    print(f"\n正在查询第 1 台设备 (index=0) 的详细信息...")
    device = device_list.get_device_by_index(0)

    # 收集所有数据
    result = {
        "device_info":           collect_device_info(device),
        "sensor_profiles":       collect_sensor_profiles(device),
        "rgb_stream_info":       collect_rgb_stream_info(device),
        "depth_stream_info":     collect_depth_stream_info(device),
        "rgb_exposure_info":     collect_rgb_exposure_info(device),
        "all_properties":        collect_all_properties(device),
    }

    # 写入 YAML
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(result, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n设备信息已保存到: {output_path}")
    print("=" * 64)


if __name__ == "__main__":
    main()