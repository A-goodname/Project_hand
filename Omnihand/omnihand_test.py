# -*- coding: utf-8 -*-
"""
OmniHand O10 灵巧手 USB/RS485 通信 Demo
========================================
使用前请先安装依赖:
    pip install pyserial

修改下方 PORT 为实际的串口号:
    - Windows: "COM5" (查看设备管理器)
    - Linux:   "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00" (by-id 稳定路径, 禁 ttyACM 编号)

安全注意事项:
    - 首次使用请确认灵巧手已正确连接电源
    - 使能 (enable) 前确保手部处于安全位置,防止意外动作
    - 按下 Ctrl+C 可随时中断程序,会执行失能操作
"""

import sys
import os
import time
import traceback

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# Windows GBK 编码兼容: 替换 emoji 为普通字符
if sys.platform == "win32" and os.environ.get("PYTHONIOENCODING") is None:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    CheckMark = "[OK]"
    CrossMark = "[FAIL]"
else:
    CheckMark = "\u2705"
    CrossMark = "\u274c"


# ============================ 配置 ============================
# 请根据实际情况修改以下配置

PORT = "/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00"  # 串口 (本机 Linux 稳定路径, 其他机器按实机枚举修改)
BAUDRATE = 460800       # 波特率
NODE_ID = 0x01          # 节点ID (默认 0x01)
VERBOSE = True          # 是否打印详细日志

# 关节中间位置 (0~4096, 2048 为中间位置)
HOME_POS = 2048

# ============================ 导入 ============================

from Omnihand.omnihand_usb_serial import (
    OmniHandSerial,
    OmniHandError,
    TimeoutError,
    ProtocolError,
)


# ============================ 辅助函数 ============================

def separator(title: str, char: str = "="):
    """打印分隔线"""
    print(f"\n{char * 20} {title} {char * 20}")


def print_positions(label: str, positions, unit: str = ""):
    """格式化打印 10 个关节位置"""
    parts = [f"{label}:"]
    for i, p in enumerate(positions):
        parts.append(f"  J{i + 1}={p}{unit}")
    print("".join(parts))


def print_sensor_matrix(label: str, matrix):
    """打印传感器矩阵"""
    print(f"  {label}:")
    for row in matrix:
        print(f"    {row}")


def print_device_info(info: dict):
    """打印设备信息"""
    print(f"  设备类型:   {info.get('device_type_name', 'N/A')}")
    print(f"  产品状态:   {info.get('product_status', 'N/A')}")
    print(f"  软件版本:   {info.get('software_version', 'N/A')}")
    print(f"  硬件版本:   {info.get('hardware_version', 'N/A')}")
    if info.get("dof") is not None:
        print(f"  自由度:     {info['dof']}")
    if info.get("serial_number"):
        print(f"  序列号:     {info['serial_number']}")


# ============================ 主函数 ============================

def main():
    hand = OmniHandSerial(
        port=PORT,
        baudrate=BAUDRATE,
        node_id=NODE_ID,
        verbose=VERBOSE,
    )

    try:
        # ---- 打开串口 ----
        separator("步骤 1/6: 打开串口")
        print(f"正在打开串口 {PORT} @ {BAUDRATE}bps ...")
        hand.open()
        print("[OK] 串口已打开")
        hand.flush_input()

        # ---- 查询设备信息 ----
        separator("步骤 2/6: 查询设备信息")
        try:
            info = hand.get_device_info()
            print_device_info(info)
        except (TimeoutError, ProtocolError) as e:
            print(f"[WARN] 查询设备信息失败: {e}")
            print("[INFO] 请确认串口号/波特率是否正确,灵巧手是否已连接")
            return

        # ---- 查询使能状态 ----
        separator("步骤 3/6: 查询使能状态")
        state = hand.query_enable()
        state_map = {0: "失能", 1: "使能", 2: "校准"}
        print(f"  当前状态: {state_map.get(state, f'未知({state})')}")

        # ---- 读取各关节状态 ----
        separator("步骤 4/6: 读取关节状态")

        # 4a. 读取位置
        try:
            positions = hand.read_all_positions()
            print_positions("当前位置", positions)
        except (TimeoutError, ProtocolError) as e:
            print(f"[WARN] 读取位置失败: {e}")

        # 4b. 读取电流
        try:
            current = hand.read_all_current_amp()
            print_positions("各关节电流", current, unit="A")
        except (TimeoutError, ProtocolError) as e:
            print(f"[WARN] 读取电流失败: {e}")

        # 4c. 读取温度
        try:
            temps = hand.read_all_temp()
            print_positions("各关节温度", temps, unit="°C")
        except (TimeoutError, ProtocolError) as e:
            print(f"[WARN] 读取温度失败: {e}")

        # 4d. 读取错误
        try:
            err = hand.read_error()
            print(f"  错误码: {err} ({hand.error_to_text(err)})")
        except (TimeoutError, ProtocolError) as e:
            print(f"[WARN] 读取错误失败: {e}")

        # ---- 使能并控制 ----
        separator("步骤 5/6: 位置控制演示")

        if state != 1:
            print("  正在使能设备...")
            success = hand.enable()
            print(f"  [{'OK' if success else 'FAIL'}] 使能结果")
            time.sleep(1.0)

        # 5a. 单轴位置控制
        print("\n  演示 1: 单轴位置控制 (关节 1 -> 1500)")
        try:
            axis, fb = hand.set_single_position(1, 1500)
            print(f"    命令: 关节{axis} -> 1500, 反馈: {fb}")
            time.sleep(1.5)
            # 归位
            axis, fb = hand.set_single_position(1, HOME_POS)
            print(f"    命令: 关节{axis} -> {HOME_POS}, 反馈: {fb}")
            time.sleep(1.0)
        except (TimeoutError, ProtocolError) as e:
            print(f"    [ERROR] 单轴控制失败: {e}")

        # 5b. 全轴位置控制
        print("\n  演示 2: 全轴位置控制")
        try:
            # 设置所有关节到中间偏右位置
            targets = [HOME_POS + 200] * 10
            result = hand.set_all_positions(targets)
            print("    反馈位置:", [f"J{i + 1}={result['position_feedback'][i]}"
                                    for i in range(10)])
            time.sleep(1.5)

            # 归位到中间
            targets = [HOME_POS] * 10
            result = hand.set_all_positions(targets)
            print("    归位反馈:", [f"J{i + 1}={result['position_feedback'][i]}"
                                    for i in range(10)])
            time.sleep(1.0)
        except (TimeoutError, ProtocolError) as e:
            print(f"    [ERROR] 全轴控制失败: {e}")

        # 5c. 读取更新后的位置
        print("\n  演示 3: 读取更新后的位置")
        try:
            positions = hand.read_all_positions()
            print_positions("读取位置", positions)
        except (TimeoutError, ProtocolError) as e:
            print(f"    [ERROR] 读取位置失败: {e}")

        # ---- 触觉传感器 (可选, 根据硬件支持情况) ----
        separator("步骤 6/6: 触觉传感器 (可选)")

        try:
            # 读取单个传感器 (指尖)
            print("  演示 1: 读取拇指传感器")
            sensor = hand.read_single_sensor(0)  # 0=拇指
            print(f"    传感器: {sensor['sensor_label']}")
            print(f"    类型: {sensor['type']}")
            print_sensor_matrix("压力矩阵", sensor["matrix"])

            # 读取传感器组1 (拇指+食指+中指)
            print("\n  演示 2: 批量读取传感器组1")
            group1 = hand.read_all_sensor_group_1()
            for name, matrix in group1.items():
                print_sensor_matrix(name, matrix)
        except (TimeoutError, ProtocolError) as e:
            print(f"  [WARN] 读取传感器失败 (可能不支持): {e}")

        # ---- 清理 ----
        separator("清理: 失能并关闭")
        try:
            if hand.query_enable() == 1:
                hand.disable()
                print("[OK] 设备已失能")
        except Exception:
            pass

        print(f"\n{CheckMark} Demo 运行完成!")

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断, 进行安全清理...")
        try:
            hand.disable()
            print("[OK] 设备已失能")
        except Exception:
            pass

    except Exception as e:
        print(f"\n[ERROR] 发生异常: {e}")
        traceback.print_exc()
        try:
            hand.disable()
            print("[OK] 设备已失能 (异常清理)")
        except Exception:
            pass

    finally:
        try:
            hand.close()
            print("[OK] 串口已关闭")
        except Exception:
            pass


# ============================ 简易快速测试 ============================

def quick_test():
    """
    极简快速测试 - 仅做基础通信验证
    适用场景: 首次连接确认通信是否正常
    """
    print("=" * 50)
    print("OmniHand O10 快速通信测试")
    print("=" * 50)

    hand = OmniHandSerial(
        port=PORT,
        baudrate=BAUDRATE,
        node_id=NODE_ID,
        verbose=True,
    )

    try:
        hand.open()
        hand.flush_input()

        # 查询设备信息
        info = hand.get_device_info()
        print(f"\n{CheckMark} 通信成功!")
        print(f"   设备:   {info['device_type_name']}")
        print(f"   软件:   v{info['software_version']}")
        print(f"   硬件:   v{info['hardware_version']}")
        print(f"   序列号: {info.get('serial_number', 'N/A')}")

    except Exception as e:
        print(f"\n{CrossMark} 通信失败: {e}")
        print("   请检查:")
        print("   1. 串口号是否正确")
        print("   2. 灵巧手是否已连接电源")
        print("   3. USB-串口驱动是否已安装")

    finally:
        hand.close()


if __name__ == "__main__":
    separator("OmniHand O10 USB/RS485 Demo", "★")

    if len(sys.argv) > 1 and sys.argv[1] in ("--quick", "-q"):
        quick_test()
    elif len(sys.argv) > 1 and sys.argv[1] == "--port":
        # python test.py --port /dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00
        PORT = sys.argv[2] if len(sys.argv) > 2 else PORT
        main()
    else:
        print("用法:")
        print("  python test.py                # 运行完整Demo (需修改上方PORT)")
        print("  python test.py --port /dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00    # 指定串口运行完整Demo")
        print("  python test.py --quick        # 快速通信测试")
        print("  python test.py -q             # 同上")
        print("")
        main()
