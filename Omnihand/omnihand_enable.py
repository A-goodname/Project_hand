# -*- coding: utf-8 -*-
"""
交互式使能/失能/校准控制
"""
try:
    from .omnihand_usb_serial import OmniHandSerial  # 作为包被导入时
except ImportError:
    from omnihand_usb_serial import OmniHandSerial   # 直接运行脚本时
import time

PORT = "COM5"           # 串口号 (Windows: COMx, Linux: /dev/ttyUSB0)
BAUDRATE = 460800       # 波特率
NODE_ID = 0x01          # 节点ID (默认 0x01)
VERBOSE = True          # 是否打印详细日志

ENABLE_STATUS_TEXT = {
    0: "失能 (Disabled)",
    1: "使能 (Enabled)",
    2: "校准 (Calibration)",
}


def check_enable(hand: OmniHandSerial):
    """查询并打印使能状态"""
    try:
        status = hand.query_enable()
        text = ENABLE_STATUS_TEXT.get(status, f"未知({status})")
        print(f"  → 使能状态: {status} ({text})")
        return status
    except Exception as e:
        print(f"  → 查询失败: {e}")
        return -1


def print_menu():
    print()
    print("=" * 50)
    print("  命令:")
    print("    1  → 使能 (ENABLE)")
    print("    0  → 失能 (DISABLE)")
    print("    2  → 校准 (CALIBRATE)")
    print("    q  → 退出")
    print("=" * 50)


if __name__ == "__main__":
    hand = OmniHandSerial(
        port=PORT,
        baudrate=BAUDRATE,
        node_id=NODE_ID,
        verbose=VERBOSE,
    )
    hand.open()
    try:
        # 查询初始状态
        print("\n  查询初始使能状态")
        check_enable(hand)

        while True:
            print_menu()
            cmd = input("  请输入命令: ").strip().lower()

            if cmd == "q":
                print("  退出")
                break
            elif cmd == "0":
                hand.disable()
                print("  disable() 命令已发送")
            elif cmd == "1":
                hand.enable()
                print("  enable() 命令已发送")
            elif cmd == "2":
                hand.calibrate()
                print("  calibrate() 命令已发送")
            else:
                print(f"  未知命令: {cmd}")
                continue

            # 等待设备状态切换
            time.sleep(0.3)

            # 查询实际使能状态
            check_enable(hand)

    except KeyboardInterrupt:
        print("\n  用户中断")
    except Exception as e:
        print(f"\n  发生错误: {e}")
    finally:
        hand.close()