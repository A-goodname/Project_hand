# -*- coding: utf-8 -*-
"""
OmniHand O10 手势控制模块
=========================
基于原项目 kinematics_solver 角度->编码器转换逻辑,
使用 omnihand_usb_serial.py 实现 USB/RS485 通信的手势控制。

关节映射 (10个主动关节):
  0: Thumb Roll       (拇指旋转)
  1: Thumb Ab/Ad      (拇指外展/内收)
  2: Thumb MCP        (拇指掌指关节)
  3: Index Ab/Ad      (食指外展/内收)
  4: Index PIP        (食指近端指间关节)
  5: Middle PIP       (中指近端指间关节)
  6: Ring Ab/Ad       (无名指外展/内收)
  7: Ring PIP         (无名指近端指间关节)
  8: Pinky Ab/Ad      (小指外展/内收)
  9: Pinky PIP        (小指近端指间关节)

角度单位: 弧度 (rad)
编码器范围: 0~4096

用法:
    from omnihand_gesture import OmniHandGesture, Gesture
    
    with OmniHandGesture(port="COM5") as hand:
        hand.enable()
        hand.do_gesture(Gesture.PAPER)    # 手掌展开
        hand.do_gesture(Gesture.FIST)     # 握拳
        hand.do_gesture(Gesture.MIDDLE)   # 竖中指
"""

import time
from enum import Enum
from typing import List, Dict, Optional, Tuple

try:
    from .omnihand_usb_serial import OmniHandSerial  # 作为包被导入时 (from Omnihand.xxx)
except ImportError:
    from omnihand_usb_serial import OmniHandSerial   # 直接运行脚本时 (python xxx.py)


# ============================================================================
# 手势枚举
# ============================================================================

class Gesture(Enum):
    """手势枚举"""
    RESET = 0       # 重置 (回到零位)
    PAPER = 1       # 手掌展开
    FIST = 2        # 握拳
    OK = 3          # OK手势
    HEART = 4       # 单手比心
    LIKE = 5        # 点赞
    ILY = 6         # I Love You 手势
    NUM1 = 7        # 数字1
    NUM2 = 8        # 数字2
    NUM3 = 9        # 数字3
    NUM4 = 10       # 数字4
    NUM6 = 11       # 数字6
    NUM8 = 12       # 数字8
    CLASPING = 13   # 合十
    MIDDLE = 14     # 竖中指 (Middle Finger)
    POINT = 15      # 指人
    GRIP = 16       # 握住 (如握住水瓶/水杯)
    EXIT = 99       # 退出


# ============================================================================
# 角度->编码器转换
# ============================================================================

class KinematicsConverter:
    """
    运动学转换器 (角度->编码器位置)
    复现 C++ OmnihandCtrl::ActiveJointPos2ActuatorInput() 逻辑
    """

    # ---- 右手参数 (与 C++ SDK 一致) ----
    ACTUATOR_MAX = [0, 0, 0, 0, 0, 0, 4096, 0, 0, 0]
    ACTUATOR_MIN = [4096, 4096, 4096, 4096, 4096, 4096, 0, 4096, 4096, 4096]

    # 关节角度限位 (rad)
    ACTIVE_JOINT_MAX = [1.12, 0.05, 0.8416, 0, 1.48, 1.48, 0.17, 1.48, 0.19, 1.48]
    ACTIVE_JOINT_MIN = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # 电机映射范围 (rad)
    MOTOR_MAX = [1.12, 0.05, 1.33, 0, 1.43, 1.43, 0.17, 1.43, 0.19, 1.43]
    MOTOR_MIN = [-0.03, -1.64, 0.0, -0.16, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    # ---- 多项式映射系数 ----
    # MCP->Motor: 4次多项式 (用于拇指/各手指)
    FINGER_MCP2MOTOR_POLY = [
        0.00944480234881967, 0.455882677008572, 0.683758090072141,
        -0.916673507519311, 0.459387725400186
    ]

    # 右手拇指 MCP->Motor
    RIGHT_THUMB_MCP2MOTOR_POLY = [
        0.00126371020922368, 0.919140692758276, 0.550958572722048,
        -0.785384985903032, 1.25635285116862
    ]

    # ---- 手势角度定义 (弧度) ----
    # 与 C++ OmnihandCtrl::SetHandGesture() 和 python/example/demo_set_motion.py 一致
    GESTURE_ANGLES: Dict[int, List[float]] = {
        Gesture.RESET.value:      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Gesture.PAPER.value:      [0.58, -0.21, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Gesture.FIST.value:       [0.50, -1.0, 0.75, 0.0, 1.48, 1.48, 0.0, 1.48, 0.0, 1.48],
        Gesture.OK.value:         [-0.03, -1.51, 0.7, -0.16, 0.85, 0.21, 0.07, 0.153, 0.107, 0.1],
        Gesture.HEART.value:      [0.8, -0.4, 0.47, 0.0, 0.82, 1.48, 0.0, 1.48, 0.0, 1.48],
        Gesture.LIKE.value:       [0.27, 0.0, 0.0, 0.0, 1.48, 1.48, 0.0, 1.48, 0.0, 1.48],
        Gesture.ILY.value:        [0.33, 0.0, 0.0, -0.1, 0.0, 1.48, 0.07, 1.48, 0.11, 0.0],
        Gesture.NUM1.value:       [0.32, -1.12, 0.79, -0.06, 0.0, 1.48, 0.0, 1.48, 0.0, 1.48],
        Gesture.NUM2.value:       [0.48, -1.5, 0.79, -0.16, 0.0, 0.0, 0.0, 1.48, 0.0, 1.48],
        Gesture.NUM3.value:       [0.64, -1.48, 0.81, -0.16, 0.0, 0.0, 0.09, 0.0, 0.09, 1.48],
        Gesture.NUM4.value:       [0.64, -1.48, 0.81, -0.16, 0.0, 0.0, 0.07, 0.0, 0.15, 0.0],
        Gesture.NUM6.value:       [0.40, 0.0, 0.0, 0.0, 1.48, 1.48, 0.05, 1.48, 0.17, 0.0],
        Gesture.NUM8.value:       [0.40, 0.0, 0.0, 0.0, 0.0, 1.48, 0.0, 1.48, 0.0, 1.48],
        Gesture.CLASPING.value:   [0.5, -0.8, 0.2, -0.16, 0.6, 0.6, 0.17, 0.6, 0.17, 0.6],
        # 竖中指: 握拳但中指伸直 (joint 5 = Middle PIP = 0.0)
        Gesture.MIDDLE.value:     [0.5, -1.0, 0.75, 0.0, 1.48, 0.0, 0.0, 1.48, 0.0, 1.48],
        # 指人: 食指伸出，其他握拳
        Gesture.POINT.value:      [0.27, 0.0, 0.0, 0.0, 0.0, 1.48, 0.0, 1.48, 0.0, 1.48],
        # 握住 (如水瓶/水杯): 手指大幅度弯曲包裹 + 拇指环绕对握, 形成圆柱形抓握
        Gesture.GRIP.value:       [0.50, -0.6, 0.6, 0.0, 1.3, 1.3, 0.0, 1.3, 0.0, 1.2],
    }

    @staticmethod
    def calculate_power(x: float, coeffs: List[float]) -> float:
        """多项式计算: y = c0 + c1*x + c2*x^2 + c3*x^3 + ..."""
        result = 0.0
        power = 1.0
        for c in coeffs:
            result += c * power
            power *= x
        return result

    @classmethod
    def clamp(cls, value: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, value))

    @classmethod
    def joint_angle_to_actuator(cls, angles: List[float],
                                 is_right_hand: bool = True) -> List[int]:
        """
        角度列表 -> 编码器位置列表
        对应 C++ OmnihandCtrl::ActiveJointPos2ActuatorInput()
        
        参数:
            angles: 10个主动关节角度 (弧度)
            is_right_hand: 右手 (False 为左手)
            
        返回:
            10个编码器位置 (0~4096)
        """
        if len(angles) != 10:
            raise ValueError(f"需10个关节角度, 收到 {len(angles)}")

        # 1. 限位
        clamped = [cls.clamp(a, cls.ACTIVE_JOINT_MIN[i], cls.ACTIVE_JOINT_MAX[i])
                   for i, a in enumerate(angles)]

        # 2. 多项式映射: 拇指MCP、各手指PIP
        hand_pos = list(clamped)

        # 拇指 MCP (joint 2) -> 拇指电机
        if is_right_hand:
            hand_pos[2] = cls.calculate_power(
                clamped[2], cls.RIGHT_THUMB_MCP2MOTOR_POLY)
        else:
            # 左手系数 (符号取反)
            left_poly = [-c if i % 2 == 1 else c
                         for i, c in enumerate(cls.RIGHT_THUMB_MCP2MOTOR_POLY)]
            hand_pos[2] = cls.calculate_power(clamped[2], left_poly)

        # 各手指 PIP -> 电机映射
        for idx in [4, 5, 7, 9]:  # Index, Middle, Ring, Pinky PIP
            hand_pos[idx] = cls.calculate_power(
                clamped[idx], cls.FINGER_MCP2MOTOR_POLY)

        # 3. 线性插值: 电机范围->编码器范围
        actuator = []
        for i in range(10):
            ratio = ((hand_pos[i] - cls.MOTOR_MIN[i]) /
                     (cls.MOTOR_MAX[i] - cls.MOTOR_MIN[i]))
            val = int(ratio * (cls.ACTUATOR_MAX[i] - cls.ACTUATOR_MIN[i])
                      + cls.ACTUATOR_MIN[i])
            # 限位到 0~4096
            val = max(0, min(4096, val))
            actuator.append(val)

        return actuator

    # ---- 直接编码器位置 (已知标定值) ----
    # 来自 test.json 的实际设备位姿
    DIRECT_POSITIONS: Dict[int, List[int]] = {
        Gesture.PAPER.value: [
            4095,  # J0 Thumb Roll (axis_1)
            4095,  # J1 Thumb Ab/Ad (axis_2)
            4095,  # J2 Thumb MCP (axis_3)
            3258,  # J3 Index Ab/Ad (axis_4)
            4095,  # J4 Index PIP (axis_5)
            4095,  # J5 Middle PIP (axis_6)
            1484,  # J6 Ring Ab/Ad (axis_7)
            4095,  # J7 Ring PIP (axis_8)
            1861,  # J8 Pinky Ab/Ad (axis_9)
            4095,  # J9 Pinky PIP (axis_10)
        ],
        Gesture.GRIP.value: [
            4095,  # J0 Thumb Roll (axis_1)
            4015,  # J1 Thumb Ab/Ad (axis_2)
            691,  # J2 Thumb MCP (axis_3)
            0,  # J3 Index Ab/Ad (axis_4)
            1728,  # J4 Index PIP (axis_5)
            1782,  # J5 Middle PIP (axis_6)
            89,  # J6 Ring Ab/Ad (axis_7)
            1728,  # J7 Ring PIP (axis_8)
            3965,  # J8 Pinky Ab/Ad (axis_9)
            1622,  # J9 Pinky PIP (axis_10)
        ],
    }

    @classmethod
    def get_gesture_positions(cls, gesture: Gesture,
                              is_right_hand: bool = True) -> List[int]:
        """
        获取指定手势的 10个编码器位置值
        
        优先使用 DIRECT_POSITIONS (已知标定值),
        否则通过角度转换计算得到。
        
        返回:
            List[int]: 10 个 0~4096 的编码器位置
        """
        # 优先使用直接编码器位置 (如 GRIP 来自 test.json Close 标定值)
        direct = cls.DIRECT_POSITIONS.get(gesture.value)
        if direct is not None:
            return list(direct)
        
        # 否则通过角度转换计算
        angles = cls.GESTURE_ANGLES.get(gesture.value)
        if angles is None:
            raise ValueError(f"未知手势: {gesture}")
        return cls.joint_angle_to_actuator(angles, is_right_hand)


# ============================================================================
# 手势中文名
# ============================================================================

GESTURE_NAMES: Dict[Gesture, str] = {
    Gesture.RESET:   "重置位置",
    Gesture.PAPER:   "手掌展开",
    Gesture.FIST:    "握拳",
    Gesture.OK:      "OK手势",
    Gesture.HEART:   "单手比心",
    Gesture.LIKE:    "点赞",
    Gesture.ILY:     "ILY手势",
    Gesture.NUM1:    "数字1",
    Gesture.NUM2:    "数字2",
    Gesture.NUM3:    "数字3",
    Gesture.NUM4:    "数字4",
    Gesture.NUM6:    "数字6",
    Gesture.NUM8:    "数字8",
    Gesture.CLASPING: "合十",
    Gesture.MIDDLE:  "竖中指",
    Gesture.POINT:   "指人",
    Gesture.GRIP:    "握住",
}


# ============================================================================
# 手势控制类
# ============================================================================

class OmniHandGesture:
    """
    灵巧手手势控制封装
    
    基于 KinematicsConverter 将手势角度转换为编码器位置,
    通过 OmniHandSerial 发送到灵巧手。

    使用示例::
    
        with OmniHandGesture(port="COM5") as hand:
            hand.enable()
            # 单个手势
            hand.do_gesture(Gesture.PAPER)
            time.sleep(1)
            # 手势序列
            hand.do_gesture_sequence([
                (Gesture.FIST, 1.0),
                (Gesture.PAPER, 1.0),
                (Gesture.MIDDLE, 2.0),
            ])
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        node_id: int = 0x01,
        is_right_hand: bool = True,
        verbose: bool = True,
    ):
        self.serial = OmniHandSerial(
            port=port,
            baudrate=baudrate,
            node_id=node_id,
            verbose=verbose,
        )
        self.is_right_hand = is_right_hand
        self._enabled = False

    # ---- 串口生命周期 ----

    def open(self):
        """打开串口"""
        self.serial.open()

    def close(self):
        """关闭串口"""
        try:
            if self._enabled:
                self.disable()
        except Exception:
            pass
        self.serial.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- 设备状态 ----

    def enable(self) -> bool:
        """使能设备"""
        self._enabled = self.serial.enable()
        time.sleep(1.0)
        return self._enabled

    def disable(self) -> bool:
        """失能设备"""
        self._enabled = False
        return self.serial.disable()

    def calibrate(self) -> bool:
        """进入校准模式"""
        return self.serial.calibrate()

    def is_enabled(self) -> bool:
        """查询使能状态"""
        return self.serial.query_enable() == 1

    # ---- 位置控制 ----

    def set_positions(self, positions: List[int],
                      delay: float = 1.0) -> dict:
        """
        设置全部关节位置
        
        参数:
            positions: 10 个编码器位置 (0~4096)
            delay: 设置后的等待时间 (秒)
            
        返回:
            反馈数据字典
        """
        result = self.serial.set_all_positions(positions)
        if delay > 0:
            time.sleep(delay)
        return result

    def read_positions(self) -> List[int]:
        """读取全部关节位置"""
        return self.serial.read_all_positions()

    def read_current(self) -> List[float]:
        """读取全部关节电流 (A)"""
        return self.serial.read_all_current_amp()

    def read_temp(self) -> List[int]:
        """读取全部关节温度 (C)"""
        return self.serial.read_all_temp()

    # ---- 手势执行 ----

    def do_gesture(self, gesture: Gesture, delay: float = 1.5) -> dict:
        """
        执行指定手势
        
        参数:
            gesture: 手势枚举
            delay: 执行后等待时间 (秒)
            
        返回:
            反馈数据字典
        """
        positions = KinematicsConverter.get_gesture_positions(
            gesture, self.is_right_hand)
        
        name = GESTURE_NAMES.get(gesture, gesture.name)
        print(f"  [手势] {name}")
        print(f"  [位置] {positions}")
        
        result = self.set_positions(positions, delay=delay)
        return result

    def do_gesture_sequence(self, sequence: List[Tuple[Gesture, float]]):
        """
        执行手势序列
        
        参数:
            sequence: [(手势, 持续时间), ...]
        """
        for gesture, duration in sequence:
            self.do_gesture(gesture, delay=duration)

    def do_multi_cycle(self, gesture: Gesture, cycles: int = 3,
                       hold_time: float = 1.0, release_time: float = 1.0):
        """
        重复执行一个手势多次 (循环张开-做手势)
        
        参数:
            gesture: 手势枚举
            cycles: 循环次数
            hold_time: 手势保持时间
            release_time: 复位等待时间
        """
        print(f"\n[循环] 执行 {GESTURE_NAMES.get(gesture, gesture.name)} {cycles} 次")
        for i in range(cycles):
            print(f"  --- 第 {i + 1}/{cycles} 次 ---")
            # 先复位
            self.do_gesture(Gesture.RESET, delay=release_time)
            # 做手势
            self.do_gesture(gesture, delay=hold_time)

    # ---- 状态查询 ----

    def get_status(self) -> dict:
        """获取完整设备状态"""
        info = self.serial.get_device_info()
        error = self.serial.read_error()
        state = self.serial.query_enable()
        return {
            "info": info,
            "enable_state": state,
            "error_code": error,
            "error_text": OmniHandSerial.error_to_text(error),
        }

    def print_status(self):
        """打印设备状态"""
        status = self.get_status()
        info = status["info"]
        print(f"  设备: {info.get('device_type_name', '?')} "
              f"(v{info.get('software_version', '?')})")
        print(f"  序列号: {info.get('serial_number', 'N/A')}")
        state_map = {0: "失能", 1: "使能", 2: "校准"}
        print(f"  状态: {state_map.get(status['enable_state'], '?')}")
        print(f"  错误: {status['error_code']} "
              f"({status['error_text']})")


# ============================================================================
# 交互式菜单
# ============================================================================

def print_menu():
    """打印手势菜单"""
    print("\n" + "=" * 50)
    print("     OmniHand O10 手势控制菜单")
    print("=" * 50)
    items = [
        (0, "重置位置"),
        (1, "手掌展开"),
        (2, "握拳"),
        (3, "OK手势"),
        (4, "单手比心"),
        (5, "点赞"),
        (6, "ILY手势"),
        (7, "数字1"),
        (8, "数字2"),
        (9, "数字3"),
        (10, "数字4"),
        (11, "数字6"),
        (12, "数字8"),
        (13, "合十"),
        (14, "竖中指"),
        (15, "指人"),
        (16, "握住"),
        (99, "退出程序"),
    ]
    for num, name in items:
        print(f"  {num:2d}. {name}")
    print()

    # 额外选项
    print("  特殊命令:")
    print("    s  - 显示设备状态")
    print("    p  - 读取当前位置")
    print("    c  - 读取电流")
    print("    t  - 读取温度")
    print()


def interactive_mode(port: str = "COM5"):
    """
    交互式手势控制模式
    
    参数:
        port: 串口号
    """
    print("OmniHand O10 手势控制 (交互模式)")
    print(f"串口: {port}")

    with OmniHandGesture(port=port, verbose=False) as hand:
        # 查询设备信息
        print("\n正在连接灵巧手...")
        try:
            hand.serial.flush_input()
            status = hand.get_status()
            info = status["info"]
            print(f"  设备: {info.get('device_type_name', '?')} "
                  f"(v{info.get('software_version', '?')})")
            print(f"  序列号: {info.get('serial_number', 'N/A')}")
            print(f"  自由度: {info.get('dof', '?')}")
        except Exception as e:
            print(f"  [警告] 查询设备信息失败: {e}")
            print("  确认设备已连接并已上电")

        # 使能
        try:
            if hand.is_enabled():
                print("  [设备已使能]")
                hand._enabled = True
            else:
                print("\n正在使能设备...")
                if not hand.enable():
                    print("  [错误] 使能失败")
                    return
                print("  [设备已使能]")
        except Exception as e:
            print(f"  [错误] 使能失败: {e}")
            return

        while True:
            print_menu()
            try:
                cmd = input("请选择手势 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                break

            # 特殊命令
            if cmd == "s":
                hand.print_status()
                continue
            elif cmd == "p":
                pos = hand.read_positions()
                print(f"  当前位置: {pos}")
                continue
            elif cmd == "c":
                cur = hand.read_current()
                print(f"  各关节电流: {[f'{c:.2f}A' for c in cur]}")
                continue
            elif cmd == "t":
                temp = hand.read_temp()
                print(f"  各关节温度: {[f'{t}C' for t in temp]}")
                continue

            # 数字命令
            try:
                choice = int(cmd)
            except ValueError:
                print("  无效输入")
                continue

            if choice == 99:
                print("  退出程序")
                break

            try:
                gesture = Gesture(choice)
                hand.do_gesture(gesture, delay=1.5)
            except (ValueError, KeyError):
                print(f"  无效手势编号: {choice}")

        # 清理
        print("\n正在失能设备...")
        try:
            hand.disable()
            print("  设备已失能")
        except Exception:
            pass


# ============================================================================
# 测试 Demo
# ============================================================================

def demo(port: str = "COM5"):
    """
    手势演示: 依次展示各手势
    """
    print("=" * 50)
    print("  OmniHand O10 手势演示")
    print("=" * 50)

    with OmniHandGesture(port=port, verbose=True) as hand:
        # 连接
        print("\n[1/4] 连接并查询设备...")
        hand.serial.flush_input()
        status = hand.get_status()
        hand.print_status()

        # 使能
        print("\n[2/4] 使能设备...")
        if not hand.enable():
            print("  [错误] 使能失败")
            return
        print("  [OK]")

        # 手势展示
        print("\n[3/4] 手势展示...")
        
        demo_gestures = [
            Gesture.PAPER,
            Gesture.FIST,
            Gesture.OK,
            Gesture.LIKE,
            Gesture.ILY,
            Gesture.NUM1,
            Gesture.NUM2,
            Gesture.NUM3,
            Gesture.HEART,
            Gesture.CLASPING,
            Gesture.MIDDLE,
        ]

        for gesture in demo_gestures:
            hand.do_gesture(gesture, delay=1.5)

        # 复位
        print("\n  复位...")
        hand.do_gesture(Gesture.RESET, delay=0.5)

        # 清理
        print("\n[4/4] 失能并关闭...")
        hand.disable()
        print("  完成!")


if __name__ == "__main__":
    import sys

    port = "COM5"
    if len(sys.argv) > 1 and sys.argv[1] == "--port":
        port = sys.argv[2] if len(sys.argv) > 2 else port
    elif len(sys.argv) > 1 and sys.argv[1] in ("--demo", "-d"):
        port = sys.argv[2] if len(sys.argv) > 2 else port
        demo(port)
        sys.exit(0)

    # 检查是否有 COM 参数
    for arg in sys.argv[1:]:
        if arg.upper().startswith("COM") or arg.startswith("/dev/"):
            port = arg

    interactive_mode(port=port)
