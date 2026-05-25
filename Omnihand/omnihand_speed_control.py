# -*- coding: utf-8 -*-
"""
OmniHand O10 速度控制模块 — 软件插值版
========================================
基于 omnihand_usb_serial.py 和 omnihand_gesture.py 实现灵巧手的
软件插值速度控制（硬件速度控制在此设备上不可用）。

速度等级 (3级):
    VERY_SLOW < MEDIUM < VERY_FAST

用法::

    from speed_control import OmniHandSpeedController, SpeedLevel

    with OmniHandSpeedController(port="COM5") as ctrl:
        ctrl.enable()
        # 软件插值平滑控制
        ctrl.do_gesture_with_speed(Gesture.FIST, speed_level=SpeedLevel.VERY_SLOW)
        ctrl.do_gesture_with_speed(Gesture.PAPER, speed_level=SpeedLevel.MEDIUM)
"""

import time
import math
from enum import Enum
from typing import List, Dict, Optional, Tuple, Callable

try:
    from .omnihand_usb_serial import (
        OmniHandSerial,
        OmniHandError,
        TimeoutError,
        ProtocolError,
    )
except ImportError:
    from omnihand_usb_serial import (
        OmniHandSerial,
        OmniHandError,
        TimeoutError,
        ProtocolError,
    )

try:
    from .omnihand_gesture import (
        OmniHandGesture,
        Gesture,
        KinematicsConverter,
        GESTURE_NAMES,
    )
except ImportError:
    from omnihand_gesture import (
        OmniHandGesture,
        Gesture,
        KinematicsConverter,
        GESTURE_NAMES,
    )


# ============================================================================
# 速度等级枚举 (3级)
# ============================================================================

class SpeedLevel(Enum):
    """
    速度等级枚举 (3级)

    速度值对应 0x32 命令中的 speeds 参数 (范围 -4095~4096):
    - 绝对值越大, 关节运动越快
    - 负值表示反向运动
    """
    VERY_SLOW = 0     # 极慢
    MEDIUM = 1        # 中速 (默认)
    VERY_FAST = 2     # 极速


# 速度等级 -> 软件插值参数映射
SPEED_EASING_MAP: Dict[SpeedLevel, Dict] = {
    SpeedLevel.VERY_SLOW: {"duration": 3.0, "steps": 60, "easing": "ease_in_out"},
    SpeedLevel.MEDIUM:    {"duration": 1.0, "steps": 20, "easing": "ease_in_out"},
    SpeedLevel.VERY_FAST: {"duration": 0.3, "steps": 6,  "easing": "linear"},
}

# 速度等级中文名
SPEED_NAMES: Dict[SpeedLevel, str] = {
    SpeedLevel.VERY_SLOW: "极慢",
    SpeedLevel.MEDIUM:    "中速",
    SpeedLevel.VERY_FAST: "极速",
}

# 演示用动作列表: (名称, 手势, 中文描述)
DEMO_GESTURES: List[Tuple[str, Gesture, str]] = [
    ("home", Gesture.RESET, "复位 (回到零位)"),
    ("张开", Gesture.PAPER, "手掌张开"),
    ("握住", Gesture.FIST, "握拳"),
]


# ============================================================================
# 缓动函数 (Easing Functions)
# ============================================================================

def ease_linear(t: float) -> float:
    """线性插值"""
    return t


def ease_in(t: float) -> float:
    """加速 (先慢后快)"""
    return t * t


def ease_out(t: float) -> float:
    """减速 (先快后慢)"""
    return t * (2 - t)


def ease_in_out(t: float) -> float:
    """先加速后减速"""
    if t < 0.5:
        return 2 * t * t
    else:
        return -1 + (4 - 2 * t) * t


# 缓动函数映射
EASING_FUNCTIONS: Dict[str, Callable[[float], float]] = {
    "linear": ease_linear,
    "ease_in": ease_in,
    "ease_out": ease_out,
    "ease_in_out": ease_in_out,
}


# ============================================================================
# 速度控制类
# ============================================================================

class OmniHandSpeedController:
    """
    灵巧手速度控制封装 — 纯软件插值版

    继承 OmniHandGesture 的所有功能, 并增加软件插值速度控制接口。
    通过将行程拆分为多个中间帧逐步下发, 实现平滑的速度控制。

    使用示例::

        ctrl = OmniHandSpeedController(port="COM5")
        ctrl.open()
        ctrl.enable()

        # 软件插值: 以不同速度执行手势
        ctrl.do_gesture_with_speed(Gesture.FIST, speed_level=SpeedLevel.VERY_SLOW)

        # 直接插值: 从当前位置平滑过渡到目标位置
        ctrl.set_positions_interpolated(target_pos, duration=3.0)

        ctrl.disable()
        ctrl.close()
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        node_id: int = 0x01,
        is_right_hand: bool = True,
        verbose: bool = True,
    ):
        self.gesture = OmniHandGesture(
            port=port,
            baudrate=baudrate,
            node_id=node_id,
            is_right_hand=is_right_hand,
            verbose=verbose,
        )
        self.serial = self.gesture.serial
        self.is_right_hand = is_right_hand
        self._enabled = False

    # ---- 串口生命周期 ----

    def open(self):
        """打开串口"""
        self.gesture.open()

    def close(self):
        """关闭串口"""
        self.gesture.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *args):
        self.close()

    # ---- 设备状态 ----

    def enable(self) -> bool:
        """使能设备"""
        self._enabled = self.gesture.enable()
        return self._enabled

    def disable(self) -> bool:
        """失能设备"""
        self._enabled = False
        return self.gesture.disable()

    def calibrate(self) -> bool:
        """进入校准模式"""
        return self.gesture.calibrate()

    def is_enabled(self) -> bool:
        """查询使能状态"""
        return self.gesture.is_enabled()

    def read_positions(self) -> List[int]:
        """读取全部关节位置"""
        return self.gesture.read_positions()

    def read_current(self) -> List[float]:
        """读取全部关节电流 (A)"""
        return self.gesture.read_current()

    def read_temp(self) -> List[int]:
        """读取全部关节温度 (C)"""
        return self.gesture.read_temp()

    def get_status(self) -> dict:
        """获取完整设备状态"""
        return self.gesture.get_status()

    def print_status(self):
        """打印设备状态"""
        self.gesture.print_status()

    # ================================================================
    # 软件插值平滑控制 (唯一控制方式)
    # ================================================================

    @staticmethod
    def interpolate_positions(
        start: List[int],
        end: List[int],
        t: float,
        easing: str = "linear",
    ) -> List[int]:
        """
        插值两个位置之间

        参数:
            start: 起始位置 (10个编码器值)
            end: 目标位置 (10个编码器值)
            t: 插值进度 0.0~1.0
            easing: 缓动函数 "linear"/"ease_in"/"ease_out"/"ease_in_out"

        返回:
            插值后的10个编码器值
        """
        fn = EASING_FUNCTIONS.get(easing, ease_linear)
        et = fn(max(0.0, min(1.0, t)))

        result = []
        for s, e in zip(start, end):
            val = int(s + (e - s) * et)
            val = max(0, min(4096, val))
            result.append(val)
        return result

    def set_positions_interpolated(
        self,
        positions: List[int],
        duration: float = 1.0,
        steps: int = 20,
        easing: str = "ease_in_out",
    ) -> List[int]:
        """
        软件插值平滑过渡到目标位置

        将行程拆分为多个中间帧逐步下发, 实现平滑运动。

        参数:
            positions: 10个目标位置 (0~4096)
            duration: 总过渡时间 (秒)
            steps: 插值步数 (越多越平滑)
            easing: 缓动函数

        返回:
            最终位置反馈 (10个编码器值)
        """
        # 读取当前位置作为起点
        try:
            current = self.serial.read_all_positions()
        except (TimeoutError, ProtocolError):
            print("  [警告] 无法读取当前位置, 使用默认零位")
            current = [0] * 10

        print(f"  [插值控制] {current} -> {positions}")
        print(f"  [参数] duration={duration}s, steps={steps}, easing={easing}")

        feedback = None
        for i in range(1, steps + 1):
            t = i / steps
            interpolated = self.interpolate_positions(
                current, positions, t, easing)

            try:
                feedback = self.serial.set_all_positions(interpolated)
            except (TimeoutError, ProtocolError) as e:
                print(f"  [警告] 插值第 {i}/{steps} 步失败: {e}")
                break

            time.sleep(duration / steps)

        if feedback is None:
            # 如果插值失败, 直接设置目标
            try:
                feedback = self.serial.set_all_positions(positions)
            except (TimeoutError, ProtocolError) as e:
                print(f"  [错误] 设置目标位置失败: {e}")
                return list(positions)

        return feedback.get("position_feedback", list(positions))

    # ================================================================
    # 手势 + 速度控制
    # ================================================================

    def do_gesture_with_speed(
        self,
        gesture: Gesture,
        speed_level: SpeedLevel = SpeedLevel.MEDIUM,
        delay: float = 0.5,
    ) -> dict:
        """
        执行指定手势, 带软件插值速度控制

        参数:
            gesture: 手势枚举
            speed_level: 速度等级
            delay: 执行后的等待时间 (秒)

        返回:
            反馈数据字典
        """
        positions = KinematicsConverter.get_gesture_positions(
            gesture, self.is_right_hand)

        name = GESTURE_NAMES.get(gesture, gesture.name)
        speed_name = SPEED_NAMES.get(speed_level, "未知")
        print(f"\n  [手势] {name} (速度: {speed_name})")
        print(f"  [位置] {positions}")

        interp_params = SPEED_EASING_MAP.get(
            speed_level, SPEED_EASING_MAP[SpeedLevel.MEDIUM])
        feedback = self.set_positions_interpolated(
            positions,
            duration=interp_params["duration"],
            steps=interp_params["steps"],
            easing=interp_params["easing"],
        )
        result = {"position_feedback": feedback}

        if delay > 0:
            time.sleep(delay)

        return result


# ============================================================================
# 演示函数 (3速度等级 x 3动作)
# ============================================================================

def _print_separator(title: str):
    """打印分隔线"""
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


def _run_speed_demo(ctrl: OmniHandSpeedController):
    """
    对3种速度等级依次演示 张开→握住→张开 往返运动.

    每个速度等级做:
      1. 握住 (FIST)
      2. 张开 (PAPER) — 回到张开状态

    注意: 假设调用者已在张开位姿, 且每个速度等级之间自动间隔 2s.
    """
    speed_levels = [SpeedLevel.VERY_SLOW, SpeedLevel.MEDIUM, SpeedLevel.VERY_FAST]

    for idx, sl in enumerate(speed_levels):
        if idx > 0:
            print(f"\n  --- 间隔 2s ---")
            time.sleep(2.0)

        speed_name = SPEED_NAMES[sl]
        print(f"\n  ═══ {speed_name} ═══")

        # 握住
        print(f"\n  >>> 握住")
        ctrl.do_gesture_with_speed(
            Gesture.FIST,
            speed_level=sl,
            delay=0.8,
        )

        # 回到张开
        print(f"\n  >>> 张开")
        ctrl.do_gesture_with_speed(
            Gesture.PAPER,
            speed_level=sl,
            delay=1.0,
        )


def demo_speed(port: str = "COM5"):
    """
    演示: 软件插值速度控制

    完整流程:
      home → 张开 → [3种速度等级: 张开→握住→张开, 等级间间隔2s] → 张开 → home
    """
    _print_separator("OmniHand O10 软件插值速度控制演示")

    with OmniHandSpeedController(port=port, verbose=False) as ctrl:
        # 连接并查询
        print("\n[1/5] 连接设备...")
        try:
            ctrl.serial.flush_input()
            status = ctrl.get_status()
            ctrl.print_status()
        except Exception as e:
            print(f"  [警告] 查询失败: {e}")

        # 使能
        print("\n[2/5] 使能设备...")
        if not ctrl.enable():
            print("  [错误] 使能失败")
            return
        print("  [OK] 已使能")

        # 第一步: home → 张开
        print(f"\n{'─' * 50}")
        print("\n[3/5] 初始化: home → 张开")
        ctrl.do_gesture_with_speed(
            Gesture.RESET,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )
        ctrl.do_gesture_with_speed(
            Gesture.PAPER,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )

        # 第二步: 3种速度等级演示
        print(f"\n{'─' * 50}")
        print("\n[4/5] 速度对比演示 (3种速度等级)")
        print("  对每个等级: 张开 → 握住 → 张开\n")
        _run_speed_demo(ctrl)

        # 第三步: 张开 → home
        print(f"\n{'─' * 50}")
        print("\n[5/5] 复位: 张开 → home")
        ctrl.do_gesture_with_speed(
            Gesture.RESET,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )

        # 清理
        print(f"\n{'─' * 50}")
        print("\n失能并关闭...")
        ctrl.disable()
        print("  完成!")


# ============================================================================
# 交互式测试
# ============================================================================

def print_menu():
    """打印菜单"""
    print("\n" + "=" * 50)
    print("     OmniHand O10 速度控制测试菜单")
    print("=" * 50)
    print("  速度等级 (3级):")
    for sl in SpeedLevel:
        print(f"    {sl.name:12s} = {SPEED_NAMES[sl]} "
              f"(duration={SPEED_EASING_MAP[sl]['duration']}s)")
    print()
    print("  动作:")
    for gesture_name, gesture, gesture_desc in DEMO_GESTURES:
        print(f"    {gesture_name:6s} - {gesture_desc}")
    print()
    print("  常用命令:")
    print("    home/张开/握住  - 执行动作")
    print("    s              - 切换速度等级")
    print("    d              - 运行完整速度演示")
    print("    status         - 显示设备状态")
    print("    pos            - 读取当前位置")
    print("    q              - 退出")
    print()


# 动作名称 -> 手势映射
GESTURE_NAME_MAP: Dict[str, Gesture] = {
    "home": Gesture.RESET,
    "张开": Gesture.PAPER,
    "握住": Gesture.FIST,
}


def interactive_mode(port: str = "COM5"):
    """
    交互式速度控制测试模式
    """
    print("OmniHand O10 速度控制测试 (软件插值版)")
    print(f"串口: {port}")

    # 当前设置
    current_speed = SpeedLevel.MEDIUM
    speed_list = list(SpeedLevel)

    with OmniHandSpeedController(port=port, verbose=False) as ctrl:
        # 连接并查询
        print("\n正在连接灵巧手...")
        try:
            ctrl.serial.flush_input()
            status = ctrl.get_status()
            info = status["info"]
            print(f"  设备: {info.get('device_type_name', '?')} "
                  f"(v{info.get('software_version', '?')})")
            print(f"  序列号: {info.get('serial_number', 'N/A')}")
        except Exception as e:
            print(f"  [警告] 查询失败: {e}")

        # 使能
        try:
            if ctrl.is_enabled():
                print("  [设备已使能]")
                ctrl._enabled = True
            else:
                print("\n正在使能设备...")
                if not ctrl.enable():
                    print("  [错误] 使能失败")
                    return
                print("  [设备已使能]")
        except Exception as e:
            print(f"  [错误] 使能失败: {e}")
            return

        while True:
            speed_name = SPEED_NAMES[current_speed]
            interp_dur = SPEED_EASING_MAP[current_speed]["duration"]
            print(f"\n当前设置: 速度={speed_name} (duration={interp_dur}s)")
            print_menu()

            try:
                cmd = input("请输入 > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n退出")
                break

            # ---- 特殊命令 ----
            if cmd == "q":
                print("  退出程序")
                break

            elif cmd == "s":
                # 切换速度等级
                idx = speed_list.index(current_speed)
                idx = (idx + 1) % len(speed_list)
                current_speed = speed_list[idx]
                speed_name = SPEED_NAMES[current_speed]
                interp_dur = SPEED_EASING_MAP[current_speed]["duration"]
                print(f"  速度等级已切换为: {speed_name} (duration={interp_dur}s)")

            elif cmd == "d":
                print("  释放串口, 运行完整速度演示...")
                ctrl.disable()
                ctrl.close()
                time.sleep(0.3)
                demo_speed(port=port)
                print("  重新连接...")
                ctrl.open()
                ctrl.enable()

            elif cmd == "status":
                ctrl.print_status()

            elif cmd == "pos":
                pos = ctrl.read_positions()
                print(f"  当前位置: {pos}")

            elif cmd in GESTURE_NAME_MAP:
                gesture = GESTURE_NAME_MAP[cmd]
                try:
                    ctrl.do_gesture_with_speed(
                        gesture,
                        speed_level=current_speed,
                    )
                except (TimeoutError, ProtocolError) as e:
                    print(f"  [错误] 通信失败: {e}")
                except OmniHandError as e:
                    print(f"  [错误] {e}")

            else:
                print("  无效输入, 可用命令: home / 张开 / 握住 / s / d / status / pos / q")

        # 清理
        print("\n正在失能设备...")
        try:
            ctrl.disable()
            print("  设备已失能")
        except Exception:
            pass


# ============================================================================
# 快速测试
# ============================================================================

def quick_test(port: str = "COM5"):
    """
    快速验证速度控制功能

    完整流程:
      home → 张开 → [3种速度等级: 张开→握住→张开, 等级间间隔2s] → 张开 → home
    """
    print("=" * 50)
    print("  OmniHand O10 速度控制快速测试 (软件插值版)")
    print("=" * 50)

    with OmniHandSpeedController(port=port, verbose=True) as ctrl:
        print("\n[1/5] 连接并使能...")
        try:
            ctrl.serial.flush_input()
        except Exception:
            pass
        if not ctrl.enable():
            print("  [FAIL] 使能失败")
            return
        print("  [OK]\n")

        # home → 张开
        print("\n[2/5] 初始化: home → 张开")
        ctrl.do_gesture_with_speed(
            Gesture.RESET,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )
        ctrl.do_gesture_with_speed(
            Gesture.PAPER,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )

        # 速度测试 (3种速度等级)
        print("\n[3/5] 速度测试...")
        speed_levels = [SpeedLevel.VERY_SLOW, SpeedLevel.MEDIUM, SpeedLevel.VERY_FAST]

        for idx, sl in enumerate(speed_levels):
            if idx > 0:
                print(f"\n  --- 间隔 2s ---")
                time.sleep(2.0)

            speed_name = SPEED_NAMES[sl]
            print(f"\n  ═══ {speed_name} ═══")

            # 握住
            print(f"  >>> 握住")
            ctrl.do_gesture_with_speed(
                Gesture.FIST,
                speed_level=sl,
                delay=0.8,
            )

            # 张开
            print(f"  >>> 张开")
            ctrl.do_gesture_with_speed(
                Gesture.PAPER,
                speed_level=sl,
                delay=1.0,
            )

        # 张开 → home
        print("\n[4/5] 复位: 张开 → home")
        ctrl.do_gesture_with_speed(
            Gesture.RESET,
            speed_level=SpeedLevel.MEDIUM,
            delay=1.0,
        )

        print("\n[5/5] 失能...")
        ctrl.disable()
        print("  [OK] 完成!")


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import sys

    port = "COM5"

    # 解析命令行参数
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg.upper().startswith("COM") or arg.startswith("/dev/"):
                port = arg

        if "--demo" in sys.argv or "-d" in sys.argv:
            demo_speed(port)
            sys.exit(0)
        elif "--quick" in sys.argv or "-q" in sys.argv:
            quick_test(port)
            sys.exit(0)

    print("OmniHand O10 速度控制模块 (软件插值版)")
    print("=" * 50)
    print(f"串口: {port} (可通过命令行参数修改)")
    print()
    print("  演示命令:")
    print("    python speed_control.py --demo       # 完整速度演示")
    print("    python speed_control.py --quick      # 快速测试")
    print("    python speed_control.py --port COM3  # 指定串口进入交互模式")
    print()
    print("  交互模式中可用命令: home / 张开 / 握住 / s / d / status / pos / q")
    print()

    interactive_mode(port=port)