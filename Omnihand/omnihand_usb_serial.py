# -*- coding: utf-8 -*-
"""
OmniHand 2025 - USB/RS485 串口通信封装库
=========================================
基于项目 C++ SDK (rs_485_device / c_agibot_hand_rs) 和官方文档逆向实现的纯 Python 通信库。

协议规范
--------
- 串口: 460800 8N1
- 帧格式:
    ┌────────┬──────────┬──────┬──────┬──────────┬──────────┐
    │ HEADER │ NODE_ID  │ LEN  │ CMD  │ DATA     │ CRC16    │
    │ 2B     │ 2B LE    │ 1B   │ 1B   │ 0~62B    │ 2B LE    │
    └────────┴──────────┴──────┴──────┴──────────┴──────────┘
  - HEADER: 固定 0xAAEE (发送为 EE AA)
  - NODE_ID: 节点ID (0~0x7FF), 0x7FF = 广播地址
  - LEN: CMD + DATA 的总长度 (1~64)
  - CRC16: CCITT, poly=0x1021, init=0x0000, MSB-first
  - 调试旁路: 接收 CRC = 0x5555 时跳过校验

基于 OmniHand 2025 SDK 分析实现
"""

import struct
import time
from typing import List, Optional, Dict, Any, Tuple, Union


# ============================================================================
# 自定义异常
# ============================================================================
class OmniHandError(Exception):
    """OmniHand 通信基础异常"""
    pass


class TimeoutError(OmniHandError):
    """通信超时异常"""
    pass


class CrcError(OmniHandError):
    """CRC 校验失败异常"""
    pass


class ProtocolError(OmniHandError):
    """协议错误异常"""
    pass


# ============================================================================
# CRC16 工具函数
# ============================================================================
def crc16_ccitt_msb(data: bytes) -> int:
    """
    CRC16-CCITT (MSB-first)
    多项式: 0x1021
    初始值: 0x0000
    异或输出: 0x0000
    """
    crc = 0x0000
    for byte in data:
        crc ^= (byte << 8)
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc & 0xFFFF


def _u16_le(v: int) -> bytes:
    """uint16 -> 小端 2 字节"""
    return struct.pack("<H", v & 0xFFFF)


def _i16_le(v: int) -> bytes:
    """int16 -> 小端 2 字节"""
    return struct.pack("<h", int(v))


def _le_u16(b: bytes) -> int:
    """小端 2 字节 -> uint16"""
    return struct.unpack("<H", b)[0]


def _le_i16(b: bytes) -> int:
    """小端 2 字节 -> int16"""
    return struct.unpack("<h", b)[0]


# ============================================================================
# 常量定义
# ============================================================================

# 帧常量
HEADER_WORD = 0xAAEE
HEADER_BYTES = b"\xEE\xAA"
DEBUG_CRC_BYPASS = 0x5555
BROADCAST_NODE_ID = 0x7FF
MAX_PAYLOAD_LEN = 64
MIN_FRAME_LEN = 7  # HEADER(2) + NODE_ID(2) + LEN(1) + CMD(1) + CRC(2)

# 设备使能状态
DISABLE = 0x00
ENABLE = 0x01
CALIBRATE = 0x02

# 控制源
CONTROL_SOURCE_ROBOT = 0  # 机器人本体控制
CONTROL_SOURCE_UI = 1     # 人机界面操作员控制

# 命令码
CMD_ENABLE = 0x01              # 使能/失能/校准
CMD_QUERY_ENABLE = 0x02        # 查询使能状态
CMD_SET_SINGLE_POS = 0x06      # 设置单轴位置
CMD_READ_SINGLE_POS = 0x07     # 读取单轴位置
CMD_SET_ALL_POS = 0x08         # 设置全部轴位置
CMD_READ_ALL_POS = 0x09        # 读取全部轴位置
CMD_READ_ALL_CURRENT = 0x0A    # 读取全部轴电流
CMD_READ_ALL_SPEED = 0x0B      # 读取全部轴速度
CMD_READ_ALL_TEMP = 0x0C       # 读取全部轴温度
CMD_READ_ERROR = 0x0D          # 读取错误码
CMD_CLEAR_ERROR = 0x0E         # 清除错误
CMD_READ_SINGLE_SENSOR = 0x11  # 读取单个触觉传感器
CMD_READ_ALL_SENSOR_1 = 0x12   # 读取传感器组1 (拇指/食指/中指)
CMD_READ_ALL_SENSOR_2 = 0x13   # 读取传感器组2 (无名指/小指)
CMD_READ_ALL_SENSOR_3 = 0x14   # 读取传感器组3 (手心/手背)
CMD_SET_ALL_POS_FORCE_MIX = 0x32  # 力位混合模式
CMD_SET_CONTROL_SOURCE = 0x80  # 设置控制源
CMD_GET_CONTROL_SOURCE = 0x81  # 查询控制源
CMD_READ_SERIAL = 0xC2         # 读取产品序列号
CMD_QUERY_MODEL_VERSION = 0xCD # 查询设备型号与版本

# 关节数量
JOINT_COUNT = 10
POSITION_MAX = 4096


# ============================================================================
# 主类
# ============================================================================
class OmniHandSerial:
    """
    OmniHand O10 USB/RS485 串口通信封装

    使用示例::

        with OmniHandSerial(port="COM3") as hand:
            hand.enable()
            hand.set_all_positions([2048] * 10)
            positions = hand.read_all_positions()
            hand.disable()

    参数:
        port: 串口号 (Windows: COM3, Linux: /dev/ttyUSB0)
        baudrate: 波特率 (默认 460800)
        node_id: 节点 ID (默认 0x01)
        timeout: 读取超时 (秒)
        write_timeout: 写入超时 (秒)
        verbose: 是否打印调试日志
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 460800,
        node_id: int = 0x01,
        timeout: float = 0.5,
        write_timeout: float = 0.5,
        verbose: bool = False,
    ):
        self.port = port
        self.baudrate = baudrate
        self.node_id = node_id
        self.timeout = timeout
        self.write_timeout = write_timeout
        self.verbose = verbose

        self._ser: Optional["serial.Serial"] = None
        self._is_owner = False  # 是否由我们打开串口

    # ======================== 属性 ========================

    @property
    def is_open(self) -> bool:
        """串口是否已打开"""
        return self._ser is not None and self._ser.is_open

    # ======================== 串口生命周期 ========================

    def open(self):
        """打开串口"""
        if self.is_open:
            return
        try:
            import serial
        except ImportError:
            raise ImportError("请安装 pyserial: pip install pyserial")

        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=self.timeout,
            write_timeout=self.write_timeout,
        )
        self._is_owner = True
        self._log(f"[OPEN] {self.port} @ {self.baudrate} 8N1")

    def close(self):
        """关闭串口"""
        if self._is_owner and self._ser is not None and self._ser.is_open:
            self._ser.close()
            self._log("[CLOSE]")
        self._ser = None
        self._is_owner = False

    def __enter__(self) -> "OmniHandSerial":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def attach(self, ser: "serial.Serial") -> "OmniHandSerial":
        """
        附加到外部已打开的串口对象（不拥有所有权，不会自动关闭）
        """
        self._ser = ser
        self._is_owner = False
        return self

    # ======================== 日志 ========================

    def _log(self, msg: str):
        if self.verbose:
            print(f"[OmniHand] {msg}")

    # ======================== 串口底层操作 ========================

    def flush_input(self):
        """清空输入缓冲区"""
        if not self.is_open:
            raise OmniHandError("串口未打开")
        self._ser.reset_input_buffer()

    def flush_output(self):
        """清空输出缓冲区"""
        if not self.is_open:
            raise OmniHandError("串口未打开")
        self._ser.reset_output_buffer()

    # ======================== 协议编解码 ========================

    def build_frame(self, cmd: int, data: bytes = b"",
                    node_id: Optional[int] = None) -> bytes:
        """
        构建协议帧

        参数:
            cmd: 命令码
            data: 数据载荷
            node_id: 目标节点ID (默认 self.node_id)

        返回:
            完整的帧字节序列 (含CRC)
        """
        if node_id is None:
            node_id = self.node_id
        if not (0 <= node_id <= 0x7FF):
            raise ProtocolError(f"node_id 范围 0~0x7FF, 收到 {node_id}")

        payload = bytes([cmd]) + data
        payload_len = len(payload)
        if not (1 <= payload_len <= MAX_PAYLOAD_LEN):
            raise ProtocolError(
                f"载荷长度需 1~{MAX_PAYLOAD_LEN}, 收到 {payload_len}")

        frame_wo_crc = (
            HEADER_BYTES
            + _u16_le(node_id)
            + bytes([payload_len])
            + payload
        )
        crc = crc16_ccitt_msb(frame_wo_crc)
        return frame_wo_crc + _u16_le(crc)

    def send_frame(self, cmd: int, data: bytes = b"",
                   node_id: Optional[int] = None):
        """发送协议帧"""
        if not self.is_open:
            raise OmniHandError("串口未打开")
        frame = self.build_frame(cmd, data, node_id)
        self._ser.write(frame)
        self._ser.flush()
        self._log(f">>> SEND [{cmd:#04x}]: {frame.hex(' ').upper()}")

    def _read_exact(self, n: int) -> bytes:
        """精确读取 n 字节"""
        buf = bytearray()
        while len(buf) < n:
            part = self._ser.read(n - len(buf))
            if not part:
                break
            buf.extend(part)
        if len(buf) != n:
            raise TimeoutError(
                f"读取超时: 期望 {n} 字节, 实际 {len(buf)} 字节")
        return bytes(buf)

    def recv_frame(self, timeout: Optional[float] = None) -> Dict[str, Any]:
        """
        接收并解析一帧

        返回:
            {
                "node_id": int,
                "length": int,
                "cmd": int,
                "data": bytes,
                "raw": bytes,
                "recv_crc": int,
                "calc_crc": int,
                "crc_ok": bool,
            }
        """
        if not self.is_open:
            raise OmniHandError("串口未打开")

        old_timeout = self._ser.timeout
        if timeout is not None:
            self._ser.timeout = timeout

        try:
            # 1. 搜索帧头 0xEE 0xAA
            while True:
                b1 = self._read_exact(1)
                if b1[0] != 0xEE:
                    continue
                b2 = self._read_exact(1)
                if b2[0] == 0xAA:
                    header = b1 + b2
                    break

            # 2. 读取 NODE_ID(2) + LEN(1)
            fixed = self._read_exact(3)
            node_id = _le_u16(fixed[:2])
            length = fixed[2]

            if not (1 <= length <= MAX_PAYLOAD_LEN):
                raise ProtocolError(
                    f"非法数据长度: {length} (期望 1~{MAX_PAYLOAD_LEN})")

            # 3. 读取 payload + CRC(2)
            remain = self._read_exact(length + 2)
            payload = remain[:length]
            recv_crc = _le_u16(remain[length:length + 2])

            # 4. 拼回完整帧用于 CRC 校验
            raw = header + fixed + remain
            calc_crc = crc16_ccitt_msb(raw[:-2])

            # 5. CRC 校验 (支持调试旁路 0x5555)
            crc_ok = (recv_crc == calc_crc) or (recv_crc == DEBUG_CRC_BYPASS)
            if not crc_ok:
                raise CrcError(
                    f"CRC 校验失败: 收到 0x{recv_crc:04X}, 计算 0x{calc_crc:04X}")

            cmd = payload[0]
            data = payload[1:]

            self._log(f"<<< RECV [{cmd:#04x}]: {raw.hex(' ').upper()}")

            return {
                "node_id": node_id,
                "length": length,
                "cmd": cmd,
                "data": data,
                "raw": raw,
                "recv_crc": recv_crc,
                "calc_crc": calc_crc,
                "crc_ok": crc_ok,
            }

        finally:
            if timeout is not None:
                self._ser.timeout = old_timeout

    def request(
        self,
        cmd: int,
        data: bytes = b"",
        node_id: Optional[int] = None,
        timeout: float = 0.8,
        check_node: bool = True,
    ) -> Dict[str, Any]:
        """
        发送请求并等待匹配回复 (请求-应答模式)

        匹配规则:
        - 应答 CMD 必须与请求相同
        - 非广播时, 应答 node_id 必须匹配
        - 广播地址 0x7FF 允许任意节点回复

        参数:
            cmd: 命令码
            data: 请求数据
            node_id: 目标节点 (默认 self.node_id)
            timeout: 超时时间 (秒)
            check_node: 是否检查节点ID

        返回:
            应答帧字典 (同 recv_frame)
        """
        target_node = self.node_id if node_id is None else node_id
        self.send_frame(cmd, data, node_id)

        deadline = time.time() + timeout
        last_unmatched = None

        while time.time() < deadline:
            left = max(0.01, deadline - time.time())
            try:
                rsp = self.recv_frame(timeout=left)
            except (TimeoutError, CrcError):
                # 接收超时或CRC错误，继续等待
                continue

            # 过滤: CMD 必须匹配
            if rsp["cmd"] != cmd:
                last_unmatched = rsp
                continue

            # 过滤: 节点ID匹配 (广播模式不过滤)
            if check_node and target_node != BROADCAST_NODE_ID:
                if rsp["node_id"] != target_node:
                    last_unmatched = rsp
                    continue

            return rsp

        # 超时
        if last_unmatched is not None:
            raise TimeoutError(
                f"等待命令 0x{cmd:02X} 回复超时, "
                f"最后收到不匹配帧: cmd=0x{last_unmatched['cmd']:02X}, "
                f"node=0x{last_unmatched['node_id']:03X}")
        raise TimeoutError(f"等待命令 0x{cmd:02X} 回复超时 (未收到任何帧)")

    def request_raw(
        self,
        cmd: int,
        data: bytes = b"",
        node_id: Optional[int] = None,
        timeout: float = 0.8,
    ) -> Dict[str, Any]:
        """
        透传请求: 发送任意命令并接收回复
        不做 CMD/节点过滤, 只返回收到的第一条有效帧
        """
        target_node = self.node_id if node_id is None else node_id
        self.send_frame(cmd, data, node_id)

        deadline = time.time() + timeout
        while time.time() < deadline:
            left = max(0.01, deadline - time.time())
            try:
                return self.recv_frame(timeout=left)
            except (TimeoutError, CrcError):
                continue

        raise TimeoutError(f"命令 0x{cmd:02X} 透传请求超时")

    # ======================== 校验辅助 ========================

    @staticmethod
    def _require_len(data: bytes, n: int):
        if len(data) < n:
            raise ProtocolError(
                f"返回数据长度不足: 期望 >= {n}, 实际 {len(data)}")

    @staticmethod
    def _check_rsp_cmd(rsp: Dict[str, Any], expected: int):
        if rsp["cmd"] != expected:
            raise ProtocolError(
                f"返回命令码不匹配: 期望 0x{expected:02X}, "
                f"实际 0x{rsp['cmd']:02X}")

    @staticmethod
    def _check_axis(axis: int):
        if not (1 <= axis <= JOINT_COUNT):
            raise ProtocolError(f"轴索引需 1~{JOINT_COUNT}, 收到 {axis}")

    @staticmethod
    def _check_position(pos: int):
        if not (0 <= pos <= POSITION_MAX):
            raise ProtocolError(f"位置需 0~{POSITION_MAX}, 收到 {pos}")

    @staticmethod
    def _check_positions(positions: List[int]):
        if len(positions) != JOINT_COUNT:
            raise ProtocolError(
                f"positions 需 {JOINT_COUNT} 个, 收到 {len(positions)}")
        for p in positions:
            OmniHandSerial._check_position(p)

    # ======================== 设备使能 ========================

    def set_enable(self, enable: int,
                   node_id: Optional[int] = None) -> bool:
        """设置使能状态: 0=失能, 1=使能, 2=校准

        设备对 CMD_ENABLE 回复 ACK (data[0]=0x01) 即表示命令已送达,
        本方法仅确认 ACK, 不额外查询实际状态。
        """
        if enable not in (DISABLE, ENABLE, CALIBRATE):
            raise ProtocolError("enable 仅支持 0/1/2")
        rsp = self.request(CMD_ENABLE, bytes([enable]), node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_ENABLE)
        self._require_len(rsp["data"], 1)
        return rsp["data"][0] == 0x01

    def enable(self, node_id: Optional[int] = None) -> bool:
        """使能设备"""
        return self.set_enable(ENABLE, node_id=node_id)

    def disable(self, node_id: Optional[int] = None) -> bool:
        """失能设备"""
        return self.set_enable(DISABLE, node_id=node_id)

    def calibrate(self, node_id: Optional[int] = None) -> bool:
        """进入校准模式"""
        return self.set_enable(CALIBRATE, node_id=node_id)

    def query_enable(self, node_id: Optional[int] = None) -> int:
        """查询使能状态: 返回 0/1/2"""
        rsp = self.request(CMD_QUERY_ENABLE, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_QUERY_ENABLE)
        self._require_len(rsp["data"], 1)
        return rsp["data"][0]

    # ======================== 节点发现 ========================

    def discover_nodes(self, timeout_window: float = 1.0
                       ) -> List[Dict[str, Any]]:
        """
        广播查询节点: 发送 0x7FF 广播, 收集所有节点的回复

        参数:
            timeout_window: 收集窗口 (秒)

        返回:
            [{"node_id": int, "enable_state": int}, ...]
        """
        if not self.is_open:
            raise OmniHandError("串口未打开")

        self.send_frame(CMD_QUERY_ENABLE, b"", node_id=BROADCAST_NODE_ID)
        deadline = time.time() + timeout_window
        found: Dict[int, Dict[str, Any]] = {}

        while time.time() < deadline:
            left = max(0.01, deadline - time.time())
            try:
                rsp = self.recv_frame(timeout=left)
            except (TimeoutError, CrcError):
                break

            if rsp["cmd"] != CMD_QUERY_ENABLE or len(rsp["data"]) < 1:
                continue

            found[rsp["node_id"]] = {
                "node_id": rsp["node_id"],
                "enable_state": rsp["data"][0],
            }

        return [found[k] for k in sorted(found.keys())]

    # ======================== 单轴位置 ========================

    def set_single_position(self, axis: int, position: int,
                            node_id: Optional[int] = None
                            ) -> Tuple[int, int]:
        """
        设置单轴位置

        返回:
            (axis, position_feedback)
        """
        self._check_axis(axis)
        self._check_position(position)
        data = bytes([axis]) + _u16_le(position)
        rsp = self.request(CMD_SET_SINGLE_POS, data, node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_SET_SINGLE_POS)
        self._require_len(rsp["data"], 3)
        return rsp["data"][0], _le_u16(rsp["data"][1:3])

    def read_single_position(self, axis: int,
                             node_id: Optional[int] = None
                             ) -> Tuple[int, int]:
        """
        读取单轴位置

        返回:
            (axis, position)
        """
        self._check_axis(axis)
        rsp = self.request(CMD_READ_SINGLE_POS, bytes([axis]),
                           node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_SINGLE_POS)
        self._require_len(rsp["data"], 3)
        return rsp["data"][0], _le_u16(rsp["data"][1:3])

    # ======================== 全轴位置 ========================

    def set_all_positions(self, positions: List[int],
                          node_id: Optional[int] = None
                          ) -> Dict[str, Any]:
        """
        设置全部轴位置 (0x08)

        返回:
            {
                "position_feedback": List[int],  # 10个位置反馈
                "speed_feedback": List[int],      # 10个速度反馈
                "torque_feedback": List[int],     # 10个力矩反馈
                "fault_feedback": List[int],      # 10个故障反馈
            }
        """
        self._check_positions(positions)

        buf = bytearray()
        for p in positions:
            buf += _u16_le(p)

        rsp = self.request(CMD_SET_ALL_POS, bytes(buf),
                           node_id=node_id, timeout=1.0)
        self._check_rsp_cmd(rsp, CMD_SET_ALL_POS)
        self._require_len(rsp["data"], 60)

        data = rsp["data"]
        return {
            "position_feedback": [_le_u16(data[i:i + 2])
                                  for i in range(0, 20, 2)],
            "speed_feedback": [_le_u16(data[i:i + 2])
                               for i in range(20, 40, 2)],
            "torque_feedback": list(data[40:50]),
            "fault_feedback": list(data[50:60]),
        }

    def read_all_positions(self, node_id: Optional[int] = None
                           ) -> List[int]:
        """读取全部轴位置, 返回 10 个 uint16"""
        rsp = self.request(CMD_READ_ALL_POS, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_POS)
        self._require_len(rsp["data"], 20)
        return [_le_u16(rsp["data"][i:i + 2]) for i in range(0, 20, 2)]

    # ======================== 电流 / 速度 / 温度 ========================

    def read_all_current(self, node_id: Optional[int] = None
                         ) -> List[int]:
        """读取全部轴电流 (原始值)"""
        rsp = self.request(CMD_READ_ALL_CURRENT, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_CURRENT)
        self._require_len(rsp["data"], 20)
        return [_le_u16(rsp["data"][i:i + 2]) for i in range(0, 20, 2)]

    def read_all_current_amp(self, node_id: Optional[int] = None
                             ) -> List[float]:
        """读取全部轴电流 (安培, 除 100)"""
        return [x / 100.0 for x in self.read_all_current(node_id=node_id)]

    def read_all_speed(self, node_id: Optional[int] = None
                       ) -> List[int]:
        """读取全部轴速度"""
        rsp = self.request(CMD_READ_ALL_SPEED, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_SPEED)
        self._require_len(rsp["data"], 20)
        return [_le_u16(rsp["data"][i:i + 2]) for i in range(0, 20, 2)]

    def read_all_temp(self, node_id: Optional[int] = None
                      ) -> List[int]:
        """读取全部轴温度 (摄氏度)"""
        rsp = self.request(CMD_READ_ALL_TEMP, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_TEMP)
        self._require_len(rsp["data"], 10)
        # 温度值为有符号字节
        return [struct.unpack("b", bytes([b]))[0] for b in rsp["data"][:10]]

    # ======================== 错误码 ========================

    def read_error(self, node_id: Optional[int] = None) -> int:
        """读取错误码"""
        rsp = self.request(CMD_READ_ERROR, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ERROR)
        self._require_len(rsp["data"], 2)
        return _le_u16(rsp["data"][:2])

    def clear_error(self, node_id: Optional[int] = None) -> bool:
        """清除错误, 成功返回 True"""
        rsp = self.request(CMD_CLEAR_ERROR, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_CLEAR_ERROR)
        self._require_len(rsp["data"], 1)
        return rsp["data"][0] == 0x01

    @staticmethod
    def error_to_text(err: int) -> str:
        """将错误码转换为可读文本"""
        if err == 0:
            return "无错误"
        if 1 <= err <= 10:
            return f"电机 {err} 堵转"
        if 21 <= err <= 30:
            return f"电机 {err - 20} 过温"
        if 41 <= err <= 50:
            return f"电机 {err - 40} 电流异常"
        if 61 <= err <= 70:
            return f"电机 {err - 60} 编码器异常"
        if 81 <= err <= 90:
            return f"电机 {err - 80} 通讯异常"
        mapping = {
            101: "通讯异常",
            102: "电机初始化故障",
            103: "触觉传感器初始化故障",
            104: "自动校准故障",
            105: "参数错误",
        }
        return mapping.get(err, f"未知错误码: {err}")

    # ======================== 控制源 ========================

    def set_control_source(self, source: int,
                           node_id: Optional[int] = None) -> int:
        """设置控制源: 0=机器人本体, 1=人机界面"""
        if source not in (CONTROL_SOURCE_ROBOT, CONTROL_SOURCE_UI):
            raise ProtocolError("source 仅支持 0 或 1")
        rsp = self.request(CMD_SET_CONTROL_SOURCE, bytes([source]),
                           node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_SET_CONTROL_SOURCE)
        self._require_len(rsp["data"], 1)
        return rsp["data"][0]

    def get_control_source(self, node_id: Optional[int] = None) -> int:
        """查询控制源: 返回 0 或 1"""
        rsp = self.request(CMD_GET_CONTROL_SOURCE, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_GET_CONTROL_SOURCE)
        self._require_len(rsp["data"], 1)
        return rsp["data"][0]

    # ======================== 产品信息 ========================

    def read_serial_number(self, node_id: Optional[int] = None) -> str:
        """读取产品序列号"""
        rsp = self.request(CMD_READ_SERIAL, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_SERIAL)
        self._require_len(rsp["data"], 19)
        return bytes(rsp["data"][:19]).decode("ascii", errors="ignore"
                                               ).rstrip("\x00")

    def query_model_version(self, node_id: Optional[int] = None
                            ) -> Dict[str, Any]:
        """查询设备型号与版本信息"""
        rsp = self.request(CMD_QUERY_MODEL_VERSION, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_QUERY_MODEL_VERSION)

        d = rsp["data"]
        if len(d) < 9:
            raise ProtocolError(f"版本信息返回长度异常: {len(d)}")

        return {
            "device_type": d[0],  # 1=O12, 其他=O10
            "device_type_name": "O12" if d[0] == 1 else "O10",
            "product_status": bytes(d[1:3]).decode("ascii",
                                                    errors="ignore"),
            "software_version": f"{d[3]}.{d[4]}.{d[5]}",
            "hardware_version": f"{d[6]}.{d[7]}.{d[8]}",
            "dof": d[9] if len(d) >= 10 else None,
        }

    def get_device_info(self, node_id: Optional[int] = None
                        ) -> Dict[str, Any]:
        """获取完整设备信息"""
        info = self.query_model_version(node_id=node_id)
        try:
            info["serial_number"] = self.read_serial_number(
                node_id=node_id)
        except Exception:
            info["serial_number"] = None
        return info

    # ======================== 触觉传感器 ========================

    def read_single_sensor(
        self,
        sensor_index: int,
        node_id: Optional[int] = None,
        index_base: str = "auto",
    ) -> Dict[str, Any]:
        """
        读取单个触觉传感器

        参数:
            sensor_index: 传感器索引 (0~6 或 1~7)
            index_base: 索引基准
                - "zero": 直接使用 0~6
                - "one":  1~7 自动转 0~6
                - "auto": 自动检测 (推荐)
            node_id: 目标节点

        返回:
            {
                "sensor_index": int,     # 协议索引 (0-based)
                "sensor_label": str,     # 传感器名称
                "type": str,             # finger_4x4 / palm_5x5
                "raw": List[int],        # 原始数据
                "matrix": List[List[int]],  # 矩阵排列
            }
        """
        SENSOR_NAMES = ["拇指", "食指", "中指", "无名指",
                        "小指", "手心", "手背"]

        if index_base == "zero":
            proto_idx = sensor_index
        elif index_base == "one":
            proto_idx = sensor_index - 1
        elif index_base == "auto":
            if 0 <= sensor_index <= 6:
                proto_idx = sensor_index
            elif 1 <= sensor_index <= 7:
                proto_idx = sensor_index - 1
            else:
                raise ProtocolError(
                    f"sensor_index 应为 0~6 或 1~7, 收到 {sensor_index}")
        else:
            raise ProtocolError(
                "index_base 需为 'zero' / 'one' / 'auto'")

        rsp = self.request(CMD_READ_SINGLE_SENSOR, bytes([proto_idx]),
                           node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_SINGLE_SENSOR)

        data = rsp["data"]

        if len(data) == 17:
            # 手指 4x4: data[0]=索引, data[1:17]=16个压力值
            idx = data[0]
            values = list(data[1:17])
            matrix = [values[i:i + 4] for i in range(0, 16, 4)]
            return {
                "sensor_index": idx,
                "sensor_label": SENSOR_NAMES[idx] if idx < 7 else f"传感器{idx + 1}",
                "type": "finger_4x4",
                "raw": values,
                "matrix": matrix,
            }
        elif len(data) == 26:
            # 手心/手背 5x5: data[0]=索引, data[1:26]=25个压力值
            idx = data[0]
            values = list(data[1:26])
            matrix = [values[i:i + 5] for i in range(0, 25, 5)]
            return {
                "sensor_index": idx,
                "sensor_label": SENSOR_NAMES[idx] if idx < 7 else f"传感器{idx + 1}",
                "type": "palm_5x5",
                "raw": values,
                "matrix": matrix,
            }
        else:
            raise ProtocolError(f"传感器返回长度异常: {len(data)}")

    def read_all_sensor_group_1(self, node_id: Optional[int] = None
                                ) -> Dict[str, Any]:
        """
        读取传感器组1: 拇指(4x4) + 食指(4x4) + 中指(4x4)
        返回 48 字节 (16*3)
        """
        rsp = self.request(CMD_READ_ALL_SENSOR_1, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_SENSOR_1)
        data = list(rsp["data"])
        if len(data) != 48:
            raise ProtocolError(f"传感器组1返回长度异常: {len(data)}")

        def row_major_4x4(arr):
            return [arr[i:i + 4] for i in range(0, 16, 4)]

        return {
            "thumb": row_major_4x4(data[0:16]),
            "index": row_major_4x4(data[16:32]),
            "middle": row_major_4x4(data[32:48]),
        }

    def read_all_sensor_group_2(self, node_id: Optional[int] = None
                                ) -> Dict[str, Any]:
        """
        读取传感器组2: 无名指(4x4) + 小指(4x4)
        返回 32 字节 (16*2), 按列主序排列
        """
        rsp = self.request(CMD_READ_ALL_SENSOR_2, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_SENSOR_2)
        data = list(rsp["data"])
        if len(data) != 32:
            raise ProtocolError(f"传感器组2返回长度异常: {len(data)}")

        def col_major_4x4(arr):
            """按列主序转矩阵"""
            m = [[0] * 4 for _ in range(4)]
            idx = 0
            for col in range(4):
                for row in range(4):
                    m[row][col] = arr[idx]
                    idx += 1
            return m

        return {
            "ring": col_major_4x4(data[0:16]),
            "little": col_major_4x4(data[16:32]),
        }

    def read_all_sensor_group_3(self, node_id: Optional[int] = None
                                ) -> Dict[str, Any]:
        """
        读取传感器组3: 手心(5x5) + 手背(5x5)
        返回 50 字节 (25*2)
        """
        rsp = self.request(CMD_READ_ALL_SENSOR_3, b"", node_id=node_id)
        self._check_rsp_cmd(rsp, CMD_READ_ALL_SENSOR_3)
        data = list(rsp["data"])
        if len(data) != 50:
            raise ProtocolError(f"传感器组3返回长度异常: {len(data)}")

        def row_major_5x5(arr):
            return [arr[i:i + 5] for i in range(0, 25, 5)]

        return {
            "palm": row_major_5x5(data[0:25]),
            "back": row_major_5x5(data[25:50]),
        }

    # ======================== 力位混合控制 ========================

    def set_all_pos_speed_torque(
        self,
        positions: List[int],
        speeds: List[int],
        torques: List[int],
        node_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        力位混合模式 (0x32)
        同时下发位置 + 速度 + 力矩目标

        参数:
            positions: 10个位置目标 (0~4096)
            speeds: 10个速度目标 (-4095~4096)
            torques: 10个力矩目标 (0~255)
            node_id: 目标节点

        返回:
            {
                "position_feedback": List[int],
                "speed_feedback": List[int],
                "torque_feedback": List[int],
                "fault_feedback": List[int],
            }
        """
        self._check_positions(positions)
        if len(speeds) != JOINT_COUNT:
            raise ProtocolError(f"speeds 需 {JOINT_COUNT} 个")
        if len(torques) != JOINT_COUNT:
            raise ProtocolError(f"torques 需 {JOINT_COUNT} 个")

        buf = bytearray()
        for p in positions:
            buf += _u16_le(p)
        for s in speeds:
            if not (-4095 <= s <= 4096):
                raise ProtocolError(f"速度范围 -4095~4096, 收到 {s}")
            buf += _i16_le(s)
        for t in torques:
            if not (0 <= t <= 255):
                raise ProtocolError(f"力矩范围 0~255, 收到 {t}")
            buf += bytes([t])

        rsp = self.request(CMD_SET_ALL_POS_FORCE_MIX, bytes(buf),
                           node_id=node_id, timeout=1.0)
        self._check_rsp_cmd(rsp, CMD_SET_ALL_POS_FORCE_MIX)

        data = rsp["data"]
        if len(data) < 60:
            raise ProtocolError(f"0x32 返回长度异常: {len(data)}")

        return {
            "position_feedback": [_le_u16(data[i:i + 2])
                                  for i in range(0, 20, 2)],
            "speed_feedback": [_le_i16(data[i:i + 2])
                               for i in range(20, 40, 2)],
            "torque_feedback": list(data[40:50]),
            "fault_feedback": list(data[50:60]),
        }
