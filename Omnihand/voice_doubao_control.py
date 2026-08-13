# -*- coding: utf-8 -*-
"""
豆包端到端实时语音大模型 (全双工) 语音灵巧手控制 — Voice Doubao Control
=====================================================================
Seeduplex S2S: 单 WebSocket 完成 ASR + LLM + TTS (语音到语音).
手势动作通过 Function Calling 执行, speed 三档真正落地.

流程:
  Mic (16k int16, 20ms/包) ──WebSocket──▶ input_audio_buffer.append
    ◀── ASR transcription / output_text / output_audio.delta (24k PCM)
    ◀── function_call_arguments.done → 手势队列执行 → tool 回传 (call_id 配对)
  TTS 音频 → StreamingPlayer (sounddevice) 流式播放, 支持 barge-in 打断

无任何回退/降级: 无 key、连接失败、播放失败、灵巧手失败 → 直接报错退出.

配置: Omnihand/voice_llm_config.json → doubao 段 (复用 voice_llm_config.local.json)
API Key: --api-key > .local.json > base json > 环境变量 DOUBAO_API_KEY

依赖:
  pip install websocket-client sounddevice

使用:
  python Omnihand/voice_doubao_control.py
  python Omnihand/voice_doubao_control.py --mode ptt --port COM5
  python Omnihand/voice_doubao_control.py --no-hand   # 不接灵巧手 (模拟执行)
  python Omnihand/voice_doubao_control.py --mock      # 离线联调 (自动起本地 mock 服务器)
"""

import os
import sys
import json
import queue
import time
import base64
import threading
import argparse
import collections

import numpy as np
import msvcrt
import sounddevice as sd
import websocket  # websocket-client (同步, 纯 threading 模型)

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Omnihand.omnihand_gesture import Gesture, KinematicsConverter
from Omnihand.omnihand_speed_control import OmniHandSpeedController, SpeedLevel
from Omnihand.voice_llm_control import GestureExecutor, _load_config
from Omnihand.voice_llm_prompt import (
    DOUBAO_INSTRUCTIONS,
    DOUBAO_TOOLS,
    validate_tool_call,
    parse_fc_items,
)

# ============================================================
# 配置加载 (doubao 段, 复用 voice_llm_config.json + .local.json)
# ============================================================

CFG = _load_config()
DB = CFG.get("doubao", {}) or {}

_WS_URL = DB.get("ws_url", "wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue")
_MODEL = DB.get("model", "1.2.6.1")
_VOICE = DB.get("voice", "")
_SPEED = DB.get("speed", 0)
_LOUDNESS = DB.get("loudness", 0)
_AUDIO_IN_FORMAT = DB.get("audio_input_format", "pcm")
_AUDIO_OUT_FORMAT = DB.get("audio_output_format", "pcm_s16le")

_SAMPLE_RATE = 16000          # 麦克风/上行音频
_OUT_SAMPLE_RATE = 24000      # 下行音频 (服务端固定 24k)
_CHUNK_MS = int(DB.get("chunk_ms", 20))
_IN_BLOCKSIZE = int(_SAMPLE_RATE * _CHUNK_MS / 1000)      # 320
_OUT_BLOCKSIZE = int(_OUT_SAMPLE_RATE * _CHUNK_MS / 1000)  # 480
_MIC_Q_MAX = 100              # 2s 音频, 满丢旧
_PLAY_Q_MAX = 200             # 4s 音频
_PREBUFFER = int(DB.get("prebuffer_chunks", 3))

_CONNECT_TIMEOUT = float(DB.get("connect_timeout_s", 10))
_PING_INTERVAL = float(DB.get("ping_interval_s", 20))
_MAX_RETRIES = int(DB.get("reconnect_max_retries", 5))
_BACKOFF_BASE = float(DB.get("backoff_base_s", 1.0))
_BACKOFF_MAX = float(DB.get("backoff_max_s", 8.0))

# ============================================================
# 日志
# ============================================================

_log_dir = os.path.join(_project_root, "logs")
os.makedirs(_log_dir, exist_ok=True)
_log_path = os.path.join(
    _log_dir, f"voice_doubao_{time.strftime('%Y%m%d_%H%M%S')}.log")
_log_lock = threading.Lock()


def _log(msg: str):
    """写日志 (终端 + 文件)."""
    ts = time.strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    with _log_lock:
        try:
            with open(_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass


# ============================================================
# StreamingPlayer — 24kHz int16 流式播放 (替代 Windows MCI)
# ============================================================

class StreamingPlayer:
    """sounddevice 队列流式播放器. 流常开, 队列空输出静音, 免重开设备延迟."""

    def __init__(self, sample_rate: int = _OUT_SAMPLE_RATE,
                 blocksize: int = _OUT_BLOCKSIZE,
                 prebuffer: int = _PREBUFFER,
                 max_chunks: int = _PLAY_Q_MAX):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.prebuffer = prebuffer
        self.max_chunks = max_chunks
        self.epoch = 0                     # barge-in 世代号
        self._q = queue.Queue(maxsize=max_chunks)
        self._pending = collections.deque()  # 预缓冲 (未达阈值不播)
        self._leftover = None              # (epoch, pcm_bytes) 跨块余量
        self._stream = None
        self._lock = threading.Lock()

    def open(self):
        """打开输出流. 失败直接抛异常 (无降级)."""
        try:
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate, channels=1, dtype="int16",
                blocksize=self.blocksize, callback=self._cb)
            self._stream.start()
        except Exception as e:
            raise RuntimeError(
                f"无法打开播放设备 ({self.sample_rate}Hz): {e}") from e

    def close(self):
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.flush()

    def _clear(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._pending.clear()
        self._leftover = None

    def enqueue(self, pcm: bytes):
        """pcm: int16 LE bytes. 前 prebuffer 块预缓冲后一次性放行."""
        with self._lock:
            self._pending.append((self.epoch, pcm))
            if len(self._pending) >= self.prebuffer:
                while self._pending:
                    if self._q.full():
                        try:
                            self._q.get_nowait()
                        except queue.Empty:
                            pass
                    self._q.put(self._pending.popleft())

    def flush(self):
        """barge-in: 清空未播放音频 (含预缓冲)."""
        with self._lock:
            self._clear()

    def bump_epoch(self):
        """打断: 世代号 +1, 旧音频全部丢弃."""
        self.epoch += 1
        self.flush()

    def _cb(self, outdata, frames, time_info, status):
        """服务端 audio.delta 包大小不固定, 须处理跨块余量 (不能丢尾部)."""
        outdata.fill(0)
        pos = 0
        while pos < frames:
            # 1. 先消费上块余量
            if self._leftover is not None:
                ep, buf = self._leftover
                if ep != self.epoch:
                    self._leftover = None
                    continue
                n = min(frames - pos, len(buf) // 2)
                outdata[pos:pos + n, 0] = np.frombuffer(
                    buf[:n * 2], dtype=np.int16)
                pos += n
                self._leftover = (ep, buf[n * 2:]) if buf[n * 2:] else None
                continue
            # 2. 取队列下一块
            try:
                ep, pcm = self._q.get_nowait()
            except queue.Empty:
                return               # 剩余输出静音
            if ep != self.epoch:
                continue             # 旧世代丢弃
            self._leftover = (ep, pcm)


# ============================================================
# GestureQueueExecutor — 手势执行队列线程 (串口唯一持有者)
# ============================================================

class GestureQueueExecutor(GestureExecutor):
    """继承 MiMo 版 GestureExecutor (SEQUENCE_STEPS/握手压力检测), 队列化执行.

    OmniHandSerial 内部无锁, 串口只允许本线程持有 — FC 调用一律入队.
    speed 语义在此真正落地 (原 MiMo 链路 speed 只打印未生效):
      null → 直接到位; fast → 0.75s 插值; medium → 1.0s; slow → 3.0s
    """

    _FAST_DURATION = 0.75
    _FAST_STEPS = 15

    def __init__(self, port: str = "COM5", on_stop_cb=None):
        super().__init__(port)
        self._q = queue.Queue()
        self._exit = threading.Event()
        self._thread = None
        self._on_stop_cb = on_stop_cb

    def start(self):
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="gesture-exec")
        self._thread.start()

    def stop(self):
        self._exit.set()
        if self._thread is not None:
            self._thread.join(timeout=3)

    def submit(self, action: dict, call_id: str = None):
        """投递动作, 微秒级返回 (FC 立即回传模式)."""
        self._q.put((action, call_id))

    # ---- 线程体 ----

    def _run(self):
        while not self._exit.is_set():
            try:
                action, call_id = self._q.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                self._execute_one(action, call_id)
            except Exception as e:
                _log(f"[执行失败] {action}: {e}")

    def _execute_one(self, action: dict, call_id: str = None):
        typ = action.get("type")
        tag = f" (call_id={call_id})" if call_id else ""
        if typ == "stop":
            print(f"  [执行] 停止{tag}")
            self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
            if self._on_stop_cb is not None:
                self._on_stop_cb()
            return
        if typ == "gesture":
            print(f"  [执行] {action['gesture']} speed={action.get('speed')}{tag}")
            self._do_gesture_speed(
                action["gesture"], action.get("speed"), hold=0.5)
            return
        if typ == "sequence":
            seq = action.get("sequence")
            if seq == "custom":
                steps = action.get("steps", [])
                print(f"  [执行] 自定义序列 ({len(steps)}步){tag}")
                for s in steps:
                    self._do_gesture_speed(
                        s["gesture"], None, hold=s.get("duration", 0.5))
            elif seq == "握手":
                print(f"  [执行] 握手序列 (压力检测){tag}")
                self._do_handshake()
            elif seq in self.SEQUENCE_STEPS:
                print(f"  [执行] 序列: {seq}{tag}")
                for g, d in self.SEQUENCE_STEPS[seq]:
                    self.ctrl.gesture.do_gesture(g, delay=d)
            return
        raise ValueError(f"未知 action type: {typ}")

    def _do_gesture_speed(self, name: str, speed, hold: float = 0.5):
        """speed: None=直接 / fast=0.75s / medium=1.0s / slow=3.0s."""
        g = Gesture[name]
        if speed is None or speed == "null":
            self.ctrl.gesture.do_gesture(g, delay=hold)
        elif speed == "fast":
            positions = KinematicsConverter.get_gesture_positions(g)
            self.ctrl.set_positions_interpolated(
                positions, duration=self._FAST_DURATION,
                steps=self._FAST_STEPS, easing="ease_in_out")
        elif speed == "medium":
            self.ctrl.do_gesture_with_speed(g, SpeedLevel.MEDIUM, delay=hold)
        elif speed == "slow":
            self.ctrl.do_gesture_with_speed(g, SpeedLevel.VERY_SLOW, delay=hold)
        else:
            self.ctrl.gesture.do_gesture(g, delay=hold)


# ============================================================
# DoubaoVoiceSession — WS 会话 + 事件分发状态机
# ============================================================

class DoubaoVoiceSession:
    """线程模型 (4 线程):

      Main:             'q' 退出 / ptt 按键检测
      PortAudio 回调:   麦克风 16k → mic_q (ptt 模式按 rec_flag)
      WS send 线程:     mic_q → input_audio_buffer.append; ctrl 事件/重连/优雅关闭
      WS recv 线程:     阻塞 recv → 事件分发 (音频→player; FC→手势队列+tool回传)
      手势执行线程:     GestureQueueExecutor (串口唯一持有者)
    """

    def __init__(self, cfg_db: dict, api_key: str, mode: str,
                 hand: GestureQueueExecutor = None,
                 ws_url: str = None, stop_event: threading.Event = None,
                 instructions: str = None):
        self.cfg = cfg_db
        self.api_key = api_key
        self.mode = mode                # "auto" | "ptt"
        self.hand = hand
        self.ws_url = ws_url or cfg_db.get("ws_url", _WS_URL)
        self.stop_event = stop_event or threading.Event()
        self.instructions = instructions or DOUBAO_INSTRUCTIONS

        self._ws = None
        self._ws_lock = threading.Lock()
        self._send_q = queue.Queue()    # ("audio", pcm) / ("event", dict) / ("reconnect", reason) / ("close", None)
        self._mic_q = queue.Queue(maxsize=_MIC_Q_MAX)
        self._seq = 0
        self.session_id = None
        self._responding = False        # 服务端播报中 (barge-in 判定)
        self._session_closed = threading.Event()
        self.rec_flag = threading.Event()  # ptt: True=录音中
        self.rec_flag.set()               # auto 恒录; ptt 由主线程切换

        self.player = StreamingPlayer()
        self._mic_stream = None
        self._threads = []
        self._close_done = threading.Event()

    # ---- 生命周期 ----

    def start(self):
        self.player.open()              # 失败抛 RuntimeError, 主程序报错退出
        try:
            self._mic_stream = sd.InputStream(
                samplerate=_SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=_IN_BLOCKSIZE, callback=self._mic_callback)
            self._mic_stream.start()
        except Exception as e:
            self.player.close()
            raise RuntimeError(f"无法打开麦克风 (16kHz): {e}") from e

        self._threads = [
            threading.Thread(target=self._ws_recv_loop, daemon=True, name="ws-recv"),
            threading.Thread(target=self._ws_send_loop, daemon=True, name="ws-send"),
        ]
        for t in self._threads:
            t.start()
        self._send_q.put(("reconnect", "初始连接"))

    def stop(self):
        """优雅关闭: session.close → 等 session.closed → 关 socket → 停音频流."""
        self._send_q.put(("close", None))
        deadline = time.time() + 6
        while time.time() < deadline and self._session_closed.is_set() is False \
                and self._close_done.is_set() is False:
            time.sleep(0.1)
        if self._mic_stream is not None:
            try:
                self._mic_stream.stop()
                self._mic_stream.close()
            except Exception:
                pass
            self._mic_stream = None
        self.player.close()

    # ---- WS 连接管理 ----

    def _get_ws(self):
        with self._ws_lock:
            return self._ws

    def _set_ws(self, ws):
        with self._ws_lock:
            self._ws = ws

    def _new_event_id(self) -> str:
        self._seq += 1
        return f"evt_{self._seq}_{int(time.time() * 1000)}"

    def _build_session_create(self) -> dict:
        out_fmt = {"type": _AUDIO_OUT_FORMAT, "sample_rate": _OUT_SAMPLE_RATE,
                   "channels": 1}
        audio_out = {"format": out_fmt}
        if _VOICE:
            audio_out["voice"] = _VOICE
        audio_out["speed"] = _SPEED
        audio_out["loudness"] = _LOUDNESS

        session = {
            "model": _MODEL,
            "audio": {
                "input": {"format": {"type": _AUDIO_IN_FORMAT,
                                     "sample_rate": _SAMPLE_RATE, "channels": 1}},
                "output": audio_out,
            },
            "instructions": self.instructions,
            "tools": DOUBAO_TOOLS,
        }
        if self.session_id:
            session["id"] = self.session_id   # 接续历史会话
        return {"type": "session.create", "event_id": self._new_event_id(),
                "session": session}

    def _do_reconnect(self, reason: str):
        """连接 + session.create. 初始连接失败 → 报错退出; 运行中断线 → 指数退避重试."""
        attempt = 0
        backoff = _BACKOFF_BASE
        while not self.stop_event.is_set():
            attempt += 1
            old = self._get_ws()
            if old is not None:
                try:
                    old.close()
                except Exception:
                    pass
                self._set_ws(None)

            if attempt > 1:
                _log(f"[WS] 重连 {attempt - 1}/{_MAX_RETRIES}: {reason}")
                time.sleep(min(backoff, _BACKOFF_MAX))
                backoff *= 2

            try:
                ws = websocket.create_connection(
                    self.ws_url,
                    header={"X-Api-Key": self.api_key},
                    timeout=_CONNECT_TIMEOUT,
                    enable_multithread=True,
                )
                # recv 无限阻塞 (帧中途超时会损坏数据流, 死链靠 ping + 服务端关闭检测)
                ws.settimeout(None)
            except Exception as e:
                if attempt == 1:
                    _log(f"[ERROR] 初始连接失败: {e}")
                    self.stop_event.set()
                    return
                _log(f"[WS] 连接失败: {e}")
                if attempt - 1 >= _MAX_RETRIES:
                    _log("[ERROR] 重连次数超限, 退出")
                    self.stop_event.set()
                    return
                continue

            self._set_ws(ws)
            try:
                ws.send(json.dumps(self._build_session_create(),
                                   ensure_ascii=False))
            except Exception as e:
                _log(f"[WS] session.create 发送失败: {e}")
                self._set_ws(None)
                continue
            self._last_ping = time.time()
            _log(f"[WS] 已连接 ({'重连成功' if attempt > 1 else '初始'}) "
                 f"attempt={attempt}")
            return

    # ---- 线程体 ----

    def _mic_callback(self, indata, frames, time_info, status):
        if self.stop_event.is_set():
            raise sd.CallbackStop
        if self.mode == "ptt" and not self.rec_flag.is_set():
            return
        chunk = indata[:, 0].copy()
        pcm = (np.clip(chunk, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
        while True:                      # 有界队列, 满丢旧 (宁丢旧不累积延迟)
            try:
                self._mic_q.put_nowait(pcm)
                break
            except queue.Full:
                try:
                    self._mic_q.get_nowait()
                except queue.Empty:
                    pass

    def _ws_send_loop(self):
        self._last_ping = time.time()
        while True:
            try:
                kind, payload = self._send_q.get(timeout=0.5)
            except queue.Empty:
                if self.stop_event.is_set():
                    return
                self._ping()
                continue

            if kind == "close":
                # 优雅关闭优先于 stop_event (避免 55000001)
                self._graceful_close()
                self._close_done.set()
                return

            if self.stop_event.is_set():
                continue              # 其余项丢弃

            if kind == "reconnect":
                self._do_reconnect(payload)
                continue

            ws = self._get_ws()
            if ws is None:
                continue                  # 未连接: 音频/事件丢弃

            if kind == "audio":
                evt = {"type": "input_audio_buffer.append",
                       "event_id": self._new_event_id(),
                       "audio": base64.b64encode(payload).decode("ascii")}
                try:
                    ws.send(json.dumps(evt))
                except Exception as e:
                    _log(f"[WS] 音频发送失败: {e}")
                    self._send_q.put(("reconnect", f"发送失败: {e}"))
            elif kind == "event":
                try:
                    ws.send(json.dumps(payload, ensure_ascii=False))
                except Exception as e:
                    _log(f"[WS] 事件发送失败 ({payload.get('type')}): {e}")
                    self._send_q.put(("reconnect", f"发送失败: {e}"))

    def _ping(self):
        ws = self._get_ws()
        if ws is None:
            return
        if time.time() - getattr(self, "_last_ping", 0) >= _PING_INTERVAL:
            try:
                ws.ping()
                self._last_ping = time.time()
            except Exception as e:
                _log(f"[WS] ping 失败: {e}")
                self._send_q.put(("reconnect", f"ping 失败: {e}"))

    def _graceful_close(self):
        """发 session.close 并等 session.closed, 再关 socket (防 55000001)."""
        ws = self._get_ws()
        if ws is None:
            return
        self._session_closed.clear()
        try:
            ws.send(json.dumps({"type": "session.close",
                                "event_id": self._new_event_id()},
                               ensure_ascii=False))
            deadline = time.time() + 3
            while time.time() < deadline and not self._session_closed.is_set():
                time.sleep(0.1)
        except Exception as e:
            _log(f"[WS] session.close 异常: {e}")
        finally:
            try:
                ws.close()
            except Exception:
                pass
            self._set_ws(None)

    def _ws_recv_loop(self):
        while not self.stop_event.is_set():
            ws = self._get_ws()
            if ws is None:
                time.sleep(0.1)
                continue
            try:
                msg = ws.recv()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as e:
                _log(f"[WS] recv 异常: {e}")
                self._send_q.put(("reconnect", f"recv 异常: {e}"))
                time.sleep(0.5)
                continue
            if not msg:
                _log("[WS] 服务端关闭连接")
                self._send_q.put(("reconnect", "服务端关闭连接"))
                time.sleep(0.5)
                continue
            try:
                evt = json.loads(msg)
            except json.JSONDecodeError:
                continue
            try:
                self._handle_event(evt)
            except Exception as e:
                _log(f"[事件处理异常] {evt.get('type')}: {e}")

    # ---- 事件处理 ----

    def _handle_event(self, evt: dict):
        typ = evt.get("type", "")

        if typ == "session.created":
            self.session_id = (evt.get("session") or {}).get("id", "")
            _log(f"[会话] 已创建 {self.session_id}")
            print("  [豆包] 会话已建立, 请说话 ('q' 退出)\n")

        elif typ == "conversation.item.input_audio_transcription.started":
            self._on_transcription_started(evt)

        elif typ == "conversation.item.input_audio_transcription.delta":
            txt = evt.get("delta") or evt.get("transcript") or evt.get("text") or ""
            if txt:
                print(f"  [ASR] {txt}", end="", flush=True)

        elif typ == "conversation.item.input_audio_transcription.completed":
            txt = evt.get("transcript") or evt.get("text") or ""
            print("")
            _log(f"ASR 终稿: {txt}")

        elif typ == "response.output_text.delta":
            txt = evt.get("delta") or evt.get("text") or ""
            if txt:
                print(f"  [文本] {txt}", end="", flush=True)

        elif typ == "response.output_text.done":
            print("")

        elif typ == "response.output_audio.started":
            self._responding = True

        elif typ == "response.output_audio.delta":
            self._on_audio_delta(evt)

        elif typ == "response.output_audio.done":
            self._responding = False

        elif typ == "response.function_call_arguments.done":
            self._on_fc_done(evt)

        elif typ == "response.done":
            _log(f"usage: {json.dumps(evt.get('usage', {}), ensure_ascii=False)}")

        elif typ == "session.closed":
            self._session_closed.set()
            _log("[会话] 已关闭")

        elif typ == "error":
            self._on_error(evt)

    def _on_transcription_started(self, evt: dict):
        """用户开口 → barge-in: 停止播放 + response.cancel (仅在服务端播报中)."""
        if self._responding:
            _log("[打断] 用户开口, 取消当前播报")
            self.player.bump_epoch()
            self._responding = False
            self._send_q.put(("event", {"type": "response.cancel",
                                        "event_id": self._new_event_id()}))

    def _on_audio_delta(self, evt: dict):
        b64 = evt.get("delta", "")
        if not b64:
            return
        try:
            pcm = base64.b64decode(b64)
        except Exception as e:
            _log(f"[音频] Base64 解码失败: {e}")
            return
        self.player.enqueue(pcm)

    def _on_fc_done(self, evt: dict):
        items = parse_fc_items(evt)
        if not items:
            _log("[FC] 收到空 FC 事件")
            return
        for call_id, name, args in items:
            ok, err, action = validate_tool_call(name, args)
            if not ok:
                _log(f"[FC] 非法调用 {name}: {err}")
                self._send_tool_result(call_id, False, err)
                continue
            _log(f"[FC] {name}: {json.dumps(args, ensure_ascii=False)}")
            if self.hand is not None:
                self.hand.submit(action, call_id)
                self._send_tool_result(call_id, True, "已接收")
            else:
                _log(f"  [模拟] 执行: {action}")
                self._send_tool_result(call_id, True, "已接收(模拟)")

    def _send_tool_result(self, call_id: str, ok: bool, detail: str):
        """FC 结果回传 (role=tool, call_id 配对)."""
        evt = {
            "type": "conversation.item.create",
            "event_id": self._new_event_id(),
            "items": [{
                "call_id": call_id,
                "role": "tool",
                "content": [{
                    "type": "input_text",
                    "text": json.dumps({"ok": ok, "detail": detail},
                                      ensure_ascii=False),
                }],
            }],
        }
        self._send_q.put(("event", evt))

    def _on_error(self, evt: dict):
        err = evt.get("error") or {}
        code = err.get("code", evt.get("code", ""))
        msg = err.get("message", evt.get("message", ""))
        _log(f"[WS] error {code}: {msg}")

        # 鉴权/参数类错误: 不重试, 直接报错退出 (无降级)
        _fatal_codes = {401, 403}
        _fatal_hints = ("ApiKey", "api key", "鉴权", "Unauthorized",
                        "InvalidSpeaker", "invalid speaker")
        if (isinstance(code, int) and code in _fatal_codes) or \
                any(h in str(msg) for h in _fatal_hints):
            _log(f"[ERROR] 鉴权/参数错误, 请检查 API Key 与 voice 配置: {code} {msg}")
            self.stop_event.set()
            return
        # 超时/静默/服务端错误: 走重连
        self._send_q.put(("reconnect", f"error {code}: {msg}"))


# ============================================================
# API Key 解析
# ============================================================

def resolve_api_key(cli_key: str = None) -> str:
    """优先级: --api-key > .local.json > base json > 环境变量 DOUBAO_API_KEY."""
    return (cli_key or DB.get("api_key", "")
            or os.environ.get("DOUBAO_API_KEY", ""))


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="豆包端到端语音灵巧手控制")
    parser.add_argument("--port", default="COM5", help="灵巧手串口号")
    parser.add_argument("--api-key", default=None,
                        help="豆包 X-Api-Key (优先级最高)")
    parser.add_argument("--mode", default="auto", choices=["ptt", "auto"],
                        help="输入模式: auto=服务端VAD, ptt=按键说话")
    parser.add_argument("--no-hand", action="store_true",
                        help="不接灵巧手, 模拟执行 (显式参数)")
    parser.add_argument("--mock", action="store_true",
                        help="离线联调: 自动启动本地 mock 服务器并连接")
    args = parser.parse_args()

    api_key = resolve_api_key(args.api_key)

    print("=" * 55)
    print("  语音灵巧手控制 — 豆包端到端实时语音 (全双工)")
    print(f"  model={_MODEL}  voice={_VOICE or '(默认)'}  mode={args.mode}")
    print("=" * 55)

    if not api_key:
        print("\n[ERROR] 未找到豆包 API Key. 请通过以下任一方式提供:")
        print("  1. --api-key sk-xxx")
        print('  2. voice_llm_config.local.json → "doubao": {"api_key": "sk-xxx"}')
        print('  3. voice_llm_config.json → "doubao": {"api_key": "sk-xxx"}')
        print("  4. 环境变量 DOUBAO_API_KEY")
        print("  (火山引擎控制台 → API Key 管理 获取 X-Api-Key)")
        return 1

    # mock 模式: 自动起本地 mock 服务器
    ws_url = None
    mock_server = None
    if args.mock:
        sys.path.insert(0, os.path.join(_project_root, "Omnihand", "tests"))
        from mock_doubao_server import start_mock_server
        mock_server = start_mock_server()
        ws_url = mock_server.ws_url
        print(f"\n[mock] 本地 mock 服务器: {ws_url}")

    # 灵巧手
    stop_event = threading.Event()
    hand = None
    if not args.no_hand:
        print("\n[Hand] 初始化...")
        try:
            hand = GestureQueueExecutor(port=args.port,
                                        on_stop_cb=stop_event.set)
            hand.start()
            print("[Hand] 手势执行线程已启动")
        except Exception as e:
            print(f"[ERROR] 灵巧手初始化失败: {e}")
            if mock_server:
                mock_server.shutdown()
            return 1
    else:
        print("\n[Hand] --no-hand: 模拟执行模式")

    # 会话
    session = DoubaoVoiceSession(DB, api_key, args.mode, hand=hand,
                                 ws_url=ws_url, stop_event=stop_event)
    print(f"\n[模式] {'Auto (服务端 VAD)' if args.mode == 'auto' else 'PTT 按键说话'} "
          f"→ 豆包 S2S → 手势执行 + 流式播放")
    if args.mode == "ptt":
        print("       回车开始录音 / 再按回车结束; 'q' 退出")
    print()
    try:
        session.start()
    except RuntimeError as e:
        print(f"[ERROR] {e}")
        if hand:
            hand.stop()
            hand.close()
        if mock_server:
            mock_server.shutdown()
        return 1

    # 主循环
    try:
        if args.mode == "ptt":
            session.rec_flag.clear()
        while not stop_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch.lower() == b"q":
                    print("\n[q] 退出")
                    break
                if args.mode == "ptt" and ch in (b"\r", b"\n", b" "):
                    if not session.rec_flag.is_set():
                        session.rec_flag.set()
                        print("\n  ● 录音中... 再按回车结束", flush=True)
                    else:
                        session.rec_flag.clear()
                        session._send_q.put((
                            "event",
                            {"type": "input_audio_buffer.commit",
                             "event_id": session._new_event_id()}))
                        print("\n  ■ 录音结束, 已发送", flush=True)
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\n[中断]")

    print("\n清理...")
    session.stop()
    if hand:
        hand.stop()
        hand.close()
    if mock_server:
        mock_server.shutdown()
    print("退出.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
