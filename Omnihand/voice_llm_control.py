# -*- coding: utf-8 -*-
"""
LLM 语音灵巧手控制 — Voice LLM Control
========================================
MiMo (多模态, speech+text-in, text-out) + edge-tts (语音合成).
语义理解替代关键词匹配, 支持自然语言指令.

流程:
  Mic → WAV → MiMo API → JSON动作 + 回复文本
    ├→ HandController 执行手势
    └→ edge-tts 合成语音 → 播放

配置: Omnihand/voice_llm_config.json
API Key: 环境变量 MIMO_API_KEY

依赖:
  pip install openai edge-tts sounddevice

使用:
  set MIMO_API_KEY=sk-xxx
  python Omnihand/voice_llm_control.py
  python Omnihand/voice_llm_control.py --port COM5
"""

import os
import sys
import io
import json
import time
import wave
import base64
import tempfile
import threading
import argparse
import subprocess
from typing import Optional, List

import numpy as np
import msvcrt
import sounddevice as sd
from openai import OpenAI

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from Omnihand.omnihand_gesture import Gesture
from Omnihand.omnihand_speed_control import OmniHandSpeedController
from Omnihand.voice_llm_prompt import (
    SYSTEM_PROMPT,
    parse_llm_response,
    validate_actions,
    extract_reply_text,
)

# ============================================================
# 配置加载
# ============================================================

def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并 override 到 base."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            base[k] = _deep_merge(dict(base[k]), v)
        else:
            base[k] = v
    return base


def _load_config() -> dict:
    """加载 base + local 覆盖 (local 含 API Key, 不提交)."""
    _dir = os.path.dirname(os.path.abspath(__file__))

    # 1. base config
    base_path = os.path.join(_dir, "voice_llm_config.json")
    try:
        with open(base_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[Config] 已加载 {base_path}")
    except Exception as e:
        print(f"[Config] 加载失败 ({e}), 使用默认值")
        cfg = {}

    # 2. local override (gitignored, 含真 API Key)
    local_path = os.path.join(_dir, "voice_llm_config.local.json")
    if os.path.exists(local_path):
        try:
            with open(local_path, "r", encoding="utf-8") as f:
                local = json.load(f)
            cfg = _deep_merge(cfg, local)
            print(f"[Config] 已合并 {local_path}")
        except Exception as e:
            print(f"[Config] 合并 local 失败 ({e})")

    return cfg


CFG = _load_config()
MIMO = CFG.get("mimo", {})
TTS = CFG.get("tts", {})
REC = CFG.get("record", {})
GDEF = CFG.get("gesture_defaults", {})

# API Key — 优先级: CLI arg > local.json > base.json > env var
_MIMO_API_KEY_FROM_CFG = CFG.get("api_key", "") or os.environ.get("MIMO_API_KEY", "")
_MIMO_BASE_URL = MIMO.get("base_url", "https://api.xiaomimimo.com/v1")
_MIMO_MODEL = MIMO.get("model", "mimo-v2.5")
_MIMO_MAX_TOKENS = MIMO.get("max_tokens", 1024)
_MIMO_TIMEOUT = MIMO.get("timeout_s", 15)

_TTS_VOICE = TTS.get("voice", "zh-CN-XiaoxiaoNeural")
_TTS_RATE = TTS.get("rate", "+10%")
_TTS_VOLUME = TTS.get("volume", "+0%")
_TTS_PITCH = TTS.get("pitch", "+0Hz")

_SAMPLE_RATE = REC.get("sample_rate", 16000)
_MAX_RECORD_S = REC.get("max_duration_s", 10.0)
_MIN_RECORD_S = REC.get("min_duration_s", 0.3)

_GESTURE_DURATION = GDEF.get("duration", 0.5)


# ============================================================
# 录音
# ============================================================

def record_audio() -> Optional[np.ndarray]:
    """PTT 模式: 按回车开始录音, 再按回车结束."""
    print("\n[录音] 按回车开始 (最长{}s)...".format(_MAX_RECORD_S))
    input()
    print("  ● 录音中... 按回车结束", end="", flush=True)

    sr = _SAMPLE_RATE
    buffer: List[np.ndarray] = []
    stop_flag = threading.Event()

    def _cb(indata, frames, time_info, status):
        buffer.append(indata[:, 0].copy())
        if stop_flag.is_set():
            raise sd.CallbackStop

    stream = sd.InputStream(
        samplerate=sr, channels=1, dtype="float32", callback=_cb
    )
    stream.start()

    deadline = time.time() + _MAX_RECORD_S
    while time.time() < deadline and not stop_flag.is_set():
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            if ch in (b'\r', b'\n', b' '):
                stop_flag.set()
        time.sleep(0.05)

    stream.stop()
    stream.close()

    if not buffer:
        print("  (无音频)")
        return None

    audio = np.concatenate(buffer)
    duration = len(audio) / sr
    print(f"\r  已录音 {duration:.1f}s")

    if duration < _MIN_RECORD_S:
        print("  [跳过] 太短")
        return None
    return audio


def audio_to_wav_bytes(audio: np.ndarray) -> bytes:
    """float32 → 16-bit PCM WAV bytes."""
    buf = io.BytesIO()
    data = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(_SAMPLE_RATE)
        wf.writeframes(data.tobytes())
    return buf.getvalue()


# ============================================================
# MiMo API
# ============================================================

def call_mimo(audio_wav: bytes, api_key: str) -> Optional[str]:
    """发送音频+system prompt → 返回文本."""

    audio_b64 = base64.b64encode(audio_wav).decode("ascii")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "请根据语音指令做出回应."},
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": f"data:audio/wav;base64,{audio_b64}"
                    },
                },
            ],
        },
    ]

    extra = {}
    if MIMO.get("thinking_disabled", True):
        extra["thinking"] = {"type": "disabled"}

    try:
        client = OpenAI(api_key=api_key, base_url=_MIMO_BASE_URL)
        completion = client.chat.completions.create(
            model=_MIMO_MODEL,
            messages=messages,
            max_completion_tokens=_MIMO_MAX_TOKENS,
            stream=False,
            timeout=_MIMO_TIMEOUT,
            extra_body=extra,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"  [MiMo Error] {e}")
        return None


# ============================================================
# TTS (edge-tts)
# ============================================================

def speak(text: str):
    """edge-tts 合成并播放 (异步)."""
    text = text.strip()
    if not text:
        return
    print(f"  [TTS] {text}")

    def _run():
        try:
            fd, mp3_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            cmd = [
                sys.executable, "-m", "edge_tts",
                "--voice", _TTS_VOICE,
                "--rate", _TTS_RATE,
                "--volume", _TTS_VOLUME,
                "--pitch", _TTS_PITCH,
                "--text", text,
                "--write-media", mp3_path,
            ]
            subprocess.run(cmd, capture_output=True, timeout=30)
            if os.path.exists(mp3_path) and os.path.getsize(mp3_path) > 0:
                import winsound
                winsound.PlaySound(mp3_path, winsound.SND_FILENAME)
            try:
                os.unlink(mp3_path)
            except Exception:
                pass
        except Exception as e:
            print(f"  [TTS Error] {e}")

    threading.Thread(target=_run, daemon=True).start()


# ============================================================
# 手势执行
# ============================================================

class GestureExecutor:
    """执行 LLM 返回的 actions."""

    SEQUENCE_STEPS = {
        "打招呼": [
            (Gesture.RESET, 0.5),
            (Gesture.HELLO_1, 0.5), (Gesture.HELLO_2, 0.5),
            (Gesture.HELLO_1, 0.5), (Gesture.HELLO_2, 0.5),
            (Gesture.HELLO_1, 0.5), (Gesture.HELLO_2, 0.5),
            (Gesture.RESET, 0.5),
        ],
        "数字手势": [
            (Gesture.RESET, 0.8),
            (Gesture.NUM1, 0.8), (Gesture.NUM2, 0.8),
            (Gesture.NUM3, 0.8), (Gesture.NUM4, 0.8),
            (Gesture.NUM6, 0.8), (Gesture.NUM8, 0.8),
            (Gesture.CLASPING, 0.8),
            (Gesture.RESET, 0.8),
        ],
    }

    def __init__(self, port: str = "COM5"):
        self.ctrl = OmniHandSpeedController(port=port, verbose=False)
        self.ctrl.open()
        self._init_hand()

    def _init_hand(self):
        for attempt in range(1, 11):
            self.ctrl.enable()
            if self.ctrl.is_enabled():
                print(f"[Hand] 已使能 (port={self.ctrl.gesture.serial.port})")
                break
            time.sleep(0.5)
        else:
            print("[Hand] 警告: 使能失败")
        self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
        print("[Hand] 初始化完成 (RESET)")

    def execute(self, actions: list):
        for a in actions:
            typ = a.get("type")
            if typ == "stop":
                print("  [执行] 停止 RESET")
                self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
                return
            elif typ == "gesture":
                name = a["gesture"]
                print(f"  [执行] {name} speed={a.get('speed')}")
                self.ctrl.gesture.do_gesture(Gesture[name], delay=0.3)
            elif typ == "sequence":
                seq_name = a["sequence"]
                if seq_name == "custom":
                    steps = a.get("steps", [])
                    print(f"  [执行] 自定义序列 ({len(steps)}步)")
                    for s in steps:
                        g = Gesture[s["gesture"]]
                        d = s.get("duration", _GESTURE_DURATION)
                        self.ctrl.gesture.do_gesture(g, delay=d)
                elif seq_name == "握手":
                    print("  [执行] 握手序列 (压力检测)")
                    self._do_handshake()
                elif seq_name in self.SEQUENCE_STEPS:
                    print(f"  [执行] 序列: {seq_name}")
                    for g, d in self.SEQUENCE_STEPS[seq_name]:
                        self.ctrl.gesture.do_gesture(g, delay=d)

    def _do_handshake(self):
        from Omnihand.omnihand_gesture import Gesture as G
        self.ctrl.gesture.do_gesture(G.RESET, delay=0.5)
        print("  等待握手压力...")

        def _read():
            sensors = self.ctrl.gesture.serial.read_all_sensor_group_3()
            return int(np.asarray(sensors.get("back", []), dtype=np.int32).sum())

        bl = float(np.median([_read() for _ in range(5)]))
        deadline = time.time() + 30
        grabbed = False
        while time.time() < deadline:
            v = _read()
            if (bl < 50 and v >= 100) or (v - bl >= 100):
                print(f"  抓握! (value={v})")
                grabbed = True
                break
            time.sleep(0.15)
        if not grabbed:
            print("  [WARN] 握手超时")
            self.ctrl.gesture.do_gesture(G.RESET, delay=0.3)
            return
        self.ctrl.gesture.do_gesture(G.HAND_SHAKE, delay=1.0)
        time.sleep(2.0)
        self.ctrl.gesture.do_gesture(G.RESET, delay=0.5)
        print("  握手结束")

    def close(self):
        try:
            self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
        except Exception:
            pass
        try:
            self.ctrl.close()
        except Exception:
            pass
        print("[Hand] 已关闭")


# ============================================================
# 主入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="LLM 语音灵巧手控制")
    parser.add_argument("--port", default="COM5", help="灵巧手串口号")
    parser.add_argument("--api-key", default=None,
                        help="MiMo API Key (优先级最高, 覆盖 config 和环境变量)")
    args = parser.parse_args()

    # API Key 优先级: --api-key > voice_llm_config.local.json > voice_llm_config.json > MIMO_API_KEY
    api_key = args.api_key or _MIMO_API_KEY_FROM_CFG

    print("=" * 55)
    print("  语音灵巧手控制 — Voice LLM Control")
    print(f"  MiMo ({_MIMO_MODEL}) + edge-tts ({_TTS_VOICE})")
    print("=" * 55)

    if not api_key:
        print("\n[ERROR] 未找到 API Key. 请通过以下任一方式提供:")
        print("  1. --api-key sk-xxx")
        print("  2. voice_llm_config.local.json → \"api_key\": \"sk-xxx\"")
        print("  3. voice_llm_config.json → \"api_key\": \"sk-xxx\"")
        print("  4. 环境变量 MIMO_API_KEY")
        return
    print(f"\n[API] key={api_key[:16]}... (来源: "
          f"{'--api-key' if args.api_key else 'config'})")

    # System prompt
    print(f"[Prompt] {len(SYSTEM_PROMPT)} chars, ~{len(SYSTEM_PROMPT)//3} tokens")

    # 灵巧手
    print("\n[Hand] 初始化...")
    try:
        hand = GestureExecutor(port=args.port)
    except Exception as e:
        print(f"[ERROR] 灵巧手初始化失败: {e}")
        print("[WARN] 模拟模式")
        hand = None

    print(f"\n[模式] 按回车录音 → MiMo 理解 → 执行 + TTS")
    print("       'q' 退出\n")

    try:
        while True:
            if msvcrt.kbhit() and msvcrt.getch().lower() == b'q':
                print("\n[q] 退出")
                break

            audio = record_audio()
            if audio is None:
                continue

            # MiMo
            print("  [MiMo] 推理中...", end="", flush=True)
            t0 = time.time()
            wav_bytes = audio_to_wav_bytes(audio)
            response = call_mimo(wav_bytes, api_key)
            if response is None:
                print(f"\r  [MiMo] 失败")
                continue
            elapsed = time.time() - t0
            print(f"\r  [MiMo] {elapsed:.1f}s")
            print(f"  ── response ──\n{response}\n  ──────────────")

            # 解析
            data = parse_llm_response(response)
            actions = validate_actions(data) if data else []
            reply = extract_reply_text(response)

            # TTS (异步)
            if reply:
                speak(reply)
            elif actions:
                speak("好的")

            # 执行
            if hand and actions:
                hand.execute(actions)
            elif actions:
                names = []
                for a in actions:
                    t = a.get("type")
                    if t == "gesture":
                        names.append(a["gesture"])
                    elif t == "sequence":
                        names.append(a["sequence"])
                    elif t == "stop":
                        names.append("STOP")
                print(f"  [模拟] {' → '.join(names) if names else str(actions)}")

    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        print("\n清理...")
        if hand:
            hand.close()
        print("退出.")


if __name__ == "__main__":
    main()
