# -*- coding: utf-8 -*-
"""
语音灵巧手控制 — Voice OmniHand Control
========================================
基于 sherpa-onnx SenseVoice + 能量阈值 VAD, 语音指令控制 OmniHand 手势。

三种输入模式:
  - 空格键按下说话，松开识别（最可靠，默认）
  - 自动 VAD 检测语音段落
  - 'q' 退出

使用:
  python Omnihand/voice_omnihand_control.py
  python Omnihand/voice_omnihand_control.py --port COM5 --mode auto
"""

import os
import sys
import time
import wave
import queue
import threading
import argparse
from collections import OrderedDict
from typing import Optional, Tuple, List

import numpy as np
import msvcrt
import sounddevice as sd

# ---- 项目根目录 ----
_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import sherpa_onnx
from Omnihand.omnihand_gesture import Gesture, OmniHandGesture
from Omnihand.omnihand_speed_control import OmniHandSpeedController, SpeedLevel

# ============================================================
# 速度映射（覆盖 VERY_FAST: 5 steps × 0.15s = 0.75s）
# ============================================================

VOICE_SPEED_MAP = {
    "fast":       {"duration": 0.75, "steps": 5,  "easing": "linear"},
    "medium":     {"duration": 1.0,  "steps": 20, "easing": "ease_in_out"},
    "very_slow":  {"duration": 3.0,  "steps": 60, "easing": "ease_in_out"},
}

SPEED_PREFIXES = OrderedDict({
    "fast":      ["快速", "快"],
    "medium":    ["中速", "普通"],
    "very_slow": ["慢速", "慢"],
})
# 无前缀 → do_gesture() 直接动作 (最快, 无插值)

# ============================================================
# 指令表
# ============================================================

SINGLE_GESTURES: List[Tuple[Gesture, List[str]]] = [
    (Gesture.RESET,      ["复位", "恢复", "归零", "重置"]),
    (Gesture.PAPER,      ["张开", "松手", "手掌", "摊开"]),
    (Gesture.FIST,       ["握拳"]),
    (Gesture.OK,         ["OK", "ok"]),
    (Gesture.HEART,      ["比心", "爱心"]),
    (Gesture.LIKE,       ["点赞", "厉害"]),
    (Gesture.ILY,        ["我爱你", "ILY"]),
    (Gesture.NUM1,       ["数字1", "一"]),
    (Gesture.NUM2,       ["数字2", "二"]),
    (Gesture.NUM3,       ["数字3", "三"]),
    (Gesture.NUM4,       ["数字4", "四"]),
    (Gesture.NUM6,       ["数字6", "六"]),
    (Gesture.NUM8,       ["数字8", "八"]),
    (Gesture.CLASPING,   ["合十", "拜拜", "双手合十"]),
    (Gesture.MIDDLE,     ["中指", "竖中指"]),
    (Gesture.POINT,      ["指人", "指向", "指着"]),
    (Gesture.GRIP,       ["抓取", "握住"]),
    (Gesture.GRIPDONE,   ["抓紧"]),
    (Gesture.PREGRIP,    ["半握", "准备抓"]),
    (Gesture.HAND_SHAKE, ["握手"]),
    (Gesture.HELLO_1,    ["挥手1", "打招呼1"]),
    (Gesture.HELLO_2,    ["挥手2", "打招呼2"]),
]

SEQUENCES = OrderedDict({
    "打招呼": [
        (Gesture.RESET, 0.5),
        (Gesture.HELLO_1, 0.5),
        (Gesture.HELLO_2, 0.5),
        (Gesture.HELLO_1, 0.5),
        (Gesture.HELLO_2, 0.5),
        (Gesture.HELLO_1, 0.5),
        (Gesture.HELLO_2, 0.5),
        (Gesture.RESET, 0.5),
    ],
    "握手": [
        # 特殊序列 — 需要压力检测，调用 Phase 0 逻辑
    ],
    "数字手势": [
        (Gesture.RESET, 0.5),
        (Gesture.NUM1, 0.5),
        (Gesture.NUM2, 0.5),
        (Gesture.NUM3, 0.5),
        (Gesture.NUM4, 0.5),
        (Gesture.NUM6, 0.5),
        (Gesture.NUM8, 0.5),
        (Gesture.CLASPING, 0.5),
        (Gesture.RESET, 0.5),
    ],
})

# 序列语音触发词
SEQUENCE_TRIGGERS = OrderedDict({
    "打招呼":   ["打招呼", "挥手"],
    "握手":     ["握手序列", "完整握手"],
    "数字手势": ["数字手势", "一到十"],
})

# ============================================================
# 命令匹配
# ============================================================

def match_command(text: str) -> Optional[Tuple[str, Optional[Gesture], Optional[str]]]:
    """
    匹配语音文本 → (命令类型, Gesture, 速度键).

    防误触: 关键词必须在文本中占显著比例, 避免长对话中短词命中.

    返回:
      ("gesture", gesture, speed_key)  — 单手势
      ("sequence", None, 序列名)       — 手势序列
      ("exit", None, None)             — 退出
      None                             — 未匹配
    """
    if not text:
        return None

    text_lower = text.lower().strip()
    text_len = len(text_lower)

    # 停止命令 (全局生效, 触发词较长, 直接子串匹配)
    for kw in ["停止", "急停", "暂停", "停", "停下", "别动", "不动", "不要动", "别做了", "快停", "停一下"]:
        if kw in text_lower:
            return ("stop", None, None)

    # 退出命令
    for kw in ["退出", "结束", "关闭"]:
        if kw in text_lower:
            return ("exit", None, None)

    # 检查速度前缀
    speed_key = None
    clean_text = text_lower
    for sk, prefixes in SPEED_PREFIXES.items():
        for prefix in prefixes:
            if clean_text.startswith(prefix):
                speed_key = sk
                clean_text = clean_text[len(prefix):]
                break
        if speed_key:
            break

    # 匹配手势 — 关键词占比检查
    best_match = None
    best_len = 0
    for gesture, phrases in SINGLE_GESTURES:
        for phrase in phrases:
            if phrase in text_lower:
                # 防误触: 关键词占比检查
                # 文本 > 6 字且关键词占比 < 30% → 视为对话中的偶然命中, 跳过
                ratio = len(phrase) / text_len
                if text_len > 6 and ratio < 0.3:
                    continue
                if len(phrase) > best_len:
                    best_len = len(phrase)
                    best_match = gesture

    if best_match:
        return ("gesture", best_match, speed_key)

    # 匹配序列 (通过触发词)
    for seq_name, triggers in SEQUENCE_TRIGGERS.items():
        for trigger in triggers:
            if trigger in text_lower:
                return ("sequence", None, seq_name)

    return None


# ============================================================
# 手控制
# ============================================================

class HandController:
    """灵巧手 + 语音速度控制."""

    def __init__(self, port: str = "COM5"):
        self.ctrl = OmniHandSpeedController(port=port, verbose=False)
        self.ctrl.open()
        self._stop_requested = False
        self._init_hand()

    def _init_hand(self):
        """初始化灵巧手，带使能重试."""
        for attempt in range(1, 11):
            self.ctrl.enable()
            if self.ctrl.is_enabled():
                print(f"[Hand] 灵巧手已使能 (port={self.ctrl.gesture.serial.port})")
                break
            print(f"[Hand] 使能重试 {attempt}/10 ...")
            time.sleep(0.5)
        else:
            print("[Hand] 警告: 使能失败，将以模拟模式运行")

        self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
        print("[Hand] 初始化完成 (RESET)")

    def do_gesture(self, gesture: Gesture, speed_key: Optional[str] = None):
        """执行单手势.

        speed_key=None → 直接 do_gesture() (最快, 无插值)
        speed_key=fast/medium/very_slow → 软件插值
        """
        name = {g: p[0] for g, p in SINGLE_GESTURES}.get(gesture, gesture.name)

        if speed_key is None:
            print(f"[执行] {name} (直接, 无插值)")
            self.ctrl.gesture.do_gesture(gesture, delay=0.3)
        else:
            speed_params = VOICE_SPEED_MAP[speed_key]
            print(f"[执行] {name} speed={speed_key} "
                  f"({speed_params['duration']}s, {speed_params['steps']} steps)")
            self.ctrl.set_positions_interpolated(
                KinematicsConverter.get_gesture_positions(gesture, self.ctrl.is_right_hand),
                duration=speed_params["duration"],
                steps=speed_params["steps"],
                easing=speed_params["easing"],
            )

    def do_sequence(self, seq_name: str):
        """执行手势序列.

        特殊处理 "握手" — 需要压力检测.
        """
        if seq_name == "握手":
            self._do_handshake()
            return

        seq = SEQUENCES.get(seq_name)
        if not seq:
            print(f"[WARN] 未知序列: {seq_name}")
            return

        self._stop_requested = False
        print(f"[执行] 序列: {seq_name} ({len(seq)} 步)")
        for gesture, duration in seq:
            if self._stop_requested:
                print("  [中断] 序列停止")
                self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
                return
            self.ctrl.gesture.do_gesture(gesture, delay=duration)

    def _do_handshake(self):
        """握手序列 — 移植自 Phase 0 GreetingHandler."""
        from Omnihand.omnihand_gesture import Gesture as G

        print("[执行] 序列: 握手 (含压力检测)")
        self.ctrl.gesture.do_gesture(G.RESET, delay=0.5)
        print("  等待握手压力...")

        # 压力检测参数 (对齐 Phase 0)
        force_timeout = 30.0
        force_abs_threshold = 100
        force_delta_threshold = 100
        force_baseline_samples = 5
        force_poll_interval = 0.15
        force_release_timeout = 60.0
        force_release_delta = 50
        force_release_hold = 2.0

        def _read_back_pressure():
            sensors = self.ctrl.gesture.serial.read_all_sensor_group_3()
            return int(np.asarray(sensors.get("back", []), dtype=np.int32).sum())

        # 基线
        bl_samples = []
        for _ in range(force_baseline_samples):
            bl_samples.append(_read_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(bl_samples, dtype=np.float64)))

        # 等抓握
        deadline = time.time() + force_timeout
        grabbed = False
        while time.time() < deadline:
            value = _read_back_pressure()
            delta = value - baseline
            abs_trigger = (baseline < force_abs_threshold * 0.5
                           and value >= force_abs_threshold)
            delta_trigger = (delta >= force_delta_threshold)
            if abs_trigger or delta_trigger:
                print(f"  检测到抓握! (value={value}, delta={delta:.0f})")
                grabbed = True
                break
            time.sleep(force_poll_interval)

        if not grabbed:
            print(f"[WARN] 握手压力超时 ({force_timeout}s)")
            self.ctrl.gesture.do_gesture(G.RESET, delay=0.3)
            return

        # 执行握手
        self.ctrl.gesture.do_gesture(G.HAND_SHAKE, delay=1.0)
        print("  等待松手...")

        # 等松手
        bl_samples = []
        for _ in range(force_baseline_samples):
            bl_samples.append(_read_back_pressure())
            time.sleep(0.05)
        baseline = float(np.median(np.asarray(bl_samples, dtype=np.float64)))
        released_since = None
        deadline = time.time() + force_release_timeout
        released = False
        while time.time() < deadline:
            value = _read_back_pressure()
            delta = value - baseline
            if delta <= force_release_delta:
                if released_since is None:
                    released_since = time.time()
                if time.time() - released_since >= force_release_hold:
                    print(f"  已松手! (value={value}, delta={delta:.0f})")
                    released = True
                    break
            else:
                released_since = None
            time.sleep(force_poll_interval)

        if not released:
            print(f"[WARN] 松手超时 ({force_release_timeout}s)")

        self.ctrl.gesture.do_gesture(G.RESET, delay=0.5)
        print("[执行] 握手结束")

    def stop(self):
        """急停 — 中断序列 + 直接 RESET."""
        self._stop_requested = True
        self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
        print("  [Hand] 已停止 (RESET)")

    def close(self):
        """安全关闭."""
        try:
            self.ctrl.gesture.do_gesture(Gesture.RESET, delay=0.3)
        except Exception:
            pass
        try:
            self.ctrl.disable()
        except Exception:
            pass
        try:
            self.ctrl.close()
        except Exception:
            pass
        print("[Hand] 已关闭")


# 为了访问 KinematicsConverter
from Omnihand.omnihand_gesture import KinematicsConverter


# ============================================================
# ASR
# ============================================================

class SpeechRecognizer:
    """sherpa-onnx SenseVoice 语音识别器."""

    def __init__(self, model_dir: str):
        model_path = os.path.join(model_dir, "model.int8.onnx")
        tokens_path = os.path.join(model_dir, "tokens.txt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型未找到: {model_path}")
        if not os.path.exists(tokens_path):
            raise FileNotFoundError(f"词表未找到: {tokens_path}")

        self.recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=model_path,
            tokens=tokens_path,
            num_threads=2,
            use_itn=True,
            language="zh",
        )
        self._temp_wav = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "_voice_temp.wav"
        )
        print(f"[ASR] SenseVoice int8 已加载")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """转写音频片段."""
        # 保存临时 WAV
        data = (audio * 32767).astype(np.int16).tobytes()
        with wave.open(self._temp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(data)

        # ASR 推理
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio.astype(np.float32))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def close(self):
        if os.path.exists(self._temp_wav):
            try:
                os.remove(self._temp_wav)
            except Exception:
                pass


# ============================================================
# 麦克风设备选择
# ============================================================

def find_mic(keywords: Optional[list] = None) -> Optional[int]:
    """查找麦克风设备 ID.

    优先级: UGREEN Camera Audio > 关键词匹配 > 系统默认.
    返回 None 表示用系统默认.
    """
    if keywords is None:
        keywords = ["UGREEN"]
    devices = sd.query_devices()
    for keyword in keywords:
        for i, d in enumerate(devices):
            if d["max_input_channels"] > 0 and keyword.lower() in d["name"].lower():
                print(f"[Mic] 选择: [{i}] {d['name']}")
                return i
    # fallback: 系统默认
    default = sd.default.device[0]
    if default is not None:
        try:
            name = devices[default]["name"]
        except (IndexError, TypeError):
            name = f"device {default}"
        print(f"[Mic] 默认: [{default}] {name}")
    return None  # None = 用 sd.default


# ============================================================
# 输入模式
# ============================================================

class VoiceInputBase:
    """输入模式基类."""

    def __init__(self, asr: SpeechRecognizer):
        self.asr = asr
        self.running = True
        self.speech_queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_worker, daemon=True)
        self._worker.start()
        self._mic_device = find_mic()  # None = 系统默认

    def _process_worker(self):
        """ASR 处理线程（避免 CFFI 回调中做文件 IO）."""
        while self.running:
            try:
                audio = self.speech_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if audio is None:
                break
            text = self.asr.transcribe(audio)
            if text:
                print(f"\n>> {text}")
                self.on_result(text)

    def on_result(self, text: str):
        """子类重写，处理识别结果."""
        pass

    def stop(self):
        self.running = False
        self.speech_queue.put(None)
        self._worker.join(timeout=2)


class PushToTalkInput(VoiceInputBase):
    """
    空格键按下说话，松开识别。

    主线程阻塞在 cv2.waitKey（需要 cv2 窗口，或用 keyboard 库）。
    简化版：用 input() 模拟 — 按回车开始录音，回车结束。
    """

    SAMPLERATE = 16000
    DURATION = 3.0  # 单次录音最长时长

    def run(self):
        print("\n[模式] 回车键 = 开始录音(3s)，说话后自动识别")
        print("       'q' = 退出\n")

        while self.running:
            cmd = input("按回车开始录音 (q=退出): ").strip().lower()
            if cmd == "q":
                self.running = False
                break

            print("  [录音 3s...]")
            audio = sd.rec(
                int(self.DURATION * self.SAMPLERATE),
                samplerate=self.SAMPLERATE,
                channels=1,
                dtype="float32",
                device=self._mic_device,
            )
            sd.wait()
            audio = audio.flatten()

            # VAD 裁剪：去掉首尾静音
            rms = np.sqrt(np.mean(audio ** 2))
            if rms < 0.01:
                print("  [跳过] 未检测到语音")
                continue

            self.speech_queue.put(audio)


class AutoVADInput(VoiceInputBase):
    """
    自动 VAD 模式：持续监听，能量阈值检测语音段落。
    """

    SAMPLERATE = 16000
    BLOCKSIZE = 512
    SPEECH_THRESHOLD = 0.015
    SILENCE_S = 1.2
    MAX_SPEECH_S = 10.0
    MIN_SPEECH_S = 0.3

    def __init__(self, asr: SpeechRecognizer):
        super().__init__(asr)
        self._buffer: List[np.ndarray] = []
        self._in_speech = False
        self._silence_count = 0
        self._silence_blocks = int(self.SILENCE_S * self.SAMPLERATE / self.BLOCKSIZE)
        self._max_blocks = int(self.MAX_SPEECH_S * self.SAMPLERATE / self.BLOCKSIZE)
        self._min_blocks = int(self.MIN_SPEECH_S * self.SAMPLERATE / self.BLOCKSIZE)

    def _callback(self, indata, frames, time_info, status):
        if not self.running:
            raise sd.CallbackStop

        chunk = indata[:, 0].copy()
        rms = float(np.sqrt(np.mean(chunk ** 2)))

        if rms > self.SPEECH_THRESHOLD:
            if not self._in_speech:
                self._in_speech = True
                self._buffer = []
                print("\n[VAD]", end="", flush=True)
            self._buffer.append(chunk)
            self._silence_count = 0
        elif self._in_speech:
            self._buffer.append(chunk)
            self._silence_count += 1

            # 超长截断
            if len(self._buffer) >= self._max_blocks:
                audio = np.concatenate(self._buffer)
                self.speech_queue.put(audio)
                self._buffer = []
                self._in_speech = False
                self._silence_count = 0
            # 静音结束
            elif self._silence_count >= self._silence_blocks:
                if len(self._buffer) >= self._min_blocks:
                    audio = np.concatenate(self._buffer)
                    self.speech_queue.put(audio)
                else:
                    print(" (too short)", end="", flush=True)
                self._buffer = []
                self._in_speech = False
                self._silence_count = 0

    def run(self):
        print(f"\n[模式] 自动VAD (阈值={self.SPEECH_THRESHOLD}, "
              f"静音={self.SILENCE_S}s, 最长={self.MAX_SPEECH_S}s)")
        print("       直接说话即可，'q' 键退出\n")

        try:
            with sd.InputStream(
                samplerate=self.SAMPLERATE,
                channels=1,
                dtype="float32",
                blocksize=self.BLOCKSIZE,
                callback=self._callback,
                device=self._mic_device,
            ):
                while self.running:
                    if msvcrt.kbhit() and msvcrt.getch().lower() == b'q':
                        print("\n[q] 退出")
                        self.running = False
                        break
                    time.sleep(0.1)
        except KeyboardInterrupt:
            pass


# ============================================================
# 主入口
# ============================================================

def load_asr_model() -> str:
    """查找 SenseVoice 模型目录."""
    candidates = [
        os.path.join(_project_root, "model",
                     "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09"),
        os.path.join(_project_root, "model",
                     "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"),
    ]
    for d in candidates:
        if os.path.exists(os.path.join(d, "model.int8.onnx")):
            return d
    raise FileNotFoundError(
        "SenseVoice int8 模型未找到。请下载到 model/ 目录。\n"
        "  候选路径: " + ", ".join(candidates)
    )


def main():
    parser = argparse.ArgumentParser(description="语音灵巧手控制")
    parser.add_argument("--port", default="COM5", help="灵巧手串口号")
    parser.add_argument("--mode", default="ptt",
                        choices=["ptt", "auto"],
                        help="输入模式: ptt=按键说话, auto=自动VAD")
    args = parser.parse_args()

    print("=" * 55)
    print("  语音灵巧手控制 — Voice OmniHand Control")
    print("  sherpa-onnx SenseVoice + OmniHand")
    print("=" * 55)

    # 1. 初始化手
    print("\n[1/3] 初始化灵巧手...")
    try:
        hand = HandController(port=args.port)
    except Exception as e:
        print(f"[ERROR] 灵巧手初始化失败: {e}")
        print("[WARN] 将以模拟模式运行（仅测试 ASR）")
        hand = None

    # 2. 加载 ASR
    print("\n[2/3] 加载 ASR 模型...")
    model_dir = load_asr_model()
    print(f"  模型目录: {model_dir}")
    asr = SpeechRecognizer(model_dir)

    # 3. 创建输入模式
    print(f"\n[3/3] 启动输入模式: {args.mode}")
    if args.mode == "auto":
        voice_input = AutoVADInput(asr)
    else:
        voice_input = PushToTalkInput(asr)

    def on_result(text: str):
        """处理 ASR 结果，执行手势."""
        result = match_command(text)
        if result is None:
            print(f"  [未匹配] 文本: '{text}'")
            return

        cmd_type, gesture, extra = result

        if cmd_type == "stop":
            print("  [停止] RESET")
            if hand:
                hand.stop()
            return

        if cmd_type == "exit":
            print("  [退出]")
            voice_input.running = False
            return

        if hand is None:
            print(f"  [模拟] {cmd_type}: {extra or gesture}")
            return

        if cmd_type == "gesture" and gesture:
            hand.do_gesture(gesture, speed_key=extra)
        elif cmd_type == "sequence" and extra:
            hand.do_sequence(extra)

    voice_input.on_result = on_result

    try:
        voice_input.run()
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        print("\n清理...")
        voice_input.stop()
        if hand:
            hand.close()
        asr.close()
        print("退出。")


if __name__ == "__main__":
    main()
