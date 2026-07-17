# -*- coding: utf-8 -*-
"""
语音灵巧手控制 — Voice OmniHand Control
========================================
基于 sherpa-onnx SenseVoice (全量 887MB / int8 227MB) + Silero VAD,
语音指令控制 OmniHand 手势。

两种输入模式 (默认 auto):
  - auto: Silero VAD 实时检测语音段落 → OfflineRecognizer 逐段识别
  - ptt:  回车键录音 3s 后识别（不回显，无 VAD）
  - 'q' 退出

流式方案:
  SenseVoice 仅支持 OfflineRecognizer, 通过 Silero VAD 实时切分
  实现伪流式: VAD 静音等待 1s + ASR 推理 ~200-500ms = ~1.5s 端到端延迟.

使用:
  python Omnihand/voice_omnihand_control.py
  python Omnihand/voice_omnihand_control.py --port COM5 --mode ptt
  python Omnihand/voice_omnihand_control.py --model int8 --mic ugreen
"""

import os
import sys
import time
import json
import queue
import threading
import argparse
from difflib import SequenceMatcher
from collections import OrderedDict
from typing import Optional, Tuple, List, Union, Dict, Any

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
# 配置加载 — 从 JSON 读取触发词, 拼音预计算
# ============================================================

def _load_commands() -> Dict[str, Any]:
    """加载 voice_commands.json, 失败时返回硬编码默认值."""
    json_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "voice_commands.json"
    )
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        print(f"[Config] 已加载 {json_path}")
        return cfg
    except Exception as e:
        print(f"[Config] 加载失败 ({e}), 使用内置默认值")
        return _default_commands()


def _default_commands() -> Dict[str, Any]:
    """内置默认指令配置 (与 voice_commands.json 保持同步)."""
    return {
        "stop_words": [
            "停止", "急停", "暂停", "停", "停下",
            "别动", "不动", "不要动", "别做了", "快停", "停一下",
        ],
        "exit_words": ["退出", "结束", "关闭"],
        "sequences": {
            "打招呼":   {"triggers": ["打招呼", "挥手", "招手", "打个招呼"]},
            "握手":     {"triggers": ["握手"]},
            "数字手势": {"triggers": ["数字手势", "一到十", "数字", "比数字", "数数", "一到8"]},
        },
        "gestures": {
            "RESET":      {"triggers": ["复位", "恢复", "归零", "重置"]},
            "PAPER":      {"triggers": ["张开", "松手", "手掌", "摊开", "五指张开"]},
            "FIST":       {"triggers": ["握拳", "攥拳", "拳头"]},
            "OK":         {"triggers": ["OK", "ok", "好的"]},
            "HEART":      {"triggers": ["比心", "爱心"]},
            "LIKE":       {"triggers": ["点赞", "厉害"]},
            "ILY":        {"triggers": ["我爱你", "ILY"]},
            "NUM1":       {"triggers": ["数字1", "一"]},
            "NUM2":       {"triggers": ["数字2", "二"]},
            "NUM3":       {"triggers": ["数字3", "三"]},
            "NUM4":       {"triggers": ["数字4", "四"]},
            "NUM6":       {"triggers": ["数字6", "六"]},
            "NUM8":       {"triggers": ["数字8", "八"]},
            "CLASPING":   {"triggers": ["合十", "拜拜", "双手合十", "双手合拢"]},
            "MIDDLE":     {"triggers": ["中指", "竖中指"]},
            "POINT":      {"triggers": ["指人", "指向", "指着"]},
            "GRIP":       {"triggers": ["抓取", "抓住", "握住"]},
            "GRIPDONE":   {"triggers": ["抓紧", "握紧"]},
            "PREGRIP":    {"triggers": ["半握", "准备抓", "预备抓"]},
            "HAND_SHAKE": {"triggers": ["握手手势", "简单握手", "直接握手"]},
            "HELLO_1":    {"triggers": ["挥手1", "打招呼1"]},
            "HELLO_2":    {"triggers": ["挥手2", "打招呼2"]},
        },
        "matching": {
            "char_threshold": 0.55,
            "pinyin_enabled": True,
            "pinyin_threshold": 0.75,
        },
    }


# ---- 加载配置 ----
_COMMANDS_CFG = _load_commands()
_MATCH_CFG = _COMMANDS_CFG.get("matching", {})

# ---- 构建运行时数据结构 ----
def _build_gesture_list(cfg: Dict[str, Any]) -> List[Tuple[Gesture, List[str]]]:
    """从 JSON gestures 段构建 SINGLE_GESTURES."""
    result = []
    for name, entry in cfg.get("gestures", {}).items():
        result.append((Gesture[name], entry["triggers"]))
    return result


SINGLE_GESTURES = _build_gesture_list(_COMMANDS_CFG)

SEQUENCE_TRIGGERS: "OrderedDict[str, List[str]]" = OrderedDict()
for _sn, _se in _COMMANDS_CFG.get("sequences", {}).items():
    SEQUENCE_TRIGGERS[_sn] = _se["triggers"]

# 手势名称映射 (供 HandController 使用)
GESTURE_NAMES: Dict[Gesture, str] = {
    g: p[0] for g, p in SINGLE_GESTURES
}

# ---- 拼音索引 (预计算, 供模糊匹配) ----
_PINYIN_ENABLED = False
_TRIGGER_PINYIN: List[Tuple[str, str, Union[str, Gesture], str]] = []

if _MATCH_CFG.get("pinyin_enabled", True):
    try:
        from pypinyin import pinyin as _pypinyin_fn, Style as _PinyinStyle

        def _to_pinyin(text: str) -> str:
            return "".join(p[0] for p in _pypinyin_fn(text, style=_PinyinStyle.TONE3))

        # 预计算所有 trigger 的拼音
        for seq_name, triggers in SEQUENCE_TRIGGERS.items():
            for t in triggers:
                _TRIGGER_PINYIN.append((_to_pinyin(t), "seq", seq_name, t))
        for gesture, phrases in SINGLE_GESTURES:
            for p in phrases:
                if len(p) > 1:  # 单字不参与拼音匹配
                    _TRIGGER_PINYIN.append((_to_pinyin(p), "ges", gesture, p))
        _PINYIN_ENABLED = True
        print(f"[Pinyin] 已启用 (pypinyin, {len(_TRIGGER_PINYIN)} 条拼音索引)")
    except ImportError:
        print("[Pinyin] pypinyin 未安装, 拼音匹配已禁用")
    except Exception as e:
        print(f"[Pinyin] 初始化失败 ({e}), 拼音匹配已禁用")


# ============================================================
# 指令表 (序列步骤 - 引用 Gesture 枚举, 保留在 Python)
# ============================================================

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
        (Gesture.RESET, 0.8),
        (Gesture.NUM1, 0.8),
        (Gesture.NUM2, 0.8),
        (Gesture.NUM3, 0.8),
        (Gesture.NUM4, 0.8),
        (Gesture.NUM6, 0.8),
        (Gesture.NUM8, 0.8),
        (Gesture.CLASPING, 0.8),
        (Gesture.RESET, 0.8),
    ],
})

# ============================================================
# 命令匹配
# ============================================================

def match_command(text: str) -> Optional[Tuple[str, Optional[Gesture], Optional[str]]]:
    """
    匹配语音文本 → (命令类型, Gesture, 速度键).

    匹配流水线:
      1. 全局命令 (停止/退出)
      2. 速度前缀剥离
      3. 精确子串匹配 — 字符级, 最长优先
      4. 字符模糊匹配 — SequenceMatcher ≥ char_threshold
      5. 拼音模糊匹配 — pypinyin + SequenceMatcher ≥ pinyin_threshold (新增)

    防误触:
      - 1 字触发词: 精确匹配, 不参与模糊/拼音
      - 序列 vs 手势冲突: 最长触发词优先
      - 关键词占比检查 (长对话中短词跳过)

    返回:
      ("gesture", gesture, speed_key)  — 单手势
      ("sequence", None, 序列名)       — 手势序列
      ("stop", None, None)             — 停止
      ("exit", None, None)             — 退出
      None                             — 未匹配
    """
    if not text:
        return None

    text_lower = text.lower().strip()
    text_len = len(text_lower)

    # ---- 1. 全局命令 ----
    for kw in _COMMANDS_CFG.get("stop_words", []):
        if kw in text_lower:
            return ("stop", None, None)
    for kw in _COMMANDS_CFG.get("exit_words", []):
        if kw in text_lower:
            return ("exit", None, None)

    # ---- 2. 速度前缀剥离 ----
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

    # ---- 3. 统一匹配 (精确子串 + 字符模糊 + 拼音模糊) ----
    char_threshold = _MATCH_CFG.get("char_threshold", 0.55)
    pinyin_threshold = _MATCH_CFG.get("pinyin_threshold", 0.75)

    # 构建候选列表
    candidates: List[Tuple[str, Union[str, Gesture], str, str]] = []
    for seq_name, triggers in SEQUENCE_TRIGGERS.items():
        for trigger in triggers:
            candidates.append(("seq", seq_name, trigger, text_lower))
    for gesture, phrases in SINGLE_GESTURES:
        for phrase in phrases:
            candidates.append(("ges", gesture, phrase, clean_text))

    # -- 3a. 精确子串匹配 --
    best_type, best_seq_name, best_gesture = _exact_match(candidates)
    if best_type == "seq" and best_seq_name is not None:
        return ("sequence", None, best_seq_name)
    if best_type == "ges" and best_gesture is not None:
        return ("gesture", best_gesture, speed_key)

    # -- 3b. 字符模糊匹配 --
    best_type, best_seq_name, best_gesture = _fuzzy_match(candidates, char_threshold)
    if best_type == "seq" and best_seq_name is not None:
        return ("sequence", None, best_seq_name)
    if best_type == "ges" and best_gesture is not None:
        return ("gesture", best_gesture, speed_key)

    # -- 3c. 拼音模糊匹配 --
    if _PINYIN_ENABLED:
        best_type, best_seq_name, best_gesture = _pinyin_match(
            clean_text, text_lower, pinyin_threshold
        )
        if best_type == "seq" and best_seq_name is not None:
            return ("sequence", None, best_seq_name)
        if best_type == "ges" and best_gesture is not None:
            return ("gesture", best_gesture, speed_key)

    return None


def _exact_match(
    candidates: List[Tuple[str, Union[str, Gesture], str, str]]
) -> Tuple[Optional[str], Optional[str], Optional[Gesture]]:
    """精确子串匹配, 最长触发词优先."""
    best_type: Optional[str] = None
    best_seq_name: Optional[str] = None
    best_gesture: Optional[Gesture] = None
    best_trigger_len = 0
    for typ, val, trigger, search in candidates:
        if len(trigger) == 1:
            if trigger != search:
                continue
        elif trigger not in search:
            continue
        ratio = len(trigger) / len(search) if search else 1
        if len(search) > 6 and ratio < 0.3:
            continue
        if len(trigger) > best_trigger_len:
            best_trigger_len = len(trigger)
            best_type = typ
            if typ == "seq":
                best_seq_name = val  # type: ignore[assignment]
                best_gesture = None
            else:
                best_seq_name = None
                best_gesture = val  # type: ignore[assignment]
    return best_type, best_seq_name, best_gesture


def _fuzzy_match(
    candidates: List[Tuple[str, Union[str, Gesture], str, str]],
    threshold: float,
) -> Tuple[Optional[str], Optional[str], Optional[Gesture]]:
    """字符级模糊匹配, 仅多字候选."""
    best_type: Optional[str] = None
    best_seq_name: Optional[str] = None
    best_gesture: Optional[Gesture] = None
    best_score = 0.0
    for typ, val, trigger, search in candidates:
        if len(trigger) <= 1:
            continue
        if abs(len(search) - len(trigger)) > max(len(search), len(trigger)) * 0.5:
            continue
        score = SequenceMatcher(None, search, trigger).ratio()
        if score > best_score:
            best_score = score
            best_type = typ
            if typ == "seq":
                best_seq_name = val  # type: ignore[assignment]
                best_gesture = None
            else:
                best_seq_name = None
                best_gesture = val  # type: ignore[assignment]
    if best_score >= threshold:
        return best_type, best_seq_name, best_gesture
    return None, None, None


def _pinyin_match(
    clean_text: str,
    text_lower: str,
    threshold: float,
) -> Tuple[Optional[str], Optional[str], Optional[Gesture]]:
    """拼音模糊匹配 — 整句转 TONE3 拼音, 与预计算 trigger 拼音比较.

    微小差异自动覆盖 (threshold=0.75):
      - 声调不同: "wo4shou3" vs "wo3shou3" → 7/8 = 0.875 ✓
      - 前后鼻音: "xin1" vs "xing1" → 4/5 = 0.889 ✓
      - 同音同调: "wo4shou3" vs "wo4shou3" → 1.000 ✓

    字符闸门: 输入与 trigger 原始汉字 SequenceMatcher ≥ 0.25,
    防止纯拼音碰撞误匹配 (如 "我全"→"握拳" 拼音同但汉字无关).
    """
    try:
        text_py = "".join(
            p[0] for p in _pypinyin_fn(clean_text, style=_PinyinStyle.TONE3)
        )
    except Exception:
        return None, None, None

    best_type: Optional[str] = None
    best_seq_name: Optional[str] = None
    best_gesture: Optional[Gesture] = None
    best_score = 0.0
    for trigger_py, typ, val, trigger in _TRIGGER_PINYIN:
        # 字符闸门: 原始汉字至少有一点重叠才比较拼音
        search_text = text_lower if typ == "seq" else clean_text
        if SequenceMatcher(None, search_text, trigger).ratio() < 0.25:
            continue
        score = SequenceMatcher(None, text_py, trigger_py).ratio()
        if score > best_score:
            best_score = score
            best_type = typ
            if typ == "seq":
                best_seq_name = val  # type: ignore[assignment]
                best_gesture = None
            else:
                best_seq_name = None
                best_gesture = val  # type: ignore[assignment]

    if best_score >= threshold:
        return best_type, best_seq_name, best_gesture
    return None, None, None


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
        name = GESTURE_NAMES.get(gesture, gesture.name)

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

    def __init__(self, model_dir: str, model_type: str = "auto"):
        # 模型选择: full > int8
        full = os.path.join(model_dir, "model.onnx")
        int8 = os.path.join(model_dir, "model.int8.onnx")
        if model_type == "full":
            model_path = full
        elif model_type == "int8":
            model_path = int8
        else:  # auto: 完整版优先
            model_path = full if os.path.exists(full) else int8
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
        model_name = os.path.basename(model_path)
        print(f"[ASR] SenseVoice 已加载 ({model_name})")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """转写音频片段."""
        stream = self.recognizer.create_stream()
        stream.accept_waveform(sample_rate, audio.astype(np.float32))
        self.recognizer.decode_stream(stream)
        return stream.result.text.strip()

    def close(self):
        pass


# ============================================================
# 麦克风设备选择
# ============================================================

def find_mic(preference: str = "auto") -> Optional[int]:
    """查找麦克风设备 ID.

    preference: auto=UGREEN优先, ugreen=仅UGREEN, default=系统默认.
    返回 None 表示用系统默认.
    """
    if preference == "default":
        print(f"[Mic] 系统默认")
        return None

    keywords = ["UGREEN"] if preference in ("auto", "ugreen") else []
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

    def __init__(self, asr: SpeechRecognizer, mic_pref: str = "auto"):
        self.asr = asr
        self.running = True
        self.speech_queue = queue.Queue()
        self._worker = threading.Thread(target=self._process_worker, daemon=True)
        self._worker.start()
        self._mic_device = find_mic(mic_pref)  # None = 系统默认

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
    按键说话模式 (PTT): 回车键开始录音 3s，自动识别。
    简化版实现，无回声消除，适合快速测试。
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
    自动 VAD 模式：sherpa-onnx VoiceActivityDetector + silero_vad.onnx.
    持续监听，神经网络 VAD 检测语音段落。
    """

    SAMPLERATE = 16000
    BLOCKSIZE = 512  # 对齐 Silero VAD window_size

    def __init__(self, asr: SpeechRecognizer, mic_pref: str = "auto"):
        super().__init__(asr, mic_pref)
        # Silero VAD 初始化
        silero_path = os.path.join(_project_root, "model", "silero_vad.onnx")
        if not os.path.exists(silero_path):
            raise FileNotFoundError(f"Silero VAD 模型未找到: {silero_path}")

        silero_cfg = sherpa_onnx.SileroVadModelConfig(
            model=silero_path,
            threshold=0.5,
            min_speech_duration=0.25,
            min_silence_duration=1.0,
            max_speech_duration=10.0,
        )
        vad_cfg = sherpa_onnx.VadModelConfig(
            silero_vad=silero_cfg,
            sample_rate=self.SAMPLERATE,
            num_threads=2,
        )
        self._detector = sherpa_onnx.VoiceActivityDetector(
            vad_cfg, buffer_size_in_seconds=60
        )
        self._speech_active = False
        print(f"[VAD] Silero VAD (silero_vad.onnx, threshold=0.5, "
              f"min_silence=1.0s, max_speech=10s)")

    def _callback(self, indata, frames, time_info, status):
        if not self.running:
            raise sd.CallbackStop

        chunk = indata[:, 0].copy()
        self._detector.accept_waveform(chunk)

        # 语音活动指示（终端显示）
        if self._detector.is_speech_detected():
            if not self._speech_active:
                self._speech_active = True
                print("\n[VAD]", end="", flush=True)
        else:
            self._speech_active = False

        # 检查完成的语音段
        while not self._detector.empty():
            seg = self._detector.front  # 属性，非方法
            audio = np.array(seg.samples, dtype=np.float32)
            duration = len(audio) / self.SAMPLERATE
            if duration >= 0.3:  # 过滤过短片段（咳嗽/杂音）
                self.speech_queue.put(audio)
            else:
                print(" (too short)", end="", flush=True)
            self._detector.pop()

    def run(self):
        print(f"\n[模式] Silero VAD 自动检测 (silero_vad.onnx)")
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
    candidate = os.path.join(_project_root, "model", "sherpa-onnx-sense-voice")
    if os.path.isdir(candidate) and (
        os.path.exists(os.path.join(candidate, "model.onnx"))
        or os.path.exists(os.path.join(candidate, "model.int8.onnx"))
    ):
        return candidate
    raise FileNotFoundError(
        f"SenseVoice 模型未找到: {candidate}/\n"
        "请下载 model.onnx (全量) 或 model.int8.onnx (int8) + tokens.txt 到该目录"
    )


def main():
    parser = argparse.ArgumentParser(description="语音灵巧手控制")
    parser.add_argument("--port", default="COM5", help="灵巧手串口号")
    parser.add_argument("--mic", default="auto",
                        choices=["auto", "ugreen", "default"],
                        help="麦克风: auto=UGREEN优先, ugreen=仅UGREEN, default=系统默认")
    parser.add_argument("--model", default="auto",
                        choices=["auto", "int8", "full"],
                        help="ASR模型: auto=完整版优先, int8=int8量化, full=完整版")
    parser.add_argument("--mode", default="auto",
                        choices=["ptt", "auto"],
                        help="输入模式: auto=Silero VAD, ptt=按键说话")
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
    print(f"  模型目录: {model_dir}  (--model={args.model})")
    asr = SpeechRecognizer(model_dir, model_type=args.model)

    # 3. 创建输入模式
    print(f"\n[3/3] 启动输入模式: {args.mode} (mic={args.mic})")
    if args.mode == "auto":
        voice_input = AutoVADInput(asr, mic_pref=args.mic)
    else:
        voice_input = PushToTalkInput(asr, mic_pref=args.mic)

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
