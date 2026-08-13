# -*- coding: utf-8 -*-
"""
voice_llm_prompt.py — LLM System Prompt + JSON 解析
=====================================================
紧凑 prompt (~400 tokens 动作空间), 8 个 few-shot 示例.
"""

import json as _json
import re as _re
from typing import Optional

# ============================================================
# 动作空间 (紧凑格式, 供 prompt 引用)
# ============================================================

GESTURE_LIST = (
    "FIST(握拳/攥拳), PAPER(张开/摊开/手掌), RESET(复位/恢复/归零), "
    "HEART(比心/爱心), OK(好的/OK), LIKE(点赞/厉害), ILY(我爱你), "
    "NUM1(一/数字1), NUM2(二/数字2), NUM3(三/数字3), NUM4(四/数字4), "
    "NUM6(六/数字6), NUM8(八/数字8), "
    "CLASPING(合十/拜拜/双手合十), MIDDLE(竖中指/中指), POINT(指人/指向), "
    "GRIP(抓取/抓住), GRIPDONE(抓紧/握紧), PREGRIP(半握/预备抓), "
    "HAND_SHAKE(握手手势/简单握手), HELLO_1(挥手1/打招呼1), HELLO_2(挥手2/打招呼2)"
)

SEQUENCE_LIST = (
    "打招呼(RESET→HELLO_1/2交替3轮→RESET) | "
    "握手(等压力→HAND_SHAKE→等松手→RESET) | "
    "数字手势(RESET→NUM1→2→3→4→6→8→CLASPING→RESET)"
)

# 用于验证的有效枚举
VALID_GESTURES = {
    "RESET", "PAPER", "FIST", "OK", "HEART", "LIKE", "ILY",
    "NUM1", "NUM2", "NUM3", "NUM4", "NUM6", "NUM8",
    "CLASPING", "MIDDLE", "POINT", "GRIP", "GRIPDONE", "PREGRIP",
    "HAND_SHAKE", "HELLO_1", "HELLO_2",
}
VALID_SEQUENCES = {"打招呼", "握手", "数字手势", "custom"}
VALID_SPEEDS = {None, "fast", "medium", "slow"}

# ============================================================
# System Prompt
# ============================================================

SYSTEM_PROMPT = f"""你是 Project_hand 的灵巧手(OmniHand)语音助手. 理解用户语音指令, 口语化中文回应, 输出 JSON 动作指令.

# 动作空间
手势: {GESTURE_LIST}
预定义序列: {SEQUENCE_LIST}
自定义序列: 任意手势+时长组合, 用户一次说多个手势时使用.
速度: null(直接,最快)/fast(0.75s插值)/medium(1.0s)/slow(3.0s)

# 输出格式 — 严格 JSON
只允许 6 种 action type:
- gesture: {{"type":"gesture","gesture":"FIST","speed":null}}
- sequence(预定义): {{"type":"sequence","sequence":"握手"}}
- sequence(自定义): {{"type":"sequence","sequence":"custom","steps":[{{"gesture":"FIST","duration":0.5}},{{"gesture":"PAPER","duration":0.5}}]}}
- stop: {{"type":"stop"}}
- reply: {{"type":"reply"}}

回复前先输出 JSON 代码块, 格式: ```json\\n{{...}}\\n```

# 规则
1. 安全优先: "停止/急停/停下/快停" → stop
2. 语义推断: "给他竖个中指"→MIDDLE; "比个心"→HEART; "握个手吧"→握手序列; "做个OK"→OK
3. 默认 speed=null, duration=0.5s(自定义序列)
4. 越界: "拿东西/倒水"等超出手势范围→reply 礼貌拒绝
5. 社交交互: 用户问候/聊天时, 自主判断是否配合手势. "你好/嗨/谢谢"→打招呼序列或HEART比心; "再见/拜拜"→CLASPING合十. 不确定时, 仅reply 不加动作
6. 多手势→自定义序列, 按用户说的顺序

# 示例

[语音]"握拳"
回复: 好的.```json
{{"actions":[{{"type":"gesture","gesture":"FIST","speed":null}}]}}
```

[语音]"给他竖个中指"
回复: 好的.```json
{{"actions":[{{"type":"gesture","gesture":"MIDDLE","speed":null}}]}}
```

[语音]"先握拳再张开最后比个心"
回复: 握拳→张开→比心.```json
{{"actions":[{{"type":"sequence","sequence":"custom","steps":[{{"gesture":"FIST","duration":0.5}},{{"gesture":"PAPER","duration":0.5}},{{"gesture":"HEART","duration":0.5}}]}}]}}
```

[语音]"慢慢张开手掌"
回复: 好的,慢速张开.```json
{{"actions":[{{"type":"gesture","gesture":"PAPER","speed":"slow"}}]}}
```

[语音]"握个手"
回复: 好的,请握我的手.```json
{{"actions":[{{"type":"sequence","sequence":"握手"}}]}}
```

[语音]"快停下"
回复: 已停止.```json
{{"actions":[{{"type":"stop"}}]}}
```

[语音]"帮我倒杯水"
回复: 抱歉,我只能做手势动作.```json
{{"actions":[{{"type":"reply"}}]}}
```

[语音]"你好呀"
回复: 你好!```json
{{"actions":[{{"type":"sequence","sequence":"打招呼"}}]}}
```

[语音]"拜拜"
回复: 再见!```json
{{"actions":[{{"type":"gesture","gesture":"CLASPING","speed":null}}]}}
```"""


# ============================================================
# JSON 解析
# ============================================================

def parse_llm_response(text: str) -> Optional[dict]:
    """从 LLM 回复中提取 JSON actions block."""
    m = _re.search(r'```json\s*(.*?)\s*```', text, _re.DOTALL)
    if m:
        try:
            return _json.loads(m.group(1))
        except _json.JSONDecodeError:
            pass
    m = _re.search(r'\{[\s\S]*"actions"[\s\S]*\}', text)
    if m:
        try:
            return _json.loads(m.group(0))
        except _json.JSONDecodeError:
            pass
    return None


def validate_actions(data: dict) -> list[dict]:
    """验证并提取 actions 列表, 过滤无效条目."""
    actions = data.get("actions", [])
    valid = []
    for a in actions:
        typ = a.get("type", "")
        if typ == "gesture":
            if a.get("gesture") not in VALID_GESTURES:
                continue
            if a.get("speed") not in VALID_SPEEDS:
                a["speed"] = None
        elif typ == "sequence":
            seq = a.get("sequence", "")
            if seq not in VALID_SEQUENCES:
                continue
            if seq == "custom":
                steps = a.get("steps", [])
                if not steps or not all(
                    s.get("gesture") in VALID_GESTURES for s in steps
                ):
                    continue
        elif typ == "stop":
            pass
        elif typ == "reply":
            pass
        else:
            continue
        valid.append(a)
    return valid


def extract_reply_text(response: str) -> str:
    """从完整回复中提取 TTS 文本 (去掉 JSON 代码块)."""
    # 去掉 ```json ... ``` 块
    text = _re.sub(r'```json.*?```', '', response, flags=_re.DOTALL)
    # 去掉残留的 ``` 标记
    text = _re.sub(r'```', '', text)
    return text.strip()


# ============================================================
# 豆包端到端实时语音大模型 (全双工) — Instructions + Tools
# ============================================================
# 由 SYSTEM_PROMPT 移植改造: S2S 下模型输出文本会被 TTS 朗读,
# 手势动作改走 Function Calling, 不再要求 JSON 代码块.
# 仅 voice_doubao_control.py 使用; voice_llm_control.py (MiMo) 不受影响.

DOUBAO_INSTRUCTIONS = f"""你是 Project_hand 的灵巧手(OmniHand)语音助手. 理解用户语音指令, 口语化中文回应, 需要动作时调用工具函数.

# 动作空间 (通过工具调用执行)
手势 (do_gesture 工具): {GESTURE_LIST}
预定义序列 (do_sequence 工具): {SEQUENCE_LIST}
自定义序列: 用户一次说多个手势时, sequence=custom + steps 按用户说的顺序排列.
速度: null(直接,最快)/fast(0.75s插值)/medium(1.0s)/slow(3.0s)

# 输出要求 — 口语化
1. 你的全部文本回复会被语音合成朗读, 必须口语化、简短 (1~2句).
2. 禁止输出 JSON、代码块、工具名或任何格式标记.
3. 需要动作时直接调用工具函数, 调用后用自然语言回应用户.

# 规则
1. 安全优先: "停止/急停/停下/快停" → 调用 stop_hand
2. 语义推断: "给他竖个中指"→do_gesture(MIDDLE); "比个心"→do_gesture(HEART); "握个手吧"→do_sequence(握手); "做个OK"→do_gesture(OK)
3. 默认 speed=null, 自定义序列 duration=0.5s
4. 越界: "拿东西/倒水"等超出手势范围→不调用工具, 口语礼貌拒绝
5. 社交交互: 用户问候/聊天时, 自主判断是否配合手势. "你好/嗨/谢谢"→do_sequence(打招呼)或do_gesture(HEART); "再见/拜拜"→do_gesture(CLASPING). 不确定时, 仅口语回复不加动作
6. 多手势→do_sequence(custom), 按用户说的顺序
7. 防自回声: 如果听到自己刚才说过的话被重复识别, 忽略它, 不要响应

# 示例

用户:"握拳" → 调用 do_gesture(gesture="FIST", speed="null"), 回复:"好的."
用户:"慢慢张开手掌" → 调用 do_gesture(gesture="PAPER", speed="slow"), 回复:"好的, 慢慢张开."
用户:"先握拳再张开最后比个心" → 调用 do_sequence(sequence="custom", steps=[FIST, PAPER, HEART]), 回复:"好, 依次来."
用户:"快停下" → 调用 stop_hand(), 回复:"已停止."
用户:"帮我倒杯水" → 不调用工具, 回复:"抱歉, 我只能做手势动作."
用户:"你好呀" → 调用 do_sequence(sequence="打招呼"), 回复:"你好!"
用户:"拜拜" → 调用 do_gesture(gesture="CLASPING"), 回复:"再见!"
"""


# ============================================================
# 豆包 Function Calling 工具定义 (JSON Schema)
# ============================================================

_DOUBAO_GESTURE_ENUM = sorted(VALID_GESTURES)

_DOUBAO_TOOL_GESTURE = {
    "type": "function",
    "function": {
        "name": "do_gesture",
        "description": (
            "控制灵巧手执行单个手势. 用户要求任何手势动作时调用. "
            "调用后仍需用口语自然回应用户, 不要读出工具名或JSON."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "gesture": {
                    "type": "string",
                    "enum": _DOUBAO_GESTURE_ENUM,
                },
                "speed": {
                    "type": "string",
                    "enum": ["null", "fast", "medium", "slow"],
                    "description": (
                        "null=直接到位(最快); fast=0.75秒平滑; "
                        "medium=1.0秒平滑; slow=3.0秒极慢平滑"
                    ),
                },
            },
            "required": ["gesture"],
        },
    },
}

_DOUBAO_TOOL_SEQUENCE = {
    "type": "function",
    "function": {
        "name": "do_sequence",
        "description": (
            "执行预定义手势序列或自定义多手势组合. "
            "打招呼/握手(含压力等待)/数字手势/用户一次说多个手势时使用."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sequence": {
                    "type": "string",
                    "enum": ["打招呼", "握手", "数字手势", "custom"],
                },
                "steps": {
                    "type": "array",
                    "description": "仅 sequence=custom 时必填, 按用户说的顺序排列",
                    "items": {
                        "type": "object",
                        "properties": {
                            "gesture": {
                                "type": "string",
                                "enum": _DOUBAO_GESTURE_ENUM,
                            },
                            "duration": {
                                "type": "number",
                                "minimum": 0.2,
                                "maximum": 3.0,
                            },
                        },
                        "required": ["gesture"],
                    },
                },
            },
            "required": ["sequence"],
        },
    },
}

_DOUBAO_TOOL_STOP = {
    "type": "function",
    "function": {
        "name": "stop_hand",
        "description": (
            "紧急停止. 用户说'停止/急停/停下/快停'时调用: "
            "灵巧手复位并结束程序."
        ),
        "parameters": {"type": "object", "properties": {}},
    },
}

DOUBAO_TOOLS = [
    _DOUBAO_TOOL_GESTURE,
    _DOUBAO_TOOL_SEQUENCE,
    _DOUBAO_TOOL_STOP,
]


def validate_tool_call(name: str, arguments: dict):
    """校验豆包 FC 调用参数.

    返回 (ok, err_msg, action):
      - ok: 参数合法
      - err_msg: 错误描述 (ok=False 时非空)
      - action: GestureExecutor.execute 兼容格式:
          {"type":"gesture","gesture":"FIST","speed":None}
          {"type":"sequence","sequence":"握手"}
          {"type":"sequence","sequence":"custom","steps":[{"gesture":"FIST","duration":0.5}]}
          {"type":"stop"}
    """
    if name == "do_gesture":
        gesture = arguments.get("gesture")
        if gesture not in VALID_GESTURES:
            return False, f"非法手势: {gesture}", None
        speed = arguments.get("speed", "null")
        if speed not in ("null", "fast", "medium", "slow"):
            return False, f"非法速度: {speed}", None
        return True, "", {
            "type": "gesture",
            "gesture": gesture,
            "speed": None if speed == "null" else speed,
        }

    if name == "do_sequence":
        seq = arguments.get("sequence", "")
        if seq not in VALID_SEQUENCES:
            return False, f"非法序列: {seq}", None
        if seq == "custom":
            steps = arguments.get("steps", [])
            if not steps:
                return False, "custom 序列缺少 steps", None
            clean = []
            for s in steps:
                g = s.get("gesture")
                d = s.get("duration", 0.5)
                if g not in VALID_GESTURES:
                    return False, f"非法手势: {g}", None
                if not isinstance(d, (int, float)) or d < 0.2 or d > 3.0:
                    return False, f"非法时长: {d}", None
                clean.append({"gesture": g, "duration": float(d)})
            return True, "", {"type": "sequence", "sequence": "custom",
                              "steps": clean}
        return True, "", {"type": "sequence", "sequence": seq}

    if name == "stop_hand":
        return True, "", {"type": "stop"}

    return False, f"未知工具: {name}", None


def parse_fc_items(evt: dict):
    """从 response.function_call_arguments.done 提取 FC 调用项.

    返回 [(call_id, name, arguments_dict), ...]; 无效条目跳过.
    """
    items = evt.get("items", []) or []
    out = []
    for it in items:
        call_id = it.get("call_id", "")
        name = it.get("name", "")
        raw_args = it.get("arguments", "{}")
        if isinstance(raw_args, dict):
            args = raw_args
        elif isinstance(raw_args, str):
            try:
                args = _json.loads(raw_args) if raw_args.strip() else {}
            except _json.JSONDecodeError:
                continue
        else:
            continue
        if call_id and name:
            out.append((call_id, name, args))
    return out
