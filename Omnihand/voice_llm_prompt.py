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
5. 闲聊/不涉及动作→reply
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
回复: 你好!需要我做什么手势吗?```json
{{"actions":[{{"type":"reply"}}]}}
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
