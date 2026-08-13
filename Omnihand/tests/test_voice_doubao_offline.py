# -*- coding: utf-8 -*-
"""
豆包语音控制离线测试 — 无 API key / 无串口 / 无音频设备可跑
=============================================================
覆盖:
  1. validate_tool_call / parse_fc_items 纯函数
  2. 配置 doubao 段 + API Key 优先级
  3. build_session_create 事件结构
  4. _handle_event 事件分发 (FC 执行/回传、barge-in、error)
  5. 端到端: mock 服务器 + 会话 WS 线程 (无音频设备)
     - normal: 全流程 + call_id 配对
     - bargein: 打断 → response.cancel + epoch 提升
     - fc_invalid: 非法参数拒绝
     - reconnect: 服务端断开 → 自动重连
     - 优雅关闭: session.close → session.closed

运行: python Omnihand/tests/test_voice_doubao_offline.py
"""

import os
import sys
import json
import time
import queue

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Omnihand.voice_llm_prompt import (
    validate_tool_call, parse_fc_items, DOUBAO_TOOLS, DOUBAO_INSTRUCTIONS)
import Omnihand.voice_doubao_control as vdc
from Omnihand.voice_doubao_control import DoubaoVoiceSession, resolve_api_key
from mock_doubao_server import MockDoubaoServer

# ============================================================
# 轻量测试框架
# ============================================================

_PASS = []
_FAIL = []


def _run(name, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  [PASS] {name}")
    except Exception as e:
        _FAIL.append((name, e))
        print(f"  [FAIL] {name}: {e}")


def _assert(cond, msg):
    if not cond:
        raise AssertionError(msg)


# ============================================================
# 1. 纯函数: validate_tool_call / parse_fc_items
# ============================================================

def t_validate_gesture():
    ok, err, a = validate_tool_call(
        "do_gesture", {"gesture": "FIST", "speed": "slow"})
    _assert(ok, "合法 gesture 应通过")
    _assert(a == {"type": "gesture", "gesture": "FIST", "speed": "slow"},
            f"action 格式错误: {a}")

    ok, err, a = validate_tool_call("do_gesture", {"gesture": "FIST"})
    _assert(ok and a["speed"] is None, "缺 speed 应默认 None")

    ok, err, a = validate_tool_call("do_gesture", {"gesture": "NOTEXIST"})
    _assert(not ok, "非法手势应拦截")

    ok, err, a = validate_tool_call("do_gesture", {"gesture": "FIST", "speed": "warp"})
    _assert(not ok, "非法速度应拦截")


def t_validate_sequence():
    ok, err, a = validate_tool_call("do_sequence", {"sequence": "握手"})
    _assert(ok and a["type"] == "sequence", "预定义序列应通过")

    ok, err, a = validate_tool_call("do_sequence", {
        "sequence": "custom",
        "steps": [{"gesture": "FIST", "duration": 0.5}, {"gesture": "PAPER"}]})
    _assert(ok and len(a["steps"]) == 2, "custom 序列应通过")
    _assert(a["steps"][1]["duration"] == 0.5, "缺 duration 应默认 0.5")

    ok, err, a = validate_tool_call("do_sequence", {
        "sequence": "custom", "steps": [{"gesture": "FIST", "duration": 9.9}]})
    _assert(not ok, "超长 duration 应拦截")

    ok, err, a = validate_tool_call("do_sequence", {"sequence": "custom"})
    _assert(not ok, "缺 steps 应拦截")

    ok, err, a = validate_tool_call("stop_hand", {})
    _assert(ok and a["type"] == "stop", "stop 应通过")


def t_parse_fc_items():
    items = parse_fc_items({"items": [
        {"call_id": "c1", "name": "do_gesture",
         "arguments": '{"gesture":"HEART"}'},
        {"call_id": "c2", "name": "do_sequence", "arguments": "bad json"},
        {"call_id": "", "name": "do_gesture", "arguments": "{}"},
        {"call_id": "c3", "name": "do_gesture", "arguments": {"gesture": "OK"}},
    ]})
    _assert(items == [("c1", "do_gesture", {"gesture": "HEART"}),
                      ("c3", "do_gesture", {"gesture": "OK"})],
            f"应跳过坏 JSON/缺 call_id: {items}")


# ============================================================
# 2. 配置与 key 优先级
# ============================================================

def t_config_doubao_section():
    _assert(vdc._MODEL == "1.2.6.1", f"model 错误: {vdc._MODEL}")
    _assert("openspeech.bytedance.com" in vdc._WS_URL, f"ws_url 错误: {vdc._WS_URL}")
    _assert(vdc._IN_BLOCKSIZE == 320, f"20ms/16k 应为 320: {vdc._IN_BLOCKSIZE}")
    _assert(vdc._OUT_BLOCKSIZE == 480, f"20ms/24k 应为 480: {vdc._OUT_BLOCKSIZE}")


def t_resolve_api_key():
    db_key = vdc.DB.get("api_key", "")
    _assert(resolve_api_key("cli-key") == "cli-key", "cli 优先")
    _assert(resolve_api_key(None) == db_key, "无 cli 时取配置 (.local.json)")
    # env 兜底仅在配置无 key 时生效
    if not db_key:
        old = os.environ.get("DOUBAO_API_KEY")
        try:
            os.environ["DOUBAO_API_KEY"] = "env-key"
            _assert(resolve_api_key(None) == "env-key", "env 兜底")
            _assert(resolve_api_key("cli-key") == "cli-key", "cli 覆盖 env")
        finally:
            if old is not None:
                os.environ["DOUBAO_API_KEY"] = old
            else:
                os.environ.pop("DOUBAO_API_KEY", None)


# ============================================================
# 3. session.create 事件结构
# ============================================================

def t_build_session_create():
    sess = DoubaoVoiceSession(vdc.DB, "test-key", "auto")
    evt = sess._build_session_create()
    _assert(evt["type"] == "session.create", "type 错误")
    s = evt["session"]
    _assert(s["model"] == "1.2.6.1", "model 错误")
    _assert(s["audio"]["input"]["format"]["sample_rate"] == 16000, "输入 16k")
    _assert(s["audio"]["output"]["format"]["sample_rate"] == 24000, "输出 24k")
    _assert(s["audio"]["output"]["format"]["type"] == "pcm_s16le", "输出格式")
    _assert(len(s["tools"]) == 3, f"tools 数量: {len(s['tools'])}")
    _assert(s["instructions"] == DOUBAO_INSTRUCTIONS, "instructions 错误")
    _assert("id" not in s, "无历史会话不应带 id")


# ============================================================
# 4. 事件分发 (无音频设备, 直接驱动 _handle_event)
# ============================================================

class FakeHand:
    """记录 submit 调用的假手."""

    def __init__(self):
        self.actions = []
        self.submitted = 0

    def submit(self, action, call_id=None):
        self.actions.append((action, call_id))
        self.submitted += 1


def _make_session(hand=None, ws_url="ws://127.0.0.1:1"):
    sess = DoubaoVoiceSession(vdc.DB, "test-key", "auto", hand=hand,
                              ws_url=ws_url)
    return sess


def t_event_audio_delta():
    sess = _make_session()
    import base64
    pcm = base64.b64encode(b"\x01\x00" * 480).decode()
    sess._handle_event({"type": "response.output_audio.started"})
    _assert(sess._responding is True, "responding 应置位")
    for _ in range(5):
        sess._handle_event({"type": "response.output_audio.delta",
                            "delta": pcm})
    # prebuffer=3: 前 3 块移入播放队列, 后 2 块仍在预缓冲
    _assert(sess.player._q.qsize() == 3
            and len(sess.player._pending) == 2,
            f"预缓冲机制错误: q={sess.player._q.qsize()} "
            f"pending={len(sess.player._pending)}")
    sess._handle_event({"type": "response.output_audio.done"})
    _assert(sess._responding is False, "responding 应复位")


def t_event_fc_normal():
    hand = FakeHand()
    sess = _make_session(hand=hand)
    sess._handle_event({"type": "response.function_call_arguments.done",
                        "items": [{"call_id": "call-9", "name": "do_gesture",
                                   "arguments": '{"gesture":"FIST"}'}]})
    _assert(hand.submitted == 1, "假手应收到 1 次 submit")
    action, cid = hand.actions[0]
    _assert(cid == "call-9", f"call_id 错误: {cid}")
    # tool 回传应在 send 队列
    kind, evt = sess._send_q.get(timeout=1)
    _assert(kind == "event" and evt["type"] == "conversation.item.create",
            "应排队 tool 回传")
    item = evt["items"][0]
    _assert(item["call_id"] == "call-9" and item["role"] == "tool",
            "回传应带 call_id + role=tool")


def t_event_fc_invalid():
    hand = FakeHand()
    sess = _make_session(hand=hand)
    sess._handle_event({"type": "response.function_call_arguments.done",
                        "items": [{"call_id": "call-1", "name": "do_gesture",
                                   "arguments": '{"gesture":"NOTEXIST"}'}]})
    _assert(hand.submitted == 0, "非法调用不应执行")
    kind, evt = sess._send_q.get(timeout=1)
    text = evt["items"][0]["content"][0]["text"]
    data = json.loads(text)
    _assert(data["ok"] is False, f"回传应报失败: {data}")


def t_event_bargein():
    sess = _make_session()
    sess._handle_event({"type": "response.output_audio.started"})
    _assert(sess._responding is True, "responding 应置位")
    old_epoch = sess.player.epoch
    sess._handle_event({
        "type": "conversation.item.input_audio_transcription.started"})
    _assert(sess._responding is False, "打断后 responding 复位")
    _assert(sess.player.epoch == old_epoch + 1, "epoch 应 +1")
    kind, evt = sess._send_q.get(timeout=1)
    _assert(kind == "event" and evt["type"] == "response.cancel",
            "应排队 response.cancel")


def t_event_bargein_only_when_responding():
    sess = _make_session()
    old_epoch = sess.player.epoch
    sess._handle_event({
        "type": "conversation.item.input_audio_transcription.started"})
    _assert(sess.player.epoch == old_epoch, "非播报中不应触发打断")
    _assert(sess._send_q.empty(), "不应排队 cancel")


def t_player_leftover():
    """服务端 delta 包可能远大于 20ms 块 — 播放器跨块余量不应丢数据."""
    p = vdc.StreamingPlayer(prebuffer=1)
    pcm = (np.arange(12000, dtype=np.int16) % 1000).tobytes()   # 0.5s @24k
    p.enqueue(pcm)
    collected = bytearray()
    for _ in range(30):
        outdata = np.zeros((480, 1), dtype=np.int16)
        p._cb(outdata, 480, None, None)
        collected.extend(outdata[:, 0].tobytes())
    total = np.frombuffer(bytes(collected), dtype=np.int16)
    _assert(np.array_equal(total[:12000], np.frombuffer(pcm, dtype=np.int16)),
            "跨块播放不应丢数据")
    _assert(bool(np.all(total[12000:] == 0)), "播完应为静音")


def t_event_error_fatal():
    sess = _make_session()
    sess._handle_event({"type": "error",
                        "error": {"code": 401, "message": "Invalid ApiKey"}})
    _assert(sess.stop_event.is_set(), "鉴权错误应置 stop_event")


def t_event_error_reconnect():
    sess = _make_session()
    sess._handle_event({"type": "error",
                        "error": {"code": 50000000, "message": "ServerError"}})
    kind, reason = sess._send_q.get(timeout=1)
    _assert(kind == "reconnect", f"5xx 应触发重连, 实际: {kind}")


# ============================================================
# 5. 端到端 (mock 服务器 + 会话 WS 线程, 无音频设备)
# ============================================================

def _wait(pred, timeout=6, step=0.05):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(step)
    return False


def _start_ws_threads(sess):
    sess._threads = [
        __import__("threading").Thread(
            target=sess._ws_recv_loop, daemon=True, name="ws-recv"),
        __import__("threading").Thread(
            target=sess._ws_send_loop, daemon=True, name="ws-send"),
    ]
    for t in sess._threads:
        t.start()
    sess._send_q.put(("reconnect", "测试连接"))


def t_e2e_normal():
    srv = MockDoubaoServer(scenario="normal", append_trigger=3)
    srv.start()
    hand = FakeHand()
    sess = _make_session(hand=hand, ws_url=srv.ws_url)
    _start_ws_threads(sess)
    try:
        _assert(_wait(srv.session_created.is_set), "session.created 未收到")
        for _ in range(3):
            sess._send_q.put(("audio", b"\x00" * 640))
        _assert(_wait(lambda: hand.submitted >= 1), "FC 未触发执行")
        _assert(_wait(lambda: len(srv.tool_results) >= 1),
                "tool 回传未到达服务器")
        _assert(srv.tool_results[0][0] == "call-1" and srv.tool_results[0][1],
                f"call_id 配对错误: {srv.tool_results}")
        # 优雅关闭
        sess._send_q.put(("close", None))
        _assert(_wait(srv.close_received.is_set), "session.close 未收到")
        _assert(_wait(sess._close_done.is_set), "优雅关闭未完成")
    finally:
        sess.stop_event.set()
        sess._send_q.put(("close", None))
        time.sleep(0.3)
        srv.shutdown()


def t_e2e_bargein():
    srv = MockDoubaoServer(scenario="bargein", append_trigger=3)
    srv.start()
    hand = FakeHand()
    sess = _make_session(hand=hand, ws_url=srv.ws_url)
    _start_ws_threads(sess)
    try:
        _assert(_wait(srv.session_created.is_set), "session.created 未收到")
        for _ in range(3):
            sess._send_q.put(("audio", b"\x00" * 640))
        _assert(_wait(srv.cancel_received.is_set), "response.cancel 未收到")
        _assert(sess.player.epoch >= 1, f"epoch 应提升: {sess.player.epoch}")
    finally:
        sess.stop_event.set()
        sess._send_q.put(("close", None))
        time.sleep(0.3)
        srv.shutdown()


def t_e2e_fc_invalid():
    srv = MockDoubaoServer(scenario="fc_invalid", append_trigger=3)
    srv.start()
    hand = FakeHand()
    sess = _make_session(hand=hand, ws_url=srv.ws_url)
    _start_ws_threads(sess)
    try:
        _assert(_wait(srv.session_created.is_set), "session.created 未收到")
        for _ in range(3):
            sess._send_q.put(("audio", b"\x00" * 640))
        _assert(_wait(lambda: len(srv.tool_results) >= 1),
                "非法 FC 也应回传错误")
        _assert(hand.submitted == 0, "非法参数不应执行")
        _assert(srv.tool_results[0][1] is False, "回传应为失败")
    finally:
        sess.stop_event.set()
        sess._send_q.put(("close", None))
        time.sleep(0.3)
        srv.shutdown()


def t_e2e_reconnect():
    srv = MockDoubaoServer(scenario="reconnect", append_trigger=3)
    srv.start()
    hand = FakeHand()
    sess = _make_session(hand=hand, ws_url=srv.ws_url)
    _start_ws_threads(sess)
    try:
        _assert(_wait(srv.session_created.is_set), "首次 session.created 未收到")
        for _ in range(3):
            sess._send_q.put(("audio", b"\x00" * 640))
        # reconnect 场景: 服务端一轮后主动断开 → 客户端重连 → 再次 session.create
        _assert(_wait(lambda: sum(
            1 for e in srv.received if e.get("type") == "session.create") >= 2,
            timeout=12), "未自动重连")
        print("    (重连成功, session.create 次数:",
              sum(1 for e in srv.received
                  if e.get("type") == "session.create"), ")")
    finally:
        sess.stop_event.set()
        sess._send_q.put(("close", None))
        time.sleep(0.3)
        srv.shutdown()


def t_e2e_ptt_commit():
    """ptt: commit 事件触发一轮回放 (不依赖 append 计数)."""
    srv = MockDoubaoServer(scenario="plain", append_trigger=9999)
    srv.start()
    sess = _make_session(ws_url=srv.ws_url)
    _start_ws_threads(sess)
    try:
        _assert(_wait(srv.session_created.is_set), "session.created 未收到")
        sess._send_q.put(("event", {"type": "input_audio_buffer.commit",
                                    "event_id": "evt-1"}))
        got_text = _wait(lambda: any(
            e.get("type") == "response.output_text.done" for e in srv.sent))
        _assert(got_text, "commit 应触发回复")
    finally:
        sess.stop_event.set()
        sess._send_q.put(("close", None))
        time.sleep(0.3)
        srv.shutdown()


# ============================================================
# 汇总
# ============================================================

def main():
    print("=" * 55)
    print("  豆包语音控制离线测试")
    print("=" * 55)
    _run("validate: gesture", t_validate_gesture)
    _run("validate: sequence", t_validate_sequence)
    _run("parse_fc_items", t_parse_fc_items)
    _run("config: doubao 段", t_config_doubao_section)
    _run("api key 优先级", t_resolve_api_key)
    _run("session.create 结构", t_build_session_create)
    _run("事件: 音频 delta 入队", t_event_audio_delta)
    _run("事件: FC 正常执行+回传", t_event_fc_normal)
    _run("事件: FC 非法参数拒绝", t_event_fc_invalid)
    _run("事件: barge-in 打断", t_event_bargein)
    _run("事件: 非播报中不打断", t_event_bargein_only_when_responding)
    _run("播放器: 跨块余量不丢数据", t_player_leftover)
    _run("事件: 鉴权错误退出", t_event_error_fatal)
    _run("事件: 5xx 触发重连", t_event_error_reconnect)
    _run("E2E: normal 全流程+call_id", t_e2e_normal)
    _run("E2E: bargein 打断", t_e2e_bargein)
    _run("E2E: 非法 FC 拒绝", t_e2e_fc_invalid)
    _run("E2E: 断线重连", t_e2e_reconnect)
    _run("E2E: ptt commit 触发", t_e2e_ptt_commit)

    print("-" * 55)
    print(f"通过 {len(_PASS)} / 失败 {len(_FAIL)}")
    for name, e in _FAIL:
        print(f"  FAIL {name}: {e}")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
