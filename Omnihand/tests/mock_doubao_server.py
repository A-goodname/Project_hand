# -*- coding: utf-8 -*-
"""
离线 mock 豆包全双工服务器 — 仅开发测试用
============================================
用 websockets (asyncio) 在后台线程回放标准事件流:

  session.create → session.created
  N×input_audio_buffer.append (或 input_audio_buffer.commit) → 一轮回放:
    transcription.started/delta/completed → output_text.delta/done
    → output_audio.started/delta×5/done → function_call_arguments.done → response.done
  conversation.item.create (role=tool) → 校验 call_id 配对并记录
  response.cancel → 记录
  session.close → session.closed → 关闭 (优雅关闭验证)

场景 (scenario):
  normal      — FC: do_gesture FIST (合法)
  plain       — 无 FC, 纯对话
  fc_invalid  — FC: 非法手势 NOTEXIST (验证客户端硬校验)
  bargein     — 音频播放中途再发 transcription.started (验证打断)
  reconnect   — 一轮回放后服务端主动断开 (验证重连)
"""

import asyncio
import base64
import json
import threading
import time

import numpy as np
import websockets


def _make_audio(ms: int = 20) -> str:
    """生成 24kHz 正弦 PCM int16 (20ms=480 样本), Base64."""
    sr = 24000
    t = np.arange(sr * ms // 1000) / sr
    wave = (np.sin(2 * np.pi * 440.0 * t) * 8000).astype(np.int16)
    return base64.b64encode(wave.tobytes()).decode("ascii")


class MockDoubaoServer:
    """脚本化 mock 服务器. start() 返回后 ws_url 可用."""

    _FC_MAP = {
        "normal": ("do_gesture", '{"gesture": "FIST", "speed": "null"}'),
        "plain": (None, None),
        "fc_invalid": ("do_gesture", '{"gesture": "NOTEXIST"}'),
        "bargein": ("do_gesture", '{"gesture": "HEART", "speed": "slow"}'),
        "reconnect": ("do_gesture", '{"gesture": "PAPER"}'),
    }

    def __init__(self, port: int = 0, scenario: str = "normal",
                 append_trigger: int = 5):
        self.port = port
        self.scenario = scenario
        self.append_trigger = append_trigger

        self.ws_url = None
        self.thread = None
        self.loop = None
        self._stop = threading.Event()

        # 观测点 (测试断言用)
        self.received = []                    # 全部上行事件
        self.sent = []                        # 全部下行事件
        self.tool_results = []                # 收到的 tool 回传 (call_id, ok)
        self.cancel_received = threading.Event()
        self.close_received = threading.Event()
        self.session_created = threading.Event()

        self._append_count = 0
        self._turn_played = False
        self._turn_id = 0
        self._set_evt_cb = None

    # ---- 生命周期 ----

    def start(self) -> str:
        self.thread = threading.Thread(target=self._run_loop, daemon=True,
                                       name="mock-doubao")
        self.thread.start()
        deadline = time.time() + 5
        while self.ws_url is None and time.time() < deadline:
            time.sleep(0.02)
        if self.ws_url is None:
            raise RuntimeError("mock 服务器启动超时")
        return self.ws_url

    def shutdown(self):
        self._stop.set()
        if self.loop is not None and self._set_evt_cb is not None:
            self.loop.call_soon_threadsafe(self._set_evt_cb)
        if self.thread is not None:
            self.thread.join(timeout=3)

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        async def _serve_wrapper():
            self._srv = await websockets.serve(
                self._handler, "127.0.0.1", self.port)
            bound = self._srv.sockets[0].getsockname()[1]
            self.ws_url = f"ws://127.0.0.1:{bound}"
            self._shutdown_evt = asyncio.Event()

            def _set_evt():
                self._shutdown_evt.set()
            self._set_evt_cb = _set_evt

            await self._shutdown_evt.wait()
            self._srv.close()
            await self._srv.wait_closed()
            # 清理残留任务, 避免 pending task 告警
            pending = [t for t in asyncio.all_tasks()
                       if t is not asyncio.current_task()]
            for t in pending:
                t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

        try:
            self.loop.run_until_complete(_serve_wrapper())
        finally:
            self.loop.close()

    # ---- 处理 ----

    async def _handler(self, ws):
        try:
            async for raw in ws:
                try:
                    evt = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                self.received.append(evt)
                typ = evt.get("type", "")

                if typ == "session.create":
                    self.session_created.set()
                    await self._send(ws, {
                        "type": "session.created",
                        "session": {"id": f"mock-dialog-{self._turn_id + 1}"},
                    })
                elif typ == "input_audio_buffer.append":
                    self._append_count += 1
                    if (self._append_count >= self.append_trigger
                            and not self._turn_played):
                        await self._play_turn(ws)
                elif typ == "input_audio_buffer.commit":
                    if not self._turn_played:
                        await self._play_turn(ws)
                elif typ == "conversation.item.create":
                    self._on_tool_result(evt)
                elif typ == "response.cancel":
                    self.cancel_received.set()
                elif typ == "session.close":
                    await self._send(ws, {"type": "session.closed"})
                    self.close_received.set()
                    break
        finally:
            pass

    async def _send(self, ws, evt: dict):
        self.sent.append(evt)
        await ws.send(json.dumps(evt, ensure_ascii=False))

    async def _play_turn(self, ws):
        self._turn_played = True
        self._turn_id += 1
        s = self.scenario
        fc_name, fc_args = self._FC_MAP.get(s, self._FC_MAP["normal"])

        await self._send(ws, {
            "type": "conversation.item.input_audio_transcription.started"})
        await self._send(ws, {
            "type": "conversation.item.input_audio_transcription.delta",
            "delta": "握"})
        await self._send(ws, {
            "type": "conversation.item.input_audio_transcription.completed",
            "transcript": "握拳"})
        await self._send(ws, {"type": "response.output_text.delta",
                              "delta": "好的"})
        await self._send(ws, {"type": "response.output_text.done"})
        await self._send(ws, {"type": "response.output_audio.started"})
        for i in range(5):
            await self._send(ws, {"type": "response.output_audio.delta",
                                  "delta": _make_audio()})
            if s == "bargein" and i == 2:
                # 播放中用户开口
                await self._send(ws, {
                    "type": "conversation.item.input_audio_transcription.started"})
        await self._send(ws, {"type": "response.output_audio.done"})

        if fc_name:
            await self._send(ws, {
                "type": "response.function_call_arguments.done",
                "items": [{"call_id": f"call-{self._turn_id}",
                           "name": fc_name, "arguments": fc_args}],
            })
        await self._send(ws, {"type": "response.done",
                              "usage": {"input_tokens": 11,
                                        "output_tokens": 23}})

        if s == "reconnect":
            await ws.close()

    def _on_tool_result(self, evt: dict):
        for item in evt.get("items", []) or []:
            content = item.get("content", []) or []
            text = content[0].get("text", "{}") if content else "{}"
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                data = {"ok": False, "detail": "bad json"}
            self.tool_results.append((item.get("call_id", ""), data.get("ok")))


def start_mock_server(scenario: str = "normal",
                      append_trigger: int = 5) -> MockDoubaoServer:
    """便捷入口: 起 mock 服务器并返回实例 (ws_url 已就绪)."""
    srv = MockDoubaoServer(scenario=scenario, append_trigger=append_trigger)
    srv.start()
    return srv


if __name__ == "__main__":
    # 手工验证: 起服务器, 用 websocket-client 走一遍协议
    import websocket as wsclient

    srv = start_mock_server(scenario="normal", append_trigger=3)
    print("mock url:", srv.ws_url)
    ws = wsclient.create_connection(srv.ws_url, timeout=5)
    ws.send(json.dumps({"type": "session.create", "session": {"model": "1.2.6.1"}}))
    for i in range(3):
        ws.send(json.dumps({"type": "input_audio_buffer.append",
                            "audio": base64.b64encode(b"\x00" * 640).decode()}))
    seen = []
    deadline = time.time() + 5
    while time.time() < deadline:
        msg = ws.recv()          # 阻塞读 (timeout=5 来自 create_connection)
        evt = json.loads(msg)
        seen.append(evt["type"])
        if evt["type"] == "response.function_call_arguments.done":
            ws.send(json.dumps({
                "type": "conversation.item.create",
                "items": [{"call_id": "call-1", "role": "tool",
                           "content": [{"type": "input_text",
                                        "text": '{"ok": true}'}]}]}))
        if evt["type"] == "response.done":
            break
    print("下行事件:", seen)
    deadline = time.time() + 3
    while time.time() < deadline and not srv.tool_results:
        time.sleep(0.05)
    print("tool 回传记录:", srv.tool_results)
    ws.send(json.dumps({"type": "session.close"}))
    try:
        print("关闭回复:", json.loads(ws.recv())["type"])
    except Exception as e:
        print("关闭回复超时:", e)
    ws.close()
    srv.shutdown()
    print("OK" if srv.close_received.is_set() else "FAIL: close 未收到")
