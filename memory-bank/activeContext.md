# Active Context — Project_hand

## 当前焦点

**豆包端到端实时语音大模型(全双工)替换 MiMo 语音链路** — 单 WebSocket 完成 ASR+LLM+TTS,预期端到端延迟从 5~10s 降到 0.5~1.5s。

## 最近变更

### 豆包语音控制 (2026-08-13)

1. **`Omnihand/voice_doubao_control.py`** — 新建(单文件实现)。豆包 Seeduplex 全双工 S2S:
   - WS `wss://openspeech.bytedance.com/api/v3/duplex/realtime/dialogue`,X-Api-Key,纯 JSON 文本帧,model 固定 `1.2.6.1`
   - 输入 pcm 16k int16 20ms/包 Base64;输出 pcm_s16le 24k 流式;服务端 VAD(无需本地 Silero)
   - 手势动作走 **Function Calling**(3 工具: do_gesture/do_sequence/stop_hand),入队立即回传,客户端硬校验
   - 4 线程: Main + PortAudio 回调 + WS send(唯一发送者/重连)+ WS recv(事件分发)+ 手势执行线程(串口独占)
   - StreamingPlayer(sounddevice 24k,预缓冲 3 块)替代 Windows MCI;barge-in(response.cancel + epoch 丢弃)
   - speed 三档真正落地: null 直接 / fast 0.75s 插值 / medium 1.0s / slow 3.0s
   - **无降级**: 无 key/连接失败/播放失败/灵巧手失败一律显式报错退出;鉴权 4xx 不重试;5xx 指数退避重连(上限 5 次)
2. **`Omnihand/voice_llm_prompt.py`** — 追加 DOUBAO_INSTRUCTIONS + DOUBAO_TOOLS + validate_tool_call + parse_fc_items(现有 MiMo 部分一字未动,两条链路共存)
3. **`Omnihand/voice_llm_config.json`** — 追加 `doubao` 段(api_key 空占位;真 key 由用户写入 .local.json)
4. **`Omnihand/tests/mock_doubao_server.py` + `test_voice_doubao_offline.py`** — mock WS 服务器(websockets)+ 18 用例离线测试全通过
5. **`requirements.txt`** — +websocket-client==1.8.0(运行时);websockets 仅 mock 测试用
6. **`doc/voice_control_design.md`** — 追加附录 C(协议要点、与 MiMo 对比、工具定义、测试结果)

### 关键协议细节(排坑记录)

- websocket-client 帧传输中途 recv 超时会损坏数据流 → 连接后 `settimeout(None)` 无限阻塞,死链靠 ping + 服务端关闭检测
- 下行事件名带前缀: `response.function_call_arguments.done`(不是 function_call_arguments.done)
- 优雅关闭: session.close → 等 session.closed → 再断连(防 55000001)
- mock 服务器 shutdown 需 asyncio.Event 通知 serve 退出,不能用 loop.stop() 硬停(pending task 告警)
- **全双工必须持续推流** — commit 后停止发包触发 `52000033 AudioServerNoAudioInputTooLongError`;主程序 auto 模式麦克风常开天然规避
- **tools 必须扁平格式** — `type=function` 与 `name/description/parameters` 同级;OpenAI 式 `function` 包装服务端不识别,FC 永不触发(真实 API 探测验证)
- **audio.delta 包大小不固定** — 实测远大于 20ms;StreamingPlayer 已加跨块余量缓冲(不能按一包一块丢尾部)

### 真实 API 验证 (2026-08-13)

- API key 已存入 `voice_llm_config.local.json` (doubao.api_key)
- 免麦克风探测 (临时脚本, 已删除;zh.wav + MiMo TTS 合成语音): WS 连接 ✅ 服务端 VAD+ASR ✅ LLM 对话 ✅ TTS 音频流 ✅ FC(扁平 tools)✅ 优雅关闭 ✅
- FC 实测: "握拳" 语音 → `do_gesture(FIST)` 触发成功, tool 回传后模型语音回复 "好的,已经握拳啦"

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | 豆包 API Key 申请 | 火山引擎控制台 → API Key 管理,写入 .local.json 的 doubao.api_key |
| 🔴 | 实机阶段 1 | 无手纯对话: ASR 正确性 + TTS 流式出声 + 首包延迟记录 |
| 🔴 | 实机阶段 2 | 空载: 单手势/速度三档/序列/自定义多手势/越界拒绝 |
| 🟡 | 实机阶段 3 | 握手压力检测、stop 急停、抢话打断、断网重连、音色选择(4 选 1) |
| 🟡 | 延迟对比 | 同句指令 MiMo vs 豆包各 5 次,结果写回附录 C |
| 🟡 | CRP 实机导入测试 | estun_handle_sm.crp + estun_handle_cup_sm.crp(此前遗留) |
| 🟡 | 单相机 SM 实机联调 | estun_handle_sm.py + estun_handle_cup_sm.py(此前遗留) |
