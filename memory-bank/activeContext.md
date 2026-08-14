# Active Context — Project_hand

## 当前焦点

**P2/P1 双确认协议适配（对方 D0 门禁）** — 训练侧审计 132051c 后要求：标定失败阻止启动（禁 np.eye(4)）、PREGRIP≠位姿确认、P2 后腕部重观测 + GRASP_AUTH/GRIP_AUTH 一次性双确认。交接文档见 `doc/p2p1-dual-confirm-handoff-2026-08-14.md`，CRP 修改用 `.claude/skills/estun-omnihand-robotics/`。

## 最近变更

### 训练侧 D0 门禁 + P2/P1 双确认请求 (2026-08-14)

- 对方回传两份请求（`~/VLA-BD/docs/`）：D0 执行门禁 + 控制器 P2/P1 双确认协议适配
- 已审计：标定文件合格（Daniilidis/26 样本/0.228px）；15 个 py 文件含 np.eye(4) 回退待清理；现 CRP `H_P2→H_P1→GRIPDONE` 不满足腕部重观测约束
- 要求：标定失败阻止启动、PREGRIP 仅手势、双一次性确认点（ticket 绑定、超时/断连安全停止）
- 待回传：CRP 文本版、`git diff 132051c..HEAD`、运行入口+CRP 版本、SHA-256
- 交接文档：`doc/p2p1-dual-confirm-handoff-2026-08-14.md`；CRP skill：`.claude/skills/estun-omnihand-robotics/`

### Linux 环境迁移适配 (2026-08-14)

1. **`EyeToHand/camera.py`** — 跨平台化:
   - Linux: `CAP_V4L2` + 先设 MJPG 再设分辨率 (720P/1080P 必需,参考 `Project/camera_view.py`),无需 pygrabber
   - Linux 设备名探测: 遍历 `/sys/class/video4linux/videoN/name` 匹配 "UGREEN Camera 1080P" (替代 pygrabber DSHOW)
   - Windows 原 DSHOW + pygrabber 逻辑保留, `_IS_LINUX` 分支切换
2. **依赖安装 (Proj 环境, Linux)**: opencv-python 4.13.0.92 / PyYAML 6.0.3 / pyserial 3.5 / pyrealsense2 2.56.5.9235 / ultralytics 8.4.48 / mediapipe 0.10.35;torch 2.13.0+cu126 已有
   - 排坑: PyPI 上 pyrealsense2 版本带构建后缀 (`2.56.5.9235`),requirements 的 `==2.56.5` 匹配不到
   - 排坑: mediapipe 拉入 opencv-contrib-python 5.0.0.93 覆盖锁定版本 → 降回 4.13.0.92 (cv2 与 requirements 一致)
   - 按需求排除: EpicEye SDK / Orbbec / pygrabber / comtypes / 语音控制包
3. **清理**: `__pycache__` 与 `.pyc` 全清; `reference/机械臂_同事运行包_20260713/` 下 Windows 专属二进制删除 (pyrealsense2 win .pyd + RealSense SDK 安装器 x3);logs/runs/calib_data/model 保留
4. **git 换行归一化**: 仓库 blob 为 LF、Windows 工作树为 CRLF (83 文件) → 设置 `core.autocrlf input` 后 86 个假变更归零; `core.ignorecase=true` (Windows 克隆遗留) 掩盖过 doc/ 两个文件大小写改名 → 本次提交暴露为 rename
5. **`.gitattributes`** — 新增 `* text=auto` + 二进制类型标记, 换行策略固化进仓库

### 三相机联调验证 (2026-08-14)

- 三相机同时采集: UGREEN x2 (1280x720 MJPG, 27/28fps) + D435i (640x480 color+depth, 28fps) 全速无冲突
- **排坑: /dev/videoN 编号不稳定** — 插拔 USB 和 pyrealsense2 驱动 unbind/rebind 都会重排 video 节点; 硬编码索引 (如 video6) 会开到错误的设备, 表现是 "Device or resource busy" 或 "can't open camera" (曾误判为 USB 控制器等时带宽争抢)。**结论: 一律用 `/sys/class/video4linux/videoN/name` 按名称枚举 (EyeToHand/camera.py 已实现), 禁止硬编码索引**
- `multi_camera_preview.py`: 名称枚举版三相机预览 (UGREEN 自动探测 + D435i color/depth), q/ESC 退出清理正常
- **花屏根因 (已解决)**: video0 的 "Corrupt JPEG" 是 D435i 与 UGREEN 同 hub 时的等时带宽饱和 (isoc 丢包, 内核日志无报错, 单独跑或降带宽即消失)。**修复: D435i 挪到独立 USB 控制器** (现 2e:00.3), UGREEN x2 在 2a:00.1 — 三相机 720p30 全速 0 丢包
- 现场结论: D435i (无压缩 color+depth, ~46MB/s) 必须与 UGREEN 分开 hub/控制器; 两台 UGREEN 同 hub 720p30 无问题
- **Camera B 固定**: 两台 UGREEN 名称/序列号完全相同 (假序列号 SN0001), 唯一区分手段是 USB 物理口 → `camera_path: "usb-0:4.2"` 写入 `EyeToHand/config.yaml`, `camera.py` 支持 ID_PATH 匹配 (优先级高于名称), person_detector/rps_gesture/hand_in_cup 已接入
- **cv2 Qt 后端中文窗口标题 → 全黑窗口**: Qt5 highgui 字体配置缺失时, 中文标题窗口内容渲染全黑 (白图正常、帧采集正常); 窗口标题必须用 ASCII (如 "UGREEN 4.1" 而非 "UGREEN 口4.1")

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

### 实机联调 (2026-08-13 下午)

- 实机"没回应"根因: **`_mic_callback` 写 `_mic_q` 但 send 线程只消费 `_send_q`, 麦克风数据从未上送** (离线测试手动塞 `_send_q` 未覆盖真实数据流路径). 已修: send 循环每轮 `_drain_mic` 持续推流 + 5s 电平统计日志
- 实机验证: "你好" → ASR 实时转写 → FC do_sequence(打招呼) → LLM 回复"你好呀!" 全链路 ✅
- 麦克风: 默认设备=英特尔智音阵列(MME index 1), 安静时输出数字零属正常; 环境噪声电平 ~1200-2200 偏高, 实机联调注意降噪/麦克风距离
- 教训: **离线测试全绿 ≠ 实机路径通** — 关键数据流路径必须实机跑一遍

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | Linux 实机验证: UGREEN | V4L2 720P MJPG 打开 + PersonDetector 全链路跑通 |
| 🔴 | Linux 实机验证: 机械臂链路 | estun_config.yaml 按 Linux 调整 (IP/串口 COM5 → /dev/ttyUSB*) |
| 🔴 | 豆包 API Key 申请 | 火山引擎控制台 → API Key 管理,写入 .local.json 的 doubao.api_key |
| 🔴 | 实机阶段 1 | 无手纯对话: ASR 正确性 + TTS 流式出声 + 首包延迟记录 |
| 🔴 | 实机阶段 2 | 空载: 单手势/速度三档/序列/自定义多手势/越界拒绝 |
| 🟡 | 实机阶段 3 | 握手压力检测、stop 急停、抢话打断、断网重连、音色选择(4 选 1) |
| 🟡 | 延迟对比 | 同句指令 MiMo vs 豆包各 5 次,结果写回附录 C |
| 🟡 | CRP 实机导入测试 | estun_handle_sm.crp + estun_handle_cup_sm.crp(此前遗留) |
| 🟡 | 单相机 SM 实机联调 | estun_handle_sm.py + estun_handle_cup_sm.py(此前遗留) |
