# Progress — Project_hand

### Linux 环境迁移适配 (2026-08-14)
- [x] `EyeToHand/camera.py` 跨平台: Linux V4L2 + MJPG + sysfs 设备名探测, Windows DSHOW 保留
- [x] Proj 环境依赖安装 (opencv/pyyaml/pyserial/pyrealsense2/ultralytics/mediapipe), 排坑: pyrealsense2 版本后缀 + opencv-contrib 版本冲突
- [x] 清理 __pycache__/.pyc + Windows 专属二进制 (win .pyd, RealSense SDK 安装器)
- [x] git 换行归一化: core.autocrlf input (86 假变更→3 真实) + .gitattributes (text=auto)
- [x] 灵巧手串口: by-id 稳定路径 + dialout 授权 + 只读身份 + RESET 自检 (O10, B655****0014)
- [x] ESTUN 握手: 4 步 CRP (连接/推流/'OK' 接收/关闭) 全通过, 机械臂 192.168.101.100
- [x] 三相机同时采集: UGREEN x2 (27/28fps) + D435i (28fps) 全速无冲突
- [x] 排坑: /dev/videoN 编号随插拔/驱动重绑漂移 → 必须按 sysfs 名称枚举, 禁止硬编码索引
- [x] `multi_camera_preview.py` 新建 (名称枚举版三相机预览)
- [ ] video0 UGREEN 的 MJPG 流有 JPEG 解码损坏 (花屏待排查, 疑似 USB 链路/hub 共享)
- [ ] 重登会话后验证 dialout 直连 (当前用 sg dialout 绕过)

### 豆包端到端语音大模型 (全双工) 语音控制 (2026-08-13)
- [x] `Omnihand/voice_doubao_control.py`: 单文件 S2S 实现 (WS 会话 + 流式音频 + FC + 重连 + barge-in)
- [x] `voice_llm_prompt.py` 追加 DOUBAO_INSTRUCTIONS/DOUBAO_TOOLS/validate_tool_call/parse_fc_items (MiMo 部分未动)
- [x] `voice_llm_config.json` 追加 doubao 段 (api_key 空占位)
- [x] `requirements.txt` +websocket-client==1.8.0
- [x] mock 服务器 + 18 用例离线测试全通过 (FC 配对/barge-in/重连/优雅关闭)
- [x] 回归验证: MiMo 链路导入无异常, 两链路共存
- [x] `doc/voice_control_design.md` 附录 C
- [x] 豆包 API Key 申请 (控制台) + 写入 .local.json
- [x] 真实 API 免麦克风探测: WS/ASR/LLM/TTS/FC/优雅关闭 全链路 ✅
- [x] FC 排坑: tools 必须扁平格式 (OpenAI 包装不触发) — 已修 + 真实 API 验证
- [x] 排坑: 全双工必须持续推流 (52000033) — 主程序常开推流天然规避
- [x] 排坑: audio.delta 包大小不固定 → StreamingPlayer 跨块余量缓冲
- [x] 实机根因修复: mic_q 断链 (麦克风数据从未上送) → send 循环 _drain_mic 持续推流
- [x] 实机阶段 1 (部分): "你好" → ASR+FC+LLM 回复全链路 ✅ (扬声器播放/延迟对比待专项)
- [ ] 实机阶段 1 完整: 语音回复播放确认 + 首包延迟记录
- [ ] 实机阶段 2: 空载手势/速度三档/序列
- [ ] 实机阶段 3: 握手/急停/抢话/断网/音色
- [ ] MiMo vs 豆包延迟对比 (各 5 次)

### VLA 数据采集版 + CRP 完善 (2026-07-31)
- [x] `Orchestrator/estun_handle_data_sm.py`: Handle+Cup 检测 → P3 倒水点, 无 Cup 阶段
- [x] `doc/estun_handle_data_sm.crp` (35 widgets): 含 H_P4 回退避碰点 (CposOffset 动态计算), 无 Cup 段
- [x] `doc/estun_handle_cup_sm.crp` 补 H_P4 (53 widgets): Handle 倒水避碰, Cup 段保留
- [x] `doc/estun_handle_sm.crp` P3 速度 sudu → V100 (Z-up 抬升无需低速)
- [x] 窗口标题 em dash → ASCII `|` (双窗口/乱码修复)
- [x] 生成脚本参数化: `scripts_gen/gen_handle_data_crp.py --keep-cup`
- [ ] CRP 实机导入 + SM 联调

### 单相机状态机重构 (2026-07-31)
- [x] `Orchestrator/estun_handle_sm.py`: 仅水壶抓取, HandleCupDetector + 固定ABC + P3=Z-up
- [x] `Orchestrator/estun_handle_cup_sm.py`: 水壶+水杯, 同架构, Cup P3=Z-up
- [x] `doc/estun_handle_sm.crp` (23 widgets): 对应仅水壶 CRP
- [x] `doc/estun_handle_cup_sm.crp` (41 widgets): 对应水壶+水杯 CRP
- [x] `Orchestrator/config.yaml`: 新增 `handle_lift_z` + `cup_lift_z`
- [x] `handle_cup_detector.py`: 红色优先检测 (暗光适配)
- [x] 白色把手两遍检测 (紧→松)
- [ ] CRP 实机导入 + SM 联调

### Camera B 剪刀石头布手势识别 (2026-07-20)
- [x] 方案调研: YOLOv8n (OpenI/PCL) vs CNN vs MediaPipe几何 → 选定 MediaPipe 几何特征
- [x] YOLOv8n 预训练模型测试: 实机效果差, 已删除
- [x] MediaPipe 几何特征实现: PIP 关节 3D 角度 + 距离比值, 尺度不变
- [x] 核心 bug 修复: 角度阈值反了 (`>=150°` → `<=35°`)
- [x] 多人多手: num_hands=4, 每手独立 stable 跟踪 (4 帧确认)
- [x] 'q' 退出卡住: 匹配 hand_in_cup 模式 (独立 camera + waitKey)
- [x] clearcut 退出阻塞: GLOG_logtostderr=1 + os._exit(0), 全项目 3 处统一
- [x] 实机测试: 合适角度识别效果良好
- [ ] 集成到 Orchestrator 4 阶段调度
- [ ] 加水实测完整流程

### LLM 语义理解 (2026-07-17)
- [x] MiMo v2.5: speech+text-in text-out, 语义理解替代关键词匹配
- [x] MiMo TTS: 冰糖音色, 0.8s API 合成 WAV, 与 LLM 同 endpoint; 合成阻塞手势并发
- [x] System prompt: ~1200 chars, 9 examples, 社交交互自主配合手势
- [x] 双模式: auto (Silero VAD 默认) + ptt (按键)
- [x] 推理门: LLM+TTS 期间暂停 VAD 收指令
- [x] 音频: 16kHz VAD → 8kHz API → VAD 静音裁剪
- [x] 配置: voice_llm_config.json + .local.json (key gitignored)
- [x] 日志: logs/voice_llm_*.log
- [ ] LLM 语音控制实机联调

### 语音控制先行验证完成 (2026-07-17)
- [x] 全量模型切换: `model.onnx` (887MB), auto 优先选取
- [x] Silero VAD: `VoiceActivityDetector` + `silero_vad.onnx` 替换能量阈值
- [x] 5 层匹配流水线: 全局→速度→精确→字符模糊→拼音模糊
- [x] 拼音匹配: pypinyin TONE3 + SequenceMatcher, 自动处理同音字 (阈值 0.75)
- [x] 字符闸门: 汉字重叠度 ≥0.25, 防纯拼音碰撞
- [x] JSON 配置: `voice_commands.json` + 硬编码回退
- [x] 同义扩展: 新增 "招手"/"数字"/"数数" 等触发词
- [x] 防误触: 单字精确匹配 + 最长优先 + "握手"→序列(含压力检测)
- [x] 数字手势序列 delay: 0.5s → 0.8s
- [x] 模型目录重命名: `sherpa-onnx-sense-voice` (去日期后缀)
- [x] 去临时 WAV: transcribe() 直接用 accept_waveform
- [x] pypinyin 依赖 + requirements.txt
- [x] 文档同步: `doc/voice_control_design.md` 附录A
- [x] 29 测试用例全通过 (17 回归 + 3 字符模糊 + 9 拼音匹配)
- [x] 全局命令防误触: 单字 stop/exit 仅精确匹配 ("不停"不触发"停")
- [ ] 非预定动作序列: `_extract_gestures()` 精确+滑动窗口+拼音, 边界情况待调
- [ ] 多手势滑动窗口 refine: 长文本短词匹配 + 拼音误匹配

### 主流程优化 v15 (2026-07-17)
- [x] P3/HandInCup 超时双向延长: CRP 15s→20s + Python 10s→25s
- [x] 手动 CONTINUE 按键: 空格/'c' 最高权限覆盖
- [x] 人就位简化: only_person_detect → 纯人检测, 10帧稳定
- [x] Camera B 自动探测: camera_id=-1 → pygrabber 按名查找

### 语音控制基础设施 (v14, 2026-07-16)
- [x] ASR: sherpa-onnx SenseVoice int8 (227MB, 零PyTorch)
- [x] funasr 废弃: torch 2.12 输出乱码
- [x] `Omnihand/voice_omnihand_control.py` 完成
- [x] `doc/voice_control_design.md` 完整方案
- [x] requirements.txt 更新 (sherpa-onnx, sounddevice)

### v13 人就位 GPU 优化 + Cup P3 提速 (2026-07-16)
- [x] PersonDetector GPU化 + 合并双推理 + leave_frames 30→45
- [x] CRP sudu2 + 4处MovL提速

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 Linux 实机验证: UGREEN | V4L2 720P MJPG + PersonDetector 全链路 |
| 🔴 Linux 实机验证: 机械臂链路 | estun_config.yaml IP/串口 (COM5 → /dev/ttyUSB*) |
| 🔴 豆包 API Key 申请 | 火山引擎控制台 → 写入 .local.json doubao.api_key |
| 🔴 豆包实机联调 (1/2/3) | 无手对话 → 空载手势 → 握手/急停/抢话/断网/音色 |
| 🟡 MiMo vs 豆包延迟对比 | 各 5 次, 写回附录 C |
| 🔴 CRP 实机导入测试 | `estun_handle_sm.crp` + `estun_handle_cup_sm.crp` |
| 🔴 单相机 SM 实机联调 | `estun_handle_sm.py` + `estun_handle_cup_sm.py` |
| 🔴 语音控制实机联调 (MiMo) | 连 OmniHand 完整链路测试 (豆包上线后可跳过) |
| 🟡 完整语音控制接入 | voice_control.py + CRP GOTO |
| 🟡 HandInCup 优化 | 调参 + 遮挡 |
| 🟡 加水实测 | 完整流程 |
| 🟡 抓取姿态微调 | 末端 + 手势 |
| 🟢 RPS 手势集成到 Orchestrator | 4 阶段调度 |
| 🟢 序列+速度 实机测 | 连灵巧手测试 |
| 🟢 sudu2 实机验证 | CRP 导入示教器 |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 已通过 VoiceActivityDetector 接入 ✅ (之前 CFFI 回调崩溃已解决)
- ASR int8 量化精度不够 (已换全量 model.onnx 解决)
