# Progress — Project_hand

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
| 🔴 语音控制实机联调 | 连 OmniHand 完整链路测试 |
| 🟡 完整语音控制接入 | voice_control.py + CRP GOTO |
| 🟡 HandInCup 优化 | 调参 + 遮挡 |
| 🟡 加水实测 | 完整流程 |
| 🟡 抓取姿态微调 | 末端 + 手势 |
| 🟢 序列+速度 实机测 | 连灵巧手测试 |
| 🟢 sudu2 实机验证 | CRP 导入示教器 |
| 🟢 MediaPipe clearcut | |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 已通过 VoiceActivityDetector 接入 ✅ (之前 CFFI 回调崩溃已解决)
- ASR int8 量化精度不够 (已换全量 model.onnx 解决)
