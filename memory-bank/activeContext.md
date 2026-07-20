# Active Context — Project_hand

## 当前焦点

**语音控制先行验证** — Silero VAD + 全量 SenseVoice + 5 层匹配 + 拼音 + 多手势提取基本可用，滑动窗口拼音匹配仍有边界情况待调。

## 最近变更

### voice_omnihand_control.py (2026-07-17)

1. **全量模型**: `model.onnx` (887MB), `auto` 优先, 目录 `sherpa-onnx-sense-voice`
2. **Silero VAD**: `VoiceActivityDetector` + `silero_vad.onnx`, 替换能量阈值
3. **JSON 配置**: `voice_commands.json` 存储触发词 + 匹配参数, 缺失时硬编码回退
4. **5 层匹配**: 全局→速度→精确→字符模糊(0.55)→拼音模糊(pypinyin TONE3, 0.75)
5. **拼音匹配**: pypinyin + SequenceMatcher, 字符闸门 ≥0.25 防碰撞
6. **多手势提取**: `_extract_gestures()` — 精确子串 + 滑动窗口(字符+拼音), 按位置排序执行
7. **全局命令防误触**: 单字命令 (如"停") 仅精确匹配, 防 "不停"触发
8. **数字手势序列**: delay 0.5s → 0.8s
9. **去临时 WAV 写入**: transcribe() 直接 accept_waveform

### 已知问题
- `_extract_gestures` 滑动窗口 + 拼音仍有边界误匹配, 待调
- "握去然后张开比心" → 单手势未命中 (`match_command` 中 ratio 0.3 过滤长文本短词), 依赖 `_extract_gestures` 补齐

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | `_extract_gestures` 滑动窗口 refine | 减少误匹配, 确保多手势流程端到端 |
| 🔴 | 语音控制实机联调 | 连 OmniHand 测试完整链路 |
| 🟡 | 完整语音控制接入 Orchestrator | voice_control.py + CRP GOTO |
| 🟡 | HandInCup 优化 | 调参 + 遮挡 |
| 🟢 | 加水实测 | 完整流程 |
