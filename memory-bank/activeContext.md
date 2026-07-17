# Active Context — Project_hand

## 当前焦点

**语音控制先行验证完成** — Silero VAD + 全量 SenseVoice + 5 层匹配流水线 + 拼音模糊匹配全部通过。

## 最近变更

### voice_omnihand_control.py 完善 (2026-07-17)

1. **全量模型切换**: `model.onnx` (887MB) 替换 int8 (227MB), 同音字精度显著提升
2. **Silero VAD**: `VoiceActivityDetector` + `silero_vad.onnx` 替换能量阈值, 神经网络语音检测
3. **5 层匹配流水线**:
   - 全局命令 → 速度前缀 → 精确子串 → 字符模糊 (0.55) → 拼音模糊 (pypinyin, 0.75)
4. **拼音匹配**: pypinyin TONE3 + SequenceMatcher, 自动覆盖同音同调/近调/声韵母相近
5. **字符闸门**: 拼音匹配前置条件 (≥0.25), 防止纯拼音碰撞误匹配
6. **JSON 配置**: `Omnihand/voice_commands.json` — 触发词独立于代码, 缺失时硬编码回退
7. **防误触完善**: 单字精确匹配 + 最长触发词优先 + "握手" 默认走序列(含压力检测)
8. **数字手势序列 delay**: 0.5s → 0.8s
9. **去临时 WAV 写入**: transcribe() 直接 accept_waveform, 去掉无用 WAV 文件 IO

### 文档同步 (2026-07-17)
- `doc/voice_control_design.md` 附录A 更新为实际实现 (架构图/技术选型/5层匹配/指令表)

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | 语音控制实机联调 | 连 OmniHand 测试完整链路 |
| 🟡 | 完整语音控制接入 Orchestrator | voice_control.py + CRP GOTO 决策点 |
| 🟡 | HandInCup 继续优化 | landmarks-in-bbox 调参，遮挡场景 |
| 🟡 | 加水实测 | 完整流程加水倒水 |
| 🟢 | 序列+速度 实机测 | 连灵巧手测试 |
| 🟢 | MediaPipe clearcut 彻底抑制 | |
