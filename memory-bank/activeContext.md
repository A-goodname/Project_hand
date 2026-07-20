# Active Context — Project_hand

## 当前焦点

**Camera B 剪刀石头布手势识别** — MediaPipe 几何特征方案验证完成, 实机识别效果良好.

## 最近变更

### rps_gesture.py (2026-07-20)

1. **Camera B 剪刀石头布手势识别**: MediaPipe Hands 21 landmarks → 关节角度 + 距离比值 → 分类 石头/布/剪刀
2. **几何方法**: PIP 关节 3D 向量夹角 (<=35°=伸直) + tip→wrist/pip→wrist 距离比值 (>1.12=伸直), 尺度不变
3. **多人多手**: num_hands=4, 每只手独立稳定跟踪 (4 帧确认), 切换手不卡
4. **clearcut 修复**: GLOG_logtostderr=1 + minloglevel=5 → 不写磁盘日志 → clearcut 退出不阻塞. 全项目 3 处统一 (main.py / hand_in_cup.py / rps_gesture.py)
5. **YOLOv8n 预训练模型测试失败**: OpenI/PCL 模型 (rock-paper-scissors-detection-yolov8n.pt) 实机效果差, 已删除

### voice_llm_control.py (2026-07-17)

1. **LLM 语义理解**: MiMo v2.5 (多模态 speech+text-in text-out), 替换关键词匹配
2. **TTS 切换**: edge-tts → MiMo TTS (冰糖, 0.8s). 合成阻塞等 WAV 完成 → 手势+播放并发 (音画同步)
3. **双模式**: `--mode auto` (Silero VAD 自动, 默认) / `ptt` (按键)
4. **推理门**: LLM+TTS 期间暂停 VAD, 防 TTS 语音被回收
5. **音频优化**: 16kHz 录音 (VAD) → 8kHz 下采样 (API传输)
6. **Prompt 设计**: 紧凑 ~1200 chars, 语义理解 + 社交交互自主配合手势
7. **配置分层**: `voice_llm_config.json` (提交, key空) + `.local.json` (gitignored, 真key)
8. **日志**: `logs/voice_llm_*.log` + TTS 临时 WAV
9. **Stop 退出**: TTS 先播完 → 退出程序, 不等手势完成 (Auto VAD / PTT 均支持)
10. **Gesture delay**: 单手势 0.5s, 自定义序列默认 0.8s

### voice_omnihand_control.py (关键词匹配, 保留备用)
- 5 层匹配 + 拼音 + 多手势提取

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | LLM 语音控制实机联调 | 连 OmniHand |
| 🟡 | LLM 延迟优化 | 探索更快模型/流式 |
| 🟡 | 完整语音控制接入 Orchestrator | CRP GOTO 决策点 |
| 🟢 | 加水实测 | 完整流程 |
