# Active Context — Project_hand

## 当前焦点

**语音灵巧手控制先行验证 + 主流程优化**（暂停于 2026-07-16 17:55）

暂停位置: UGREEN camera DSHOW id 自动探测方案调研中。
- 硬编码 id 不稳定（昨天3今天1）
- Win32_PnPEntity 查不到 UGREEN 视频设备
- 需要按名称匹配或自动探测

## 最近变更

### v14 语音控制基础设施 (2026-07-16)

1. **ASR 方案确定**:
   - 选型: sherpa-onnx SenseVoice int8 (227MB)，纯 ONNX，零 PyTorch 依赖
   - funasr 已废弃: torch 2.12 > 2.3 导致输出乱码，PyTorch 降级风险大
   - 模型路径: `model/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2025-09-09/`
   - pip: `sherpa-onnx` v1.13.4（已安装）

2. **语音手势控制**:
   - `Omnihand/voice_omnihand_control.py` 完成（语法检查通过）
   - 22 个手势 + 2 条序列 + 3 级速度前缀
   - 两种输入模式: PTT(按键说话) + AutoVAD(能量阈值自动检测)
   - VAD 测试通过: 能量阈值 0.015, 静音1.5s 切段, "嗯"/"你好" 正确识别

3. **设计文档**: `doc/voice_control_design.md` — 完整语音控制方案 (ASR/TTS选型、决策点、CRP GOTO、命令队列、WAV分类)

### 主流程待优化 (实机测试反馈)

4. **UGREEN Camera ID**: config.yaml 设 `camera_id: 3`，但 DSHOW 枚举到 [1][2][3] 三个设备，需要实机插拔测试确认哪个是 UGREEN
5. **P3 等待时间**: CRP SocketReadStr 超时 + Python 端超时均需延长（当前 15s/10s 不够）
6. **人就位简化**: 椅子检测 + 人就位可简化为人检测（参考 Phase 0 GreetingHandler 的 YOLO person 检测），去掉椅子标定逻辑
7. **HandInCup 优化**: landmarks-in-bbox 判定需要继续调参，遮挡场景未覆盖

8. **手动 CONTINUE 按键覆盖**: Phase D 期间，空格/'c' 键手动设置 hand_on_cup=True → CONTINUE 分支，避免 HandInCup 长时间检测误判。

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | UGREEN camera ID 确认 | 实机插拔测试，确认 DSHOW id |
| 🔴 | P3 等待时间延长 | CRP SocketReadStr + Python 双向延长 |
| 🔴 | 手动 CONTINUE 按键 | 空格/'c' → hand_on_cup=True → 跳过 HandInCup 等待 |
| 🟡 | 人就位简化为 YOLO 人检测 | 参考 greeting，去椅子逻辑 |
| 🟡 | HandInCup 继续优化 | 调参 + 遮挡测试 |
| 🟡 | voice_omnihand_control 实机测 | 连灵巧手测试语音手势流程 |
| 🟢 | 完整语音控制接入 Orchestrator | voice_control.py + 决策点 CRP |
