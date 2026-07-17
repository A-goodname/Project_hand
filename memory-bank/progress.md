# Progress — Project_hand

### 语音手势控制完善 (2026-07-17)
- [x] 三档速度前缀齐全 (无前缀=直接, 快速/中速/慢速)
- [x] 预定义序列: 打招呼, 握手(压力检测), 数字手势
- [x] 停止指令全局生效 (10+关键词)
- [x] 防误触: 序列优先匹配, 关键词占比过滤, 删高频词
- [x] mic 可选 (--mic auto/ugreen/default)
- [x] 模型可选 (--model auto/full/int8)
- [x] AutoVAD 'q'键退出
- [ ] silero_vad 接入 (失败, is_speech无状态+CFFI崩溃, 已回退)
- [ ] 换完整 model.onnx 提升 ASR 精度
- [ ] 序列+速度 实机测试

### 主流程优化 v15 (2026-07-17)
- [x] P3/HandInCup 超时双向延长: CRP 15s→20s + Python 10s→25s (config-driven)
- [x] 手动 CONTINUE 按键: 空格/'c' 最高权限覆盖
- [x] 人就位简化: `only_person_detect: true` → 纯人检测, 10帧稳定
- [x] Camera B 自动探测: `camera_id: -1` → pygrabber 按名查找

### 语音控制基础设施 (v14, 2026-07-16)
- [x] ASR: sherpa-onnx SenseVoice int8 (227MB, 零PyTorch)
- [x] funasr 废弃: torch 2.12 输出乱码
- [x] `Omnihand/voice_omnihand_control.py` 完成
- [x] `doc/voice_control_design.md` 完整方案
- [x] 单手势测试通过 ✅
- [x] requirements.txt 更新 (sherpa-onnx, sounddevice)

### v13 人就位 GPU 优化 + Cup P3 提速 (2026-07-16)
- [x] PersonDetector GPU化 + 合并双推理 + leave_frames 30→45
- [x] CRP sudu2 + 4处MovL提速

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 ASR 精度优化 | 换完整 model.onnx (895MB) |
| 🔴 默认 mic 测试 | --mic default 对比 UGREEN |
| 🟡 序列+速度 实机测 | 连灵巧手测试 |
| 🟡 HandInCup 优化 | 调参 + 遮挡 |
| 🟡 加水实测 | 完整流程 |
| 🟡 抓取姿态微调 | 末端 + 手势 |
| 🟢 完整语音控制接入 | voice_control.py + CRP GOTO |
| 🟢 sudu2 实机验证 | CRP 导入示教器 |
| 🟢 MediaPipe clearcut | |

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 无法接入 (is_speech无状态) — 模型保留
- ASR int8 量化精度不够 (同音字混淆: "八"→"吧")
