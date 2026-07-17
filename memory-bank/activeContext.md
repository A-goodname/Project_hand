# Active Context — Project_hand

## 当前焦点

**主流程优化 + 语音控制修复** (2026-07-17) — 已实机测试通过 ✅

## 最近变更

### v15 主流程优化 (2026-07-17)

1. **P3/HandInCup 超时双向延长**:
   - CRP Phase D `SocketReadStr` mtesttime: 15000→20000 (CRP 总等待 22s = Wait 2s + SocketReadStr 20s)
   - Python 两处超时改为读 config: `EyeToHand/config.yaml` → `hand_in_cup.timeout: 25.0` (>CRP 22s, 3s 余量)
   - `main.py` `_run_hand_in_cup()` + `state_machine.py` `_handle_client_impl` READY 段均读 config

2. **手动 CONTINUE 按键**:
   - Phase D 期间，空格 或 'c' 键 → `ctx.hand_on_cup = True`
   - 最高权限：覆盖任何当前状态（None / False 均可），不等检测完成

3. **人就位简化 — `only_person_detect`**:
   - `EyeToHand/config.yaml` chair 段新增 `only_person_detect: true` (默认) + `person_stable_frames: 10`
   - `PersonDetector` 新增 `detect_person_only()` — 纯 YOLO class 0，不走椅子标定
   - `main.py` 新增 `_run_person_check()` — 独立纯人检测函数，连续 10 帧有人→就位，确认后退出 YOLO
   - phase 调度按 config 分支：`only_person_detect=true` 走 `_run_person_check`，false 走 `_run_seated_check`
   - `SharedContext` 新增 `_person_stable` 计数器

4. **语音控制防误触**:
   - `match_command()` 关键词占比检查: 文本 > 6 字且关键词占比 < 30% → 跳过 (防对话误触)
   - VAD `MAX_SPEECH_S=10.0` 不变 (长动作序列需要)
   - 两层防护: VAD 截断长时间静默段 + 匹配层占比过滤

5. **Camera B 自动探测**:
   - `camera_id: -1` (哨兵) + `camera_name: "UGREEN Camera 1080P"` → `UGreenCamera.__init__` 自动调用 `find_camera_id()` 解析
   - `EyeToHand/camera.py` 新增 `find_camera_id()` — 用 `pygrabber` 枚举 DSHOW 设备名→索引 (模仿 `find_mic` 模式)
   - PersonDetector + hand_in_cup.py ×3 调用方统一传递 `camera_name`，自动受益
   - `pip install pygrabber` (依赖 comtypes)

### 语音控制基础设施 (v14, 2026-07-16)

- ASR: sherpa-onnx SenseVoice int8 (227MB)
- `Omnihand/voice_omnihand_control.py`: 22手势+序列+速度, PTT+AutoVAD 双模式
- VAD 能量阈值 0.015, 静音 1.5s 切段, 测试通过
- `doc/voice_control_design.md`: 完整方案 (含 CRP GOTO 决策点设计)

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | HandInCup 继续优化 | landmarks-in-bbox 调参，遮挡场景 |
| 🟡 | voice_omnihand_control 序列+速度 实机测 | 连灵巧手测试 |
| 🟡 | 加水实测 | 完整流程加水倒水, 防溅 + HandInCup 判定 |
| 🟡 | 抓取姿态微调 | 末端抓取姿态 + 灵巧手手势 |
| 🟢 | 完整语音控制接入 Orchestrator | voice_control.py + CRP 决策点 GOTO |
| 🟢 | 完整流程稳定性测试 | 多次连续运行 |
| 🟢 | MediaPipe clearcut 彻底抑制 | shutdown 时 ERROR 日志泄漏 |
