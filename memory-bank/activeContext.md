# Active Context — Project_hand

## 当前焦点

**Camera B D435i→UGREEN 替换完成。 P1/P2 ✅。 CRP 协议一致性验证 ✅。**
双 Socket 架构确认 (8600+8080)。Phase 0/Phase A-C 两份 CRP+Python 对齐。
待实机双相机全流程测试。

## 最近变更

1. **CRP 协议一致性修复 — orchestrator CRP 补全 + TXT 更新** (2026-07-14):
   - orchestrator CRP Phase A-C 入口补全 3 个 widget: `SocketCreate(8080, Socket0)` / `If(socketflag)` / `GetCurCPos`
   - Phase D 3 处 socket 引用修正: `Socket1→Socket0`
   - 双 Socket 架构验证: Socket1(8600, Phase 0) + Socket0(8080, Phase A-C+D)
   - Phase 0 / Phase A-C(Cup P3前) 两份 CRP + 两份 Python 端协议对齐确认
   - TXT 伪代码同步更新: 新增 v8 changelog + Phase 0 段 + socket 引用统一

2. **HandInCup curled gate 修复** (2026-07-14):
   - 手腕在框外 + score≥5（多手指在框内）→ 直接判定 hit，跳过 curl 检测
   - `curled_threshold: 25→35`（适配 720p）
   - `bbox_margin: 20→40`（适配 720p）
   - config 标注 `[分辨率相关]` 注释

2. **挥手时序优化** (2026-07-14):
   - `do_gesture(delay=0.5)` + `time.sleep(wave_interval/2)` 替代插值控制
   - `wave_interval: 0.15→0.45` (匹配同事 hello_frame_delay)
   - 每周期 ~1.45s

3. **椅子实时检测 + 非对称阈值 + baseline 锚定** (2026-07-14):
   - 删除 `calibrate_chair()` 锁定逻辑 → 每帧实时检测椅子 (`_detect_chair_live`)
   - 非对称阈值：进 10 帧 / 出 5 帧立刻 False
   - `_chair_baseline` 锚定：仅 not person_near 时更新，person_near 时面积差 ≤30% 才更新 `_chair_ref`
   - `is_chair_calibrated` → `has_chair`
   - `PersonDetector` 加 `start_camera` 参数/方法

3. **Camera B 生命周期调整** (2026-07-14):
   - IDLE: Camera B 关闭
   - GREETING enter: 开 Camera B + 启 seated detection（不等人，直接进）
   - HANDLE: seated detection 继续 → READY 判定

4. **Camera B D435i → UGREEN 替换** (2026-07-13):
   - 新增 `EyeToHand/camera.py`, 纯 RGB 无深度
   - HandInCup 移除深度校验

## 未解决问题

- **实机双相机全流程测试待进行**

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 P0 | **双相机全流程测试** | `Orchestrator/main.py` + `handlepluscup-orchestrator.crp` |
