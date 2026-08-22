# Active Context — Project_hand

> 工作区 /data/VLA/Project_hand(2026-08-16 起唯一;~/ 旧副本仅历史)。

## 当前焦点

**⚠️ ROBOT 语音播报被截断 (融合场景, 已定位 + 已实装根修, 待实机验证)** — 融合流程已通，但 RPS 的 ROBOT 语音"只吐两个字就停"（`[Robot]` 完整文本已生成, 播放层被 `interrupt_output()` 清 buffer）。**根因 (读 /data/Speech 代码确认)**：8770 门禁被 **SIGTERM**，`realtime_voice` 只捕 KeyboardInterrupt(SIGINT)、**SIGTERM 跳过 `finally:audio.close()`** → 其 pw-cat speaker 子进程孤儿化 + 音频图不释放 → RPS voice 接手同一麦克风时, 半双工门控没拦住 JBL 回灌的机器人 TTS 回声 → 被当成新的人声 → 打断 (`[Audio] microphone peak: 28712` 即麦克风开着时捕获强回声)。**单局正常**（无门禁占音频）。**实装修复 (2026-08-22, 方案A)**：门禁 kill 改 **SIGINT 优雅关停**（触发 audio.close 干净释放麦克风/pw-cat）+ 起 RPS 前 `GATE_AUDIO_SETTLE`(2s) 音频沉降；改 `estun_handle_cup_sm.py` + `start_phase_all.sh` + `start_cup_handover.sh`，全量 168 测试绿。**下一步**：实机融合 demo 验证 RPS 交接不再"只吐两字"；若仍截断再走方案B(仅加长沉降)/C(同事改 /data/Speech 加 SIGTERM 清音频)。

## 已提交 (2026-08-21)
- `ea680f3` Phase0 语音欢迎 (复用 /data/Speech)
- `dc24475` 三 CRP 融合 + RPS 交接加固 (defer/headless/retry + review A-G)
- `7e93228` phaseE 完成判断: 端口释放→进程退出 (实机 RPS 被过早收尾)
- `c25960b` 服务终端自动关闭 (默认去 exec bash, KEEP_TERM=1 保留)

## 最近变更

### 融合语音截断根修: 门禁 SIGINT 优雅关停 + 音频沉降 (2026-08-22, 方案A)

1. **原理 (读 /data/Speech realtime_voice.py + realtime_audio.py 确认)**: 回声防护 = **半双工麦克风门控** (`_mic_enabled.clear()` while TTS 播放, 播放完 `playback_tail_guard_ms` 1000ms 重开); `interrupt_output()` 由游戏(播邀请/结算前主动)或 ASR 触发。**根因** = 门禁被 SIGTERM → `realtime_voice` **无 SIGTERM 处理器**, `finally:audio.close()` 不执行 → 其 **pw-cat speaker 子进程孤儿化** + 音频图不释放 → RPS voice 无缝接手同一麦克风(UGREEN usb 内置 mic), 卡在未释放音频图上, 半双工没拦 JBL 回灌的机器人 TTS 回声 → 当人声打断
2. **修复**: ① `estun_handle_cup_sm.py` stop() 门禁 `terminate()`(SIGTERM)→`send_signal(SIGINT)` (触发 KeyboardInterrupt→audio.close 干净释放; 保留 SIGTERM/SIGKILL 兜底) ② `start_phase_all.sh` `wait_gate_gone` pkill -TERM→**pkill -INT** (SIGTERM/KILL 兜底), 起 RPS 前加 `GATE_AUDIO_SETTLE`(默认 2s) 音频沉降 ③ `start_cup_handover.sh` cleanup_gate 同步 SIGINT
3. **验证**: bash -n ×2 + py_compile + **全量 168 测试绿** (无回归)
4. **待实机** (需训练侧批准): 融合 demo RPS 交接不再"只吐两字"

### 融合运行前音频检查 check_audio (2026-08-22, 不 sudo)

1. **现象**: 融合 demo 音响无声; 根因 = 蓝牙音箱关机 (JBL sink 消失) + 双 PipeWire (user + wangxilu) 抢 A2DP
2. **修复**: `start_phase_all.sh` 加 `check_audio` 前置检查 — 不 sudo (杀不了别人进程): 检测"多套声音服务器并存 / JBL sink 不存在" → 打印说明 + 手动修复法 → `exit 3` 停止启动 (AUDIO_CHECK=0 跳过)
3. **实机验证 (2026-08-22)**: 停 wangxilu + JBL 开机/重连 → `api.bluez5.profile=a2dp-sink` + 默认 sink=JBL, paplay 测试音确认可出声

### 三 CRP 融合 + RPS 交接 (2026-08-21, 代码完成待实机)

1. **交付物**: `doc/estun_fused_phase_all.crp` (sha `2a2ab984`) + `start_phase_all.sh` + `scripts_gen/fuse_three_crp.py`; Phase0(8600)→wait3s→A2D(8080)→wait3s→E(8700); **159 测试绿**。详细约束/端口/设备串行/进程判断见 `techContext.md` 融合段
2. **待实机**: 示教器导入 `estun_fused_phase_all.crp` + `bash start_phase_all.sh` 走通三阶段; 重点验证 A2D defer-devices 连接时开设备耗时 < Phase B 60s、RPS 晚起 auto-invite 重试成功、8770 门禁→RPS voice 麦克风交接 (即当前焦点)

### 融合实机排坑：A2D 永不 accept + RPS 崩溃根因 (2026-08-21)

1. **A2D defer 永不 accept**: `_get_frames` 在 detector None (defer-devices) 恒 None → 主循环走 continue 分支从未 `_accept_tcp()` → 8080 永不 listening。**修复**: run 循环 `continue` 前也调 `_accept_tcp()`
2. **RPS 启动崩溃根因 = 设备沉降/进程没退干净, 非服务/启动方式**: rps_stream standalone 三种方式均正常 → 是 A2D 刚用完 D435i+UGREEN+串口 ~3s 后 RPS 启动即崩溃。**修复**: sh 等 A2D 进程完全退出 (锚定 pgrep, ≤30s 硬门禁超时 exit 1) + `RPS_SETTLE=5s` 沉降; 起 RPS 前 `wait_gate_gone` 确认 8770 门禁 (占麦克风) 进程+端口都消失, 防门禁→RPS voice 交接竞态 (这是当前焦点语音截断的根因)
3. **多窗口 CPU 渲染太卡**: 融合默认 **headless** (`--no-preview`, 不建 cv2 窗), `PREVIEW=1` 可开预览
4. **语音 PID 修正**: `| tee` → `> >(tee)` 进程替换, `$!` 指向真实 python 而非 tee

### 融合服务终端自动关闭 (2026-08-21, 用户指出 A2D 终端不自动关)

1. 根因: `open_term()` 命令串末尾 `exec bash` — py 退出后拉起交互 bash, 终端永久停留
2. 修复: 默认去 `exec bash` (py 退出 → 终端自动关); `KEEP_TERM=1` 时保留 (调试看日志); RPS 服务终端同理 (helper 前台 wait, shutdown_rps 后自动关)

### phaseE 完成判断修复: 端口释放 → 进程退出 (2026-08-21, 实机定位)

1. 根因: phaseE `serve()` accept 到机器人连接后 `break` → `finally: server.close()` **立即关闭监听 socket** (单连接模式) → 8700 从 game 一开始就不监听, 端口释放≠完成 → sh 误判过早停 RPS
2. 修复: `start_phase_all.sh` 改**进程判断** (`pgrep -f "$PHASEE_PATTERN"`); 8600 提示处同步; A2D 8080 释放信号仍可靠 (cup_sm run() 一直非阻塞 accept 保持监听, 仅 stop() 关)

### Phase 0 语音欢迎融入 (2026-08-21, 复用 /data/Speech 全部方案)

1. 文案零自定义: `_announce_voice()` 调 `start_greeting.sh --no-enter` **不传 --message**, 走对方 `config.json greeting.message`
2. 接入点: `estun_greeting_phase0.py` `_wait_for_person` 检测到人进门 (stable≥3) 成功后 `subprocess.Popen` 非阻塞 + `start_new_session=True` (独立进程组); 脚本不存在/失败安全跳过; 不改 CRP/状态机/sh

### start_cup_handover.sh 门禁清理对齐 (2026-08-22)

1. **现象**: 独立 Phase D 递杯门禁原先在单独窗口跑 `realtime_voice.py --config`, 窗口末尾 `exec bash` 常开不关; 且 cup_sm 退出后**不 kill 门禁** → 门禁持续占麦克风/JBL, 接着跑 Phase E RPS 会撞"门禁→RPS voice 回声自打断"同款竞态
2. **修复**: 去掉窗口 `exec bash` (门禁退出→窗口自动关) + `trap cleanup_gate EXIT` 在 cup_sm 退出后 kill 8770 门禁 (匹配模式同 wait_gate_gone); 与融合 start_phase_all.sh 两种门禁路径行为统一 (预启仍保留, 方便看门禁日志)
3. **验证**: bash -n + GATE_PATTERN 命中实际命令行 + trap 结构

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | **三阶段融合实机联调 + 语音截断** | `start_phase_all.sh` 走通 Phase0→A2D→E; 实测方案A (门禁 SIGINT 优雅关停 + `GATE_AUDIO_SETTLE` 沉降) 是否消除"只吐两字"; 仍截断走方案B/C |
| 🟡 | 证据合同 5→10 申报（训练侧） | ROT_THETA_WIN 对齐 10 已定案; 随 18 回合采集一并申报（协作线，非实机新动作） |
| 🟡 | 异常点 110950 重采 | pos_side 数据点 d=0.531 近距; 可随 18 回合采集顺带 |
| 🟡 | Phase 0 握手超时备选 | 若握手场景变慢：方案A=CRP 90→110~120s / 方案B=py force_release_timeout 60→45s（现冗余足够，未触发） |
| 🟡 | 豆包实机联调 + MiMo 对比 | 阶段 1/2/3 实机 + 延迟对比（遗留，非主线，去留待定） |

已通过/已闭环（2026-08-21 用户确认）：Phase E 单局 + 多局制复测 ✅ · data_sm 各档 P1 复测 ✅ · pos_side 只读验证 ✅ · 10 帧稳定窗口实机验证 ✅ · θ 稳定性记录工具逐档运行 ✅ · Phase D 失败根因 ✅ · 三 CRP 融合代码 ✅（待实机）· 单相机 SM 联调 ❌ 排除 · CRP 实机导入测试 ❌ 过时（Cup 阶段已实机走通） · sudu2 实机验证 ✅（随 Phase D CRP 实机已覆盖）
