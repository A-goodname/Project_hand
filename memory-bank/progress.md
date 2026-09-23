# Progress — Project_hand

## Windows 再移植 (2026-09-23)

- [x] 用户实机确认：独立 Phase0 全流程正常；独立 A2D 的 PhaseD ABORT 分支正常。
- [x] 用户实机确认：融合版 Phase0→A2D 交接和 CONTINUE 分支通过。
- [x] 再次实机确认融合 CONTINUE；日志证实 COM5 复用、两相机生命周期与松夹收尾。Handle 判稳慢主因是 θ 极差 5–12° 超过 5° 门槛，Cup/Handle 位置与姿态已稳定；先保留安全门槛。
- [ ] 融合版 ABORT 分支实机验证；如需优化 Handle θ，进一步采集同一摆位的独立 θ 稳定性数据。
- [x] 服务器 `VLA` 分支取回本机，确认相对旧 `main` 线性领先 88 个提交
- [x] Windows 设备枚举：D435i ×1、UGREEN ×2、OmniHand `COM5`、ESTUN `192.168.101.100`
- [x] 双 UGREEN 实机预览定案：Phase0=`UGREEN Camera 1080P`(当前 4)，PhaseD=`UGREEN Camera`(当前 3)
- [x] Windows DSHOW 完整名称优先匹配，避免两只 UGREEN 子串误匹配
- [x] 端口/现场门禁相关 40 个无动作测试通过
- [ ] 完成语音模块移除并清理依赖/文档
- [ ] 完成 Phase0 + Phase A-D 离线协议验证
- [ ] 经现场安全确认后进行全流程实机验证

## 待续 (2026-08-22): ROBOT 语音截断 (融合场景) — 方案A 已实装, 待实机验证
- [x] 现象: 融合流程已通, 但 RPS ROBOT 语音只吐两个字就停; 单局正常融合异常
- [x] 诊断 (读 /data/Speech 代码确认): 8770 门禁被 SIGTERM → realtime_voice 无 SIGTERM 处理器, finally:audio.close() 不执行 → pw-cat speaker 子进程孤儿化 + 音频图不释放 → RPS voice 接手同一麦克风, 半双工没拦 JBL 回灌机器人 TTS 回声 → 当人声打断 (peak 28712 佐证)
- [x] 修复 (方案A, 2026-08-22): 门禁 kill 改 SIGINT 优雅关停 (触发 audio.close 干净释放) + 起 RPS 前 GATE_AUDIO_SETTLE(2s) 沉降; 改 estun_handle_cup_sm.py + start_phase_all.sh + start_cup_handover.sh; 全量 168 测试绿
- [ ] 实机验证 (需训练侧批准): 融合 demo RPS 交接不再"只吐两字"; 仍截断走方案B(加长沉降)/C(同事改 /data/Speech)

### 融合服务终端自动关闭 (2026-08-21)
- [x] open_term() 默认去 exec bash → py 退出终端自动关; KEEP_TERM=1 保留 (调试)
- [x] bash -n + 命令串模拟测试 PASS

### 融合 phaseE 完成判断修复 (2026-08-21, 实机定位)
- [x] 现象: RPS 可起 (第 2 次重试成功), 但 sh 过早"phaseE 已完成 (8700 释放)→停 RPS"
- [x] 根因: phaseE serve() accept 后立即 close 监听 socket (单连接), 8700 从 game 一开始就不监听 → 端口释放≠完成
- [x] 修复: start_phase_all.sh 改 pgrep 进程判断 (PHASEE_PATTERN/PHASE0_PATTERN/A2D_PATTERN); 8600 提示处同步
- [x] bash -n + 进程判断模拟测试 PASS
- [ ] 实机复测: 融合 demo RPS 交接不再过早收尾 (等 phaseE 真正结束才停 RPS)

### 融合 RPS 交接加固 (2026-08-21, 同事 review + 用户实测驱动)
- [x] 用户实测: hand_in_cup.py 按 q 关闭后 0s 即起 PhaseE 成功 → 推翻"USB 沉降 15s 不够"; 根因 = A2D 进程没退干净仍起 RPS
- [x] start_phase_all.sh 等 A2D 退出硬门禁 (超时 30s → exit 1); RPS_SETTLE 15→5s
- [x] cup_sm stop() teardown 补齐 (_hic_active / join 线程 / close 检测器, 对齐状态机 finally); Phase0 stop() 显式 disable_hand()
- [x] helper RPS 失败重试 ≤3 次 (等 10s 沉降 + 8768 确认); 语音 PID >(tee); wait_gate_gone 轮询确认 8770; 启动前清旧实例 pkill -9
- [x] bash -n + 模拟测试 + 全量 168 测试绿
- [ ] 实机验证 (需训练侧确认): start_phase_all.sh 走通三阶段, 重点 RPS 交接不再崩

### Phase 0 语音欢迎融入 (2026-08-21)
- [x] 用户定案: 复用 /data/Speech 全部方案, 文案零自定义 (走对方 config.json greeting.message)
- [x] estun_greeting_phase0.py `_announce_voice()`: Popen start_greeting.sh --no-enter (非阻塞 + 独立进程组); 失败安全跳过
- [x] 触发点: `_wait_for_person` 检测到人进门 (stable≥3) 成功后异步播报; 不改 CRP/状态机/sh
- [x] 新测试 5 例 + 全量 166 测试绿
- [ ] 实机验证 (需训练侧确认): 导入 estun_greeting_phase0.crp 跑 Phase0, 人进门时 JBL 播报对方问候文案

### start_cup_handover.sh 门禁清理对齐 (2026-08-22)
- [x] 独立 Phase D 递杯门禁: 去掉窗口末尾 exec bash (门禁退出→窗口自动关)
- [x] cup_sm 退出后 `trap cleanup_gate EXIT` kill 8770 门禁 + 轮询确认 (匹配模式同 wait_gate_gone), 释放麦克风/JBL
- [x] 防接着跑 Phase E RPS 撞"门禁→RPS voice 回声自打断"同款竞态
- [x] bash -n + GATE_PATTERN 命中断言 + trap 结构 PASS

### Phase0 语音欢迎出声修复: greeting_announcer 去 pipewire_target (2026-08-22)
- [x] 现象: Phase0 人进门语音欢迎 JBL 无声, 但 paplay/pw-cat 直连都能出声
- [x] 根因: greeting_announcer 用默认 config(带 pipewire_target=JBL) → `pw-cat --target` 在 A2DP sink 静默失败 (同 realtime_audio 已知坑)
- [x] 修复: `estun_greeting_phase0.py _announce_voice` 复制 config 清空 pipewire_target → `--config /tmp/greet_config.json` 走 default sink=JBL
- [x] py_compile + 实测: `start_greeting.sh --no-enter --config <去target> ` 再播 问候语可听到 (2026-08-22)
- [ ] 实机融合: Phase0 人进门 JBL 播"来自同济大学的各位老师晚上好呀！"

### PhaseD 门禁(8770) TTS 出声修复: 同 pw-cat --target 坑 (2026-08-22)
- [x] 现象: 门禁 TTS("拿稳了吗？/快来接一下。/收到。") JBL 无声; RPS voice(--rps-game) 正常 (已用去 target config)
- [x] 根因: 门禁启动用 --config /data/Speech/config.json (带 pipewire_target=JBL) → realtime_audio pw-cat --target 在 A2DP 静默失败
- [x] 修复: 门禁改 --config /tmp/gate_config.json (复制 config.json 清空 pipewire_target, 走 default sink=JBL); cup_sm _spawn_handover_gate + start_cup_handover.sh 预启都改; GATE_PATTERN 同步改成 /tmp/gate_config.json (唯一匹配不误伤 RPS voice)
- [x] bash -n ×2 + py_compile + 无残留旧 pattern
- [ ] 实机融合: 门禁提示音可听到

### 融合运行前音频检查 check_audio (2026-08-22, 不 sudo)
- [x] 现象: 融合 demo 音响没声音; 根因 = ①蓝牙音箱关机 (JBL sink 消失) ②双 PipeWire (user + wangxilu) 抢 A2DP
- [x] 手动修复验证: 停 wangxilu 音频 (sudo pkill -u wangxilu -x pipewire/wireplumber/pipewire-pulse, 需 sudo) + JBL 开机 + bluetoothctl connect → JBL 回 user 实例, api.bluez5.profile=a2dp-sink, 默认 sink=JBL, paplay 测试音可出声 (实测)
- [x] start_phase_all.sh 加 check_audio 前置检查: 不用 sudo — 检测"多套声音服务器并存 / JBL sink 不存在" → 打印说明 + 手动修复法 + exit 3 停止启动 (AUDIO_CHECK=0 可跳过)
- [x] bash -n + 检测逻辑干跑验证 (正确识别多余实例/排除当前 user)
- [ ] 实机融合: 音频修复后重跑 start_phase_all.sh 确认 Phase0 语音欢迎 + RPS voice 正常出声

### 融合语音截断根修: 门禁 SIGINT 优雅关停 (2026-08-22, 方案A)
- [x] 原理: 半双工麦克风门控 (TTS 播放时 _mic_enabled.clear, 播放完 tail_guard 1000ms 重开); interrupt 由游戏/ASR 触发
- [x] cup_sm stop() 门禁 terminate()(SIGTERM)→send_signal(SIGINT) (触发 KeyboardInterrupt→audio.close 干净释放; SIGTERM/SIGKILL 兜底)
- [x] start_phase_all.sh wait_gate_gone pkill -TERM→-INT (SIGTERM/KILL 兜底); 起 RPS 前加 GATE_AUDIO_SETTLE(默认 2s) 音频沉降
- [x] start_cup_handover.sh cleanup_gate 同步 SIGINT (统一两种门禁路径)
- [x] bash -n ×2 + py_compile + 全量 168 测试绿 (无回归)
- [ ] 实机验证 (需训练侧批准): 融合 demo RPS 交接不再"只吐两字"

### Phase E RPS 接入 (2026-08-21, 已交付+复测, 待三 CRP 融合)
- [x] 用户写 CRP → `doc/estun_rps_phaseE.crp` (Down-667, sha a5054c91, 端口 8700; 独立 whileflag 法); py `estun_rps_phaseE.py` + config rps 段 + `start_rps_phaseE.sh` 一键启动 (148 测试绿)
- [x] 单局 demo 实机走通 (2026-08-21) + **多局制 rounds=0 等人退出 + RPS 服务优雅收尾 de143d1 实机复测通过** (用户确认)
- [x] 排坑: 位姿米/度 · while 只循环一次 (socketreadstr operateval 不能驱动 while 条件) · RPS 相机 usb-0:2 路径解析 · 双 PipeWire 抢 JBL A2DP · pw-cat --target 静默失败 → 清 pipewire_target 走 default sink
- [ ] 三 CRP 融合实机联调 (见"待完成")

## 历史已完成段 (08-15 ~ 08-20, 旋转判档/抓取/示教)
> 已固化到 `systemPatterns.md`（检测链/5 档判档/C=C0-θ/t_offset 插值/Z 钳位）与 `techContext.md`（约束/路径/排坑），逐日排坑叙事不再在此重复。

- ✅ 旋转判档 5 档简化定案 ([-8,8]=0 / (8,22]=15 / >22=30, 负侧对称; 无 ±45 物理摆位限定 [-35,+35]) + 逐档 t_offset 示教 + data_sm band 查表/θ 插值/C 解析式/P1 Z 钳位 [0.360,0.362]
- ✅ 姿态规律 C=C0-θ (误差<0.2°) 不逐档示教姿态
- ✅ d2 阶段: 9 事件 CRP-207 审计 + EpisodeRecorder + raw/ 流 + canonical-a 10 段 (wrist_depth 955 PNG/段, depth_skip=2)
- ✅ Phase A-C 移植实机走通 + Phase D 失败根因 (HandInCup gate `_hic_active` + cup 裁剪/杯壁 + P3 前移) + 语音门禁 8770 接入 (语音优先/视觉兜底)
- ✅ 波动根因修复 (anchor 固定 seed) · 姿态几何法重构 (Y=壶身轴×环法线) · 检测器简化 (geometry 主成分) · HSV 青色贴纸链替代 SAM

## 待完成

| 任务 | 说明 |
|------|------|
| 🔴 三阶段融合实机联调 | **主线已改同事单进程 `single_all.py`(`bash /data/xilu/single/start_single.sh`, 多次实机验证通过)**; CRP 不动 estun_fused_phase_all.crp; 多进程方案甲 start_phase_all.sh 被取代(保留参考, 备份 reference/single/) |
| 🟡 证据合同 5→10 申报 (训练侧) | ROT_THETA_WIN 对齐 10; 随 18 回合采集一并申报 (协作线) |
| 🟡 异常点 110950 重采 | pos_side 数据点 d=0.531 近距; 可随 18 回合采集顺带 |
| 🟡 Phase 0 握手超时备选 | 若握手变慢: 方案A=CRP 90→110~120s / 方案B=py force_release_timeout 60→45s (现冗余足够) |
| 🟡 豆包实机联调 (1/2/3) + MiMo 对比 | 无手对话 → 空载手势 → 握手/急停/抢话/断网/音色; 延迟对比 (遗留, 去留待定) |

已通过/已闭环 (2026-08-21 用户确认): Phase E 单局 + 多局制复测 ✅ (de143d1) · data_sm 各档 P1 复测 ✅ · pos_side 只读验证 ✅ · 10 帧稳定窗口实机验证 ✅ · θ 稳定性记录工具逐档运行 ✅ · Phase D 失败根因 ✅ (cup 裁剪+杯壁 + P3 前移 + HandInCup gate fix + 语音门禁) · sudu2 实机验证 ✅ (随 Phase D CRP 实机覆盖) · **三 CRP 融合代码 ✅ (待实机)** · 单相机 SM 联调 ❌ 排除 · CRP 实机导入测试 ❌ 过时 (estun_handle_cup_sm.crp 已实机走通)

## 已知问题
- RealSense_R rs.align 不可用 (内参损坏)
- UGREEN Camera B 纯RGB无深度 (P3 X全示教)
- silero_vad.onnx 已通过 VoiceActivityDetector 接入 ✅ (之前 CFFI 回调崩溃已解决)
- ASR int8 量化精度不够 (已换全量 model.onnx 解决)
