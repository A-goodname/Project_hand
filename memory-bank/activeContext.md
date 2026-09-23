# Active Context — Project_hand

> 当前迁移工作区 `D:\Downloads\Project_hand`；服务器基线 `/data/VLA/Project_hand` 已取回为 `server/VLA`。

## 当前焦点

**服务器 VLA → Windows 再移植**：先完成设备/端口映射，再移除语音模块，最终恢复 Phase0 + Phase A-D 全流程。服务器 `VLA` 比本机旧 `main` 线性领先 88 个提交，迁移分支为 `codex/server-vla-windows`。

**2026-09-23 当前新主线**：控制器 `D:/Download` 的 Phase0、PhaseA2D、PhaseAll 导出为准。已生成 `doc/controller_exports/PhaseAll-NoRPS.crp`（84 个顶层块，去除 RPS 段/点/变量/8700，PhaseD 用 `<C_P3>` / `[C_P3]` 的 TranStrToCpos 分支）。Windows 入口 `scripts_gen/run_phase0_a2d.py` 只融合 Phase0+A2D；不启动语音或 RPS。PhaseD 必须键盘 C/回车 CONTINUE 或 A ABORT，视觉仅提示。操作见 `doc/phase0_a2d_windows.md`。已完成两台 UGREEN 抓帧、Phase0 YOLO 单帧、COM5 初始化 RESET、PhaseD HandInCup 单帧、D435i 枚举和无运动测试；**融合 ALL 尚未在控制器导入/执行**。RealSense Windows 通过根目录 `pyrealsense2.cp311-win_amd64.pyd` 导入，不从 pip 安装。

**Phase0 实机复测**：用户确认识人、挥手、握手和 Home 均正常。此前一次 Step4 的 COM5 `WriteFile failed` 未再现，暂按偶发问题记录，不能认定阈值故障。融合生命周期进一步改为主进程全程持有同一灵巧手串口，迎宾 UGREEN+YOLO 由 `Estun/phase0_camera_worker.py` 独立子进程持有，识人后立即关闭；已实测子进程独立打开/预热/关闭相机。融合 ALL 仍未上机执行。

**PhaseA2D 下一步**：控制器原版 `PhaseA2D-Down-444.crp` 派生了 `doc/controller_exports/PhaseA2D-KeyboardGate.crp`（57 块，`<C_P3>` / `[C_P3]` 分支），供单独实测。`estun_handle_cup_sm.py` 启动时预开 UGREEN PhaseD 相机+加载 HandInCup，但 READY 前不推理；PhaseB 位姿发送后关闭 D435i，READY 后丢弃 3 帧缓冲再开始视觉辅助。用户将测试 PhaseA2D，结果待回报。

**PhaseD 分支新路径（用户指定）**：`CONTINUE` 在 Python 松夹后由 CRP Wait 1s → MovL Home；`ABORT` 先持杯 MovL C_P1（sudu2），CRP 发 RELEASE（Python 此时松夹）→ Wait 1s → MovL C_P2 → 发 RESET → Wait 1s → MovL Home。两份派生 CRP 均已更新，原始控制器导出未动；静态结构与 Python 后续 RELEASE/RESET 协议测试已通过，实机仍待验证。

## 已提交 (2026-08-21)
- `ea680f3` Phase0 语音欢迎 (复用 /data/Speech)
- `dc24475` 三 CRP 融合 + RPS 交接加固 (defer/headless/retry + review A-G)
- `7e93228` phaseE 完成判断: 端口释放→进程退出 (实机 RPS 被过早收尾)
- `c25960b` 服务终端自动关闭 (默认去 exec bash, KEEP_TERM=1 保留)

## 最近变更

### Windows 双 UGREEN 物理角色确认 (2026-09-23)

1. DSHOW 实机预览确认：`UGREEN Camera 1080P`（当前索引 4）= Phase0 迎宾相机；`UGREEN Camera`（当前索引 3）= PhaseD HandInCup 判断相机。
2. 配置使用完整设备名自动搜索，索引只作现场诊断，避免重启后编号漂移。
3. 灵巧手为 `COM5`（STM32 S/N `304137433235`）；Socket 服务配置保持 `192.168.101.96:8080`，机械臂为 `192.168.101.100`。

### 同事单进程融合 single_all 多次实机验证通过 + 环境差异/备份 (2026-08-23)

**同事把多进程融合收拢为单进程 (`/data/xilu/single/single_all.py`)，多次实机验证通过。** 原多进程(P0 py + A2D py + phaseE py + rps_stream_merged + realtime_voice×2 + 8770门禁TCP + /tmp触发文件 + FIFO)全部收进一个 Python 进程；**CRP 完全不动** (`estun_fused_phase_all.crp` 依次连 8600/8080/8700)。入口 `bash /data/xilu/single/start_single.sh`。

**关键设计**：
- 灵巧手 `SingleHandHub` **唯一持有者**：open→enable→`is_enabled()`确认(未确认重试 30×0.5s)、**全程保持使能、阶段间只做手势绝无 disable/close**；`ensure_enabled()` 阶段边界校验；仅进程退出 `shutdown()` RESET→disable→close。RPS 出拳走共享代理(同实例+互斥锁)。
- `rps_active` 由编排器 `activate_rps()` **进程内调用**(无 /tmp 触发文件)；门禁 `HandoverConfirmationGate` **进程内对象**(无 8770 TCP)；语音 `VoiceEngine` quiet→gate→game 三段；完成/邀请信号进程内(无日志tail/FIFO)。
- 设备单进程持有：手(唯一)/迎宾cam/ D435i / RPS cam usb-0:2 / 麦克风音响 / 8768 Flask。

**→ 这解决了我们多进程"方案甲"追的坑**：手全程共享不失能(无跨进程交接、无"疯狂使能")、触发→RPS 进程内无缝(无文件时序)、RELEASE 无 RESET/无提前 disable、无 8770 TCP/日志tail/FIFO。**多进程方案甲(`start_phase_all.sh` + estun_handle_cup_sm handover/触发文件/假失能)被单进程取代为融合主线**(保留作参考)。

**环境差异(两 conda)**: Proj 与 VLA 的 torch **相同**(2.13.0+cu126)、python 3.11.15 相同、mediapipe 0.10.35 / ultralytics 8.4.48 相同。差异= **`sherpa-onnx==1.13.4`(requirements.txt line30, ASR, onnx推理、非 funasr) 只在 VLA**；Proj 未装但可装(它是项目依赖 single 用 `/data/xilu/conda-envs/VLA/bin/python` 即因含 ASR)。

**备份**: `/data/xilu/single` 已复制到 `reference/single/`(21文件, gitignore 忽略)。

### 融合方案甲重排 + 实机联调修复 (2026-08-22, 代码完成待实机复测)

**架构(方案甲)**: merged 进程(`rps_stream_merged.py`)全程拥有 usb-0:2(单相机 HandInCup 递杯 + RPS 切换); A2D py 让出 usb-0:2; voice(`--rps-game`)内建 8770 门禁供递杯确认; 灵巧手在 A2D 递杯段后交给 merged(RPS 手势)。

**A2D py (`estun_handle_cup_sm.py`)**:
- `_init_camera_b` no-op(让 usb-0:2); READY 写 `/tmp/a2d_ready`(通知 sh 起 RPS 服务); `_gate_setup` 融合(defer)不 spawn 门禁(等 sh 的 voice)。
- PhaseD 去 ABORT/Continue; RELEASE 用 `handover_ready` 区分**两次 RELEASE**(CRP 序 CUP→RELEASE(松杯)→GRIP→READY→RELEASE(递杯交接))—— 第1次只松夹, 第2次才有门禁确认。
- **第2次 RELEASE 先等门禁就绪**(voice 晚起, RELEASE 常先到)再 `wait_confirmed("拿好了")` → 松夹 → 发结束串。
- `_finalize_handover()` 在 `_handle_session` finally(socket 断开=递杯段真结束)才 `disable_hand`+CANCEL+写触发文件(灵巧手交接, 防"RELEASE 后马上 RESET")。
- `_is_person_seated=True`; 删 unused import + 死代码(Camera B/HandInCup/`_camera_b_loop`)。

**sh `start_phase_all.sh`**:
- 监控段: 等 `[ -f /tmp/a2d_ready ]` → 起 RPS 服务; 删"A2D 退出→起 RPS"后置链路; 8770 门禁=voice 内建(A2D 不再 spawn), 删 `wait_gate_gone`/`GATE_PATTERN`。
- HELPER 里 **voice 先起**(8770 门禁+日志就绪)再 merged(照同事 startup); 用 `rps_stream_merged.py --enable-hand RPS_HANDOVER_MODE=1`。
- py 服务输出 `tee` 日志 + `PYTHONUNBUFFERED=1`(否则 pipe 块缓冲, 终端日志延迟); 启动清 `a2d_ready/handover_ready/trigger/released` + 日志。
- `shutdown_rps`/cleanup 改杀 merged+voice。

**对方(wangxilu)**: `rps_stream_merged.py` 灵巧手 open 延迟到 `rps_active`(触发才 configure_hand); 修 `capture_loop` 缺 `import socket`(NameError)。

**实机踩坑(2026-08-22 定位)**: ①merged `socket` NameError(capture_loop) ②上次 `/tmp/a2d_ready` 残留→RPS 提前启(启动清) ③RELEASE 处 `disable_hand` 内部发 RESET→"RELEASE后马上RESET"(移到段尾) ④CRP 两次 RELEASE(`hand_judged` 跳过第2次→改 `handover_ready`) ⑤voice 晚起→RELEASE 到门禁没就绪(改先等门禁)。协作修改方案见 `/data/VLA/协作-灵巧手交接-A2D与merged-2026-08-22.md`。

### 音频会话验证 + check_audio 放宽 + 清孤儿 (2026-08-22)

1. **现象**: 融合 PhaseALL 无声排查。JBL 蓝牙音箱被"对方(wangxilu, 训练侧)会话"占/抢同一 A2DP, user 本机播落到无后端 sink 无声; 期间 user 的 PipeWire 守护进程一度卡死 → 系统设置"无输入/输出设备"
2. **验证 (结论)**: 语音(千寻=`realtime_voice --rps-game`)/greeting/门禁**统一走 user(uid 1000) 会话 → 默认 sink=JBL 才出声**。本机控制台登录=user(seat0/tty2)+SSH user 共享同一 `/run/user/1000` PipeWire; 对方(训练侧)先后用了两个用户: **之前 wangxilu**(uid 1001) 会话 → 独立 PipeWire 抢同一 BT A2DP → 无声源; **刚才的交接测试 sh 则以 user 身份跑**(见下). **音频跟着持有 A2DP 的会话走**（印证 start_greeting.sh 注释: 跨会话 sudo 播放无声）
3. **修复**: `start_phase_all.sh check_audio` 判据放宽 — **只要 user 默认 sink 是 JBL 即放行**, 多实例(wangxilu)只警告不阻断; 唯一硬失败=user 实例无该 JBL sink → exit 3; bash -n + 干跑 PASS
4. **清理**: `/data/xilu/test/start_rps_after_handover.sh`(对方交接测试脚本, **以 user 身份跑**)起的孤儿 — 千寻(21322)/rps_stream(21071)/FIFO holder(21069)/脚本(21318)全清, 8768/8770 释放, FIFO/`rps_config_phaseE.json` 删除; 该脚本前台 trigger 等"拿好了"+回车、交接完才退出 → 常驻非 bug, 但跑 PhaseALL 前须清
5. **实机验证**: user 会话独立播 greeting("来自同济大学的各位老师晚上好呀！")→ JBL 出声 ✅ (2026-08-22)

### 融合语音截断根修: 门禁 SIGINT 优雅关停 + 音频沉降 (2026-08-22, 方案A)

1. **原理 (读 /data/Speech realtime_voice.py + realtime_audio.py 确认)**: 回声防护 = **半双工麦克风门控** (`_mic_enabled.clear()` while TTS 播放, 播放完 `playback_tail_guard_ms` 1000ms 重开); `interrupt_output()` 由游戏(播邀请/结算前主动)或 ASR 触发。**根因** = 门禁被 SIGTERM → `realtime_voice` **无 SIGTERM 处理器**, `finally:audio.close()` 不执行 → 其 **pw-cat speaker 子进程孤儿化** + 音频图不释放 → RPS voice 无缝接手同一麦克风(UGREEN usb 内置 mic), 卡在未释放音频图上, 半双工没拦 JBL 回灌的机器人 TTS 回声 → 当人声打断
2. **修复**: ① `estun_handle_cup_sm.py` stop() 门禁 `terminate()`(SIGTERM)→`send_signal(SIGINT)` (触发 KeyboardInterrupt→audio.close 干净释放; 保留 SIGTERM/SIGKILL 兜底) ② `start_phase_all.sh` `wait_gate_gone` pkill -TERM→**pkill -INT** (SIGTERM/KILL 兜底), 起 RPS 前加 `GATE_AUDIO_SETTLE`(默认 2s) 音频沉降 ③ `start_cup_handover.sh` cleanup_gate 同步 SIGINT
3. **验证**: bash -n ×2 + py_compile + **全量 168 测试绿** (无回归)
4. **待实机** (需训练侧批准): 融合 demo RPS 交接不再"只吐两字"

### Phase0 语音欢迎出声修复 (2026-08-22)

1. **现象**: Phase0 人进门语音欢迎 JBL 无声, 但 paplay/pw-cat 直连都能出声
2. **根因**: greeting_announcer 用默认 config(带 pipewire_target=JBL) → `pw-cat --target` 在 A2DP sink 静默失败 (同 realtime_audio 坑)
3. **修复**: `estun_greeting_phase0.py _announce_voice` 复制 config 清空 pipewire_target → `--config /tmp/greet_config.json` 走 default sink=JBL; py_compile + 实测 (start_greeting.sh --no-enter --config 去target 可听到)
4. **待实机**: 融合 Phase0 人进门 JBL 播问候

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
