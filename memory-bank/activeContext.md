# Active Context — Project_hand

> 工作区 /data/VLA/Project_hand(2026-08-16 起唯一;~/ 旧副本仅历史)。

## 当前焦点

**Phase 0 打招呼独立交付(✅ 已完成 2026-08-21)** — `Estun/estun_greeting_phase0.py` + `doc/estun_greeting_phase0.crp` 实机 4 步走通(等人→挥手→握手→Home)。时序核对:CRP SocketReadStr 10/130/30/90s vs py recv 5/130/30/90s;Step 4 py 握手最坏 92.5s 略超 CRP 90s(~2.5s 缺口),实机握手几秒内完成,用户判定冗余足够,暂不改(方案 A=CRP 90→110~120s / 方案 B=py force_release_timeout 60→45s 备选)。

**传统路线 Cup 阶段移植** — `estun_handle_cup_sm.py` + CRP 移植 Handle+Cup 全流程。Phase A-C（Handle 抓取→倒水→Cup 抓取→递送）已成功实机运行；**Phase D（READY→HandInCup→CONTINUE/ABORT）失败** — CRP 到 C_P3 后 ~2s 自动停止运动，未发送 READY 到 Python，连接关闭。根因待查（CRP SocketSendStr/SocketReadStr 配置或时序问题）。

**多视角朝向示教（本地主线，抓取链路已打通）** — 旋转判档定版（2026-08-19）：纯 θ 区间 **5 档** `[-8,8]=0, (8,22]=15, >22=30`（负侧对称，无 ±45，+45 统一落 +30）。检测链 `handle_cup_detector.py`：HSV 青色贴纸 → ratio → θ 反解 → band，模型/门限在 `Orchestrator/config.yaml handle_rotation`。**抓取姿态规律**：绕竖直轴旋转只改 C（`C=C0-θ`，误差<0.2°），A/B 恒定，姿态不逐档示教。**t_offset 5 档示教完成**（p1_offsets 表 -30/-15/0/+15/+30）。**data_sm 已接入**：θ 稳定窗口中位数 → band → 查 p1_offsets + θ 档内线性插值（`_interp_p1_offset`）+ C 解析式 + P1 Z 钳位 [0.360,0.362]。稳定窗口 5→10 / skip 3（2026-08-20，4 SM 共享，稳定耗时 ~1s）。pos_side v2（08-20）：双阈值 rd_a/rd_b + 分段线性连续角度，中间角度不再二值化。待办：实机复测各档 P1 落点（尤其 +15 档 Z 天然 364mm 观察钳位）、pos_side 只读验证、异常点 110950 重采。

训练侧线（**已完成**）：P2/P1 双确认协议适配（D0 门禁）协作线已交付完毕。

**Phase E RPS 接入（2026-08-21，单局已走通 + 多局制/收尾代码完成待实机）** — `/data/Speech` RPS 服务（语音/视觉/灵巧手出拳）之上，机械臂负责端手到 P1 示教点位 + 游戏保持 + 回 Home。用户写 CRP → `doc/estun_rps_phaseE.crp`（sha `a5054c91`，端口 8700）：open→P1→While(独立 whileflag)→Home(存储点)→close；**位姿米/度**。py 单一入口 `Estun/estun_rps_phaseE.py` + config + `start_rps_phaseE.sh` 已交付。**完整单局已实机走通**（臂到位→auto-invite→JBL 播邀请→"开始"→判拳→结算→DONE→Home）；**多局制 rounds=0 等人退出**（"退出游戏/不玩了"→日志裸"退出"行→DONE）+ **RPS 服务优雅收尾** 代码完成，153 测试绿。排坑全记录：while 独立标志法 / 相机 EyeToHand 路径解析 / 音响双 PipeWire + pw-cat default sink。待办：多局制实机复测、三 CRP 融合。

## 最近变更

### Phase E 多局制 + RPS 服务优雅收尾 (2026-08-21, 代码完成待实机)

1. **多局制 rounds=0 等人退出 (用户要求)**: config `rps.rounds 1→0` (0=等人退出 默认; N≥1 固定 N 局保留) + 新增 `quit_keyword: "退出"`; py `_wait_round_done` 同时 tail 两个信号 — `[RPS][RESULT]` (计数 N 局) 与 **裸行 == "退出"** (玩家在 replay-decision 窗口说 "退出游戏/不玩了" → `rps_voice_game.py` handle_user_text quit 分支 `print("退出")`, 精确匹配不误伤含"退出"字样其它行) → 任一命中/超时 → 发 DONE 回 Home
2. **收尾机制 (同事侧确认)**: 退出仅 honor 在 `awaiting_replay_decision` 窗口内 (结果播报后到下次开始); 退出后 `invited=False` 需新邀请才可再开; 每局续玩说 "开始" 即可, 无需局间 FIFO 再邀请
3. **RPS 服务优雅收尾**: `start_rps_phaseE.sh` 新增 `shutdown_rps()` — phaseE py 退出后 (无论 rc) SIGTERM `rps_stream.py` + `realtime_voice.py --rps-game` (atexit 释放相机/MediaPipe/串口), sleep 1 兜底 `fuser -k 8768`; `--rps-game` 限定不误伤递杯门禁 (8770 realtime_voice --config); helper 语音管道包 `set +e` 防 SIGTERM 下 `set -e+pipefail` 提前 abort 丢退出信息
4. **验证**: py_compile + bash -n + 全量 **153 测试绿** (新增 5: rounds=0 退出触发 / 一局后退出 / 精确匹配不误伤 / rounds=0 超时兜底 / rounds=2 两局后 DONE)
5. **待办**: 实机复测多局制 (第 1 局→说"开始"→第 2 局→说"退出游戏"→DONE→Home→RPS 服务自动收尾); 三 CRP 融合 (Phase0 8600 + A2D 8080 + PhaseE 8700)

### Phase E RPS 接入: py + config + 一键脚本 (2026-08-21)

1. **背景**: RPS 语音/视觉/灵巧手出拳由 /data/Speech 服务负责 (rps_stream.py 8768 + realtime_voice.py --rps-game); Phase E 只负责机械臂 — 端手到 P1 → 游戏保持 → 回 Home
2. **用户定案 (CRP 用户自己写)**: `doc/estun_rps_phaseE.crp` (= `~/下载/PhaseERPS-Down-667.crp`, sha `a5054c91`) 端口 **8700**: open → P1(sendpos→getpos→**movj** RPSpos) → While(**whileflag==1** 独立标志) → **Home 存储点直 movj** → close; 12 widgets (socketcreate/getcurcpos/cpostostr/socketsendstr/socketreadstr/transtrtocpos/movj×2/equal×2/while/socketclose)
3. **位姿格式 = [x,y,z,a,b,c] 米,度** (cpostostr/transtrtocpos lengthunit=m angleunit=deg, 非 Phase 0 的 mm) — py `_fmt_pose6` 用 `.3f`(xyz 米)/`.2f`(abc 度)
4. **py** `Estun/estun_rps_phaseE.py` (单一入口, 镜像 Phase 0 结构): TCP 8700; 收 X1 → 回 P1(米,度); tail 语音日志等 `[RPS][RESULT]` → 发 DONE (任意非空串, **内容不比**); **DONE 后等 EOF 优雅退出 (Home 存储点, py 不回位姿)**
5. **config** `estun_config.yaml rps:` 段: port/done_flag/voice_log_path/arm_settle_delay/max_game_wait/rounds/poses.play(占位待示教)/auto_invite(fifo_path)
6. **一键脚本** `start_rps_phaseE.sh`: 终端① RPS 服务 (rps_stream --enable-hand + realtime_voice --rps-game, 语音 stdout→终端+/tmp/rps_voice_phaseE.log, PYTHONUNBUFFERED=1; auto_invite=1 时 stdin→FIFO); 本终端 phaseE py (一轮 socket 断开自动退出)
7. **排坑**: ①**位姿单位米/度** (Phase 0 是 mm, 用户 CRP 是 m) ②游戏完成信号 = 语音日志 [RPS][RESULT] (HTTP 8768 无胜负, 不改同事代码) ③**日志 tail 起始位置须在连接开始时记录** (进 _wait_round_done 才记 → settle 窗口内到达的结果被当旧内容跳过, 已修) ④realtime_voice 回车邀请 = asyncio add_reader(stdin), FIFO 写 \n 可触发; 但 FIFO 需持写端 (tail -f /dev/null > fifo) 防读端 EOF 停读 ⑤语音进程[RESULT]打印无 flush, 需 PYTHONUNBUFFERED=1 ⑥**socketreadstr operateval 不能驱动 while 条件**: 成功=0 / 超时=非0, 直接当条件会"只循环一次" (详见下条 while 修复)
8. **验证**: py_compile + 新测试 6 例 (正常流程米度/no_rps/mock/全零拒绝/auto_invite FIFO/DONE 后 EOF) + **全量 148 测试绿**
9. **约束**: phaseE **禁 open 灵巧手** (RPS 服务独占 STM32 串口) / **禁碰相机** (RPS 独占 /dev/video6)
10. **待办**: ①实机示教 P1 (灵巧手落 RPS 相机 FOV) 写入 config ✓ (已填 `rps.poses.play` 米值) ②导入 `doc/estun_rps_phaseE.crp` 跑 start_rps_phaseE.sh 走通完整 demo ③三 CRP 融合预留 (Phase0 8600 + A2D 8080 + PhaseE 8700, 段间 wait 1s)

### Phase E 实机: while 只循环一次 → 独立标志法修复 (2026-08-21)

1. **实机现象**: 臂到 RPSpos 后 ~2s 自动回 Home (CRP while 只循环一次), 游戏无法开始; 语音服务正常 (Press Enter 邀请 + 麦克风收声)
2. **根因**: 原 CRP `while endflag==1` 直接用 socketreadstr 的 operateval→endflag; ESTUN operateval **成功=0 / 超时=非0** → 首次读取(游戏期间 py 静默→2s 超时)后 endflag≠1 → 循环退出 → movl Home
3. **Codroid 限制**: while 条件**无法比较字符串** (不能 `while rpsstr3 != "DONE"`)
4. **修复 (用户改 CRP → Down-667, 入库 `doc/estun_rps_phaseE.crp` sha `a5054c91`)**: 独立循环标志 — 顶层 `equal whileflag=1` 初始化; `while whileflag==1`; 循环内 `socketreadstr(ov→endflag, 不进条件)` + `if endflag==0`(成功读到数据) → `equal whileflag=0` → 退出 → movj Home
5. **py 无需改**: DONE 内容不比, 任意非空消息读到即 break; done_flag 仅作标识
6. **验证**: py_compile + 148 测试绿 (P1→等游戏→DONE→EOF 逻辑不变)

### Phase E 实机: 音响/相机排坑 (2026-08-21)

1. **音响无播报根因 = 双声音服务器抢蓝牙**: 系统同时跑 `wangxilu` 和 `user` 两个 PipeWire, JBL(A2DP) 传输被两边抢 → user wireplumber 报 *"Multiple sound server instances ... trying to use Bluetooth audio at the same time"* → 挂不上 bluez sink → 无声。**修复**: 停 wangxilu 音频 (pipewire/wireplumber/pipewire-pulse, 需 sudo) → 重连 JBL → `bluez_output.78_44_05_FA_12_B9.1` 出现 → `pactl set-default-sink` 设默认 + 80%
2. **RPS 相机 = Camera B (Phase A2D 同相机)**: `EyeToHand/config.yaml` `camera_path: usb-0:2` + `camera_name: UGREEN Camera` → `/dev/video6`（**非 4.2**；全局记忆里 "Camera B=4.2" 是旧的）。`rps_stream.py` 属 wangxilu(664) 改不了
3. **相机打开方式改稳定路径解析**: `start_rps_phaseE.sh` 用 EyeToHand `find_camera_id(keywords=['UGREEN Camera'], path_keyword='usb-0:2')` 解析 → `RPS_CAMERA_INDEX=6` 传 rps_stream，不再硬编码 /dev/videoN（编号漂移排坑）。⚠️ find_camera_id 的 V4L2 清单打印走 **stdout** → 捕获须 `| tail -1` 取最后一行
4. **语音不出声再排 (pw-cat --target)**: realtime_audio 用 `pw-cat --playback --target bluez_output... --rate 24000 --channels 1 --channel-map MONO --format s16` 播 TTS → **A2DP sink 上静默失败** (paplay / pw-cat 不带 --target 走 default sink 正常, 实测 30/30 帧相机 + paplay 有声)。**修复**: 复制 config 清空 `realtime.pipewire_target` → `--config /tmp/rps_config_phaseE.json` 传语音进程 → pw-cat 走 **default sink**; 启动时 `pactl set-default-sink bluez_output.78_44_05_FA_12_B9.1`。不改 wangxilu 文件 (config.json/realtime_audio.py 均 664)
5. **单局 demo 走通 (2026-08-21)**: 完整流程 — 臂到 P1 → auto-invite → JBL 播邀请 (`default sink`) → 人说"开始" → 倒计时/视觉判拳 → `[RPS][RESULT] outcome=win` → phaseE 发 DONE → 臂回 Home; 浏览器画面 (8768) + JBL 出声均正常; phaseE 退出后 RPS 服务继续等下局"开始" (Ctrl+C 收尾)
6. **待办**: ①RPS 服务收尾 (demo 后需手动 Ctrl+C 停 RPS 终端) ②多局制 (rounds>1 或等人退出) ③三 CRP 融合 (Phase0 8600 + A2D 8080 + PhaseE 8700, 段间 wait 1s)

### Cup 状态机同步 data_sm: CRP + py (2026-08-21, Phase D 排查前置)

1. **CRP (`doc/estun_handle_cup_sm.crp`, sha256 `41e7e147`)**: ①H_P3 倒水后 wait 1000→**3000ms** ②`movl H_P2`×2 + `movl Home(sudu2)` → **MovJ**(mblendtype→mjblendtype); 旧版 `.bak`
   - **2026-08-21 更新: 用户手动补 `DCp1.dcpos.dc` 计算(示教器导出 `PhaseA2D-Down-442` → doc 版替换, 现 sha256 `fefef5fd`, 57 widgets)** — 原 doc 版缺 DCp1 绕 Z(dc) 分量, 倒水避碰点 H_P4 姿态需它
2. **py (`estun_handle_cup_sm.py`)**: ①**修 skip_frames 双计数** — `_detect` 删 `frame_counter+=1`(run 保留; 消除偶数 skip 永不 detect 隐患, 与 data_sm 对齐) ②**PoseStabilizer 升级增强版** — θ 并入窗口(仅收 pos/euler/θ 齐全帧, θ 极差<5°, 取 θ 中位数→band), get_mean 返回 3 元组 ③**_detect Handle 段** 加 θ 稳定/代际失效作废/`_resolve_rot_band` 判档; Cup 段 get_mean 兼容 3 元组 ④**新增** `_resolve_rot_band`/`_resolve_p1_offsets`/`_interp_p1_offset`/`_clamp_p1_z` ⑤config `handle_teach.p1_offsets`/`p1_z_check` 加载 ⑥**Phase B 抓取策略同步 data_sm** — θ 插值 offset + `C=C0-θ` 解析式 + P1 Z 钳位[0.360,0.362] + B3 无有效 band REJECT(不得默认 0°)
3. **验证**: py_compile + import OK + 全量 **142 测试绿**
4. **Camera A 生命周期 (2026-08-21)**: `_detect_active` 标志 — Phase B 完成 → `False` 停 Handle/Cup 检测(`[CAM-A] Phase B 完成 → 停止...`); socket 断开(finally) → `True` 恢复(`[CAM-A] socket 断开 → 恢复...`); run 主循环条件调用 `_detect`
5. **Phase D 排查打印增强**: HandInCup 结果打印耗时(`[INFO] HandInCup: TRUE (X.Xs)` / 超时); `[WARN] HandInCup 已 18s+ → 逼近 CRP 20s 窗口` 预警(直接验证 25s vs 22s 疑点)
6. **实机日志分析 (2026-08-21)**: 
   - 首测: GRIP 后连接关, READY 未发 → CRP C_P3 后 sendstr 无返回值(socket 层); 用户确认"后面的 socketsendstr 没有返回值"
   - **次测(CRP 更新 fefef5fd 后)**: **READY 到达 ✓**(`[RECV] READY` → 人就位 → 启动 HandInCup)→ 通信链路通了; 但 **HandInCup 18s 无结果**(`[WARN] HandInCup 已 18.0s`), `_hand_on_cup` 一直 None
7. **HandInCup gate bug 修复 (2026-08-21, 检测失败根因)**: `_camera_b_loop` 的 `active = _hand_on_cup is not None` 与 READY 激活(设 None)矛盾 → **`update()` 永不调用** → 检测永不运行 → 18s 超时。修复: 加**独立 `_hic_active` 标志**(READY 激活置 True, 检测到/超时/人未就位置 False); Camera B 线程 `active = self._hic_active`, 检测到 True 清标志。142 测试绿
8. **同事 HANDOVER 方案 (/data/Speech/)**: Phase D 视觉(计分制连续5帧命中)之上加**语音门禁** — 状态机 IDLE→WAITING_HAND→WAITING_CONFIRMATION→CONFIRMED; 视觉命中后仍需语音确认("拿好了/接住了"等)才 release permission; 未确认/超时 → 保持夹持 → 返回 `[位姿]` ABORT 路径。**接入点**: cup_sm READY 后 `HandoverVoiceClient(127.0.0.1:8770)` start, 视觉 True → hand_detected, RELEASE 时 `wait_confirmed` → confirmed 才 `release_object()`(PROJECT_HAND_PHASE_D_PATCH.md 最小接入)。待确认是否接入
9. **cup 检测优化 (2026-08-21, 裁剪+杯壁)**: `_compute_cup_3d` 裁剪宽度 `2×rim_px` → **+杯壁 5mm 像素**(thick_px, config `cup_wall_thickness: 0.005`); 解决 **bbox 靠近杯缘/把手非正对左侧时采样偏窄 → 杯体被裁 → 倒水点偏**。142 测试绿
10. **P3 已调 (2026-08-21, 用户定)**: `p3_teach_offset.left.30.t_offset` X 0.0686→**0.0886**(+2cm), Z 0.2536→**0.2636**(+1cm) — 若裁剪+杯壁仍偏, P3 前移补偿
11. **Phase D 打通 (2026-08-21, 用户确认)**: 完整流程成功 — cup 裁剪+杯壁 + P3 X+2cm/Z+1cm + HandInCup gate fix(`_hic_active`)三处生效。**Camera B 实时显示**: 线程缓存 `_last_frame_b` + 主循环 imshow `HandleCupSM | Camera B`
12. **语音门禁接入 (2026-08-21, 同事方案 /data/Speech, 方案B=cup_sm+独立门禁, 最终定案)**: 
    - cup_sm 接入 `HandoverVoiceClient`(127.0.0.1:8770): READY 时 `_init_handover_voice()`(**服务不可用回退纯视觉**); 视觉 True → `hand_detected()`; RELEASE 时 `wait_confirmed(30s)` → **confirmed 才 release + CONTINUE**, 未确认 → 保持夹持 + ABORT
    - **client 内联进 cup_sm**(`_HandoverClient` 类, 无独立文件/文件夹; Handover/ 已删) — 最小 TCP client(发命令/轮询确认)
    - **8770 服务端 = 绝对路径直接跑同事的 `realtime_voice.py`**(= gate-only, 不开机器人/相机, 与 cup_sm 共存):
      `/data/xilu/conda-envs/VLA/bin/python /data/Speech/realtime_voice.py --config /data/Speech/config.json`
    - **验证通过 (2026-08-21)**: realtime_voice 直接跑 → 8770 监听; client(Proj)协议 `HANDOVER_START`→WAITING_HAND / `STATUS`→confirmed=False / `CANCEL`→IDLE 全对
    - **语音优先 (2026-08-21 用户定)**: RELEASE 判定改为**语音门禁优先, 视觉兜底** — 语音 confirmed → 松手(**即使视觉未命中**, WAITING_HAND 状态也接受确认词); 语音未确认/门禁不可用 → 才看视觉
    - **单次执行模式 + 一键脚本 (2026-08-21)**: cup_sm `_handle_session` finally 设 `_stop_event` — **一轮 CRP socket 断开后自动退出**; `start_cup_handover.sh` 一键启动(门禁弹 gnome-terminal 窗口, **cup_sm 本终端**)
    - **实机测试通过 (2026-08-21, 用户确认)**: 语音门禁流程通(含语音优先: 视觉未命中 + 喊"拿好了" → 松手)
13. **RPS 了解 (2026-08-21, /data/Speech/RPS_SERVICE_HANDOVER.md)**: RPS 剪刀石头布 = `realtime_voice.py --rps-game` + HTTP 8768 + `rps_stream.py`(唯一持摄像头/MediaPipe/手势动作队列); 统一启动 `start_rps_voice.sh`; 邀请→倒计时→视觉判断→结算; 与递杯(8770 门禁)是 realtime_voice 不同模式(--rps-game 切换), 完整服务 run_handover_service 整合递杯; **RPS 接入 Project_hand 待定**
14. **待办**: ①提交语音门禁这批改动 ②若语音确认慢 > CRP 20s 窗口需评估 CRP 超时(confirmed 在视觉命中前可能已达成, WAITING_HAND 即收确认) ③RPS 接入 Project_hand 待定

### Phase 0 实机验证通过 (2026-08-21)

1. **实机 4 步走通**: 示教器导入 `doc/estun_greeting_phase0.crp` (3c3f2bf5) + PC `python Estun/estun_greeting_phase0.py` — Step1 初始位姿→回 ZHAOSHOU1→MovL / Step2 等人进门→回 ZHAOSHOU2→MovL / Step3 挥手→回 WOSHOU→MovL / Step4 握手→回 Home→MovL+SocketClose, 全程正常
2. **socket 时序核对** (CRP SocketReadStr 等待窗口 vs py 端处理耗时, 方向相反各自独立):
   - Step1: CRP 10s 等 py 回复 (py 毫秒级回) / py recv 5s 等 CRP 发初始位姿 (socketcreate 后立即 send) — 冗余足 ✅
   - Step2: CRP 130s vs py 人检测≤120s + YOLO 首次加载~2-3s — 余量仅 ~7s ⚠️ (正常情况人几秒内进门, 无碍)
   - Step3: CRP 30s vs py 挥手 3×0.45s≈3.7s + RESET — 冗余足 ✅
   - Step4: CRP 90s vs py 握手最坏 92.5s (force_timeout 30 + release_timeout 60 + 基线采样 0.5 + HAND_SHAKE/RESET 2.0) — **py 最坏略超 ~2.5s**, 若人迟迟不伸手/不松手, CRP 第 90s 先超时→回 Home+Close, py sendall(Home) 报 BrokenPipe; 实机握手几秒内完成, 未触发
3. **结论 (用户判定)**: 当前不修改, 冗余足够; 备选修法 — 方案A = CRP SocketReadStr 90→110~120s (给足握手, 需示教器重新导入), 方案B = py `force_release_timeout 60→45s` 或 `force_timeout 30→20s` (py 先超时优雅收尾, 仅改 config)
4. **验证**: 此前 py_compile + mock TCP 4 步回复全对 + 142 测试绿

### Phase 0 打招呼独立交付: py + CRP (2026-08-20)

1. **更新 doc CRP (状态机单独保存)**: `doc/state_machine_reference.crp` ← 示教器下载 `~/下载/IDLEtemp-Down-645.crp` (sha256 `a8a8ee00...`, 80块/22点, 含 GH_* 与双 Socket); 旧版备份 `.bak` (`4d92f70d`); `data_sm-Down-946.crp` 与 `doc/estun_handle_data_sm.crp` **sha256 相同** (`4227c66b`) 已是最新未动
2. **独立 Phase 0 CRP**: `scripts_gen/gen_greeting_phase0_crp.py` 抽取 → 后被**示教器导出版 `Phase0-Down-224.crp` 取代** (用户实机改过) → **socket IP 修正** `192.168.101.2 → 192.168.101.96:8600` (旧 IP 通信失败根因) → **GH_* 存储坐标同步为新示教值** → 当前 `doc/estun_greeting_phase0.crp` sha256 **`3c3f2bf5`** (26 块: Socket1:8600 四步, 超时 10/130/30/90s, MovL GH_*/Home, socketclose; 6 点: GH_CUR/Z1/Z2/WS/HOME2/Home)。旧版 `.bak` (`97b2c2e0`)
3. **独立 py**: `Estun/estun_greeting_phase0.py` — 抽取 state_machine GreetingHandler 独立化 (无 SharedContext/协作线依赖)。**相机=迎宾 UGREEN `usb-0:4.1`** (EyeToHand UGreenCamera 按路径枚举, 人进门检测, 非 D435i); `--mock` 模拟灵巧手; `--camera-path/--port` 覆写; 4 步 TCP + YOLO 人检测 + Gesture 挥手/握手 + 压力双阈值
4. **config**: `Estun/estun_config.yaml greeting.camera_path: usb-0:4.1`; **2026-08-20 重新示教三打招呼位姿** (取代同事值, mm/deg): Z1 `[-239.73,339.759,852.73,-89.008,1.6,11.114]` / Z2 `[-204.105,204.746,928.191,-46.92,-3.076,3.518]` / WS `[-206.149,809.121,602.386,-173.773,65.873,-87.126]` — **CRP GH_* 存储坐标已同步同值** (通信失败兜底一致)
5. **排坑 (socket 通信失败)**: CRP socketcreate IP 是 `192.168.101.2` (旧) 而 PC 是 `192.168.101.96` → 机器人连不上, TranStrToCpos 拿不到数据回退 CRP 存储旧坐标 → 表现"Z1 还是旧位姿"。已改 .96
6. **验证**: py_compile + 离线 mock TCP 4 步回复全对 + **142 测试绿**
7. **待办 (实机)**: 示教器导入 `doc/estun_greeting_phase0.crp` 运行 + PC `python Estun/estun_greeting_phase0.py`, 走通 4 步 (等人→挥手→握手)

### config: 稳定窗口 5→10 + 跳帧 5→3 (2026-08-20, 保险定案)

1. **config.yaml**: `skip_frames 5→3` (≈10Hz@30fps) + `stabilization.window 5→10` — 依据 θ 稳定 log 分析: 5 帧窗口稳定后 |偏差|≤0.4°、band 判对率 100%, 但用户为保险加大到 10
2. **影响范围 (共享 config, 4 状态机读)**: data_sm / handle_sm / cup_sm / state_machine 的 handle+cup 稳定窗口全变 10、跳帧全变 3 → **稳定耗时 1/6s×5=0.83s → 1/10s×10=1s** (用户确认)
3. **ROT_THETA_WIN 已对齐 10 (用户定)**: `_rot_sample_buf` maxlen 5→10, round 记录 `theta_samples` 现=10 样本 → **18 回合证据合同 "5 原始样本" → 10**, 采集前需训练侧知晓 (addendum §3 变更)
4. **验证**: config 解析 OK + 全量 142 测试绿
5. **待办**: 实机验证 10 帧窗口稳定行为; 训练侧证据合同 5→10 申报

### θ 稳定性记录工具 (2026-08-20, 新建脚本, 待实机)

1. **新建 `Orchestrator/tests/theta_stability_log.py`** (未改动检测器本体): import 复用 HandleCupDetector, 参考 test_tag_points 风格记录每帧检测结果到 `logs/theta_<ts>.log`
2. **记录字段**: 帧号 / pos_cam / euler_cam / ratio / rd / θ / band / d / handle_side + 累积 θ 统计 (N/mean/std, 'r' 按姿势重置); **逐帧结果只写 log 文件, 不刷终端**
3. **按键**: q 退出 | r 重置累积 (换角度后重新采样) | s 存帧; 退出输出 SUMMARY
4. **参数**: `--skip N` (检测间隔, 默认 1=逐帧; 用 `%N==0` 语义, 避开 `%1==1` 恒 False 坑)
5. **目的**: 实机摆各档位 → 记录**原始 θ 逐帧变化** → 观察波动规律 → 再定稳定窗口/skip 参数 (**滚动窗口逻辑已去除**, 先只看原始数据, 找到规律再改)
6. **脚本整理 (2026-08-20)**: `test_tag_points.py` / `mark_roi_hsv.py` / `theta_stability_log.py` 三个独立工具脚本 (均无运行时 import) 移入 `Orchestrator/tests/`; 三者 `_project_root` 路径深度 2→3 层; handle_cup_detector 两处注释路径同步
7. **验证**: py_compile + import OK + 全量 **142 测试绿** (unittest, pytest 未装)
8. **待办**: 实机逐档运行记录 → 分析 θ 逐帧 std → 再定稳定窗口与 skip 参数

### pos_side v2: 双阈值判档 + 分段线性连续角度 (2026-08-20, 解决中间角度二值化)

1. **v1 问题 (用户反馈)**: 单阈值 rd 判档对 ±15/±30 精确, 但中间角度 (+20~+25) 被二值化归 15/30, 正侧角度不连续 — 实际抓取摆位可能非整档
2. **物理限制确认**: rd(θ) 在 +20° 后每度变化 <0.001 (噪声 0.002-0.003), **+22~+30 物理不可分**; +15 与 +20 实测 rd 重合 (0.3164 vs 0.3162); 实测 rd 比理论低 0.002-0.007 (位置影响)
3. **补 +20 数据 (144654, rd 0.3162)**: v2 json 现 **24 点**; 确认中间角度 rd 与 +15 档上限重合
4. **pos_side v2 设计**: 双阈值 **rd_a=0.314 (15 上限) / rd_b=0.320 (30 下限)**; 角度 = **分段线性** (nodes: 0.2964→0°, 0.3111→15°, 0.3162→20°, 0.3229→30°) 若 rd≤rd30, rd>rd30 用 **30 档仿射** (30+50·(rd−0.3229)); 中间区 (rd_a<rd<rd_b) 连续解算 (θ<22.5→band15, else 30)
5. **效果**: +20 摆位解算 19.96° (连续, 不再二值化); 中间 rd 0.318→22.7°; +30 仿射 30.52 (无外推 51°); 30 节点衔接无跳变; +15 档个别点 (rd 与 +20 重合) 连续漂到 20° (用户接受); 判档 15/16 (仅异常 110950)
6. **实现**: config pos_side 改 rd_a/rd_b/rd30/k/nodes; detector `_interp_pos` 分段线性 + `_solve_theta` 正侧分支 + `_classify_rotation` 双阈值判档 (getattr 防御, 无 pos_side 回退 θ 区间); 135 测试全绿
7. **待办**: 实机只读验证各档 band/θ_med (尤其中间角度); 异常点 110950 重采; 中间角度精确度仍受 rd 噪声限制 (物理天花板)

### v2 模型重拟合 + g 校正 (2026-08-20, g 校正已被 pos_side 替代)

### 青色贴纸搜索区域扩展 10% (2026-08-20, 未实机验证)

1. **区域百分比化**: `_detect_tag_region` 从固定"bbox 右 30%"改为按 handle_side 的百分比区间 — 右侧 **[70%,110%]** (向右扩 10%), 左侧 **[-10%,30%]** (向左扩 10%); 区间 clamp 到图像边界, 空区域 (rx1<=rx0) 返回 None
2. **接入**: detect() 调用传 `self._handle_side` (config `handle_side`, 默认 right); 常量 `HANDLE_ROI_X_RIGHT/LEFT`, 删 `HANDLE_SIDE_RATIO`; `test_tag_points.py` 独立副本同步同区间
3. **交互注意 (B2 后 +45 已工艺排除, 2026-08-20)**: 协作线 B2 (`c97605a`) 已将 +45 排除 — 工艺摆位限定 **θ∈[-35,+35]** (源头上 +45 不出现), 且 `_classify_rotation` `ratio>0.34` → REJECT 兜底 (防 YOLO 框裁虚高, 合法档位 max 0.3177 留 0.02 余量 0 误杀)。扩区域让贴纸更完整 → ratio 回落正常范围, **不触发 0.34 REJECT, 判档更准**; 不再依赖"裁切虚高"区分 +45
4. **验证**: 新测试 6 例 (右/左外延拾取/区域外不拾取/右缘 clamp/空区域/缺省右侧) + 全量 **130 测试全绿** (124 + 6)
5. **待办**: 实机只读跑检测器, 对比扩前后各朝向 ratio/θ/band (尤其 +45 摆位)

### PoseStabilizer 合并 θ 稳定窗口 (2026-08-19, 未实机验证)

1. **合并 (data_sm)**: θ 并入 PoseStabilizer — 窗口内每帧同时收 pos+euler+θ (仅 θ 非 None 帧入窗), 稳定判定同时约束三者 (θ 极差<5°), 直接取同一窗口 θ 中位数 → band; 删独立 `_rot_theta_win` deque + spread 手检查
2. **收益**: 消除"位姿已稳定但 θ 窗口不满"错位 — θ 缺失帧 (=不可达/异常 ratio, e.g. θ<-40) 不再参与稳定统计, 正是 B2 REJECT 方向; 窗口内帧全带 θ (θ非None ⟺ fitEllipse 成功 ⟺ euler 动态)
3. **config**: `stabilization.window` 3→5 (与 ROT_THETA_WIN 统一, B3 恰 5 帧; **cup 稳定窗口随之变 5**, 倒水点确定稍慢可接受); `_init_stabilizer_and_calib` handle 加 theta_thresh=ROT_THETA_SPREAD_MAX
4. **删双计数**: `_detect` 里 `_frame_counter += 1` 删除 (run 主循环保留) — frame_idx 每帧+1 更连续, 消除偶数 skip_frames 永不 detect 隐患 (skip 保持 5 本无 bug; 曾记录 teach_p1 `%1==1` 同款坑)
5. **B3 契约保持**: 恰满窗 (window=5) / θ 分歧 (极差<5°) / 代际失效作废 / Phase B 无候选禁 0° offset / 1.5s 新鲜度 — 全保留; REJECT 原因 "θ 样本不足" 并入"未稳定" (θ 与帧同窗绑定后不再单独出现)
6. **验证**: 新测试 9 例 (θ 一致/中位数/分歧/不满窗/向后兼容无θ/代际失效/滑动窗口/阈值覆盖) + 全量 **101 测试全绿**
7. **待办**: 实机只读跑 data_sm 观察 [Handle STABLE] θ_med/band 输出 + 稳定耗时 (位姿窗口 3→5 帧, ~0.5s→0.83s@skip5/30fps)

### 抓取成功 + 交接文档 (2026-08-19)

1. **实机抓取成功** (分档 t_offset + θ 插值 + C 解析式 + Z 钳位全链路)
2. **已知问题 (待改进)**: 抓取姿态完全依赖 θ 检测, **θ 波动偏大** (ratio 噪声 → θ 抖动 → C 补偿/插值受影响); 候选: θ 窗口 5→8 帧 / ratio EMA 平滑 / 检测器稳定 ratio
3. **交接文档已发训练侧**: `doc/project-hand-rotation-band-route-2026-08-19.md` (路线/模型/档位表/data_sm集成/实机结果/已知问题), sha256 `9a51c96e...` → `nickwang@192.168.112.82:.../inbox/from-real-host/`

### θ 插值实装 + Z 范围 [360,362] (2026-08-19)

1. **θ 档内线性插值 (data_sm)**: 新增 `_interp_p1_offset(th)` — X/Y/Z 在相邻档中心线性混合, θ 超出端点钳到端点档; Phase B 有 θ 时用插值 (替代纯查表), 消除档边界跳变 (X 边界误差 10~18mm→~2mm)
2. **Z 范围调整**: `p1_z_check` [0.3599,0.3629] → **[0.360, 0.362]** (实机标定收紧)
3. **+15 摆位手动验证**: θ_med=16.2°, 插值 off=(+8.5,-89.8,+2.5)mm; P1 插值 (49.7,781.6,362.9) vs 查表 (52.1,781.3,362.9) — 差 2.4mm (θ 贴近档中心); Z 原始 364.1 超限钳到 362.9
4. **注意**: +15 档该摆位 Z 天然 364.1 > 362 上限, 钳位生效; 若实测不可接受, +15 offset.Z 需微调 (-2mm 量级)
5. **待办**: 实机复测各档 P1 (插值 + C 解析式 + Z 钳位)

### data_sm C 解析式 + P1 Z 钳位 (2026-08-19)

1. **实机验证**: 0° 摆位 P1 Z=361.4mm 落在标定范围 [359.9,362.9]; +30 档此前 Z 高 5mm 的 config 修改**已撤回** (band30 p1_offset.Z 还原 -0.0005 — 原判定有误, 0° 实测证明 offset 正确)
2. **C 解析式 (data_sm Phase B)**: `C = C0 - θ_med`, C0 = 0° 档 p1_abc.C = -21.1 (2026-08-19); 档中心与查表一致 (θ=30→C=-51.1), 档边界更平滑 (替代档内 p1_abc.C)
3. **P1 Z 钳位**: config `handle_teach.p1_z_check: [0.3599, 0.3629]` (m); Phase B 算完 P1 后超出范围**钳到最近边界** (非仅警告), P2 随之
4. **待办**: 实机复测各档 P1 落点 → 确认 C 解析式 + Z 钳位生效

### data_sm 接入 band 查表 t_offset (纯查表, 2026-08-19)

1. **estun_handle_data_sm.py 实现 band 查表 P1**: θ 5 帧中位数 → band → `handle_teach.p1_offsets[band]`
   - `ROT_THETA_WIN=5` + `self._rot_theta_win` deque; `_detect` 每帧 push θ, 稳定后 `median(θ)` → `_resolve_rot_band` ([-8,8]=0, (8,22]=15, >22=30, 负侧对称)
   - `_resolve_p1_offsets(band)`: 查 p1_offsets 拿 p1_offset + p1_abc (C 已含 C=C0-θ); 缺档回退 0° → 平铺字段
   - Phase B: `R_base_fix`/A1/B1/C1 用 band_abc, `pos_base_1 = detect_base + band_off`; 旧链 (teach 关闭) 回退固定 ABC
2. **子 agent 角度-偏移关系分析 (2026-08-19)**: X 对称线性 (-1.72mm/°, R²=0.991), Y 强不对称弯曲 (线性 R²=0.82 失败, 二次 R²=0.998), C 严格线性 (C=-21.14-1.002θ, R²=0.99998, 残差<0.2°), Z 小量
3. **后续增强 (当前未实现, 纯查表已够用)**: ①**档内 θ 线性插值** (相邻档 offset 按 θ 混合, 消除档边界跳变 X 10~18mm→~2mm) ②**C 用解析式 C=C0-θ** (查表在档边界有 7~8° C 误差) ③Y 连续二次/正余弦模型 (R²=0.998, 若需连续)
4. **待办**: 实机验证 data_sm band 查表 (各档 P1 落点) → 若边界误差不可接受再上插值

### t_offset 5 档示教完成 (2026-08-19)

1. **teach_P1 --angle 逐档示教完成**: p1_offsets 表 5 档 (-30/-15/0/+15/+30) 全齐
   - 0°: [33.0,-83.4,2.2]mm / C=-21.1 (2026-08-17 默认值)
   - +15°: [10.8,-90.0,2.8] / C=-36.3; +30°: [-17.9,-87.3,-0.5] / C=-51.1
   - -15°: [67.0,-68.3,-0.8] / C=-6.2 (Z 方差 6.2mm 略超验收, C 预测-6.1 实测-6.2 吻合)
   - -30°: [82.9,-47.6,-5.0] / C=+9.0
2. **C=C0-θ 全档验证 (误差 ≤0.2°)**: ΔC vs 0° = [-30:+30.1, -15:+14.9, +15:-15.2, +30:-30.0] — 完美线性
3. **X 偏移单调**: -30:+83 → -15:+67 → 0:+33 → +15:+11 → +30:-18 mm (≈-1.7mm/°, 负侧略弯)
4. **-15 预测验证**: 预测 [X58/Y-65/C-6.1] vs 实测 [X67/Y-68/C-6.2] — C 完美, Y 接近, X 偏 9mm (X-θ 负侧非线性)
5. **注意**: -15 的 abc_offset C 方差 169° 是 ±180 环绕假象 (p1_abc C 干净); -15 Z 方差 6.2mm>5mm (量级小可接受)
6. **待办**: ①data_sm 稳定逻辑 (θ 5帧中位数→band, 用户要求示教完成后做) ②data_sm 查 p1_offsets 表 + band 接入 (task #4)

### teach_P1 支持角度参数 (2026-08-19)

1. **`estun_teach_handle_p1.py` 新增 `--angle`**: 示教时指定档位角度 (deg, 如 `--angle 15`), 打角度标签
2. **输出**: banner/log 头部记录示教角度; yaml 输出块改为按角度建 `handle_teach.p1_offsets.<angle>` 表结构
3. **`--apply`**: 写入 `handle_teach.p1_offsets.<angle>` (按档 t_offset 表, data_sm 用) + 平铺字段同步本档 (旧 data_sm 兼容)
4. **用途**: 逐档示教 t_offset (task #3) — 5 档各跑一次 `teach_P1 --angle <band>`; 完成后 data_sm 按检测 band 查 `p1_offsets` 表

### 5 档简化定案: 纯 θ 区间, 无 ±45 (2026-08-19)

1. **用户定案 (简化)**: 旋转判档只保留 **5 档 -30/-15/0/+15/+30** (无 ±45); +45 物理摆位统一落 +30 (此前已接受); 每档分别示教 t_offset
2. **判档边界 (用户 2026-08-19)**: **[-8,8]=0, (8,22]=15, >22=30**; 负侧对称 [-22,-8)=-15, <-22→-30, <-40 不可达 — **纯 θ 反解区间**, 删掉 per-band 位置模型 / u 门限 / 45 识别
3. **代码**: `_classify_rotation` 重写为纯 θ 区间 (5 分支); `__init__` 只加载 band0_halfwidth=8 + band15_hi=22; 删除 _rot_band/_rot_u45/_rot_u_high
4. **config**: handle_rotation 删 band_coeffs/angle45_gate/u_high_zone; 保留 global_model (28点) + band0_halfwidth + band15_hi
5. **全量验证 (35 点)**: 精确 26 / 相邻 7 / 跨档 2 (pos45/pos45_3 → 15, 低 u≈533 非操作摆位); -30/-15/0/+15 全对, +30 3/5→30 + 2/5→15 (相邻)
6. **实机含义**: 干净 +30 (u≈513, θ~16) → **15 band**; +45 (u>560, ratio 0.34+, θ=29.6) → **30 band**; 5 档对应物理 ≈ -30/-15/0/「15」=物理+15~+30/「30」=物理+45
7. **JSON**: model 段更新为 θ 区间描述 (删过时键); +45 历史点保留备查
8. **待办**: 逐档示教 t_offset (5 band, teach_p1 需支持角度参数) → C=C0-θ + band 查表 t_offset 接入 data_sm P1

### +45 真相: 裁切是高 ratio 的来源, ROI 扩展回退 (2026-08-19)

1. **u>560 门限定案**: config `angle45_gate.u_min` 530→560 (低 ratio 簇 u≤540 与 +15 物理重合, 不再认 45)
2. **高 ratio 簇确认**: pos45_1/2/100600 (u 587~613, ratio 0.34~0.36) 判 45 ✓
3. **真相发现**: 尝试 ROI 上下扩 20% (`TAG_ROI_VPAD=0.2`) 防裁切 → 同一位置 ratio 从 0.3598 掉到 0.2947~0.2995, **长轴恢复** (49.5→59.1) — **0.34~0.36 的"高 ratio"就是贴纸被 YOLO 框裁掉的产物**; 干净 +45 ratio ≈0.295~0.30 落在 +15 区间, 纯 ratio 根本分不开
4. **用户决策 (回退)**: 贴纸被裁是 +45 的**稳定特征**, 回归原 bbox 高度检索 (`TAG_ROI_VPAD` 已删, handle_cup_detector + test_tag_points 同步), 保留裁切效应 → ratio 虚高 0.34~0.36 才能与 +15 (0.30) 区分; 校准数据本就是旧管道采集, 回退后操作与校准一致
5. **扩展代码采集的两个干净点 (pos45_101756 u602/r0.2947, pos45_102144 u599/r0.2995) 不作数** — 管道与操作不一致, 未入档
6. **+45 定案 (u>586 高角度区, 2026-08-19)**: `_classify_rotation` + config 新增 `u_high_zone: 586` — **u>586 时候选只留 {30,45}** (per-band 比 r̂30/r̂45); 依据: 校准中 +45 独有 u≥587, +15/+30/+0 最高 u=586。补入裁切点 pos45_101155 (0.3598)/pos45_102457 (0.3144); pos45_093515 (u540) 标 excluded (+45 域外, 反推 θ<10 判 0)
7. **全量复验 (35 有效点)**: +45 7/7 达标 (4 判 45 / 3 判 30, 无 45→15/0); 其余 30 精确 + 3 相邻 (neg15_094913→0, pos15_3→0 可接受) — **真实跨档 0**
8. **注意 (脆弱点)**: u_high=586 与 0°/30° 最高 u (585.8/586) 相邻 — 实机 0°/30° 摆位必须保持 u≤586, 否则被误路由到 30/45 区
9. **待办**: 实机验证 u_high 门限 (各朝向 band 输出) → 逐档示教 t_offset → C 补偿接入

### -15 补数据 + 28点重拟合 + 相邻档可接受标准 (2026-08-19)

1. **-15° 三点入档** (用户提供 3 个实机 log 提炼, std≈0.001): neg15_094705 (ratio0.2567/u541.6/d0.653) + neg15_094913 (0.2650/u552.6/d0.627) + neg15_095007 (0.2536/u525.6/d0.682); JSON data_points 36 点
2. **重拟合 -30~+30 (28点: -30×5/-15×3/0×11/15×4/30×5)** — `cos+sin+dd`: c0=0.1071, c1=0.1739, c2=0.0988, c3=-0.0768, d_ref=0.63; R²=0.9780 (含 -15 后略降, 因 -15 物理点非纯正弦), LOOCV ratio MAE=0.0046; φ=29.61°, A=0.2000
3. **反解角度偏差 (每档)**: -30 [+0.3,-2.0] / **-15 [+3.4,+6.6] 系统性低估~5°** / 0 [-4.8,+2.1] / +15 [-5.3,+1.6] / **+30 [-11.1,-0.4] 正侧饱和** — 与实机 -15→0、+45→15 现象同源 (ratio 响应在正侧/负15 端被压缩)
4. **三种判档方案对比 (28点)**: 方案A 现状代码 (正半侧per-band+负半侧反解区间) 26/28; 方案B 档中心最近邻 (全局模型在每档角度预测ratio, argmin|实测-预测|) 27/28; 方案C 全档per-band u,d 27/28
5. **用户新标准 (2026-08-19)**: **相邻档误判可接受** (如 -15→0/-30 均可), 只需无跨档 → 三方案全达标 (误判全为 ±1 档); **之前标记的 -15 实机 76%→0、pos15_3→0 均撤销为可接受**
6. **config.yaml 同步**: handle_rotation.global_model 更新为新系数; 注释标注 28点重拟合 + 相邻档可接受标准; JSON model 段已更新 (bands/band_centers/classification=档中心最近邻建议)
7. **唯一硬伤仍是 +45→15 (跨两档)**: 全局模型预测 45° 档 ratio≈0.300 与 +15(0.301)/+30(0.307) 几乎重合, 纯 ratio 不可分 → 需 u/v/d 单独区分 (task #7)
8. **推荐判档方案**: 方案B 最简单且无 per-band 过拟合 (C 的 -15 档 3 点 3 参数是退化拟合), 正负两侧统一; 是否把代码从方案A改B待用户确认
9. **待办**: +45 单独区分 → 逐档示教 t_offset → C 补偿接入

### band 实机验证 + 方向调整: -30~+30 主模型 + +45 单独区分 (2026-08-19)

1. **实机验证四档** (handle_cup_detector 独立跑, 只读相机):
   - **0° ✅**: band=0 全对, ratio≈0.279 (落标定 0° 档中心), th 波动 0~7.8 但稳判 0
   - **+15° ⚠️**: 稳定段 34 帧 32 判 15 (94%), 2 帧 ratio 掉 0.2897/0.2906 → th 9.0/9.5 → 判 0。0° 最大 th=7.8 vs 15° 最小 9.0, 缝隙仅 ~1.2°
   - **-15° ❌**: 42 帧仅 24% 判 -15, 76% 判 0。th 只有 -8.4~-10.6 (期望 -15), **反解系统性低估 ~5°** + **-15° 档 JSON 零数据** (纯反解区间映射)
   - **+45° ❌**: 实机 log (u≈540, d≈0.626) 稳定 ratio≈0.2944。per-band 最近邻**判 15** (r̂45=0.311 过拟合偏高, r̂15=0.294 与实测重合), **无一帧判 45**。违反硬约束 45→15
2. **+45 物理混淆根因**: 45° 遮挡致 fitEllipse 短轴未按理论 cos45° 收缩 (理论 ratio≈0.19, 实测 0.294), 与 +15° 档实测 (0.293) **几乎重合**。标定数据 +15 [0.2947,0.3038] 与 +45 [0.299,0.3556] 本就重叠
3. **实现与设计不符**: 标定阶段 45 门限 = `ratio≥0.2925 且 u≥530`, 但 `_classify_rotation` 实现成 **per-band 最近邻** (u≥530 仅把 45 加入候选, 无 ratio 门限); JSON band_coeffs 无 45/-15, config band_coeffs 有 45 (4 点拟合, 过拟合, b1=0.000764 是其它档 5~60 倍)
4. **用户定方向 (2026-08-19)**: ①补 -15° 数据 ②拟合 **-30~+30 主模型** (不含 +45) ③**+45 与易混淆档用 u/v/d 等参数单独区分**
5. **JSON 已补 +45 实机点** (pos45_093515_实机: u539.9/d0.626/ratio0.2944/std0.0031/n50), 现 33 点; 验证重拟合 45 (5 点 R²=0.85) 后实机点仍判 15 → 证实补数据不能解决, 需换区分方案。config.yaml 未动 (恢复备份)
6. **待办**: 等用户补 -15 → 重拟合 -30~+30 → 设计 +45 单独区分 → 逐档示教 t_offset → C 补偿接入

### T-Offset 方案定案: 每档示教 + 姿态规律 C=C0-θ (2026-08-19)

1. **方案定案 (用户)**: 每档抓取**姿态 ABC 不用示教** — 参考 C 变化规律 (`C=C0-θ`, A/B 恒定) 自动补偿; 但**新的档位需重新示教 t_offset** (位置偏移方向随把手旋转, 固定 p1_offset 不成立)
2. **实机验证结果**: -30 档可行; **+45→+30 可容许** (45 判成 30 是用户明确可接受的, 与离线判档结论一致)
3. **待办**: 逐档示教 t_offset → data_sm P1 = 检测位置 + 该档 t_offset (姿态 C 按 C=C0-θ 补偿)

## 下一步

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| 🔴 | **Phase E 多局制实机复测** | rounds=0 等人退出已实现: 第 1 局→"开始"→第 2 局→"退出游戏"→DONE→Home→RPS 服务自动收尾；单局 demo 已走通 (2026-08-21) |
| 🟡 | 三 CRP 融合 | Phase0(8600)+A2D(8080)+PhaseE(8700) 融合一个, 段间 wait 1s; 串口归属交接 (cup_sm→RPS 服务) 是难点 |
| 🔴 | **Phase D 失败根因待查** | CRP 到 C_P3 后 ~2s 自动停、READY 未发、连接关闭；查 SocketSendStr/SocketReadStr 配置或时序 |
| 🔴 | **data_sm 实机复测各档 P1** | θ 插值 + C 解析式 + Z 钳位链路落地验证；尤其 +15 档 Z 天然 364mm 超限观察钳位 |
| 🔴 | **pos_side v2 实机只读验证** | 中间角度连续解算；异常点 110950 重采 |
| 🟡 | 实机验证 10 帧稳定窗口 | config 5→10 已定案；训练侧申报证据合同 5→10 |
| 🟡 | θ 稳定性记录工具实机逐档运行 | 分析 θ 逐帧 std → 再定稳定窗口/skip |
| 🟡 | Phase 0 握手超时备选 | 若握手场景变慢：方案A=CRP 90→110~120s / 方案B=py force_release_timeout 60→45s（现冗余足够，未触发） |
| 🟡 | CRP 实机导入测试 | estun_handle_sm.crp + estun_handle_cup_sm.crp（遗留） |
| 🟡 | 单相机 SM 实机联调 | estun_handle_sm.py + estun_handle_cup_sm.py（遗留） |
| 🟡 | 豆包实机联调 + MiMo 对比 | 阶段 1/2/3 实机 + 延迟对比（遗留，非主线） |
