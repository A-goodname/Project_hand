# Tech Context — Project_hand

## 技术栈

| 类别 | 技术 |
| ------ | ------ |
| 语言 | Python 3.11+ |
| 深度学习 | PyTorch 2.x (CUDA 12.6), Ultralytics YOLOv8 / YOLOv8-Pose |
| 计算机视觉 | OpenCV, NumPy, MediaPipe Hands |
| 3D 处理 | pyrealsense2 (仅 Camera A), 深度反投影 |
| 通信 | TCP socket, USB 串口 (PySerial), OpenCV VideoCapture (Camera B UGREEN: Linux V4L2 / Windows DSHOW) |
| 设备探测 | Linux: `/sys/class/video4linux` 名称匹配 + ID_PATH 物理口 (同名相机区分); Windows: pygrabber (DSHOW) |
| 配置 | PyYAML |

## 开发环境 (Linux 主用, 2026-08-16 起唯一工作区)

- **工作区**: `/data/VLA/Project_hand` (唯一; Windows 副本仅历史)
- **OS**: Ubuntu (Linux 7.0.0-29-generic)
- **Conda 环境**: `/home/user/miniconda3/envs/Proj` (Python 3.11.15); 运行时 `conda run -n Proj python ...`
- **GPU**: 可用 (torch 2.13.0+cu126, CUDA 可用)
- **已装依赖**: numpy 2.4.4, opencv-python / opencv-contrib-python 4.13.0.92, PyYAML 6.0.3, pyserial 3.5, pyrealsense2 2.56.5.9235, ultralytics 8.4.48, mediapipe 0.10.35
- **未装 (按需)**: EpicEye SDK, pyorbbecsdk2, pygrabber/comtypes (Windows 专属), 语音控制包 (sherpa-onnx / sounddevice / pypinyin / openai / websocket-client)
- **git 配置**: 身份 `qaq <huangzhiwuming@163.com>`; `core.autocrlf input` + `.gitattributes` (`* text=auto`); **本地提交不 push**
- **灵巧手串口**: `/dev/serial/by-id/usb-STMicroelectronics_STM32_Virtual_ComPort_304137433235-if00` (STM32 VCP, 枚举为 /dev/ttyACM0; **禁 ttyACM 编号**); 460800; 节点 0x01; 当前用户已在 dialout 组直连 (2026-08-16 起全项目默认端口已统一为此路径)
- **机械臂**: 192.168.101.100 ↔ 本机 `192.168.101.96:8080` (本机 IP 为 DHCP 动态, 启动前确认; Phase 0 独立服务用 8600)

## 项目目录结构

- `Orchestrator/`: **主入口** — 4 阶段调度 + 双相机检测 + TCP 服务
- `Estun/`: 单相机入口 (运行/演示/工具); 示教工具 `estun_teach_handle_p1.py`/`estun_teach_cup.py`/`estun_teach_p3.py`; 独立交付 `estun_greeting_phase0.py` (Phase 0 打招呼) / `estun_rps_phaseE.py` (Phase E RPS 机械臂接入, 端口 8700, 不依赖协作线)
- `Orchestrator/tests/`: 独立工具脚本 `test_tag_points.py` (HSV 判档验证) / `mark_roi_hsv.py` (ROI 分析) / `theta_stability_log.py` (θ 稳定性记录)
- `RealSense_R/`: Camera A D435i 旧版 (手动像素映射)
- `RealSense/`: Camera A D435i 新版 (rs.align)
- `EyeToHand/`: Camera B UGREEN 1080P — PersonDetector + HandInCup
- `EpicEye/`: Epic Eye 灰度相机 (非主用)
- `Orbbec/`: Orbbec Gemini 215 微距相机 (非主用)
- `Omnihand/`: 灵巧手控制
- `calib_data/`: 手眼标定工具 + 标定结果
- `KettleTraining/`: 水壶检测模型微调
- `doc/`: 文档 + CRP 文件
- `logs/`: 运行日志
- `start_cup_handover.sh` / `start_rps_phaseE.sh`: 一键启动脚本 (Phase D 递杯 / Phase E RPS)

## 依赖安装

1. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
2. EpicEye SDK (手动打包 whl)
3. `pip install -r requirements.txt`
4. 注意: pyrealsense2 需带构建后缀版本 (`==2.56.5.9235`),`==2.56.5` 匹配不到; mediapipe 会拉 opencv-contrib-python,需与 opencv-python 版本对齐 (当前均 4.13.0.92)

## 关键约束

- RealSense_R 不依赖 `rs.align` (旧相机内参损坏)
- Camera B 为 UGREEN 1080P, 纯 RGB 无深度 — 3D→1D 人体标定已废弃
- 水壶微调模型路径: `model/kettle_yolov8n_finetune.pt`
- 手眼标定结果存储在 YAML 文件, 各模块独立读取
- HandInCup: YOLOv8n (杯检测, class=41) + MediaPipe Hands (手部 21 landmarks, hand_landmarker.task)
- HandInCup 模型路径可配置: `EyeToHand/config.yaml` → `hand_in_cup` 段
- **/dev/videoN 编号不稳定** (插拔/驱动重绑会重排): 严禁硬编码索引, Camera B 必须按名称枚举 (EyeToHand/camera.py 的 sysfs 探测)
- **Handle 检测 = HSV 青色贴纸链** (把手面贴标准矩形青贴纸, H 40-110/S/V≥50, 质心=中心 + 长短轴=6D 姿态); 旋转判档 = ratio→θ 反解→5 档 band (`handle_rotation` config: `band0_halfwidth=8`/`band15_hi=22`, pos_side 双阈值 + 分段线性); 抓取姿态 `C=C0-θ` 不逐档示教
- **Phase 0 独立服务**: TCP 8600 单 Socket 4 步, socketcreate IP 必须 = 本机当前 IP (`192.168.101.96`, DHCP 动态); 迎宾相机 = 迎宾 UGREEN USB 口 `usb-0:4.1` (非 D435i); 三打招呼位姿在 `estun_config.yaml greeting.` (Z1/Z2/WS, mm/deg) + CRP GH_* 存储坐标兜底 (通信失败时回退); **socket 时序** (CRP SocketReadStr vs py recv, 各段独立): Step1 10s/5s 冗余足、Step2 130s vs 人检测120s+YOLO加载 余量仅~7s、Step3 30s 冗余足、**Step4 py 握手最坏 92.5s 略超 CRP 90s (~2.5s 缺口)** — 实机几秒完成冗余足够; 若变慢备选 A=CRP 90→110~120s / B=py force_release_timeout 60→45s
- **RPS 服务 (/data/Speech, 同事代码)**: `rps_stream.py` (HTTP 127.0.0.1:8768, 视觉+灵巧手动作队列; 相机 **Camera B = usb-0:2 = /dev/video6**, 与 Phase A2D 同相机; rps_stream.py 属 wangxilu 改不了, 由 `start_rps_phaseE.sh` 用 EyeToHand `find_camera_id` 按路径解析后传 `RPS_CAMERA_INDEX`) + `realtime_voice.py --rps-game` (豆包语音+游戏状态机); **灵巧手被 RPS 独占** (STM32 串口冲突 → phaseE 禁 open); 游戏结果只在语音进程 (`[RPS][RESULT]` 日志 + TTS), HTTP 8768 `/round/status` 不含最终胜负; 回车邀请 = asyncio `add_reader(stdin)`, FIFO 写 `\n` 可程序化触发 (但需持写端 `tail -f /dev/null > fifo` 防读端 EOF 停读); `[RESULT]` 打印无 flush → 重定向需 `PYTHONUNBUFFERED=1`
- **蓝牙音响 (JBL GO 78:44:05:FA:12:B9) 排坑 (2026-08-21)**: 无播报根因 = ①**多声音服务器抢 A2DP** (系统有 `wangxilu` 和 `user` 两个 PipeWire/wireplumber, wireplumber 报 "Multiple sound server instances") → 停多余那方的 pipewire/wireplumber/pipewire-pulse (sudo) → 重连 JBL → bluez sink 出现; ②**realtime_audio `pw-cat --playback --target bluez_output... --rate 24000 --channels 1 --channel-map MONO` 在 A2DP sink 上静默失败** (paplay / pw-cat 不带 --target 走 default sink 正常) → 复制 config 清空 `realtime.pipewire_target` + `--config` 传语音进程 → 走 default sink; `pactl set-default-sink bluez_output.78_44_05_FA_12_B9.1`
- **2026-08-22 再来 (仍无声)**: ①**蓝牙音箱本身关机** — JBL sink 会整个消失 (挂在 wangxilu 那套 PipeWire 上, 一停那套就掉) → 先确认音箱开机 + `bluetoothctl connect 78:44:05:FA:12:B9`; ②杀 wangxilu (sudo pkill -u wangxilu -x pipewire/wireplumber/pipewire-pulse) 只清双实例, 停后 JBL 需重连回 user 实例; ③**`start_phase_all.sh` 新增 `check_audio` 前置检查 (不 sudo)** — 判据 2026-08-22 放宽(实测): **只要当前 user 实例默认 sink 是 JBL (`bluez_output.78_44_05_FA_12_B9.1`) 即 PASS 继续**; 多套声音服务器并存(如对方用 wangxilu 会话, 会抢同一 BT A2DP)只**警告不阻断**(实测 JBL 在 user 实例+默认 sink 时 wangxilu 并存仍能出声); **唯一硬失败 = user 实例找不到该 JBL sink**(音箱关机/未连/被 wangxilu 抢走) → 打印修复法 + `exit 3` (除非 `AUDIO_CHECK=0` 跳过). 只用 ps/pactl 检测, 不代杀别的用户进程. JBL 重连成功后 `api.bluez5.profile="a2dp-sink"` + 默认 sink = JBL, paplay 测试音可出声 (2026-08-22 实测验证); **greeting_announcer (Phase0 语音欢迎) 同样踩 pw-cat `--target` 坑** — `estun_greeting_phase0.py _announce_voice` 原先调 `start_greeting.sh` 用默认 config(带 pipewire_target=JBL) → `pw-cat --target` A2DP 上静默无声; **修复 = 复制 config 清空 `pipewire_target` → `--config /tmp/greet_config.json` 走 default sink(=JBL) 出声** (2026-08-22 实测确认); 文案仍是 config.json greeting.message 不自定义; **PhaseD 门禁(8770) 同坑** — `estun_handle_cup_sm.py _spawn_handover_gate` + `start_cup_handover.sh` 预启都改用 `--config /tmp/gate_config.json` (复制 config.json 清空 pipewire_target, 独立路径供 GATE_PATTERN 唯一匹配不误伤 RPS voice); **RPS voice(--rps-game) 本就 OK** (用 `/tmp/rps_config_phaseE.json` 去 target)
- **Phase E RPS 机械臂接入 (2026-08-21)**: TCP **8700** (≠8600/8080, 三 CRP 融合预留, 段间 wait 1s); CRP = `doc/estun_rps_phaseE.crp` (用户写, sha `a5054c91`); 位姿协议 **[x,y,z,a,b,c] 米,度** (CRP cpostostr/transtrtocpos lengthunit=m, 非 Phase 0 的 mm); py `Estun/estun_rps_phaseE.py` tail `/tmp/rps_voice_phaseE.log` 等 `[RPS][RESULT]`; Home 为 CRP 存储点直 movj (py 不回位姿); auto_invite FIFO `/tmp/rps_voice_stdin`; **socketreadstr operateval: 成功=0 / 超时=非0, 不能驱动 while 条件** (实机"只循环一次"); **Codroid while 无法比较字符串** → 用独立循环标志法 (whileflag 初始=1, `if operateval==0` 置 0 退出)
- **Phase E 多局制 (2026-08-21, de143d1)**: config `rps.rounds` — **0=等人退出** (默认, 玩家每局说"开始"续局, 说"退出游戏/不玩了"结束) / N≥1=固定 N 局; **退出信号 = 语音日志裸行 `"退出"`** (rps_voice_game.py handle_user_text quit 分支 `print("退出")`, 仅 honor 在结果播报后 replay-decision 窗口, 第 1 局结果前说"退出"不生效); py 精确匹配 `line.strip()==quit_keyword` (防误伤含"退出"字样其它行); 臂全程保持 P1, 收到 RESULT N 次或退出行或超时 → 发 DONE → Home; `start_rps_phaseE.sh` 新增 `shutdown_rps()` 优雅收尾 (phaseE 退出后 SIGTERM `rps_stream.py`+`realtime_voice.py --rps-game`, atexit 释放相机/MediaPipe/串口; `--rps-game` 限定不误伤递杯 8770 的 `--config` 进程)
- **8770 语音门禁两种启动路径 (共用同一 py `_init_handover_voice`) — "服务已在跑则只连, 不在则自启兜底"; 门禁 kill 统一 SIGINT 优雅关停 (方案A, 2026-08-22)**: ①**独立 Phase D (`start_cup_handover.sh`)** = **预启** — 单独 gnome-terminal 窗口跑 `realtime_voice.py --config` (窗口无 exec bash, 门禁退出→自动关), cup_sm 直接连接; `trap cleanup_gate EXIT` 在 cup_sm 退出后 kill 8770 门禁。②**融合 (`start_phase_all.sh`)** = cup_sm **自启** — 不预启, cup_sm 在 READY (Phase D) `_spawn_handover_gate()` Popen `realtime_voice.py --config` (stdout DEVNULL), sh `wait_gate_gone` 在起 RPS 前 kill 门禁。**⚠️ 门禁 kill 必须用 SIGINT (方案A)**: `realtime_voice` 只捕 KeyboardInterrupt(SIGINT), **SIGTERM 跳过 `finally:audio.close()`** → 其 pw-cat speaker 子进程孤儿化 + 音频图不释放 → RPS voice 接手同一麦克风(UGREEN usb 内置 mic)时半双工没拦 JBL 回灌的 TTS 回声 → 当人声打断 ("只吐两字"); SIGINT→KeyboardInterrupt→audio.close 干净释放麦克风/pw-cat; SIGTERM/SIGKILL 兜底
- **三 CRP 融合 (2026-08-21, 待实机)**: `doc/estun_fused_phase_all.crp` (sha `2a2ab984`) = **Socket1:8600 → wait 3s → Socket0:8080 → wait 3s → Socket2:8700**; **Home 去重** (P0/A2D 共享点 ID `ptmquqgw91h2026u`, E 的 Home `ptmt2eglfiqt8c3e` 重指后丢弃); 变量全保留 (projectcfg.vars.data 三文件逐字节相同; data.vars.project = A2D 15 + E 5 并集); 位姿单位各段保留 (段1 mm / 段2,3 m); **A2D Phase D socketreadstr mtesttime 20s→60s** (py 已默认 ABORT: HandInCup 25s + 语音门禁 30s → 最坏 ~55s 判决; 60s 覆盖, 不动 CRP 框架); 融合脚本 `scripts_gen/fuse_three_crp.py` 可复现
- **A2D py `--defer-devices`** (融合专用): `start()` 只 `_load_configs`+`_init_stabilizer_and_calib`(标定/D0 门禁不占设备)+`_init_tcp`; **不开灵巧手/D435i/Camera B**; `_handle_session` 入口调 `_ensure_devices()` 打开 (幂等); `_get_frames` 在 detector None 时返回 None → 主循环空转 (run 无需改); 位姿稳定靠 PoseStabilizer
- **phaseE py auto-invite 重试**: `_auto_invite_trigger(quiet=)` + `_auto_invite_with_retry()` — ENXIO(FIFO 无读端) 每 3s 重试 ≤60s; 融合场景 RPS 服务在 A2D 退出后才起 (~10s), phaseE 已提前监听 8700 → 首邀常 ENXIO, 重试兜底
- **融合 sh `start_phase_all.sh`**: 预启动 Phase0(8600)/A2D(--defer-devices,8080)/phaseE(8700); **8770 门禁不预启** — 由 A2D py 在 READY (Phase D) 时自启 (异步, 服务不在则 Popen realtime_voice --config 并轮询等就绪); 设备/麦克风严格串行; 监控**进程退出**做交接 (phaseE/Phase0 单连接模式 accept 即关监听, 不能用端口释放判完成) — 8080 释放→等 A2D 进程退出→wait_gate_gone SIGINT 优雅关停 8770 门禁→`GATE_AUDIO_SETTLE`(默认 2s) 音频沉降→起 RPS 服务(8768), 8700 释放(phaseE 进程退出)→停 RPS; `doc/*.crp` 在 .gitignore (CRP 不入 git)
- **⚠️ 融合主线已改单进程 (2026-08-23, 同事多次实机验证通过)**: `/data/xilu/single/single_all.py`(单 Python 进程, `bash start_single.sh`) **取代上述多进程 start_phase_all.sh + estun_handle_cup_sm handover/触发文件/假失能** (多进程保留作参考). 单进程: 灵巧手 SingleHandHub 唯一持有(open→enable→is_enabled确认 30×0.5s, **全程保持使能阶段间只做手势无 disable/close**, ensure_enabled 边界校验, 进程退出才 RESET→disable→close); `activate_rps()` 进程内切换(无 /tmp 触发文件); 门禁 `HandoverConfirmationGate` 进程内对象(无 8770 TCP); 语音 VoiceEngine quiet→gate→game; 完成/邀请信号进程内(无日志tail/FIFO). CRP 完全不动(estun_fused_phase_all.crp 依次连 8600/8080/8700). 备份在 `reference/single/`.
- **两 conda 环境**: Proj 与 VLA torch **相同**(2.13.0+cu126)/python 3.11.15 相同/mediapipe 0.10.35/ultralytics 8.4.48; 差异= **`sherpa-onnx==1.13.4`(requirements.txt line30, ASR onnx 推理、非 funasr) 只在 VLA**, Proj 未装但可装(可 `pip install sherpa-onnx==1.13.4`). single 用 `/data/xilu/conda-envs/VLA/bin/python` 即因含 ASR.
