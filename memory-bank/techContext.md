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
- **Phase 0 独立服务**: TCP 8600 单 Socket 4 步, socketcreate IP 必须 = 本机当前 IP (`192.168.101.96`, DHCP 动态); 迎宾相机 = 迎宾 UGREEN USB 口 `usb-0:4.1` (非 D435i); 三打招呼位姿在 `estun_config.yaml greeting.` (Z1/Z2/WS, mm/deg) + CRP GH_* 存储坐标兜底 (通信失败时回退)
- **RPS 服务 (/data/Speech, 同事代码)**: `rps_stream.py` (HTTP 127.0.0.1:8768, 视觉+灵巧手动作队列; 相机 **Camera B = usb-0:2 = /dev/video6**, 与 Phase A2D 同相机; rps_stream.py 属 wangxilu 改不了, 由 `start_rps_phaseE.sh` 用 EyeToHand `find_camera_id` 按路径解析后传 `RPS_CAMERA_INDEX`) + `realtime_voice.py --rps-game` (豆包语音+游戏状态机); **灵巧手被 RPS 独占** (STM32 串口冲突 → phaseE 禁 open); 游戏结果只在语音进程 (`[RPS][RESULT]` 日志 + TTS), HTTP 8768 `/round/status` 不含最终胜负; 回车邀请 = asyncio `add_reader(stdin)`, FIFO 写 `\n` 可程序化触发 (但需持写端 `tail -f /dev/null > fifo` 防读端 EOF 停读); `[RESULT]` 打印无 flush → 重定向需 `PYTHONUNBUFFERED=1`
- **蓝牙音响 (JBL GO 78:44:05:FA:12:B9) 排坑 (2026-08-21)**: 无播报根因 = ①**多声音服务器抢 A2DP** (系统有 `wangxilu` 和 `user` 两个 PipeWire/wireplumber, wireplumber 报 "Multiple sound server instances") → 停多余那方的 pipewire/wireplumber/pipewire-pulse (sudo) → 重连 JBL → bluez sink 出现; ②**realtime_audio `pw-cat --playback --target bluez_output... --rate 24000 --channels 1 --channel-map MONO` 在 A2DP sink 上静默失败** (paplay / pw-cat 不带 --target 走 default sink 正常) → 复制 config 清空 `realtime.pipewire_target` + `--config` 传语音进程 → 走 default sink; `pactl set-default-sink bluez_output.78_44_05_FA_12_B9.1`
- **Phase E RPS 机械臂接入 (2026-08-21)**: TCP **8700** (≠8600/8080, 三 CRP 融合预留, 段间 wait 1s); CRP = `doc/estun_rps_phaseE.crp` (用户写, sha `a5054c91`); 位姿协议 **[x,y,z,a,b,c] 米,度** (CRP cpostostr/transtrtocpos lengthunit=m, 非 Phase 0 的 mm); py `Estun/estun_rps_phaseE.py` tail `/tmp/rps_voice_phaseE.log` 等 `[RPS][RESULT]`; Home 为 CRP 存储点直 movj (py 不回位姿); auto_invite FIFO `/tmp/rps_voice_stdin`; **socketreadstr operateval: 成功=0 / 超时=非0, 不能驱动 while 条件** (实机"只循环一次"); **Codroid while 无法比较字符串** → 用独立循环标志法 (whileflag 初始=1, `if operateval==0` 置 0 退出)
