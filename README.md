# 机械臂+灵巧手抓取系统

双相机 (眼在手上 + 眼在手外) 全流程自动抓取-倒水-递杯. Camera A 检测把手+水杯 3D 位姿, Camera B 检测人就位+HandInCup 判定, TCP 发送给 ESTUN 机器人控制 OmniHand 灵巧手.

## 硬件需求

| 设备 | 型号 | 连接 | 用途 |
|------|------|------|------|
| Camera A | RealSense D435i (serial `342222071364`) | USB 3.0 | 眼在手上: Handle+Cup 3D 检测 |
| Camera B | UGREEN 1080P | USB | 眼在手外: 人就位 + HandInCup |
| 灵巧手 | OmniHand | USB 串口 (当前 Windows 配置 COM4) | 抓取/手势/压力检测 |
| 机械臂 | ESTUN | TCP (LAN) | 执行抓取动作 |
| GPU | NVIDIA (CUDA 12.6) | — | YOLO 推理加速 |

## 运行前准备

首次运行前按顺序完成:

### 1. 环境安装

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

EpicEye SDK 需要手动打包 whl 安装 (SDK 位于机器本地目录, 非本仓库内容).

根目录的 `requirements.txt` 用于完整工程。Phase A–C Windows 独立包使用 `packaging/phase_a2c_windows/requirements.txt`，其中固定了通过 pip 安装的 RealSense 版本。倒水试采的范围和数据要求见 [Phase A–C 运行说明](doc/phase_a2c_handle_windows.md#第一轮数据采集)。

### 2. 确认硬件连接

- [ ] Camera A USB 连接, serial 与 `RealSense_R/D435i_info.yaml` 一致
- [ ] Camera B USB 连接 (默认按名称自动搜索, 无需手动指定索引)
- [ ] OmniHand 串口连接, 确认 `Estun/estun_config.yaml` → `hand.port`
- [ ] 机械臂与控制 PC 在同一局域网

### 3. 配置 IP 与端口

编辑 `Estun/estun_config.yaml`:

| 参数 | 说明 |
|------|------|
| `estun.tcp_server.host` | **本机 IP** (机械臂作为客户端连接) |
| `estun.tcp_server.port` | 8080 (Phase A-D 抓取) |
| `estun.greeting.port` | 8600 (Phase 0 打招呼) |
| `estun.hand.port` | OmniHand 串口号 (当前为 COM4) |

编辑 `EyeToHand/config.yaml`:

| 参数 | 说明 |
|------|------|
| `ugreen.camera_id` | `-1` (默认) = 按 `camera_name` 自动搜索; 指定 `0/1/2...` = 直接用设备索引 |
| `ugreen.camera_name` | 自动搜索时的设备名称 (默认 `UGREEN Camera 1080P`) |

### 4. 标定 Camera B 杯 ROI

UGREEN Camera 无深度, YOLO 检测不到杯时回退到固定 ROI:

```bash
python EyeToHand/hand_in_cup.py --calibrate-roi
```

1. 确保 Camera B 对准杯放置区域
2. 鼠标拖拽框选杯区域
3. 按 `s` 保存 → 写入 `EyeToHand/config.yaml` → `cup_roi_fallback`

### 5. 手眼标定确认

- [ ] `calib_data/calibration_result.yaml` 存在
- [ ] `T_cam_to_gripper` 矩阵非单位阵 (查看 `handeye.use_method`)
- [ ] 如更换相机/重新安装 → 运行 `calib_data/collect_data.py` + `calib_data/calibrate_handeye.py`

### 6. 抓取偏移微调

首次部署或机械结构变更时必须重调:

| 偏移 | 配置文件 | 关键参数 |
|------|----------|----------|
| Handle 抓取 | `Estun/estun_config.yaml` | `handle_grasp.gripper_offset.t_handle_offset` |
| Cup 抓取 (back) | `Estun/estun_config.yaml` | `cup_grasp.gripper_offset.t_cup_offset` |
| Cup 抓取 (left) | `Estun/estun_config.yaml` | `cup_grasp.gripper_offset.t_cup_offset_left` |

微调方法: 运行 `Estun/` 下对应 demo (如 `Estun/estun_handle_cup_demo.py`), 启用 `fine_tune_mode`, 手动微调后自动回算新偏移.

### 7. 示教位姿 + 模型确认

- [ ] P3 倒水偏移: `Estun/estun_config.yaml` → `handle_grasp.p3_teach_offset` (按角度+模式索引)
- [ ] Cup P3 递送: `EyeToHand/config.yaml` → `delivery.*_TEACH`
- [ ] 打招呼位姿: `Estun/estun_config.yaml` → `greeting.zhaoshou*_pose / woshou_pose`
- [ ] 模型文件: `model/kettle_yolov8n_finetune.pt`, `model/hand_landmarker.task`
- [ ] YOLO 基础模型首次运行自动下载

## 快速开始

### 主入口: 双相机全流程

```bash
python Orchestrator/main.py
```

4 阶段顺序执行: `greeting → handle_detect → seated_check → hand_in_cup`.

1. 先启动 Python 端, 再运行机器人 CRP (示教器工程文件在工作区 `doc/` 下, 不入版本库)
2. 按 `q` 退出

### 单相机状态机 (2026-07-31 重构)

从双相机 Orchestrator 拆分, 检测器 = `HandleCupDetector` (YOLO+HSV 轻量), 位姿 = 固定 ABC (`Orchestrator/config.yaml`), 仅用 Camera A.

| 脚本 | 场景 | P3 语义 | Phase C 指令 |
|------|------|---------|--------------|
| `Orchestrator/estun_handle_sm.py` | 仅水壶抓取 | Z-up 抬升 (`handle_lift_z`) | PREGRIP/GRIPDONE/RESET |
| `Orchestrator/estun_handle_cup_sm.py` | 水壶+水杯 | 倒水点 (杯位+示教偏移) | + CUP/RELEASE/GRIP |
| `Orchestrator/estun_handle_data_sm.py` | VLA 数据采集 | 倒水点 (同 cup_sm) | 同 handle_sm |

```bash
python Orchestrator/estun_handle_sm.py
python Orchestrator/estun_handle_cup_sm.py
python Orchestrator/estun_handle_data_sm.py
```

**H_P4 回退避碰** (handle_cup_sm / handle_data_sm): 倒水段 P1→P3 经 CRP 内 CposOffset 动态计算的 H_P4 过渡 (Python 只发 3 位姿, P4 由 CRP 计算):
`P1(抓) → H_P4(V100) → P3(倒水,sudu) → H_P4(sudu) → P1`, 回退量按倒水角 B 动态: <25°→150mm / <17.5°→175mm / <12.5°→200mm.

每个脚本对应一个示教器 CRP 工程文件 (工作区 `doc/`, 不入版本库): 仅水壶 / 水壶+水杯 / 数据采集 各一个.

### 语音控制

**LLM 语义理解** (推荐):
```bash
set MIMO_API_KEY=sk-xxx
python Omnihand/voice_llm_control.py                       # Auto VAD, MiMo + MiMo TTS
python Omnihand/voice_llm_control.py --mode ptt            # 按键模式
```

**关键词匹配** (离线):
```bash
python Omnihand/voice_omnihand_control.py                  # Silero VAD, 全量模型
python Omnihand/voice_omnihand_control.py --mode ptt       # 按键模式
```

触发词配置: `Omnihand/voice_commands.json` | LLM配置: `Omnihand/voice_llm_config.json`

### 单独测试

```bash
# Camera A Handle+Cup 检测
python Orchestrator/handle_cup_detector.py

# Camera B 人就位检测
python EyeToHand/person_detector.py

# Camera B HandInCup 检测
python EyeToHand/hand_in_cup.py
```

## 配置速查

| 你想改什么 | 去哪改 |
|-----------|--------|
| 本机 IP / 串口号 | `Estun/estun_config.yaml` |
| Handle/Cup 抓取偏移 | `Estun/estun_config.yaml` → `handle_grasp` / `cup_grasp` |
| P3 倒水角度 | `Estun/estun_config.yaml` → `p3_teach_offset.pour_angle` |
| Camera B 自动搜索 | `EyeToHand/config.yaml` → `ugreen.camera_id: -1` + `camera_name` |
| Cup P3 递送位姿 | `EyeToHand/config.yaml` → `delivery.*_TEACH` |
| 杯 ROI 回退 | `python EyeToHand/hand_in_cup.py --calibrate-roi` |
| 检测模型/固定ABC/跳帧 | `Orchestrator/config.yaml` |
| 白色把手/红色胶带 HSV | `Orchestrator/config.yaml` → `white_handle` / `red_handle` |
| 人就位/HandInCup 阈值 | `EyeToHand/config.yaml` → `chair` / `hand_in_cup` |
| 语音控制触发词 | `Omnihand/voice_commands.json` |
| 语音控制设计 | `doc/voice_control_design.md` |

## 关键概念

- **4 阶段调度**: greeting → handle_detect → seated_check → hand_in_cup, 任何时刻只有一个检测器运行
- **pack_three**: `<P1>[P2]{P3}` 格式发送三组抓取位姿, CRP `TranStrToCpos` 按 bracket 标记解析
- **固定 ABC**: Handle+Cup 姿态角从 `Orchestrator/config.yaml` 直接读取基系固定值, 不通过 PCA 计算
- **PoseStabilizer**: 缓存 N 帧中值稳定, 代际计数器 `_gf` 防断连后旧帧污染
- **HandInCup**: YOLOv8n(杯) + MediaPipe Hands(21 landmarks), landmarks-in-bbox 计分 + curled gate 判定
- **CONTINUE/ABORT**: Python 发 `<x,y,z,A,B,C>` (CONTINUE) 或 `[x,y,z,A,B,C]` (ABORT), CRP 两个 `TranStrToCpos` (cmd=`<` / cmd=`[`) 解析, If `strflag_cont/abort == 0` 路由分支
- **RELEASE 双语义**: HandInCup 未启动 → 松把手; 已启动 → 判定触发

## 目录结构

| 路径 | 说明 |
|------|------|
| `Orchestrator/` | **主入口**: 4 阶段调度 + 双相机检测 + TCP 服务 + 单相机状态机 ×3 |
| `Estun/` | 单相机入口 (RealSense/EpicEye), 坐标变换, 灵巧手控制 |
| `RealSense_R/` | Camera A 检测模块 (D435i 旧版, 手动像素映射) |
| `RealSense/` | Camera A 检测模块 (D435i 新版, rs.align) |
| `EyeToHand/` | Camera B: PersonDetector + HandInCup + UGREEN Camera |
| `Omnihand/` | 灵巧手 USB 串口控制 |
| `calib_data/` | 手眼标定数据 + 结果 |
| `KettleTraining/` | 水壶 YOLO 微调 |
| `EpicEye/` `Orbbec/` | 灰度/微距相机模块 (非主用) |
| `doc/` | 协议文档 + CRP 文件 + 流程文档 |
| `memory-bank/` | 跨会话项目上下文 |
| `model/` | YOLO + MediaPipe 模型文件 |

## 文档索引

| 文档 | 内容 |
|------|------|
| `doc/estun_handle_cup_code_orchestrator.txt` | CRP 伪代码 + 版本变更记录 |
| `doc/Greeting_Flow.md` | Phase 0 + Phase A-D 协议时间线 |
| `doc/phase_a2c_handle_windows.md` | Windows 水壶抓取、倒水和回 Home 的独立运行与打包 |
| `doc/Handle_Cup_Process.md` | Handle+Cup 位姿计算 + 偏移参数 |
| `doc/Handle_Process.md` | 把手检测设计决策记录 |
| `doc/Grip_Process.md` | 瓶身位姿转换算法 |
| `doc/Module_Rules.md` | 各硬件模块开发规则 |
| `doc/voice_control_design.md` | 语音控制设计方案 + 先行验证结果 |
| `doc/Project_Workflow.md` | 完整架构 + 双相机 + 标定流程 |
| `doc/Kettle_Training_Task.md` | 水壶模型微调 SOP |

> ⚠️ 示教器 CRP 工程文件 (`.crp`) 按 `.gitignore` 约定**不入版本库**, 本仓库内不存在, 仅在本地工作区 `doc/` 下.
