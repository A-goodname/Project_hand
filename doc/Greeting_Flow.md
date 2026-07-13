# 打招呼 + Handle+Cup 全流程梳理

## 整体架构

```
┌─────────────────────────────────────────────────────────┐
│ PC (Python)                                             │
│ estun_greeting_handle_cup_run.py                        │
│                                                         │
│ Camera A ──→ Handle+Cup 6D检测 ──→ 基系位姿 ──→ TCP   │
│ (眼在手上)                                              │
│                                                         │
│ 灵巧手 ←── USB串口 ←── Gesture/Pressure                │
└────────────────────┬────────────────────────────────────┘
                     │
          ┌──────────┴──────────┐
          │ TCP 8600 (Phase 0)  │  打招呼: 4步单连接
          │ TCP 8080 (Phase A-C)│  抓取:  持久连接+指令循环
          └──────────┬──────────┘
                     │
┌────────────────────┴────────────────────────────────────┐
│ ESTUN 机器人 (CRP: greeting_handlepluscup.crp)          │
│                                                         │
│ Phase 0: 30 widgets, port 8600, 单SocketCreate/Close   │
│ Phase A-C: 53 widgets, port 8080, 单SocketCreate/Close  │
└─────────────────────────────────────────────────────────┘
```

---

## 启动流程 (PC)

```
main()
  │
  ├─ [Init 1] load_cam2ee()            ← 手眼标定矩阵
  ├─ [Init 2] init_hand("COM5")        ← 灵巧手: RESET → PAPER
  │
  ├─ ═══ Phase 0 ═══
  │  Phase0Server.serve(port=8600)
  │  │
  │  ├─ _open_camera()               ← Phase 0 独立 RealSense 彩色流
  │  ├─ 非阻塞 accept() 循环 (带预览, 'q' 跳过)
  │  └─ 连接后 → _handle_session(conn) → 4步协议 (阻塞)
  │     ├─ Step 1: recv → reply ZHAOSHOU1
  │     ├─ Step 2: recv → YOLO人检测 → _close_camera() → reply ZHAOSHOU2
  │     ├─ Step 3: recv → Hello_1↔Hello_2×3 → RESET → reply WOSHOU
  │     └─ Step 4: recv → RESET→压力→HAND_SHAKE→松手→RESET → reply "Home"
  │
  ├─ [Init 3] RealSenseHandle6DDetector()  ← Camera A pipeline 启动 (Phase A-C)
  ├─ [Init 4] RealSenseCup6DDetector(start_pipeline=False) ← 复用 Camera A
  │
  ├─ ═══ Phase A-C ═══
  │  TCP Server(8080) 非阻塞 accept
  │  │
  │  └─ 主循环 (每帧):
  │     ├─ get_frames() → Handle+Cup 检测 → PoseStabilizer
  │     ├─ 可视化 (Handle 6D + Cup 6D overlay)
  │     ├─ 非阻塞 accept() → spawn handle_client 线程
  │     └─ 'q' 退出
  │
  └─ cleanup: pipeline.stop(), disable_hand()
```

---

## Phase 0 协议详解 (port 8600)

### 消息格式

| 方向 | 消息 | 含义 |
|------|------|------|
| CRP→PC | `x,y,z,a,b,c` | Step 1: 机器人当前末端位姿 (m, deg) |
| CRP→PC | `COMMAND\|x,y,z,a,b,c` | Step 2-4: 命令 + 当前末端位姿 |
| PC→CRP | `x,y,z,A,B,C` | 回复目标位姿 (m, deg), 用于 ZHAOSHOU1/2/WOSHOU |
| PC→CRP | `Home` | Step 4 完成信号, 机器人自行 MovL Home |

### 时间线

```
     CRP (Robot)                                     PC (Python)
     ════════                                       ════════
                                                    _open_camera()  ← Phase 0 独立 Camera pipeline

Step 1 — 初始位姿 → ZHAOSHOU1
─────
[SetTool(0)]                                       accept() 阻塞等待
[GetCurCPos → pt_temp]
[CPosToStr("") → "x,y,z,a,b,c"]
[SocketCreate(192.168.101.2:8600)]
[SocketSendStr] ───────── "x,y,z,a,b,c" ─────────→ 解析: 无"|" → Step 1
[SocketReadStr(10s)] ←──── "x,y,z,A,B,C" ──────── 回复 ZHAOSHOU1 位姿
[TranStrToCpos → pt_ZHAOSHOU1]                    (来自 estun_config.yaml:
[SetTool(0)]                                        zhaoshou1_pose mm→m)
[MovL(pt_ZHAOSHOU1)]        机器人移动到招手1位置


Step 2 — 人检测 → ZHAOSHOU2
─────
[GetCurCPos → pt_temp]
[CPosToStr("ZHAOSHOU1|") → "ZHAOSHOU1|x,y,z,a,b,c"]
[SocketSendStr] ─────── "ZHAOSHOU1|x,y,z..." ────→ 解析: cmd="ZHAOSHOU1"
[SocketReadStr(130s)]                             │
       │                                          ├─ YOLOv8n 人检测 (Camera A)
       │                                          │  class=0, conf≥0.45, area≥2%
       │                                          │  连续 3 帧稳定
       │                                          ├─ _close_camera()  ← 人检测完成, 关闭 pipeline
       │                                          │
       ←─────────── "x,y,z,A,B,C" ──────────────── 回复 ZHAOSHOU2 位姿
[TranStrToCpos → pt_ZHAOSHOU2]
[SetTool(0)]
[MovL(pt_ZHAOSHOU2)]        机器人移动到招手2位置


Step 3 — 挥手 → WOSHOU
─────
[GetCurCPos → pt_temp]
[CPosToStr("ZHAOSHOU2|") → "ZHAOSHOU2|x,y,z,a,b,c"]
[SocketSendStr] ─────── "ZHAOSHOU2|x,y,z..." ────→ 解析: cmd="ZHAOSHOU2"
[SocketReadStr(30s)]                              │
       │                                          ├─ Hello_1 (do_gesture, delay=wave_interval)
       │                                          ├─ Hello_2 (do_gesture, delay=wave_interval)
       │                                          │  ×3 循环
       │                                          ├─ RESET (复位)
       │                                          │
       ←─────────── "x,y,z,A,B,C" ──────────────── 回复 WOSHOU 位姿
[TranStrToCpos → pt_WOSHOU]
[SetTool(0)]
[MovL(pt_WOSHOU)]           机器人移动到握手位置


Step 4 — 握手 → Home
─────
[GetCurCPos → pt_temp]
[CPosToStr("WOSHOU|") → "WOSHOU|x,y,z,a,b,c"]
[SocketSendStr] ─────── "WOSHOU|x,y,z..." ───────→ 解析: cmd="WOSHOU"
[SocketReadStr(90s)]                              │
       │                                          ├─ RESET (复位, 等待握手)
       │                                          ├─ 等待手背压力
       │                                          │  (参数从 estun_config.yaml 读取:
       │                                          │   baseline_samples, abs_threshold,
       │                                          │   delta_threshold, poll_interval, timeout)
       │                                          ├─ HAND_SHAKE (do_gesture, 1s)
       │                                          ├─ 等待松手 (release_delta 持续 release_hold 秒)
       │                                          ├─ RESET (复位)
       │                                          │
       ←─────────────── "Home" ─────────────────── 回复 "Home" (字符串)

[MovL(pt_Home)]             机器人移动到 Home (预定义点位)
[SocketClose]               关闭 8600 连接
[Wait(1000ms)]

═══ Phase 0 结束 ═══
```

---

## Phase A-C 协议详解 (port 8080)

### 数据格式

| 方向 | 格式 | 含义 |
|------|------|------|
| Robot→PC | `x,y,z,A,B,C` | Phase A: 机器人初始末端位姿 |
| PC→Robot | `<P1>[P2]{P3}` | Phase B: Handle 3个位姿 (抓取点/预抓取/倒水) |
| Robot→PC | `PREGRIP` / `GRIPDONE` / `RESET` | Phase C: 把手阶段指令 |
| Robot→PC | `CUP` | Phase C: 请求水杯位姿 |
| PC→Robot | `<P1>[P2]{P3}` | Phase C: Cup 3个位姿 (抓取点/预抓取/递送) |
| Robot→PC | `RELEASE` / `GRIP` | Phase C: 水杯阶段指令 |

### 时间线

```
  CRP (Robot)                         PC handle_client 线程
  ════════                           ════════

Phase A — 同步初始位姿
──────
[SocketCreate(192.168.101.2:8080)]
[GetCurCPos → pt_init]
[CPosToStr → "[x,y,z,a,b,c]"]
[SocketSendStr] ─── "x,y,z,a,b,c" ──→ recv(5s) → robot_xyz, robot_ABC, R_ee_base


Phase B — 检测 & 发送 Handle 位姿
──────
[SocketReadStr(5s) while loop]       等待 Handle+Cup 双检测稳定 (30s)
       │
       │                              ┌─ Handle 6D检测 (微调YOLOv8n + SAM2)
       │                              │  → PoseStabilizer 稳定
       │                              │  → cam2ee → 基系 P1/P2
       │                              │
       │                              ├─ Cup 6D检测 (YOLOv8m COCO)
       │                              │  → PoseStabilizer 稳定
       │                              │  → cam2ee → 基系 P1/P2/P3
       │                              │
       │                              └─ Handle P3 (倒水):
       │                                 P3_TEACH: Cup P1 + 示教偏移
       │                                 非TEACH:  基于壶嘴+杯几何计算
       │
       ←── "<P1>[P2]{P3}" ──────────── 发送 Handle 3位姿 (pack_three)

[TranStrToCpos: <...> → H_P1]
[TranStrToCpos: [...] → H_P2]
[TranStrToCpos: {...} → H_P3]


Phase C — 指令循环 (Handle + Cup)
──────
[MovL(H_P1)]         接近把手
[SocketSendStr "PREGRIP"] ──────────→ hand_pre_grip()  (半闭合)
[MovL(H_P2)]         预抓取位置
[SocketSendStr "GRIPDONE"] ─────────→ hand_grip_done() (完全夹紧)
[MovL(H_P1)]         提起
[MovL(H_P3)]         移动到倒水位置 (P3)
[Wait(5000ms)]       倒水保持

[MovL(H_P1)]         回 P1
[SocketSendStr "RESET"] ────────────→ hand_reset()

[MovL(H_P2)]         接近 Home
[MovL(Home)]         回 Home
[SocketSendStr "CUP"] ──────────────→ 回复 Cup 3位姿

       ←── "<P1>[P2]{P3}" ────────── Cup P1/P2/P3

[TranStrToCpos: <...> → C_P1]
[TranStrToCpos: [...] → C_P2]
[TranStrToCpos: {...} → C_P3]

[MovL(C_P1)]         接近水杯
[SocketSendStr "RELEASE"] ──────────→ release_object() (PAPER 释放把手)
[SocketSendStr "GRIP"] ─────────────→ grip_object()    (GRIP 抓水杯)
[MovL(C_P2)]         预抓取
[MovL(C_P3)]         递送位置 (P3)
[Wait(3000ms)]       递送保持

[MovL(C_P2)]         回 P2
[MovL(C_P1)]         回 P1
[MovL(Home)]         回 Home

[SocketClose] ──────────────────────→ handle_client 线程退出
                                      PoseStabilizer.on_disconnect()
                                      _poses_cleared = True → 主循环重稳定
```

---

## 位姿来源总结

| 位姿 | 来源 | 坐标系 | 说明 |
|------|------|--------|------|
| ZHAOSHOU1 | `estun_config.yaml` greeting.zhaoshou1_pose | 基系 (mm→m) | PC 配置, 示教值 |
| ZHAOSHOU2 | `estun_config.yaml` greeting.zhaoshou2_pose | 基系 | PC 配置, 示教值 |
| WOSHOU | `estun_config.yaml` greeting.woshou_pose | 基系 | PC 配置, 示教值 |
| Home | CRP 预定义点位 `ptmquqgw91h2026u` | 基系 | 机器人端, 同 Phase A-C Home |
| Handle P1/P2/P3 | PC 计算 (检测→cam2ee→基系) | 基系 | Phase B 动态检测 |
| Cup P1/P2/P3 | PC 计算 (检测→cam2ee→基系) | 基系 | Phase C 按需发送 |

---

## 文件一览

```
Estun/
  estun_greeting_handle_cup_run.py   ← PC 主入口 (Phase0Server + Phase A-C)
  estun_handle_cup_run.py            ← 纯 Handle+Cup (无打招呼, 参考实现)
  estun_config.yaml                  ← ESTUN + greeting 配置
  utils_shared.py                    ← 坐标变换 + 灵巧手控制

doc/
  greeting_handlepluscup.crp         ← 完整 CRP (greeting 30w + Handle+Cup 53w)
  greeting_handshake_socket_server.py ← 同事原始打招呼服务器 (参考)

RealSense_R/
  test_handle_6D.py                  ← Handle 6D 检测器
  test_cup_6D.py                     ← Cup 6D 检测器
  config.yaml                        ← 相机/YOLO/SAM 配置

Omnihand/
  omnihand_gesture.py                ← Gesture 枚举 + OmniHandGesture 类
```
