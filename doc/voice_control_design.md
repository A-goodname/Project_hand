# 语音控制拓展 — 设计方案 v1

> 状态: 草案 | 日期: 2026-07-17 | 手势控制单手势测试通过 ✅

---

## 1. 概述

为 Project_hand 双相机机械臂抓取系统添加语音控制能力。**核心原则**：

- **零语音 = 零影响**：未启用语音时，行为与现有一致
- **静默降级**：任何组件失败不影响主流程
- **宁可不响应，不可误响应**

### 1.1 当前流程

```
greeting (Phase 0) → handle_detect (Phase A-B) → seated_check (Phase C) → hand_in_cup (Phase D)
     8600                      8080                     8080                   8080
```

### 1.2 语音控制范围

| 阶段 | 语音干预 | 说明 |
|------|----------|------|
| Phase 0 招呼前 | "跳过打招呼" / "端茶倒水" | 决策点 `L_DEC_GREETING_START` |
| Phase 0 招呼后 | "跳过握手" | 决策点 `L_DEC_AFTER_ZHAOSHOU` |
| Phase A-C | 无 | 端茶倒水整体执行，不接受指令 |
| Phase D | "递杯" / "放回" | 决策点 `L_DEC_AT_C_P5` |

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                   VoiceController 线程                    │
│                                                         │
│  ┌──────┐   ┌──────┐   ┌───────────────┐   ┌────────┐  │
│  │ VAD  │──→│ ASR  │──→│ CommandMatcher │──→│ Queue  │  │
│  │检测语音│   │转写  │   │ 关键词匹配     │   │命令队列 │  │
│  └──────┘   └──────┘   └───────────────┘   └───┬────┘  │
│                                                │       │
│                                    ┌───────────┘       │
│                                    │ AudioPlayer       │
│                                    │ beep / WAV / TTS  │
│                                    └──────────────────┘ │
└──────────────────────────┬──────────────────────────────┘
                           │ pop_for_phase(current_phase)
                           ▼
┌─────────────────────────────────────────────────────────┐
│                      main.py 主循环                      │
│  cmd = voice_queue.pop_for_phase(phase)                  │
│  if cmd: apply_to_context(ctx, cmd)                     │
│  # 其余流程完全不变                                      │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                   state_machine.py                       │
│  SharedContext.voice_overrides                           │
│  GreetingHandler: 语音窗口+跳过检查                       │
│  HandleGraspAndPourHandler: Phase D voice_hand_on_cup    │
└─────────────────────────────────────────────────────────┘
```

---

## 3. ASR 方案对比

不指定具体模型。按部署复杂度从低到高排列。

### 3.1 方案总览

| 方案 | 类型 | 延迟 | 网络 | GPU | 中文精度 | 部署复杂度 |
|------|------|------|------|-----|----------|------------|
| **A. KWS-only** | 本地关键词检测 | <100ms | 无 | 否 | 固定词表 | ★☆☆ |
| **B. ONNX ASR** | 本地离线 | 50-200ms | 无 | 否 | ★★★ | ★★☆ |
| **C. PyTorch ASR** | 本地离线 | 100-500ms | 无 | 可选 | ★★★★ | ★★★ |
| **D. 云端 API** | 在线流式 | 200-800ms | **需要** | 否 | ★★★★★ | ★☆☆ |

### 3.2 方案 A: KWS-only（纯关键词检测）

```
麦克风 → 关键词检测模型 → 检测到预定义词 → 触发命令
```

| 代表框架 | 模型大小 | 优点 | 缺点 |
|----------|----------|------|------|
| sherpa-onnx KWS | ~5MB | 极低延迟、极低 CPU | 只能识别固定词表、不支持自然变体 |
| openWakeWord | ~10MB | 自定义唤醒词 | 同上 |

**适用场景**：命令词少（<10个），不需要理解变体表达。

### 3.3 方案 B: ONNX 离线 ASR

```
麦克风 → VAD → 语音段 → ONNX 推理 → 文本 → 关键词匹配
```

| 代表框架 | 模型 | 优点 | 缺点 |
|----------|------|------|------|
| funasr-onnx | SenseVoiceSmall (230MB) | 不依赖 PyTorch、CPU 快 | 非流式，需 VAD |
| sherpa-onnx | SenseVoice / Zipformer | 多模型可选、流式可选 | 模型需手动下载 |

**适用场景**：当前推荐。不与 YOLO 抢 GPU，不引入 PyTorch 版本冲突。

### 3.4 方案 C: PyTorch 离线 ASR

```
麦克风 → VAD → 语音段 → PyTorch 推理 → 文本 → 关键词匹配
```

| 代表框架 | 模型 | 优点 | 缺点 |
|----------|------|------|------|
| FunASR | SenseVoiceSmall (234M) | 流式+VAD 内置、精度高 | PyTorch 版本可能与 YOLO 冲突 |
| faster-whisper | whisper-large-v3 | 多语言精度最高 | CPU 慢(数秒)、GPU 占资源 |

**适用场景**：需要高精度且不介意 PyTorch 版本管理。

### 3.5 方案 D: 云端 API

```
麦克风 → VAD → 语音段 → HTTP/WS → 云端识别 → 文本 → 关键词匹配
```

| 代表服务 | 优点 | 缺点 |
|----------|------|------|
| 阿里云 ASR / 讯飞 | 精度最高、支持方言 | 需要网络、按量付费 |
| OpenAI Whisper API | 多语言 | 需要网络+API Key |

**适用场景**：工厂有稳定网络，需要最高精度。

### 3.6 推荐路线

```
Phase 1: 方案 B (ONNX 离线) → 快速部署、无依赖冲突
Phase 2: 方案 D (云端 API，可选) → 精度提升，网络条件允许时切换
```

---

## 4. 音频输出方案

### 4.1 三种生成方式

| 方式 | 延迟 | 网络 | 自然度 | 成本 | 适用场景 |
|------|------|------|--------|------|----------|
| **代码生成** | 即时 | 无 | — | 零 | 系统提示音 (beep/error) |
| **TTS 合成** | 100-800ms | 看方案 | ★★★ | 低/零 | 功能性播报，固定文本 |
| **人声录制** | 即时 | 无 | ★★★★★ | 录制一次 | 面向用户的社交交互 |

### 4.2 WAV 文件分类

#### 代码生成 (Code-generated)

用 `numpy` 生成正弦波，`wavfile.write` 写入：

| 文件 | 内容 | 参数 | 用途 |
|------|------|------|------|
| `beep.wav` | 高音 | 880Hz, 200ms | 命令已接受 |
| `error.wav` | 低音 | 220Hz, 300ms | 识别失败 |

#### TTS 合成 (edge-tts 或本地 TTS)

功能性播报，内容固定，语音合成质量足够：

| 文件 | 内容 | 场景 |
|------|------|------|
| `skip_waving.wav` | "已跳过招手，直接握手" | 跳过打招呼 |
| `skip_handshake.wav` | "已跳过握手" | 跳过握手 |
| `pour_tea.wav` | "直接端茶倒水" | 跳过 Phase 0 |
| `putback.wav` | "杯子已放回" | 放回完成 |
| `stop.wav` | "正在停止" | 急停 |

#### 人声录制 (Human-voiced)

面向用户的社交交互，需要温暖、清晰、自然的人声：

| 文件 | 内容 | 场景 |
|------|------|------|
| `greeting.wav` | "你好，请握手" | Phase 0 检测到人进门 |
| `caution.wav` | "水倒好了，请取杯" | Phase D 递杯提醒 |

> 仅需录制 2 个。如果想进一步减少录制，可全部用 TTS，代价是社交感下降。

### 4.3 TTS 合成方案

| 方案 | 类型 | 延迟 | 网络 | 中文质量 |
|------|------|------|------|----------|
| **edge-tts** | 微软在线 | ~500ms | **需要** | ★★★★ (晓晓/云希) |
| piper-tts | 本地离线 | ~200ms | 无 | ★★ |
| sherpa-onnx TTS | 本地离线 | ~100ms | 无 | ★★ |

**推荐**：TTS 合成用 edge-tts（联网时），网络不通时跳过播报（功能性播报丢了不可惜）。

### 4.4 播放方式

Windows 上 `winsound` 或 `sounddevice` 直接播放 WAV：

```python
import winsound
winsound.PlaySound("audio/beep.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
```

异步播放 (`SND_ASYNC`) 不阻塞主流程。

### 4.5 后续增强 (Phase 2 可选)

```
系统状态 → LLM 生成语境文本 → edge-tts 朗读

示例: "检测到您已就坐，正在为您倒水，大约需要 3 秒。"
```

当前不实现。

---

## 5. 语音控制指令设计

### 5.1 指令表

| 指令 | 触发短语 | 生效决策点 | 作用 |
|------|----------|------------|------|
| **跳过打招呼** | "跳过打招呼"、"不招手" | `L_DEC_GREETING_START` | 跳过招手 (Step 1-2)，直接握手 |
| **端茶倒水** | "端茶倒水" | `L_DEC_GREETING_START` | 跳过整个 Phase 0，直入 Phase A-C |
| **跳过握手** | "跳过握手"、"不握手" | `L_DEC_AFTER_ZHAOSHOU` | 招手完成后跳过握手 |
| **递杯** | "递杯"、"给他"、"取杯" | `L_DEC_AT_C_P5` | hand_on_cup=True → CONTINUE 分支 |
| **放回** | "放回"、"不放"、"别给" | `L_DEC_AT_C_P5` | hand_on_cup=False → ABORT 分支 |
| **停止** | "停止" | 全局 | Python 安全退出 + 告警 |

### 5.2 匹配规则

- **关键词子串匹配**：指令文本中包含任一触发短语即匹配
- **大小写不敏感**
- **优先最长匹配**：如果同时命中多个，取短语最长的命令
- **未匹配 = 忽略**：不提示、不告警

### 5.3 互斥规则

| 命令组 | 互斥方式 |
|--------|----------|
| 递杯 ↔ 放回 | 同类覆盖，最后一条生效 |
| 跳过打招呼 ↔ 端茶倒水 | 同类覆盖，最后一条生效 |
| 停止 | 不覆盖，立即生效 |

---

## 6. 决策点设计

### 6.1 模块树

```
[init_hand + TCP 8600 accept]

L_DEC_GREETING_START                ← 决策点 1
├── DEFAULT → 招手
├── "跳过打招呼" → L_MOD_HANDSHAKE
└── "端茶倒水" → L_MOD_POUR_TEA

    Step 1: GetCurCPos → send → SocketReadStr(10s) → ZHAOSHOU1 → MovL
    Step 2: 等人(YOLO) → ZHAOSHOU2 → MovL

L_DEC_AFTER_ZHAOSHOU                ← 决策点 2
├── DEFAULT → 握手
└── "跳过握手" → L_MOD_POUR_TEA

    L_MOD_HANDSHAKE:                 ← 握手子模块
      Step 3: 挥手(HELLO_1/2 ×N) → WOSHOU → MovL
      Step 4: 握手(压力检测) → Home → MovL
      SocketClose(8600)
      → L_MOD_POUR_TEA

L_MOD_POUR_TEA:                     ← 端茶倒水 (Phase A-C 整体)
    TCP 8080 accept → Phase A → Phase B → Handle+Cup+倒水 → C_P5
    不可拆分, 不接受语音指令

L_DEC_AT_C_P5:                      ← 决策点 3
    语音路径: hand_on_cup 直接赋值, 跳过 HandInCup 检测器
    DEFAULT:   启动 HandInCup 检测器, 等 ≤10s 判定
├── CONTINUE (<>包装) → L_MOD_DELIVER
└── ABORT ([]包装)    → L_MOD_PUTBACK

    L_MOD_DELIVER:  RELEASE → 退避 → Home → SocketClose
    L_MOD_PUTBACK:  C_P5 → C_P3 → RELEASE → 退避 → Home → SocketClose
```

### 6.2 CRP Label 命名

| Label | 类型 | 说明 |
|-------|------|------|
| `L_DEC_GREETING_START` | 决策点 | Phase 0 入口，语音决定是否跳过招手 |
| `L_MOD_HANDSHAKE` | 模块入口 | 握手 (Step 3-4) |
| `L_DEC_AFTER_ZHAOSHOU` | 决策点 | 招手完成后，语音决定是否跳过握手 |
| `L_MOD_POUR_TEA` | 模块入口 | 端茶倒水 (Phase A-C) |
| `L_DEC_AT_C_P5` | 决策点 | Phase D 判定点 |
| `L_MOD_DELIVER` | 模块入口 | 递杯分支 |
| `L_MOD_PUTBACK` | 模块入口 | 放回分支 |

### 6.3 决策点 CRP 结构

每个决策点由以下 widget 序列组成：

```
SocketSendStr(signal)        ← 通知 Python 到达决策点
SocketReadStr(timeout)       ← 等待 Python 路由指令
If(reply == "分支1")         ← 条件判断
  GoTo(label_id_1)           ← 跳转到对应模块
If(reply == "分支2")
  GoTo(label_id_2)
GoTo(default_label_id)       ← 默认路径
```

### 6.4 决策点与 Python 的交互

```
CRP                           Python
───                           ──────

到达决策点
SocketSendStr("DECISION:GREETING_START")
                              收到信号，设置 current_module
SocketReadStr(5s)             voice_queue.pop_for_module(current_module)
                              如果有语音命令:
                                reply = 命令对应的 reply 值
                                session_log("VOICE: skip_waving")
                              如果无语音命令:
                                reply = "DEFAULT"
                              conn.sendall(reply)

收到 reply
If(reply == "SKIP_WAVING")
  GoTo(L_MOD_HANDSHAKE)
If(reply == "SKIP_ALL")
  GoTo(L_MOD_POUR_TEA)
GoTo(DEFAULT)  → 继续招手
```

**无语音时延迟为零**：Python 瞬间回复 `DEFAULT`，CRP 的 SocketReadStr 立即读到，走默认路径。

### 6.5 决策点时序

```
L_DEC_GREETING_START:
  信号: "DECISION:GREETING_START"
  SocketReadStr 超时: 5s
  默认: "DEFAULT" (继续招手)
  分支: "SKIP_WAVING" → L_MOD_HANDSHAKE
        "SKIP_ALL" → L_MOD_POUR_TEA

L_DEC_AFTER_ZHAOSHOU:
  信号: "DECISION:AFTER_ZHAOSHOU"
  SocketReadStr 超时: 5s
  默认: "DEFAULT" (继续握手)
  分支: "SKIP_HANDSHAKE" → L_MOD_POUR_TEA

L_DEC_AT_C_P5:
  信号: "READY" (复用现有协议)
  默认: "WAIT_DETECT" (启动 HandInCup 检测器)
  分支: "CONTINUE" / "ABORT" (通过现有 <> / [] 包装位姿)
```

---

## 7. 唤醒与启用机制

### 7.1 启用方式

两种方式均可，满足其一即 `enable_voice = True`：

| 方式 | 触发 | 窗口 |
|------|------|------|
| **空格键** | 按键 | Phase 0 初始化后 10s 内 |
| **唤醒词** | "开始" / "开启语音控制" | Phase 0 初始化后 10s 内 |

### 7.2 10s 窗口逻辑

```
Phase 0 开始
    │
    ├── init_hand()
    ├── Camera A open
    ├── TCP 8600 listen
    │
    ├── [10s 窗口开始]
    │     │
    │     ├── 语音唤醒词 / 空格键
    │     │   → enable_voice = True
    │     │   → 窗口延长，等待后续指令
    │     │
    │     └── 10s 无触发
    │         → enable_voice = False
    │         → 走默认线性流程，后续不接受语音指令
    │
    └── 进入决策点 (无论 enable_voice 状态)
```

### 7.3 状态机

```
              ┌─────────┐
    启动 ───→ │ WAITING │ (10s 倒计时)
              └────┬────┘
                   │
        ┌──────────┼──────────┐
        │ 唤醒词    │ 空格键    │ 10s 超时
        ▼          ▼          ▼
   ┌─────────┐ ┌─────────┐ ┌──────────┐
   │ ACTIVE  │ │ ACTIVE  │ │ DISABLED │
   │ (语音)  │ │ (按键)  │ │ 本次运行  │
   └────┬────┘ └────┬────┘ │ 无语音    │
        │          │        └──────────┘
        └────┬─────┘
             │
             ▼
    ┌───────────────┐
    │ 接受语音指令   │
    │ 直到流程结束   │
    └───────────────┘
```

### 7.4 实现伪代码

```python
class VoiceController:
    WAIT_WINDOW_S = 10.0
    
    def __init__(self):
        self.enabled = False
        self._wait_deadline = time.time() + self.WAIT_WINDOW_S
    
    def on_key_space(self):
        """空格键触发."""
        if not self.enabled and time.time() < self._wait_deadline:
            self.enabled = True
            self._play_beep()
    
    def on_wake_phrase(self, phrase: str):
        """唤醒词触发."""
        if not self.enabled and time.time() < self._wait_deadline:
            self.enabled = True
            self._play_beep()
    
    def update(self):
        """主循环每帧调用."""
        if not self.enabled and time.time() > self._wait_deadline:
            return  # 窗口已过，永不启用
```

---

## 8. 命令队列

### 8.1 数据结构

```python
@dataclass
class VoiceCommand:
    command: str        # 命令名，如 "skip_waving"
    phase_scope: str    # 作用决策点，如 "L_DEC_GREETING_START"
    timestamp: float
    ttl: float = 30.0

class CommandQueue:
    def __init__(self):
        self._queue: deque[VoiceCommand] = deque()
        self._lock = threading.Lock()
    
    def push(self, cmd: VoiceCommand):
        """入队。互斥命令覆盖前序。"""
        with self._lock:
            # 清理过期的
            self._queue = deque(
                c for c in self._queue 
                if time.time() - c.timestamp < c.ttl
            )
            # 互斥覆盖：同一 phase_scope 内同组命令只保留最新的
            group = CONTRADICTION_GROUPS.get(cmd.command)
            if group:
                self._queue = deque(
                    c for c in self._queue
                    if not (CONTRADICTION_GROUPS.get(c.command) == group
                            and c.phase_scope == cmd.phase_scope)
                )
            self._queue.append(cmd)
    
    def pop_for_module(self, module: str) -> Optional[VoiceCommand]:
        """弹出当前模块的有效命令."""
        with self._lock:
            now = time.time()
            # 清理过期
            while self._queue and now - self._queue[0].timestamp > self._queue[0].ttl:
                self._queue.popleft()
            # 取第一个匹配的
            for i, cmd in enumerate(self._queue):
                if cmd.phase_scope == module:
                    del self._queue[i]
                    return cmd
            return None
```

### 8.2 互斥组

```python
CONTRADICTION_GROUPS = {
    "skip_waving":     "greeting_skip",
    "pour_tea":        "greeting_skip",
    "skip_handshake":  "handshake",
    "deliver":         "deliver_branch",
    "putback":         "deliver_branch",
}
```

---

## 9. 相机生命周期

### 9.1 各模块相机需求

| 模块 | Camera A | Camera B | 说明 |
|------|----------|----------|------|
| 招手 (Step 1-2) | open | open(chair) | A: 等人; B: 椅子标定 |
| 握手 (Step 3-4) | open | any | A: 已不需要; B: 保持 |
| 端茶倒水 (Phase A-C) | open→closed | paused | A: Phase B后关闭; B: 暂停 |
| 递杯/放回 (Phase D) | closed | active(hic) | A: 不用; B: HandInCup |

### 9.2 模块切换时的相机调整

```python
MODULE_CAMERA_REQUIREMENTS = {
    "L_DEC_GREETING_START": {"cam_a": "open",  "cam_b": "open"},
    "L_MOD_HANDSHAKE":       {"cam_a": "open",  "cam_b": "any"},
    "L_MOD_POUR_TEA":        {"cam_a": "open",  "cam_b": "paused"},
    "L_DEC_AT_C_P5":         {"cam_a": "closed","cam_b": "active"},
    "L_MOD_DELIVER":         {"cam_a": "closed","cam_b": "active"},
    "L_MOD_PUTBACK":         {"cam_a": "closed","cam_b": "active"},
}

def transition_cameras(from_module, to_module, ctx):
    """只做必要的变化，不重启已正确的状态."""
    req = MODULE_CAMERA_REQUIREMENTS[to_module]
    cur = get_current_camera_status(ctx)
    
    if req["cam_a"] != cur["cam_a"]:
        if req["cam_a"] == "open":
            open_camera_a(ctx)
        elif req["cam_a"] == "closed":
            close_camera_a(ctx)
    
    if req["cam_b"] != cur["cam_b"]:
        # Camera B 通过 PersonDetector 控制
        adjust_camera_b(ctx, req["cam_b"])
```

---

## 10. 音频设备配置

```yaml
# voice_config.yaml — audio 段
audio:
  mic_device: "UGREEN Camera Audio"     # 首选 Camera B 自带麦克风
  mic_fallback: "default"               # 找不到 UGREEN 时用系统默认
  sample_rate: 16000                    # ASR 标准采样率
  channels: 1                           # 单声道
```

### 10.1 设备检测逻辑

```python
def find_mic(cfg: dict) -> Optional[str]:
    """按优先级查找可用麦克风."""
    devices = sd.query_devices()
    input_devices = [d for d in devices if d['max_input_channels'] > 0]
    
    # 1. 首选
    preferred = cfg.get("mic_device")
    if preferred:
        for d in input_devices:
            if preferred.lower() in d['name'].lower():
                return d['name']
    
    # 2. 兜底
    fallback = cfg.get("mic_fallback", "default")
    if fallback == "default":
        return sd.default.device[0]  # 系统默认
    
    for d in input_devices:
        if fallback.lower() in d['name'].lower():
            return d['name']
    
    # 3. 第一个可用
    if input_devices:
        return input_devices[0]['name']
    
    return None  # 无可用麦克风 → self.enabled = False
```

---

## 11. VAD 静音判定

```
麦克风 → VAD 检测 → 语音段 → 静音 > 阈值 → 段落结束 → 送 ASR
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `silence_trigger_s` | 1.5 | 静音超过此值认为一段话结束 |
| `max_speech_s` | 10 | 单段语音最长时长，超长截断 |
| `min_speech_s` | 0.3 | 语音太短忽略（咳嗽/杂音） |

```yaml
# voice_config.yaml — vad 段
vad:
  silence_trigger_s: 1.5
  max_speech_s: 10
  min_speech_s: 0.3
```

---

## 12. 命令确认

TTS 不回读指令内容。用代码生成的短促提示音：

```
用户说 "递杯"
  → VAD 检测到语音结束
  → ASR 识别 → 匹配 "递杯"
  → 队列 push("deliver")
  → 播放 beep.wav (880Hz, 200ms)   ← 代码生成
  → 执行

ASR 有文本但无匹配:
  → 播放 error.wav (220Hz, 300ms)  ← 代码生成
  → 忽略
```

| 文件 | 生成 | 内容 | 时长 |
|------|------|------|------|
| `beep.wav` | numpy 正弦波 | 高音提示 | ~200ms |
| `error.wav` | numpy 正弦波 | 低音提示 | ~300ms |

---

## 13. 降级策略

```
VAD 模型加载失败   → ASR 不启动 → self.enabled = False → 等同于 enabled=false
ASR 模型加载失败   → self.enabled = False
麦克风设备不可用   → self.enabled = False
TTS 预录 WAV 不存在 → 尝试 edge-tts
TTS edge-tts 失败  → 静默跳过播报
TTS 网络不通       → 静默跳过播报
```

**全部静默降级，不抛异常。**

---

## 14. 新增/修改文件清单

### 14.1 新增

| 文件 | 行数(估) | 内容 |
|------|----------|------|
| `Orchestrator/voice_control.py` | ~300 | VoiceController + CommandQueue + AudioPlayer |
| `Orchestrator/voice_config.yaml` | ~80 | 所有配置项 |
| `Orchestrator/audio/` | - | 预录音频目录 |
| `scripts/insert_decision_point.py` | ~100 | CRP 决策点生成脚本 |

### 14.2 修改

| 文件 | 改动量 | 内容 |
|------|--------|------|
| `Orchestrator/main.py` | +20 | 创建 VoiceController，主循环 pop，空格键处理 |
| `Orchestrator/state_machine.py` | +15 | SharedContext 加 voice 字段；GreetingHandler 加 10s 窗口；_handle_client_impl 加 voice_hand_on_cup |

### 14.3 不改

- CRP 文件（无 GOTO 方案）或 `doc/greeting_handlepluscup-goto.crp`（GOTO 方案，新文件）
- HandleCupDetector / PersonDetector / HandInCupDetector
- EyeToHand / RealSense_R / Omnihand 模块
- estun_config.yaml / orch config.yaml

---

## 15. 配置文件示例

```yaml
# Orchestrator/voice_config.yaml
voice_control:
  enabled: false                     # 默认关闭
  wait_window_s: 10.0                # Phase 0 初始化后的等待窗口

  audio:
    mic_device: "UGREEN Camera Audio"
    mic_fallback: "default"
    sample_rate: 16000
    channels: 1

  asr:
    engine: "funasr-onnx"            # 方案 B: ONNX 离线
    model: "SenseVoiceSmall"         # 根据方案调整

  vad:
    engine: "silero-vad"
    silence_trigger_s: 1.5
    max_speech_s: 10
    min_speech_s: 0.3

  wake:
    phrases: ["开始", "开启语音控制"]
    space_key: true                  # 空格键也可启用

  tts:
    audio_dir: "Orchestrator/audio/"
    fallback: "edge-tts"             # 或 "none"
    voice: "zh-CN-XiaoxiaoNeural"

  decisions:
    L_DEC_GREETING_START:
      signal: "DECISION:GREETING_START"
      default: "DEFAULT"
      commands:
        skip_waving:
          phrases: ["跳过打招呼", "不招手"]
          reply: "SKIP_WAVING"
          goto: "L_MOD_HANDSHAKE"
        pour_tea:
          phrases: ["端茶倒水"]
          reply: "SKIP_ALL"
          goto: "L_MOD_POUR_TEA"

    L_DEC_AFTER_ZHAOSHOU:
      signal: "DECISION:AFTER_ZHAOSHOU"
      default: "DEFAULT"
      commands:
        skip_handshake:
          phrases: ["跳过握手", "不握手"]
          reply: "SKIP_HANDSHAKE"
          goto: "L_MOD_POUR_TEA"

    L_DEC_AT_C_P5:
      signal: "READY"                # 复用现有协议
      default: "WAIT_DETECT"
      commands:
        deliver:
          phrases: ["递杯", "给他", "取杯", "可以给"]
          reply: "CONTINUE"
        putback:
          phrases: ["放回", "不放", "别给", "拿回来"]
          reply: "ABORT"

  confirm:
    beep_audio: "audio/beep.wav"      # 命令确认提示音
    error_audio: "audio/error.wav"    # 识别失败提示音

  test_mode: false
  test_commands: []
```

---

## 16. 测试矩阵

| # | 场景 | 语音输入 | 预期 |
|---|------|----------|------|
| 1 | 无语音，正常 | 无 | greeting → handle → seated → hic → exit (与现有一致) |
| 2 | 无语音，config enabled=false | 无 | 与 #1 完全一致 |
| 3 | 跳过打招呼 | "跳过打招呼" | 招手跳过 → 直接握手 |
| 4 | 端茶倒水 | "端茶倒水" | Phase 0 全跳过 → Phase A-C |
| 5 | 跳过握手 | "跳过握手" (招手后) | 握手跳过 → Phase A-C |
| 6 | 递杯 | "递杯" | hand_on_cup=True → CONTINUE |
| 7 | 放回 | "放回" | hand_on_cup=False → ABORT |
| 8 | 先递后反悔 | "递杯" → "放回" | 队列覆盖 → ABORT |
| 9 | 错误阶段 | Phase A-C 说"递杯" | 忽略 (阶段门控) |
| 10 | 10s 无触发 | 无 (窗口超时) | enable_voice=False，走正常线性流程 |
| 11 | 空格键唤醒 | 空格键 | enable_voice=True |
| 12 | 唤醒词唤醒 | "开始" | enable_voice=True |
| 13 | 麦克风不可用 | - | self.enabled=False，静默降级 |
| 14 | 网络断开 | "递杯" (edge-tts 不可用) | 命令执行，播报回退 WAV |
| 15 | 测试模式 | --voice-cmd "递杯" | 直接注入队列 |

---

## 附录 A: 语音手势控制 — 先行验证

在接入完整机械臂流程之前，先用灵巧手做语音控制的独立验证。
不涉及机器人运动、TCP 协议、CRP，只验证 ASR → 命令 → 执行的链路。

### A.1 目标

```
麦克风 → VAD → ASR → 关键词匹配 → Gesture 枚举 → OmniHand.execute()
                                          │
                                          └── beep.wav / error.wav
```

### A.2 手势指令

| 指令 | 触发短语 | Gesture |
|------|----------|---------|
| 握拳 | "握拳"、"攥拳" | `FIST` |
| 张开 | "张开"、"手掌"、"摊开" | `PAPER` |
| 抓取 | "抓取"、"抓住" | `GRIP` |
| 半握 | "半握"、"准备抓" | `PREGRIP` |
| 握紧 | "握紧" | `GRIPDONE` |
| 握手 | "握手" | `HAND_SHAKE` |
| 挥手1 | "挥手1" | `HELLO_1` |
| 挥手2 | "挥手2" | `HELLO_2` |
| 复位 | "复位"、"恢复" | `RESET` |

### A.3 准备工作

1. **安装 ASR 依赖**
   ```bash
   pip install funasr-onnx sounddevice
   ```

2. **创建音频目录**
   ```bash
   mkdir Orchestrator\audio
   ```

3. **代码生成提示音**（numpy 正弦波 → WAV）

4. **测试脚本**
   - `test_asr.py` — 纯 ASR 测试：VAD → 识别 → 打印文本
   - `test_voice_hand.py` — 语音手势控制：VAD → ASR → 匹配 → 执行手势

### A.4 不涉及

- ❌ CRP / 机器人运动 / TCP 协议
- ❌ 决策点 / GOTO
- ❌ 人声 WAV 录制（提示音代码生成即可）
- ❌ 唤醒词（直接说指令）
