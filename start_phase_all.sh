#!/bin/bash
# 一键启动三阶段融合 demo (2026-08-21):
#   融合 CRP: doc/estun_fused_phase_all.crp (Phase0 8600 → wait 3s → A2D 8080 → wait 3s → E 8700)
#   服务终端 (gnome-terminal 独立窗口, sh 本身只协调):
#     A: Phase 0 py        estun_greeting_phase0.py       8600  占 灵巧手 + 迎宾 UGREEN(4:1)
#     C: A2D py --defer-devices  estun_handle_cup_sm.py   8080  只初始化不占设备; 连接时才开 D435i/usb-0:2/手
#     D: phaseE py         estun_rps_phaseE.py            8700  零设备 (RPS 服务独占 usb-0:2+手)
#     E: RPS 服务           rps_stream --enable-hand + realtime_voice --rps-game  8768  A2D 退出后才起
#   8770 语音门禁: 不预启动 — 由 A2D py 在 READY (Phase D) 时自启 (异步), 不在 Phase A/B 提前启动
#   设备/麦克风严格串行: Phase0 占 → 释放 → A2D 占 → 释放 → RPS 占
# 用法: bash start_phase_all.sh
# 前提: 机器人端已导入 doc/estun_fused_phase_all.crp; 三 py 依赖的服务/相机空闲
set -e

PROJECT_DIR="/data/VLA/Project_hand"
PROJ_PY="/home/user/miniconda3/envs/Proj/bin/python"
VOICE_LOG="/tmp/rps_voice_phaseE.log"
FIFO="/tmp/rps_voice_stdin"
HELPER="/tmp/rps_phaseE_service.sh"
PREVIEW="${PREVIEW:-0}"   # 1=开 cv2 预览窗口 (调试); 0=headless (默认, GPU 不支持渲染省 CPU)
# 各阶段 py 进程匹配模式 (pgrep -f 锚定真实命令行, 用于"等进程退出"交接判断)
A2D_PATTERN="^${PROJ_PY} .*estun_handle_cup_sm\.py"
PHASE0_PATTERN="^${PROJ_PY} .*estun_greeting_phase0\.py"
PHASEE_PATTERN="^${PROJ_PY} .*estun_rps_phaseE\.py"
# 8770 语音门禁: 由 realtime_voice --rps-game 内建 (A2D 递杯确认用); 无需单独 spawn/匹配

# ---- 清理残留 (端口 + RPS 进程 + 临时文件) ----
echo "[清理] 检查残留端口/进程..."
for p in 8600 8080 8700 8768 8770; do
  if ss -ltn | grep -q ":$p "; then
    echo "[WARN] 端口 $p 被占用 → fuser -k $p/tcp"
    fuser -k "$p/tcp" 2>/dev/null || true
  fi
done
pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
pkill -TERM -f 'rps_stream_merged.py' 2>/dev/null || true
# 清理旧阶段实例 (review G 点, 2026-08-21): 多协调脚本残留会抢端口/串口/相机 → 组合启动不稳定
pkill -TERM -f 'estun_greeting_phase0.py' 2>/dev/null || true
pkill -TERM -f 'estun_handle_cup_sm.py' 2>/dev/null || true
pkill -TERM -f 'estun_rps_phaseE.py' 2>/dev/null || true
sleep 1   # 给 SIGTERM 时间走 teardown (close MediaPipe / disable_hand)
# 再 pkill -9 兜底 (确保旧实例彻底退出, 否则 pgrep 等 A2D 退出的硬门禁可能误判)
pkill -KILL -f 'estun_greeting_phase0.py' 2>/dev/null || true
pkill -KILL -f 'estun_handle_cup_sm.py' 2>/dev/null || true
pkill -KILL -f 'estun_rps_phaseE.py' 2>/dev/null || true
sleep 0.5
rm -f "$VOICE_LOG" "$HELPER" "$FIFO" /tmp/rps_voice_stdin 2>/dev/null || true
# 清手交接残留 (方案甲): 防止上次运行的 a2d_ready / merged ready+trigger 让 RPS 提前启动
rm -f /tmp/a2d_ready /tmp/rps_handover_ready /tmp/rps_handover_trigger /tmp/rps_handover_released 2>/dev/null || true
# 清阶段 py 运行日志 (调试用 tee): 每次从新日志开始
rm -f /tmp/phase0_phase0.log /tmp/phase0_a2d.log /tmp/phase0_phaseE.log 2>/dev/null || true

# ---- 运行前环境检查: 音频 (2026-08-22) ----
# 不用 sudo (杀不了其它用户的进程). 只做检测: 发现音频异常(多套声音服务器 / 无 JBL sink) →
# 打印说明 + 停止启动 (exit), 由用户手动修复后重跑; AUDIO_CHECK=0 可跳过停止(仍打印, 可能无声).
JBL_SINK=bluez_output.78_44_05_FA_12_B9.1
JBL_MAC=78:44:05:FA:12:B9
check_audio() {
  local me="$(id -un)"
  local RT=/run/user/1000    # 当前 user 的 PipeWire 实例
  echo "[音频检查] 前置环境检查 (不用 sudo, 仅检测)..."
  # ① JBL 蓝牙 sink 是否在当前 user 实例 (音箱开机 + 已连回 user; 若挂在其它声音服务器/未连接则缺)
  #   主判据: JBL sink 存在即视为可用 (下方 line set-default-sink + 播测可出声) — 2026-08-22 实测
  if ! XDG_RUNTIME_DIR=$RT pactl list short sinks 2>/dev/null | grep -q "$JBL_SINK"; then
    echo "[音频检查] ❌ 找不到 JBL sink ($JBL_SINK) — 音箱可能未开机 / 未连接 / 仍挂在其它声音服务器"
    return 1
  fi
  # ② 默认 sink 是否就是 JBL (user 实例) — "只要 default 是 user 的, 可继续"; 非 JBL 仅提醒 (下方会 set-default-sink 修正)
  local def
  def=$(XDG_RUNTIME_DIR=$RT pactl get-default-sink 2>/dev/null) || def=""
  if [ "$def" != "$JBL_SINK" ]; then
    echo "[音频检查] ⚠ 默认 sink=${def:-<未知>} ≠ JBL — 下方会 set-default-sink 修正; 若仍无声再排查"
  fi
  echo "[音频检查] ✓ JBL sink 在当前 user 实例 (默认 sink=${def:-<未知>})"
  # ③ 多套声音服务器并存: 只提示不阻断 (2026-08-22 实测 wangxilu+user 并存, JBL 在 user 实例且 default 仍可正常出声)
  local extras
  extras=$(ps -o user= -C pipewire,wireplumber,pipewire-pulse 2>/dev/null | sort -u \
             | grep -v "^${me}$" | grep -v '^root$' | tr '\n' ' ')
  [ -n "$extras" ] && echo "[音频检查]   ⚠ 另有声音服务器实例 [$extras] 并存 (不阻断, 实测可用; 若无声再查此条)"
  return 0
}

if ! check_audio; then
  echo "[音频检查] ⚠ 音频异常 — 最常见=JBL 音箱未开机/未连回当前 user 实例 (需 sudo/蓝牙, sh 不代劳). 手动修复后重跑, 或用 AUDIO_CHECK=0 跳过:"
  echo "   ① JBL 音箱关机 → 开机; 未连接 → bluetoothctl connect $JBL_MAC"
  echo "   ② JBL 仍挂在其它声音服务器 → 停多余那套 (需 sudo)   e.g.  sudo pkill -u <多余用户> -x pipewire; sudo pkill -u <多余用户> -x wireplumber; sudo pkill -u <多余用户> -x pipewire-pulse"
  echo "      (停掉后 JBL 可能需重连:  bluetoothctl connect $JBL_MAC)"
  echo "   ③ 确认 user 实例能见 JBL:  XDG_RUNTIME_DIR=/run/user/1000 pactl list short sinks | grep $JBL_SINK"
  echo "   ④ 默认 sink 设回 JBL:   XDG_RUNTIME_DIR=/run/user/1000 pactl set-default-sink $JBL_SINK"
  if [ "${AUDIO_CHECK:-1}" != "0" ]; then
    echo "[音频检查] 中断启动 (音频环境异常)"
    exit 3
  fi
  echo "[音频检查] AUDIO_CHECK=0 → 已跳过中断, 继续启动 (可能无声)"
fi

# ---- 读 config auto_invite.enabled ----
read_auto_invite() {
  "$PROJ_PY" -c "
import yaml
cfg = yaml.safe_load(open('$PROJECT_DIR/Estun/estun_config.yaml', encoding='utf-8'))
print('1' if cfg['estun'].get('rps', {}).get('auto_invite', {}).get('enabled', False) else '0')
" 2>/dev/null || echo 0
}
AUTO_INVITE=$(read_auto_invite)

# ---- RPS 相机: EyeToHand 方式解析 usb-0:2 (不硬编码 /dev/videoN) ----
RPS_CAM_INDEX=$("$PROJ_PY" -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from EyeToHand.camera import find_camera_id
idx = find_camera_id(keywords=['UGREEN Camera'], path_keyword='usb-0:2')
print(idx if idx is not None else '')
" 2>/dev/null | tail -1)
if [ -z "$RPS_CAM_INDEX" ]; then
  echo "[WARN] 找不到 RPS 相机 (usb-0:2) → 回退 RPS_CAMERA_INDEX=6"
  RPS_CAM_INDEX=6
fi
export RPS_CAM_INDEX
echo "[START] RPS 相机: usb-0:2 → /dev/video$RPS_CAM_INDEX"

# ---- 语音播报 config (清 pipewire_target 走 default sink=JBL) ----
VOICE_CONFIG="/tmp/rps_config_phaseE.json"
"$PROJ_PY" -c "
import json
d = json.load(open('/data/Speech/config.json', encoding='utf-8'))
d.get('realtime', {}).pop('pipewire_target', None)
json.dump(d, open('$VOICE_CONFIG', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
" || echo "[WARN] 语音 config 复制失败"
XDG_RUNTIME_DIR=/run/user/1000 pactl set-default-sink bluez_output.78_44_05_FA_12_B9.1 2>/dev/null \
  || echo "[WARN] 设默认 sink=JBL 失败 (JBL 未连?)"
export VOICE_CONFIG

# ---- FIFO (auto_invite=1 时 phaseE 写回车触发 RPS 邀请) ----
FIFO_HOLDER=""
if [ "$AUTO_INVITE" = "1" ]; then
  rm -f "$FIFO"; mkfifo "$FIFO"
  tail -f /dev/null > "$FIFO" &   # 持写端, 防读端 EOF 停读
  FIFO_HOLDER=$!
  echo "[START] FIFO $FIFO 已建 (持写端 pid=$FIFO_HOLDER)"
fi

# ---- RPS 服务 helper (gnome-terminal 内运行, 与 start_rps_phaseE.sh 一致) ----
cat > "$HELPER" <<'EOS'
#!/bin/bash
set -Eeuo pipefail
PYTHON=/data/xilu/conda-envs/VLA/bin/python
RPS_SCRIPT=/data/xilu/test/rps_stream_merged.py   # 方案甲: merged (HandInCup 递杯 + RPS) 同相机 usb-0:2
VOICE_SCRIPT=/data/Speech/realtime_voice.py
VOICE_LOG=/tmp/rps_voice_phaseE.log
FIFO=/tmp/rps_voice_stdin
AUTO_INVITE=$(/home/user/miniconda3/envs/Proj/bin/python -c "
import yaml
cfg = yaml.safe_load(open('/data/VLA/Project_hand/Estun/estun_config.yaml', encoding='utf-8'))
print('1' if cfg['estun'].get('rps', {}).get('auto_invite', {}).get('enabled', False) else '0')
" 2>/dev/null || echo 0)
RPS_CAM_INDEX="${RPS_CAM_INDEX:-6}"
VOICE_CONFIG="${VOICE_CONFIG:-/tmp/rps_config_phaseE.json}"
RPS_LOG=$(mktemp /tmp/rps_stream.XXXXXX.log)
RPS_PID=""
VOICE_PID=""

cleanup() { trap - INT TERM EXIT; [[ -n "$VOICE_PID" ]] && kill -TERM "$VOICE_PID" 2>/dev/null || true; [[ -n "$RPS_PID" ]] && kill -TERM "$RPS_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

# RPS 语音服务先起 (提供 8770 门禁 + voice log; merged 依赖 RPS_HANDOVER_LOG) —
# 否则 voice 晚起, A2D READY 后 20s 连不上 8770 门禁 (2026-08-22 实机定位).
echo "[RPS] 启动语音服务 realtime_voice.py --rps-game (stdout→终端+日志)..."
set +e
if [ "$AUTO_INVITE" = "1" ]; then
  echo "[RPS] 语音 stdin=FIFO (phaseE 自动邀请); 终端回车将不可用"
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$VOICE_SCRIPT" --rps-game --config "$VOICE_CONFIG" < "$FIFO" > >(tee -a "$VOICE_LOG") 2>&1 &
else
  echo "[RPS] 语音 stdin=终端 (人工按回车邀请)"
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$VOICE_SCRIPT" --rps-game --config "$VOICE_CONFIG" > >(tee -a "$VOICE_LOG") 2>&1 &
fi
VOICE_PID=$!
set -e
# 等 RPS 语音就绪 (日志出现 [System] Realtime voice started 或进程退出)
for ((i = 0; i < 80; i++)); do
  if ! kill -0 "$VOICE_PID" 2>/dev/null; then
    echo "[RPS] 语音进程提前退出 (code=$?)"
    break
  fi
  if grep -q "Realtime voice started" "$VOICE_LOG" 2>/dev/null; then
    echo "[RPS] 语音服务就绪"
    break
  fi
  sleep 0.25
done

# RPS 视觉服务 (8768) — 启动确认 + 失败重试 (review D 点, 2026-08-21):
#   启动失败(进程早退/8768 未监听) → 清理 → 等 10s (设备沉降) → 重试; 最多 RPS_MAX_ATTEMPTS 次.
RPS_MAX_ATTEMPTS="${RPS_MAX_ATTEMPTS:-3}"
rps_ready=0
for ((attempt = 1; attempt <= RPS_MAX_ATTEMPTS; attempt++)); do
  echo "[RPS] 启动视觉服务 (第 $attempt/$RPS_MAX_ATTEMPTS 次) rps_stream_merged.py --enable-hand (/dev/video$RPS_CAM_INDEX, 8768)..."
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 RPS_CAMERA_INDEX="$RPS_CAM_INDEX" RPS_WIDTH=640 RPS_HEIGHT=480 RPS_FPS=30 \
    RPS_STREAM_PORT=8768 RPS_STREAM_LABEL="Camera /dev/video$RPS_CAM_INDEX" \
    RPS_HANDOVER_MODE=1 RPS_HANDOVER_LOG="$VOICE_LOG" \
    "$PYTHON" "$RPS_SCRIPT" --enable-hand > >(tee -a "$RPS_LOG") 2>&1 &
  RPS_PID=$!

  for ((i = 0; i < 80; i++)); do
    if curl -fs --max-time 1 http://127.0.0.1:8768/round/status >/dev/null 2>&1; then
      echo "[RPS] 视觉服务就绪 (8768, 第 $attempt 次)"
      rps_ready=1
      break 2
    fi
    if ! kill -0 "$RPS_PID" 2>/dev/null; then
      echo "[RPS] 视觉服务启动失败 (第 $attempt 次):" >&2
      tail -n 40 "$RPS_LOG" >&2 || true
      break
    fi
    sleep 0.25
  done

  if [ "$rps_ready" = "1" ]; then
    break
  fi
  kill -TERM "$RPS_PID" 2>/dev/null || true
  if [ "$attempt" -lt "$RPS_MAX_ATTEMPTS" ]; then
    echo "[RPS] 等 10s 设备沉降后重试..."
    sleep 10
  fi
done

if [ "$rps_ready" != "1" ]; then
  echo "[RPS] 视觉服务 ${RPS_MAX_ATTEMPTS} 次启动均失败 → 放弃 (请检查 RPS 日志)" >&2
  exit 1
fi

if [ "${RPS_OPEN_BROWSER:-1}" = "1" ]; then
  xdg-open http://127.0.0.1:8768/ >/dev/null 2>&1 || true
fi

# 保持前台运行 (等 phaseE 发 DONE 后由主脚本 SIGTERM 收尾)
set +e
wait "$VOICE_PID" 2>/dev/null
status=$?
set -e
echo "[RPS] 语音服务退出 (code=$status)"
cleanup
exit "$status"
EOS
chmod +x "$HELPER"

# ---- 开终端 (通用) ----
# 注意: 不能用 "'$*'" 单引号包命令 (会把整条含空格命令当成一个文件名 → 127).
# 命令直接放双引号内, 由内层 bash 解析 (本项目命令均无空格/特殊字符).
# KEEP_TERM=1 时保留终端 (exec bash, 看日志); 默认 py 退出后终端自动关闭.
open_term() {
  local title="$1"; shift
  if [ "${KEEP_TERM:-0}" = "1" ]; then
    gnome-terminal --title="$title" -- bash -c "$*; echo '[$title] 已退出 (code='\$?')'; exec bash" \
      >/dev/null 2>&1 || { echo "[WARN] 无法开终端 $title → 请手动跑: $*"; return 1; }
  else
    gnome-terminal --title="$title" -- bash -c "$*; echo '[$title] 已退出 (code='\$?')'" \
      >/dev/null 2>&1 || { echo "[WARN] 无法开终端 $title → 请手动跑: $*"; return 1; }
  fi
}

echo "=============================================="
echo " 三阶段融合启动  auto_invite=$AUTO_INVITE  preview=$PREVIEW"
echo "=============================================="

# 预启动: A=Phase0  C=A2D(--defer-devices)  D=phaseE  (8770 门禁由 A2D py 在 READY 时自启, 不预开)
# 默认 headless (--no-preview): 多窗口 CPU 渲染太卡; PREVIEW=1 开预览调试
PHASE0_PREV=""; A2D_PREV=""
if [ "$PREVIEW" != "1" ]; then
  PHASE0_PREV="--no-preview"
  A2D_PREV="--no-preview"
fi
# PYTHONUNBUFFERED=1: 否则 stdout 走 tee(pipe) 变块缓冲 → 终端日志延迟/变少
open_term "Phase0 (8600)" "PYTHONUNBUFFERED=1 $PROJ_PY $PROJECT_DIR/Estun/estun_greeting_phase0.py $PHASE0_PREV > >(tee -a /tmp/phase0_phase0.log) 2>&1"
open_term "A2D-defer (8080)" "PYTHONUNBUFFERED=1 $PROJ_PY $PROJECT_DIR/Orchestrator/estun_handle_cup_sm.py --defer-devices $A2D_PREV > >(tee -a /tmp/phase0_a2d.log) 2>&1"
open_term "phaseE (8700)" "PYTHONUNBUFFERED=1 $PROJ_PY $PROJECT_DIR/Estun/estun_rps_phaseE.py > >(tee -a /tmp/phase0_phaseE.log) 2>&1"

# ---- 等各端口 ready ----
wait_port() {
  local port="$1" title="$2" timeout="${3:-30}"
  for ((i = 0; i < timeout * 2; i++)); do
    if ss -ltn | grep -q ":$port "; then
      echo "[OK] $title :$port ready"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] $title :$port 未就绪 (${timeout}s) — 请检查对应终端日志"
  return 1
}
wait_port 8600 "Phase0"  30 || { echo "[ERROR] Phase0 py 未就绪 — 查看对应终端日志"; exit 1; }
wait_port 8080 "A2D"     30 || { echo "[ERROR] A2D py 未就绪 (标定 D0 失败?) — 查看对应终端日志"; exit 1; }
wait_port 8700 "phaseE"  30 || { echo "[ERROR] phaseE py 未就绪 — 查看对应终端日志"; exit 1; }

echo ""
echo "=============================================="
echo " 全部就绪! 请在示教器上导入并启动:"
echo "   doc/estun_fused_phase_all.crp"
echo " 流程: Phase 0 打招呼 → A2D 递杯 → E 石头剪刀布 (段间各 3s)"
echo " 本终端监控设备交接并自动收尾 (Ctrl+C 可中止)"
echo "=============================================="

# ---- 监控循环: 起 RPS(JSON ready) + 收尾 (phaseE 退出) ----
shutdown_rps() {
  echo "[收尾] phaseE 结束 → 优雅停 RPS 服务 (SIGTERM merged + realtime_voice --rps-game)"
  pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
  pkill -TERM -f 'rps_stream_merged.py' 2>/dev/null || true
  sleep 1
  if ss -ltn | grep -q ':8768 '; then
    echo "[WARN] 8768 仍占用 → fuser -k"
    fuser -k 8768/tcp 2>/dev/null || true
  fi
  echo "[收尾] RPS 服务已停止 (相机/串口已释放)"
}

cleanup() {
  trap - INT TERM EXIT
  echo "[清理] 停止 RPS 服务 (merged + voice) + 释放资源..."
  pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
  pkill -TERM -f 'rps_stream_merged.py' 2>/dev/null || true
  [ -n "$FIFO_HOLDER" ] && kill "$FIFO_HOLDER" 2>/dev/null || true
  rm -f "$HELPER"
  echo "[清理] 完成"
}
trap cleanup INT TERM EXIT

rps_started=0; p0_logged=0
while true; do
  if [ "$rps_started" = "0" ] && [ -f /tmp/a2d_ready ]; then
    echo "[交接] A2D 已进 READY (PhaseD 递杯) → 起 RPS 服务 (voice+merged)..."
    # 方案甲 option(b): A2D READY 时写 /tmp/a2d_ready, 此处才起 RPS 服务.
    # merged 随相机/usb-0:2 启 + 灵巧手 defer(触发才 open, 交接给同事); voice 内建 8770 供递杯确认+RPS游戏.
    open_term "RPS 服务 (8768)" "bash $HELPER"
    rps_started=1
  fi
  if [ "$rps_started" = "1" ] && ! pgrep -f "$PHASEE_PATTERN" >/dev/null 2>&1; then
    # 不能用"8700 端口释放"判断 phaseE 完成: phaseE serve() accept 后立即 close 监听 socket
    # (单连接模式), 8700 在 game 期间就不监听 → 端口释放 ≠ 完成, 会过早停 RPS (2026-08-21 实机).
    # 正确信号 = phaseE 进程退出 (socket 断开后 _handle_session finally 退出进程).
    echo "[交接] phaseE 进程已退出 → 停 RPS, 融合 demo 结束"
    shutdown_rps
    break
  fi
  if [ "$p0_logged" = "0" ] && ! pgrep -f "$PHASE0_PATTERN" >/dev/null 2>&1; then
    echo "[交接] Phase 0 进程已退出; A2D/phaseE 已就绪"
    p0_logged=1
  fi
  sleep 1
done

# ---- 正常收尾 ----
[ -n "$FIFO_HOLDER" ] && kill "$FIFO_HOLDER" 2>/dev/null || true
rm -f "$HELPER"
echo "融合 demo 结束 (全部阶段完成, RPS 服务已释放)"
