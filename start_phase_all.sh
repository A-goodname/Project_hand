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
RPS_SETTLE="${RPS_SETTLE:-5}"   # A2D 退出后到 RPS 启动的沉降秒数 (用户实测 hand_in_cup→PhaseE 0s 即可起; 留 5s 余量, 可 env 覆盖)
# 各阶段 py 进程匹配模式 (pgrep -f 锚定真实命令行, 用于"等进程退出"交接判断)
A2D_PATTERN="^${PROJ_PY} .*estun_handle_cup_sm\.py"
PHASE0_PATTERN="^${PROJ_PY} .*estun_greeting_phase0\.py"
PHASEE_PATTERN="^${PROJ_PY} .*estun_rps_phaseE\.py"

# ---- 清理残留 (端口 + RPS 进程 + 临时文件) ----
echo "[清理] 检查残留端口/进程..."
for p in 8600 8080 8700 8768 8770; do
  if ss -ltn | grep -q ":$p "; then
    echo "[WARN] 端口 $p 被占用 → fuser -k $p/tcp"
    fuser -k "$p/tcp" 2>/dev/null || true
  fi
done
pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
pkill -TERM -f 'rps_stream.py' 2>/dev/null || true
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
RPS_SCRIPT=/data/xilu/rps_stream.py
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

# RPS 视觉服务 (8768) — 启动确认 + 失败重试 (review D 点, 2026-08-21):
#   启动失败(进程早退/8768 未监听) → 清理 → 等 10s (设备沉降) → 重试; 最多 RPS_MAX_ATTEMPTS 次.
#   兜底 RPS 服务晚起/交接未完全释放; 在 phaseE auto-invite 60s 窗口内足够.
RPS_MAX_ATTEMPTS="${RPS_MAX_ATTEMPTS:-3}"
rps_ready=0
for ((attempt = 1; attempt <= RPS_MAX_ATTEMPTS; attempt++)); do
  echo "[RPS] 启动视觉服务 (第 $attempt/$RPS_MAX_ATTEMPTS 次) rps_stream.py --enable-hand (/dev/video$RPS_CAM_INDEX, 8768)..."
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 RPS_CAMERA_INDEX="$RPS_CAM_INDEX" RPS_WIDTH=640 RPS_HEIGHT=480 RPS_FPS=30 \
    RPS_STREAM_PORT=8768 RPS_STREAM_LABEL="Camera /dev/video$RPS_CAM_INDEX" \
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
open_term() {
  local title="$1"; shift
  gnome-terminal --title="$title" -- bash -c "$*; echo '[$title] 已退出 (code='\$?')'; exec bash" \
    >/dev/null 2>&1 || { echo "[WARN] 无法开终端 $title → 请手动跑: $*"; return 1; }
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
open_term "Phase0 (8600)" "$PROJ_PY $PROJECT_DIR/Estun/estun_greeting_phase0.py $PHASE0_PREV"
open_term "A2D-defer (8080)" "$PROJ_PY $PROJECT_DIR/Orchestrator/estun_handle_cup_sm.py --defer-devices $A2D_PREV"
open_term "phaseE (8700)" "$PROJ_PY $PROJECT_DIR/Estun/estun_rps_phaseE.py"

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

# ---- 监控循环: 设备交接 (端口释放 = 该段 py 退出 = 设备已释放) ----
# 停 8770 门禁并确认退出 (review F 点, 2026-08-21): 不只用 sleep 1, 轮询确认进程和端口都消失,
# 否则残留门禁占着麦克风/音频, RPS voice 接手时出错.
wait_gate_gone() {
  local timeout="${1:-10}"
  pkill -TERM -f 'realtime_voice.py --config /data/Speech/config.json' 2>/dev/null || true
  for ((i = 0; i < timeout * 2; i++)); do
    if ! pgrep -f 'realtime_voice.py --config /data/Speech/config.json' >/dev/null 2>&1 \
       && ! ss -ltn | grep -q ':8770 '; then
      echo "[门禁] 8770 已释放 (进程+端口均退出)"
      return 0
    fi
    sleep 0.5
  done
  echo "[WARN] 门禁 ${timeout}s 内未退出 → fuser -k 8770 兜底"
  fuser -k 8770/tcp 2>/dev/null || true
  return 0
}

shutdown_rps() {
  echo "[收尾] phaseE 结束 → 优雅停 RPS 服务 (SIGTERM rps_stream + realtime_voice --rps-game)"
  pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
  pkill -TERM -f 'rps_stream.py' 2>/dev/null || true
  sleep 1
  if ss -ltn | grep -q ':8768 '; then
    echo "[WARN] 8768 仍占用 → fuser -k"
    fuser -k 8768/tcp 2>/dev/null || true
  fi
  wait_gate_gone 10
  echo "[收尾] RPS 服务已停止 (相机/串口已释放)"
}

cleanup() {
  trap - INT TERM EXIT
  echo "[清理] 停止 RPS 服务 + 释放资源..."
  pkill -TERM -f 'realtime_voice.py --rps-game' 2>/dev/null || true
  pkill -TERM -f 'rps_stream.py' 2>/dev/null || true
  wait_gate_gone 5
  [ -n "$FIFO_HOLDER" ] && kill "$FIFO_HOLDER" 2>/dev/null || true
  rm -f "$HELPER"
  echo "[清理] 完成"
}
trap cleanup INT TERM EXIT

rps_started=0; p0_logged=0
while true; do
  if [ "$rps_started" = "0" ] && ! ss -ltn | grep -q ':8080 '; then
    echo "[交接] A2D socket 已关 (8080 释放) → 等进程完全退出释放 D435i/CameraB/串口..."
    # stop() 先关 8080 监听, 相机/串口释放在后; 必须等 A2D 进程退出 (fd 全释放) 再起 RPS,
    # 否则 RPS 的 MediaPipe 初始化会因设备未释放而崩溃 (2026-08-21 实机定位)
    # 硬门禁 (2026-08-21): 等进程退出超时必须失败退出, 禁止空转后继续起 RPS —
    #   用户实测 hand_in_cup.py→PhaseE 0s 即可起 RPS, 崩的根因是"进程没退干净仍起 RPS"而非沉降;
    #   若 A2D 进程卡在 teardown (disable_hand/门禁/MediaPipe CUDA 释放), 直接报错, 不硬起.
    A2D_EXITED=0
    for ((i = 0; i < 60; i++)); do
      if ! pgrep -f "^/home/user/miniconda3/envs/Proj/bin/python .*estun_handle_cup_sm\.py" >/dev/null 2>&1; then
        A2D_EXITED=1
        break
      fi
      sleep 0.5
    done
    if [ "$A2D_EXITED" != "1" ]; then
      echo "[ERROR] A2D 进程 30s 内未退出 (可能卡在 stop() teardown) → 禁止启动 RPS"
      echo "[ERROR] 请检查 A2D 终端日志 (disable_hand / 门禁 terminate / MediaPipe CUDA 释放)"
      exit 1
    fi
    echo "[交接] A2D 已退出 → 沉降 ${RPS_SETTLE}s (余量, 实测 0s 即可起 RPS)..."
    sleep "$RPS_SETTLE"
    echo "[交接] 沉降完成 → 起 RPS 服务..."
    # 兜底停门禁 (A2D py stop() 已 terminate 自启的 8770 门禁; 残留则此处清, 让麦克风给 RPS voice)
    wait_gate_gone 20
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
