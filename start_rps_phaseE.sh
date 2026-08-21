#!/bin/bash
# 一键启动 Phase E RPS 游戏 (2026-08-21):
#   终端①(新窗口): RPS 服务 (rps_stream.py --enable-hand + realtime_voice.py --rps-game, VLA Python)
#      - realtime_voice stdout → 终端 + /tmp/rps_voice_phaseE.log (phaseE py tail 它等 [RPS][RESULT])
#      - auto_invite=1 → 语音 stdin 接 FIFO, phaseE py 写回车触发邀请; 否则人工在 RPS 终端按回车
#   本终端:        estun_rps_phaseE.py (Proj Python, 等机器人连接, 一轮 socket 断开后自动退出)
# 用法: bash start_rps_phaseE.sh
# 前提: 机器人端 CRP 已导入 (open→P1→While→Home→close, 端口 8700); RPS 相机 (Camera B = usb-0:2) 空闲
#
# 安全: RPS 服务用 --enable-hand 会驱动真实灵巧手; 首次验证先改 --hand-dry-run.

set -e

PROJECT_DIR="/data/VLA/Project_hand"
PROJ_PY="/home/user/miniconda3/envs/Proj/bin/python"
VOICE_LOG="/tmp/rps_voice_phaseE.log"
FIFO="/tmp/rps_voice_stdin"
HELPER="/tmp/rps_phaseE_service.sh"

# ---- 读 config auto_invite.enabled (主脚本与 helper 各自读同一 config, 保证一致) ----
# 注意: 用 Proj python (系统 python3 无 yaml)
read_auto_invite() {
  "$PROJ_PY" -c "
import yaml
cfg = yaml.safe_load(open('$PROJECT_DIR/Estun/estun_config.yaml', encoding='utf-8'))
print('1' if cfg['estun'].get('rps', {}).get('auto_invite', {}).get('enabled', False) else '0')
" 2>/dev/null || echo 0
}
AUTO_INVITE=$(read_auto_invite)

echo "=============================================="
echo " Phase E RPS 一键启动  auto_invite=$AUTO_INVITE"
echo "=============================================="

# ---- RPS 相机: 按 EyeToHand 方式 (sysfs 物理路径+名称) 解析稳定节点, 不硬编码 /dev/videoN ----
# Camera B (Phase A2D 同相机) = usb-0:2 / UGREEN Camera; 节点可能随插拔漂移, 用 find_camera_id 解析
RPS_CAM_PATH="${RPS_CAM_PATH:-usb-0:2}"
RPS_CAM_NAME="${RPS_CAM_NAME:-UGREEN Camera}"
# find_camera_id 的 V4L2 清单打印走 stdout, 用 tail -1 只取最后一行 index
RPS_CAM_INDEX=$("$PROJ_PY" -c "
import sys; sys.path.insert(0, '$PROJECT_DIR')
from EyeToHand.camera import find_camera_id
idx = find_camera_id(keywords=['$RPS_CAM_NAME'], path_keyword='$RPS_CAM_PATH')
print(idx if idx is not None else '')
" 2>/dev/null | tail -1)
if [ -z "$RPS_CAM_INDEX" ]; then
  echo "[WARN] 按 $RPS_CAM_PATH/$RPS_CAM_NAME 找不到 RPS 相机 → 回退 RPS_CAMERA_INDEX=6"
  RPS_CAM_INDEX=6
fi
export RPS_CAM_INDEX
echo "[START] RPS 相机: $RPS_CAM_PATH + '$RPS_CAM_NAME' → /dev/video$RPS_CAM_INDEX"

# ---- 语音播报修复 (2026-08-21): pw-cat 显式 --target bluez 不出声 → 改走 default sink (JBL) ----
# 不改 wangxilu 的 config.json; 复制一份临时 config 清空 realtime.pipewire_target, 用 --config 传语音进程
VOICE_CONFIG="/tmp/rps_config_phaseE.json"
"$PROJ_PY" -c "
import json
d = json.load(open('/data/Speech/config.json', encoding='utf-8'))
d.get('realtime', {}).pop('pipewire_target', None)   # 清空 → pw-cat 用 default sink
json.dump(d, open('$VOICE_CONFIG', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
print('语音 config → $VOICE_CONFIG (pipewire_target 已清空)')
"
# 确保默认 sink = JBL (pw-cat 无 --target 时走它)
XDG_RUNTIME_DIR=/run/user/1000 pactl set-default-sink bluez_output.78_44_05_FA_12_B9.1 2>/dev/null \
  || echo "[WARN] 设默认 sink=JBL 失败 (JBL 未连?) — 语音可能走别的输出"
export VOICE_CONFIG

# ---- 清理残留: 旧日志 / 残留 RPS 进程(8768) ----
: > "$VOICE_LOG" 2>/dev/null || true
if ss -ltn | grep -q ':8768 '; then
  echo "[WARN] 8768 被占用 → 停掉残留 RPS 服务 (fuser -k 8768/tcp)..."
  fuser -k 8768/tcp 2>/dev/null || true
  sleep 0.5
fi

# ---- 生成 RPS 服务 helper (gnome-terminal 内运行, 避免复杂引号) ----
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

cleanup() { trap - INT TERM EXIT; [[ -n "$RPS_PID" ]] && kill -TERM "$RPS_PID" 2>/dev/null || true; }
trap cleanup INT TERM EXIT

echo "[RPS] 启动视觉服务 rps_stream.py --enable-hand (/dev/video$RPS_CAM_INDEX, 8768)..."
sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
  XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
  PYTHONDONTWRITEBYTECODE=1 RPS_CAMERA_INDEX="$RPS_CAM_INDEX" RPS_WIDTH=640 RPS_HEIGHT=480 RPS_FPS=30 \
  RPS_STREAM_PORT=8768 RPS_STREAM_LABEL="Camera /dev/video$RPS_CAM_INDEX" \
  "$PYTHON" "$RPS_SCRIPT" --enable-hand > >(tee -a "$RPS_LOG") 2>&1 &
RPS_PID=$!

for ((i = 0; i < 80; i++)); do
  if curl -fs --max-time 1 http://127.0.0.1:8768/round/status >/dev/null 2>&1; then
    echo "[RPS] 视觉服务就绪 (8768)"
    break
  fi
  if ! kill -0 "$RPS_PID" 2>/dev/null; then
    echo "[RPS] 视觉服务启动失败:" >&2
    tail -n 40 "$RPS_LOG" >&2 || true
    exit 1
  fi
  sleep 0.25
done

# RPS 流无本地窗口 (headless), 画面走网页: 服务就绪后自动开浏览器 (可 RPS_OPEN_BROWSER=0 关)
if [ "${RPS_OPEN_BROWSER:-1}" = "1" ]; then
  echo "[RPS] 打开画面: http://127.0.0.1:8768/ (本机浏览器)"
  xdg-open http://127.0.0.1:8768/ >/dev/null 2>&1 || true
fi

echo "[RPS] 启动语音服务 realtime_voice.py --rps-game (stdout→终端+日志)..."
if [ "$AUTO_INVITE" = "1" ]; then
  echo "[RPS] 语音 stdin=FIFO (phaseE 自动邀请); 终端回车将不可用"
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$VOICE_SCRIPT" --rps-game --config "$VOICE_CONFIG" < "$FIFO" 2>&1 | tee -a "$VOICE_LOG"
else
  echo "[RPS] 语音 stdin=终端 (人工按回车邀请)"
  sudo -u user env -u PIPEWIRE_REMOTE -u PIPEWIRE_RUNTIME_DIR -u PULSE_SERVER -u PULSE_RUNTIME_PATH \
    XDG_RUNTIME_DIR=/run/user/1000 DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/1000/bus \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    "$PYTHON" "$VOICE_SCRIPT" --rps-game --config "$VOICE_CONFIG" 2>&1 | tee -a "$VOICE_LOG"
fi
status=$?
echo "[RPS] 语音服务退出 (code=$status)"
exit "$status"
EOS
chmod +x "$HELPER"

# ---- auto_invite: 建 FIFO + 后台持写端 (防语音端读到 EOF 停读 add_reader) ----
FIFO_HOLDER=""
if [ "$AUTO_INVITE" = "1" ]; then
  rm -f "$FIFO"; mkfifo "$FIFO"
  tail -f /dev/null > "$FIFO" &
  FIFO_HOLDER=$!
  echo "[START] FIFO $FIFO 已建 (持写端 pid=$FIFO_HOLDER)"
fi

# ---- 终端①: RPS 服务 (新窗口) ----
echo "开 RPS 服务窗口..."
gnome-terminal --title="RPS 语音服务 (8768)" -- bash -c "'$HELPER'; echo 'RPS 服务已退出 (code='\$?')'; exec bash" \
  || echo "[WARN] 无法打开 gnome-terminal → RPS 服务未启动, 请手动跑 $HELPER"

# ---- 本终端: phaseE py (等机器人连接, 一轮后自动退出) ----
echo "本终端启动 phaseE (一轮 RPS 后自动退出)..."
cd "$PROJECT_DIR"
set +e
"$PROJ_PY" "$PROJECT_DIR/Estun/estun_rps_phaseE.py"
rc=$?
set -e

# ---- 清理 ----
if [ -n "$FIFO_HOLDER" ]; then kill "$FIFO_HOLDER" 2>/dev/null || true; fi
rm -f "$HELPER"
echo "phaseE 已退出 (code=$rc)"
exit "$rc"
