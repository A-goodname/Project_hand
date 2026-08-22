#!/bin/bash
# 一键启动 Phase D 递杯流程 (2026-08-21):
#   终端 ①(新窗口): 8770 语音门禁  /data/Speech/realtime_voice.py  (VLA Python)
#   本终端:        cup_sm 状态机  Orchestrator/estun_handle_cup_sm.py  (Proj Python)
#
# cup_sm 为单次执行模式: 一轮 CRP 连接结束(socket 断开)后自动退出.
# 2026-08-22: 门禁终端去掉末尾 exec bash (门禁退出→对应窗口自动关);
#   cup_sm 退出后 kill 8770 门禁 (释放麦克风/JBL, 防接着跑 Phase E RPS 撞
#   "门禁→RPS voice 回声自打断" 同款竞态; 匹配模式对齐 start_phase_all.sh 的 wait_gate_gone).
# 用法: bash start_cup_handover.sh

set -e

VLA_PY="/data/xilu/conda-envs/VLA/bin/python"
PROJ_PY="/home/user/miniconda3/envs/Proj/bin/python"
SPEECH_DIR="/data/Speech"
PROJECT_DIR="/data/VLA/Project_hand"
# 门禁 process 匹配模式 (与 start_phase_all.sh wait_gate_gone / cup_sm 自启 一致)
GATE_PATTERN="realtime_voice.py --config /data/Speech/config.json"

# ---- ① 门禁: 新终端窗口 (独立, 便于看门禁日志); 门禁退出 → 窗口自动关 (无 exec bash) ----
echo "开 8770 门禁窗口..."
gnome-terminal --title="8770 语音门禁" -- bash -c "\
    echo '======== 8770 语音门禁 (realtime_voice) ========'; \
    '$VLA_PY' '$SPEECH_DIR/realtime_voice.py' --config '$SPEECH_DIR/config.json'; \
    echo '门禁已退出 (code='\$?')'"

# ---- ② cup_sm 退出后清理门禁 (trap: 无论 cup_sm 成功/失败都执行) ----
cleanup_gate() {
  echo "清理: 停 8770 门禁并确认退出..."
  # 2026-08-22 (方案A): SIGINT 优雅关停 — realtime_voice 只捕 KeyboardInterrupt, SIGTERM 会跳过
  # finally:audio.close() → pw-cat speaker 子进程孤儿化, 音频图不释放 → 接 RPS 会回声自打断.
  pkill -INT -f "$GATE_PATTERN" 2>/dev/null || true
  for i in $(seq 1 20); do
    pgrep -f "$GATE_PATTERN" >/dev/null 2>&1 || break
    sleep 0.5
  done
  if pgrep -f "$GATE_PATTERN" >/dev/null 2>&1; then
    echo "[WARN] 门禁 10s 内未响应 SIGINT → SIGTERM 兜底"
    pkill -TERM -f "$GATE_PATTERN" 2>/dev/null || true
    sleep 1
    if pgrep -f "$GATE_PATTERN" >/dev/null 2>&1; then
      echo "[WARN] 门禁仍未退出 → pkill -KILL 兜底"
      pkill -KILL -f "$GATE_PATTERN" 2>/dev/null || true
    fi
  fi
}
trap cleanup_gate EXIT

# ---- ③ cup_sm: 本终端前台; socket 断开后自动退出 ----
echo "本终端启动 cup_sm (一轮 CRP 结束后自动退出)..."
cd "$PROJECT_DIR"
"$PROJ_PY" "$PROJECT_DIR/Orchestrator/estun_handle_cup_sm.py"
rc=$?

echo "cup_sm 已退出 (code=$rc)"
exit "$rc"
