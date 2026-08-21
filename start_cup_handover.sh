#!/bin/bash
# 一键启动 Phase D 递杯流程 (2026-08-21):
#   终端 ①(新窗口): 8770 语音门禁  /data/Speech/realtime_voice.py  (VLA Python)
#   本终端:        cup_sm 状态机  Orchestrator/estun_handle_cup_sm.py  (Proj Python)
#
# cup_sm 为单次执行模式: 一轮 CRP 连接结束(socket 断开)后自动退出.
# 用法: bash start_cup_handover.sh

set -e

VLA_PY="/data/xilu/conda-envs/VLA/bin/python"
PROJ_PY="/home/user/miniconda3/envs/Proj/bin/python"
SPEECH_DIR="/data/Speech"
PROJECT_DIR="/data/VLA/Project_hand"

# ---- ① 门禁: 新终端窗口 (独立, 便于看门禁日志) ----
echo "开 8770 门禁窗口..."
gnome-terminal --title="8770 语音门禁" -- bash -c "\
    echo '======== 8770 语音门禁 (realtime_voice) ========'; \
    '$VLA_PY' '$SPEECH_DIR/realtime_voice.py' --config '$SPEECH_DIR/config.json'; \
    echo '门禁已退出 (code='\$?')'; exec bash"

# ---- ② cup_sm: 本终端前台; socket 断开后自动退出 ----
echo "本终端启动 cup_sm (一轮 CRP 结束后自动退出)..."
cd "$PROJECT_DIR"
"$PROJ_PY" "$PROJECT_DIR/Orchestrator/estun_handle_cup_sm.py"
rc=$?

echo "cup_sm 已退出 (code=$rc)"
exit "$rc"
