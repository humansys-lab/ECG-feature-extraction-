#!/usr/bin/env bash
# One-command startup for a model profile: launch the vLLM server in the
# background, wait for it to finish loading, then run the chat UI in the
# foreground. Ctrl+C stops both.
#
#   ./webchat/start_all.sh            # medgemma → UI :7860, model :8000
#   ./webchat/start_all.sh qwen3.8    # qwen3.8  → UI :7861, model :8001
#
# Different profiles use different ports, so several models can run at once.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-${CHAT_PROFILE:-medgemma}}"
PYTHON="${CHAT_PYTHON:-$REPO_ROOT/.venv/bin/python}"

eval "$("$PYTHON" - "$PROFILE" <<'PY'
import json, shlex, sys
from pathlib import Path

name = sys.argv[1]
profiles = json.loads(Path("webchat/models.json").read_text(encoding="utf-8"))
if name not in profiles:
    sys.exit(f"echo '未知模型档案: {name}（可用: {', '.join(sorted(profiles))}）' >&2; exit 1")
p = profiles[name]
print(f"P_LABEL={shlex.quote(p['label'])}")
print(f"P_PORT={p['port']}")
print(f"P_UI_PORT={p['ui_port']}")
PY
)"

PORT="${CHAT_PORT:-$P_PORT}"
UI_PORT="${CHAT_UI_PORT:-$P_UI_PORT}"
LOG_FILE="${CHAT_LOG:-$REPO_ROOT/webchat/model_server_$PROFILE.log}"
export CHAT_PROFILE="$PROFILE"
export CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:$PORT/v1}"

MODEL_PID=""
cleanup() {
  if [[ -n "$MODEL_PID" ]] && kill -0 "$MODEL_PID" 2>/dev/null; then
    echo "正在停止 $P_LABEL 模型服务 (pid $MODEL_PID)…"
    kill "$MODEL_PID" 2>/dev/null || true
    wait "$MODEL_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "检测到端口 $PORT 上已有模型服务，直接复用。"
else
  echo "启动 $P_LABEL 模型服务，日志: $LOG_FILE"
  "$REPO_ROOT/webchat/start_model.sh" "$PROFILE" >"$LOG_FILE" 2>&1 &
  MODEL_PID=$!

  printf "加载权重中"
  for _ in $(seq 1 240); do   # up to ~20 minutes
    if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo " 就绪"
      break
    fi
    if ! kill -0 "$MODEL_PID" 2>/dev/null; then
      echo
      echo "模型服务启动失败，日志末尾：" >&2
      tail -n 30 "$LOG_FILE" >&2
      exit 1
    fi
    printf "."
    sleep 5
  done
fi

echo
echo "  $P_LABEL 聊天界面: http://localhost:$UI_PORT"
echo
exec "$REPO_ROOT/webchat/start_ui.sh" "$PROFILE"
