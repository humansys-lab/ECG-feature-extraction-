#!/usr/bin/env bash
# Serve the chat web UI for one model profile. The model server from
# start_model.sh may still be loading; the UI shows that state.
#
#   ./webchat/start_ui.sh            # medgemma → :7860
#   ./webchat/start_ui.sh qwen3.8    # qwen3.8  → :7861
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-${CHAT_PROFILE:-medgemma}}"
shift || true
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

export CHAT_PROFILE="$PROFILE"
export CHAT_BASE_URL="${CHAT_BASE_URL:-http://127.0.0.1:$P_PORT/v1}"

HOST="${CHAT_UI_HOST:-127.0.0.1}"
PORT="${CHAT_UI_PORT:-$P_UI_PORT}"

case "$HOST" in
  127.0.0.1|::1|localhost) ;;
  *)
    AUTH_TOKEN="${CHAT_AUTH_TOKEN:-}"
    if (( ${#AUTH_TOKEN} < 16 )); then
      echo "拒绝在 $HOST 上启动未鉴权界面：请设置至少 16 字符的 CHAT_AUTH_TOKEN" >&2
      exit 1
    fi
    ;;
esac

echo "$P_LABEL 聊天界面: http://$HOST:$PORT   （模型服务: $CHAT_BASE_URL）"

exec "$PYTHON" -m uvicorn webchat.server:app --host "$HOST" --port "$PORT" "$@"
