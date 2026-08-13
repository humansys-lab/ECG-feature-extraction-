#!/usr/bin/env bash
# Serve one local checkpoint over an OpenAI-compatible HTTP API.
#
#   ./webchat/start_model.sh                 # medgemma (default profile)
#   ./webchat/start_model.sh qwen3.6
#   CHAT_GPUS=1,2 CHAT_TP=2 ./webchat/start_model.sh qwen3.6
#
# Per-model defaults (path, port, GPU, context length) live in models.json;
# any CHAT_* environment variable overrides the profile.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PROFILE="${1:-${CHAT_PROFILE:-medgemma}}"
PYTHON="${CHAT_PYTHON:-$REPO_ROOT/.venv/bin/python}"

# Read the profile once, as shell assignments.
eval "$("$PYTHON" - "$PROFILE" <<'PY'
import json, shlex, sys
from pathlib import Path

name = sys.argv[1]
profiles = json.loads(Path("webchat/models.json").read_text(encoding="utf-8"))
if name not in profiles:
    sys.exit(f"echo '未知模型档案: {name}（可用: {', '.join(sorted(profiles))}）' >&2; exit 1")
p = profiles[name]
for key, value in {
    "P_LABEL": p["label"],
    "P_PATH": p["path"],
    "P_SERVED": p["served_name"],
    "P_PORT": p["port"],
    "P_GPUS": p["gpus"],
    "P_TP": p["tensor_parallel"],
    "P_MAXLEN": p["max_len"],
    "P_UTIL": p["gpu_util"],
    "P_REASONING": p.get("reasoning_parser") or "",
}.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
)"

MODEL_PATH="${CHAT_MODEL_PATH:-$P_PATH}"
SERVED_NAME="${CHAT_SERVED_NAME:-$P_SERVED}"
HOST="${CHAT_HOST:-127.0.0.1}"
PORT="${CHAT_PORT:-$P_PORT}"
MAX_LEN="${CHAT_MAX_LEN:-$P_MAXLEN}"
TP="${CHAT_TP:-$P_TP}"
GPU_UTIL="${CHAT_GPU_UTIL:-$P_UTIL}"
REASONING="${CHAT_REASONING_PARSER:-$P_REASONING}"

export CUDA_VISIBLE_DEVICES="${CHAT_GPUS:-$P_GPUS}"
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

if [[ ! -f "$MODEL_PATH/config.json" ]]; then
  echo "找不到模型：$MODEL_PATH/config.json" >&2
  exit 1
fi

echo "$P_LABEL 模型服务"
echo "  档案       : $PROFILE"
echo "  权重       : $MODEL_PATH"
echo "  GPU        : CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (tensor-parallel=$TP)"
echo "  上下文长度 : $MAX_LEN"
echo "  监听       : http://$HOST:$PORT/v1"
[[ -n "$REASONING" ]] && echo "  思考过程   : 启用（reasoning-parser=$REASONING）"
echo "  首次加载权重通常需要几分钟，出现 'Application startup complete' 后即可对话。"

ARGS=(
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_NAME"
  --host "$HOST"
  --port "$PORT"
  --dtype bfloat16
  --max-model-len "$MAX_LEN"
  --tensor-parallel-size "$TP"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching
  --no-enable-log-requests
)
[[ -n "$REASONING" ]] && ARGS+=(--reasoning-parser "$REASONING")

exec "$PYTHON" -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
