#!/usr/bin/env bash
set -euo pipefail

# Launch TensorBoard for this project.
# Default logdir is the one used by runners: results/tb_logs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$SCRIPT_DIR"
if [[ ! -d "$ROOT_DIR/results" ]]; then
  PARENT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
  if [[ -d "$PARENT_DIR/results" ]]; then
    ROOT_DIR="$PARENT_DIR"
  fi
fi
LOGDIR_DEFAULT="$ROOT_DIR/results/tb_logs"

HOST="0.0.0.0"
PORT="6006"
LOGDIR="$LOGDIR_DEFAULT"
PURGE="false"

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Options:
  -l, --logdir DIR   TensorBoard logdir (default: $LOGDIR_DEFAULT)
  -p, --port PORT    Port to bind (default: $PORT)
      --host HOST    Host to bind (default: $HOST)
      --purge        Enable --purge_orphaned_data true
  -h, --help         Show this help

Examples:
  $(basename "$0")
  $(basename "$0") -p 7007
  $(basename "$0") -l /path/to/logs --host 127.0.0.1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -l|--logdir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      LOGDIR="$2"; shift 2;;
    -p|--port)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      PORT="$2"; shift 2;;
    --host)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      HOST="$2"; shift 2;;
    --purge)
      PURGE="true"; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown argument: $1" >&2
      usage; exit 2;;
  esac
done

mkdir -p "$LOGDIR"

if ! command -v tensorboard >/dev/null 2>&1; then
  echo "Error: 'tensorboard' not found in PATH." >&2
  echo "Install it, e.g.: pip install tensorboard" >&2
  exit 127
fi

EXTRA_ARGS=()
if [[ "$PURGE" == "true" ]]; then
  EXTRA_ARGS+=(--purge_orphaned_data true)
fi

echo "Starting TensorBoard:"
echo "  URL   : http://$HOST:$PORT"
echo "  Logdir: $LOGDIR"

exec tensorboard --logdir "$LOGDIR" --host "$HOST" --port "$PORT" "${EXTRA_ARGS[@]}"
