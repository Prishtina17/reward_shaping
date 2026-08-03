#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-reward_shaping_final}"
RESULTS_ROOT="${RESULTS_ROOT:-results/full_3a0dc33_sacred_fix}"
WAIT_SECONDS="${WAIT_SECONDS:-600}"

if ! tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Experiment session '${SESSION_NAME}' is not running."
  exit 0
fi

runner_pid="$(tmux display-message -p -t "${SESSION_NAME}" '#{pane_pid}')"
python_pid=""
if [[ -n "${runner_pid}" ]]; then
  python_pid="$(pgrep -P "${runner_pid}" -f "python.*src/main.py" | head -n 1 || true)"
fi

if [[ -z "${python_pid}" ]]; then
  echo "No active training process found; sending Ctrl-C to '${SESSION_NAME}'."
  tmux send-keys -t "${SESSION_NAME}" C-c
else
  echo "Requesting graceful checkpoint at PID ${python_pid}."
  kill -INT "${python_pid}"
fi

deadline=$((SECONDS + WAIT_SECONDS))
while tmux has-session -t "${SESSION_NAME}" 2>/dev/null; do
  if (( SECONDS >= deadline )); then
    echo "Checkpoint is still being written; session remains running." >&2
    exit 1
  fi
  sleep 1
done

resume_root="${ROOT_DIR}/${RESULTS_ROOT}/resume"
latest_checkpoint="$(find "${resume_root}" -type f -name checkpoint_complete -printf '%T@ %h\n' 2>/dev/null | sort -nr | head -n 1 | cut -d' ' -f2- || true)"
echo "Stopped cleanly. Latest resume checkpoint: ${latest_checkpoint:-not found}"
