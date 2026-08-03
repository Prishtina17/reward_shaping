#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${SESSION_NAME:-reward_shaping_final}"
RESULTS_ROOT="${RESULTS_ROOT:-results/full_3a0dc33_sacred_fix}"
RUNTIME_DIR="${ROOT_DIR}/${RESULTS_ROOT}/runtime"
LOG_PATH="${RUNTIME_DIR}/matrix.log"
PYTHON_BIN="${PYTHON_BIN:-/home/anton/miniconda3/envs/marl_sc2/bin/python}"
SC2PATH="${SC2PATH:-/home/anton/StarCraftII}"
PROTOCOL_REVISION="${PROTOCOL_REVISION:-}"

if [[ -z "${PROTOCOL_REVISION}" && -f "${ROOT_DIR}/${RESULTS_ROOT}/protocol_revision.txt" ]]; then
  PROTOCOL_REVISION="$(<"${ROOT_DIR}/${RESULTS_ROOT}/protocol_revision.txt")"
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "Experiment session '${SESSION_NAME}' is already running."
  exit 0
fi

mkdir -p "${RUNTIME_DIR}"
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${ROOT_DIR}' && exec env PYTHONUNBUFFERED=1 SC2PATH='${SC2PATH}' PYTHON_BIN='${PYTHON_BIN}' RESULTS_ROOT='${RESULTS_ROOT}' PROTOCOL_REVISION='${PROTOCOL_REVISION}' ALLOW_DIRTY_RUN='${ALLOW_DIRTY_RUN:-1}' bash run_all_shapings.sh >> '${LOG_PATH}' 2>&1"

echo "Started '${SESSION_NAME}'. Log: ${LOG_PATH}"
