#!/usr/bin/env bash
# Install PyTorch and Python packages into a conda environment.
#
# This script is intentionally runnable from a "fresh shell" without an activated env:
# it targets an env by name via `conda -n <env> ...` and runs pip inside that env.

set -euo pipefail

CONDA_EXE="${CONDA_EXE:-}"
ENV_NAME="${ENV_NAME:-${1:-}}"
USE_CUDA="${USE_CUDA:-0}"  # 0 = CPU-only (default), 1 = install CUDA runtime via conda

if [[ -z "${CONDA_EXE}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    CONDA_EXE="$(command -v conda)"
  else
    echo "Ошибка: conda не найдена. Установите conda/Miniconda и повторите запуск." >&2
    exit 127
  fi
fi

if [[ -z "${ENV_NAME}" ]]; then
  if [[ -n "${CONDA_DEFAULT_ENV:-}" && "${CONDA_DEFAULT_ENV}" != "base" ]]; then
    ENV_NAME="${CONDA_DEFAULT_ENV}"
  else
    echo "Ошибка: не указано имя окружения." >&2
    echo "Запуск: ENV_NAME=<имя> bash install_dependencies.sh" >&2
    echo "   или: bash install_dependencies.sh <имя>" >&2
    exit 2
  fi
fi

echo "[deps] Установка conda-пакетов (PyTorch) в env: ${ENV_NAME}"
# Use pip wheels for PyTorch to avoid channel/ABI issues when installing via conda-forge-only.
echo "[deps] Установка PyTorch через pip (USE_CUDA=${USE_CUDA})"
if [[ "${USE_CUDA}" == "1" ]]; then
  "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
    "torch==1.12.1+cu113" "torchvision==0.13.1+cu113" "torchaudio==0.12.1" \
    --extra-index-url https://download.pytorch.org/whl/cu113
else
  "${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
    "torch==1.12.1+cpu" "torchvision==0.13.1+cpu" "torchaudio==0.12.1" \
    --extra-index-url https://download.pytorch.org/whl/cpu
fi

echo "[deps] Установка pip-пакетов в env: ${ENV_NAME}"
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
  protobuf==3.19.5 sacred==0.7.5 numpy scipy gym==0.11 matplotlib seaborn \
  pyyaml==5.3.1 pygame pytest probscale imageio snakeviz tensorboard-logger pymongo

echo "[deps] Установка SMAC (фиксированный коммит) в env: ${ENV_NAME}"
"${CONDA_EXE}" run -n "${ENV_NAME}" python -m pip install \
  "git+https://github.com/oxwhirl/smac.git@26f4c4e4d1ebeaf42ecc2d0af32fac0774ccc678"
