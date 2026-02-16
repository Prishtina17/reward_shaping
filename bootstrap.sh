#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="${ENV_NAME:-marl_reward_shaping}"
RUN_MODE="${RUN_MODE:-smoke}" # smoke | full | none
INSTALL_GFOOTBALL="${INSTALL_GFOOTBALL:-0}"
USE_CUDA="${USE_CUDA:-0}" # 0 = CPU-only (default), 1 = install CUDA runtime via conda
SKIP_SC2="${SKIP_SC2:-0}"
SC2PATH_OPT="${SC2PATH_OPT:-}"
MINICONDA_PREFIX="${MINICONDA_PREFIX:-$HOME/.miniconda3}"
CONDA_EXE="${CONDA_EXE:-}"

usage() {
  cat <<EOF
Использование: $(basename "$0") [опции]

Скрипт "одной командой" для запуска проекта:
  1) При необходимости устанавливает Miniconda (без sudo)
  2) Создает (или использует) conda-окружение
  2) Устанавливает Python-зависимости
  3) Устанавливает StarCraft II + карты SMAC
  4) (Опционально) ставит Google Football
  5) Запускает smoke или полный прогон

Опции:
  --env-name NAME         Имя conda env (по умолчанию: ${ENV_NAME})
  --run-mode MODE         smoke | full | none (по умолчанию: ${RUN_MODE})
  --with-gfootball        Установить Google Football окружение
  --with-cuda             Установить CUDA runtime (большая загрузка). По умолчанию: CPU-only
  --sc2-path PATH         Путь к каталогу StarCraftII (по умолчанию: \$HOME/StarCraftII)
  --skip-sc2              Пропустить установку StarCraftII/SMAC (нужно, чтобы SC2 уже была установлена)
  --miniconda-prefix DIR  Куда ставить Miniconda (по умолчанию: ${MINICONDA_PREFIX})
  -h, --help              Показать справку

Примеры:
  bash bootstrap.sh
  bash bootstrap.sh --run-mode full
  bash bootstrap.sh --run-mode none
  bash bootstrap.sh --env-name pymarl --with-gfootball
  bash bootstrap.sh --sc2-path /path/to/StarCraftII
  bash bootstrap.sh --skip-sc2 --sc2-path /path/to/StarCraftII
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      ENV_NAME="$2"
      shift 2
      ;;
    --run-mode)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      RUN_MODE="$2"
      shift 2
      ;;
    --with-gfootball)
      INSTALL_GFOOTBALL=1
      shift
      ;;
    --with-cuda)
      USE_CUDA=1
      shift
      ;;
    --sc2-path)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      SC2PATH_OPT="$2"
      shift 2
      ;;
    --skip-sc2)
      SKIP_SC2=1
      shift
      ;;
    --miniconda-prefix)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
      MINICONDA_PREFIX="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ "${RUN_MODE}" != "smoke" && "${RUN_MODE}" != "full" && "${RUN_MODE}" != "none" ]]; then
  echo "Invalid --run-mode: ${RUN_MODE}. Expected smoke|full|none" >&2
  exit 2
fi

have_cmd() { command -v "$1" >/dev/null 2>&1; }

fetch() {
  # fetch URL to output path
  local url="$1"
  local out="$2"
  if have_cmd wget; then
    wget -O "$out" "$url"
  elif have_cmd curl; then
    curl -L -o "$out" "$url"
  else
    echo "Ошибка: не найдено ни wget, ни curl. Установите один из них и повторите запуск." >&2
    echo "Пример (Ubuntu): sudo apt-get update && sudo apt-get install -y wget unzip" >&2
    return 127
  fi
}

ensure_conda() {
  if [[ -n "${CONDA_EXE}" ]]; then
    return 0
  fi
  if have_cmd conda; then
    CONDA_EXE="$(command -v conda)"
    return 0
  fi

  echo "[bootstrap] Conda не найдена. Ставлю Miniforge (conda-forge) в ${MINICONDA_PREFIX}"

  local os arch installer url cache_dir
  os="$(uname -s)"
  arch="$(uname -m)"
  if [[ "${os}" != "Linux" ]]; then
    echo "Ошибка: авто-установка Miniconda поддерживается только для Linux. Текущая ОС: ${os}" >&2
    return 1
  fi
  case "${arch}" in
    x86_64) installer="Miniforge3-Linux-x86_64.sh" ;;
    aarch64|arm64) installer="Miniforge3-Linux-aarch64.sh" ;;
    *)
      echo "Ошибка: неподдерживаемая архитектура CPU: ${arch}" >&2
      return 1
      ;;
  esac

  url="https://github.com/conda-forge/miniforge/releases/latest/download/${installer}"
  cache_dir="${ROOT_DIR}/.bootstrap_cache"
  mkdir -p "${cache_dir}"

  local installer_path="${cache_dir}/${installer}"
  if [[ ! -f "${installer_path}" ]]; then
    echo "[bootstrap] Скачиваю Miniforge: ${installer}"
    fetch "${url}" "${installer_path}"
  else
    echo "[bootstrap] Miniforge installer уже скачан: ${installer_path}"
  fi

  if [[ ! -x "${installer_path}" ]]; then
    chmod +x "${installer_path}" || true
  fi

  if [[ -d "${MINICONDA_PREFIX}" && ! -x "${MINICONDA_PREFIX}/bin/conda" ]]; then
    echo "Ошибка: каталог установки уже существует, но conda не найдена: ${MINICONDA_PREFIX}" >&2
    echo "Удалите этот каталог или укажите другой путь через --miniconda-prefix." >&2
    return 1
  fi

  if [[ ! -x "${MINICONDA_PREFIX}/bin/conda" ]]; then
    echo "[bootstrap] Устанавливаю Miniforge (тихий режим)"
    # The installer fails if the target prefix directory already exists.
    mkdir -p "$(dirname "${MINICONDA_PREFIX}")"
    bash "${installer_path}" -b -p "${MINICONDA_PREFIX}"
  else
    echo "[bootstrap] Conda уже установлена: ${MINICONDA_PREFIX}"
  fi

  CONDA_EXE="${MINICONDA_PREFIX}/bin/conda"
  if [[ ! -x "${CONDA_EXE}" ]]; then
    echo "Ошибка: conda не появилась после установки Miniconda." >&2
    return 1
  fi
}

ensure_conda

if ! command -v unzip >/dev/null 2>&1; then
  echo "Ошибка: unzip не найден в PATH." >&2
  echo "Пример (Ubuntu): sudo apt-get update && sudo apt-get install -y unzip" >&2
  exit 127
fi

if ! "${CONDA_EXE}" env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  echo "[bootstrap] Создаю conda env: ${ENV_NAME}"
  # Avoid default Anaconda channels (may require ToS acceptance). Use explicit channels only.
  "${CONDA_EXE}" create -n "${ENV_NAME}" -y --override-channels -c conda-forge python=3.8
else
  echo "[bootstrap] Conda env уже существует: ${ENV_NAME}"
fi

echo "[bootstrap] Устанавливаю Python зависимости (может занять 5-20 минут)"
ENV_NAME="${ENV_NAME}" CONDA_EXE="${CONDA_EXE}" USE_CUDA="${USE_CUDA}" bash "${ROOT_DIR}/install_dependencies.sh"

SC2PATH_USED="${SC2PATH_OPT:-$HOME/StarCraftII}"
if [[ "${SKIP_SC2}" == "1" ]]; then
  echo "[bootstrap] Установка StarCraft II пропущена (--skip-sc2)."
  echo "[bootstrap] SC2PATH=${SC2PATH_USED}"
else
  echo "[bootstrap] Устанавливаю StarCraft II + карты SMAC (большая загрузка, 2-10+ ГБ)"
  SC2PATH="${SC2PATH_USED}" bash "${ROOT_DIR}/install_sc2.sh"
fi

if [[ "${INSTALL_GFOOTBALL}" == "1" ]]; then
  echo "[bootstrap] Устанавливаю Google Football зависимости"
  echo "[bootstrap] Важно: install_gfootball.sh использует sudo apt-get и может запросить пароль."
  "${CONDA_EXE}" run -n "${ENV_NAME}" bash "${ROOT_DIR}/install_gfootball.sh"
fi

case "${RUN_MODE}" in
  smoke)
    echo "[bootstrap] Запускаю smoke-эксперимент (короткий прогон, чтобы проверить установку)"
    SC2PATH="${SC2PATH_USED}" "${CONDA_EXE}" run -n "${ENV_NAME}" python "${ROOT_DIR}/src/main.py" \
      --config=qmix --env-config=sc2 \
      with env_args.map_name=2m_vs_1z seed=42 t_max=20000 test_nepisode=4 save_model=False
    ;;
  full)
    echo "[bootstrap] Запускаю полный дипломный прогон (долго)"
    SC2PATH="${SC2PATH_USED}" "${CONDA_EXE}" run -n "${ENV_NAME}" bash "${ROOT_DIR}/run_all_shapings.sh"
    ;;
  none)
    echo "[bootstrap] Установка завершена. Эксперименты запускайте вручную."
    ;;
esac

echo "[bootstrap] Готово."
