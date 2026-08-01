#!/usr/bin/env bash

set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

# Reproducible final-run protocol. Every value can be overridden from the
# environment, while the checked-in defaults remain the paper configuration.
ALG_CONFIG="${ALG_CONFIG:-qmix}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SEEDS_CSV="${SEEDS_CSV:-42,43,44,45,46}"
ENV_CONFIGS_CSV="${ENV_CONFIGS_CSV:-melee_range_control_ab,melee_range_control_ap,melee_range_control_asp,melee_range_control_pb,melee_range_control_sb,melee_range_control_sp,melee_range_control_as,sc2}"

DIPLOMA_T_MAX="${DIPLOMA_T_MAX:-2050000}"
DIPLOMA_TEST_INTERVAL="${DIPLOMA_TEST_INTERVAL:-10000}"
DIPLOMA_LOG_INTERVAL="${DIPLOMA_LOG_INTERVAL:-10000}"
DIPLOMA_RUNNER_LOG_INTERVAL="${DIPLOMA_RUNNER_LOG_INTERVAL:-10000}"
DIPLOMA_LEARNER_LOG_INTERVAL="${DIPLOMA_LEARNER_LOG_INTERVAL:-10000}"
DIPLOMA_TEST_NEPISODE="${DIPLOMA_TEST_NEPISODE:-32}"
SAVE_MODEL="${SAVE_MODEL:-True}"
SAVE_MODEL_INTERVAL="${SAVE_MODEL_INTERVAL:-${DIPLOMA_T_MAX}}"

RESULTS_ROOT="${RESULTS_ROOT:-results/final_run}"
MANIFEST="${MANIFEST:-${RESULTS_ROOT}/completed_runs.tsv}"
RESUME="${RESUME:-1}"
DRY_RUN="${DRY_RUN:-0}"
MAX_RUNS="${MAX_RUNS:-0}"

# 6h_vs_8z is intentionally excluded: it changes the unit matchup and is too
# expensive for the final home-compute protocol. It can still be passed
# explicitly as a positional map entry when needed.
DEFAULT_MAP_CONFIGS=(
  "2m_vs_1z:100000"
  "3s_vs_3z:200000"
  "3s_vs_4z:250000"
  "3s_vs_5z:300000"
)

cleanup_sc2() {
  pkill -f SC2_x64 2>/dev/null || true
  pkill -f SC2App 2>/dev/null || true
  pkill -f 'SC2.*\.exe' 2>/dev/null || true
}

trap cleanup_sc2 EXIT INT TERM

IFS=',' read -r -a SEED_VALUES <<< "${SEEDS_CSV}"
IFS=',' read -r -a ENV_CONFIGS <<< "${ENV_CONFIGS_CSV}"

if (( $# > 0 )); then
  MAP_CONFIGS=("$@")
else
  MAP_CONFIGS=("${DEFAULT_MAP_CONFIGS[@]}")
fi

if (( ${#SEED_VALUES[@]} == 0 || ${#ENV_CONFIGS[@]} == 0 )); then
  echo "SEEDS_CSV and ENV_CONFIGS_CSV must not be empty." >&2
  exit 2
fi

for seed in "${SEED_VALUES[@]}"; do
  if [[ ! "${seed}" =~ ^[0-9]+$ ]]; then
    echo "Invalid seed '${seed}'. SEEDS_CSV must contain comma-separated integers." >&2
    exit 2
  fi
done

mkdir -p "${RESULTS_ROOT}"
mkdir -p "$(dirname "${MANIFEST}")"
touch "${MANIFEST}"

run_count=0
for map_entry in "${MAP_CONFIGS[@]}"; do
  IFS=':' read -r map_name epsilon_anneal <<< "${map_entry}"
  if [[ -z "${map_name}" || ! "${epsilon_anneal:-}" =~ ^[0-9]+$ ]]; then
    echo "Invalid map entry '${map_entry}'. Expected map_name:epsilon_anneal_time." >&2
    exit 2
  fi

  for seed in "${SEED_VALUES[@]}"; do
    for env_config in "${ENV_CONFIGS[@]}"; do
      printf -v run_key '%s\t%s\t%s\t%s\t%s' \
        "${map_name}" "${env_config}" "${seed}" "${epsilon_anneal}" "${DIPLOMA_T_MAX}"
      if [[ "${RESUME}" == "1" ]] && grep -Fqx "${run_key}" "${MANIFEST}"; then
        echo "[skip] ${map_name} ${env_config} seed=${seed} already completed"
        continue
      fi

      command=(
        "${PYTHON_BIN}" src/main.py
        "--config=${ALG_CONFIG}"
        "--env-config=${env_config}"
        with
        "env_args.map_name=${map_name}"
        "env_args.seed=${seed}"
        "seed=${seed}"
        "epsilon_anneal_time=${epsilon_anneal}"
        "t_max=${DIPLOMA_T_MAX}"
        "test_interval=${DIPLOMA_TEST_INTERVAL}"
        "log_interval=${DIPLOMA_LOG_INTERVAL}"
        "runner_log_interval=${DIPLOMA_RUNNER_LOG_INTERVAL}"
        "learner_log_interval=${DIPLOMA_LEARNER_LOG_INTERVAL}"
        "test_nepisode=${DIPLOMA_TEST_NEPISODE}"
        "training_stop_mode=steps"
        "save_model=${SAVE_MODEL}"
        "save_model_interval=${SAVE_MODEL_INTERVAL}"
        "local_results_path=${RESULTS_ROOT}"
      )

      echo "[run] map=${map_name} env=${env_config} seed=${seed} epsilon_anneal=${epsilon_anneal} t_max=${DIPLOMA_T_MAX}"
      if [[ "${DRY_RUN}" == "1" ]]; then
        printf '  %q' "${command[@]}"
        printf '\n'
      else
        cleanup_sc2
        "${command[@]}"
        printf '%s\n' "${run_key}" >> "${MANIFEST}"
        cleanup_sc2
      fi

      run_count=$((run_count + 1))
      if (( MAX_RUNS > 0 && run_count >= MAX_RUNS )); then
        echo "Stopped after MAX_RUNS=${MAX_RUNS}."
        exit 0
      fi
    done
  done
done

echo "Final-run matrix completed. Manifest: ${MANIFEST}"
