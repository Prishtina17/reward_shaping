#!/usr/bin/env bash

set -euo pipefail

# =============================================================================
# Experiment protocol from Диплом §3.1
# Algorithm: QMIX (qmix.yaml already has runner=parallel, batch_size_run=8, etc.)
# Maps: all except 6h_vs_8z
# Seed: fixed for reproducibility
# =============================================================================

ALG_CONFIG="qmix"
SEED=42

# --- Per-map epsilon_anneal_time (Диплом Table 40) ---
# Format: "map_name:epsilon_anneal_time"
DEFAULT_MAP_CONFIGS=(
  "2m_vs_1z:100000"
  "3s_vs_3z:200000"
  "3s_vs_4z:250000"
  "3s_vs_5z:300000"
)

# --- Diploma §3.1 / Table 41: unified experiment parameters ---
DIPLOMA_T_MAX=2050000
DIPLOMA_TEST_INTERVAL=10000
DIPLOMA_LOG_INTERVAL=10000
DIPLOMA_RUNNER_LOG_INTERVAL=10000
DIPLOMA_LEARNER_LOG_INTERVAL=10000
DIPLOMA_TEST_NEPISODE=32

cleanup_sc2() {
  pkill -f SC2_x64 2>/dev/null || true
  pkill -f SC2App 2>/dev/null || true
  pkill -f "SC2.*\\.exe" 2>/dev/null || true
  sleep 2
}

# --- All shaping envs + baseline ---
ENV_CONFIGS=(
  melee_range_control_ab
  melee_range_control_ap
  melee_range_control_asp
  melee_range_control_pb
  melee_range_control_sb
  melee_range_control_sp
  melee_range_control_as
  sc2
)

if (( $# > 0 )); then
  MAP_CONFIGS=("$@")
else
  MAP_CONFIGS=("${DEFAULT_MAP_CONFIGS[@]}")
fi

for MAP_ENTRY in "${MAP_CONFIGS[@]}"; do
  IFS=':' read -r MAP_NAME EPSILON_ANNEAL <<< "${MAP_ENTRY}"
  if [[ -z "${MAP_NAME}" ]]; then
    echo "Skipping empty map entry." >&2
    continue
  fi
  if [[ -z "${EPSILON_ANNEAL}" ]]; then
    EPSILON_ANNEAL=100000
  fi

  # Diploma protocol: t_max, test/log intervals, test_nepisode, fixed seed
  DIPLOMA_ARGS=(
    env_args.map_name="${MAP_NAME}"
    env_args.seed="${SEED}"
    epsilon_anneal_time="${EPSILON_ANNEAL}"
    t_max="${DIPLOMA_T_MAX}"
    test_interval="${DIPLOMA_TEST_INTERVAL}"
    log_interval="${DIPLOMA_LOG_INTERVAL}"
    runner_log_interval="${DIPLOMA_RUNNER_LOG_INTERVAL}"
    learner_log_interval="${DIPLOMA_LEARNER_LOG_INTERVAL}"
    test_nepisode="${DIPLOMA_TEST_NEPISODE}"
    save_model=True
    save_model_interval=50000
  )

  for ENV_CFG in "${ENV_CONFIGS[@]}"; do
    echo "=== Running ${ENV_CFG} on map ${MAP_NAME} (epsilon_anneal=${EPSILON_ANNEAL}, t_max=${DIPLOMA_T_MAX}, seed=${SEED}) ==="
    python src/main.py --config="${ALG_CONFIG}" --env-config="${ENV_CFG}" \
      with "${DIPLOMA_ARGS[@]}"
    cleanup_sc2
  done
done
