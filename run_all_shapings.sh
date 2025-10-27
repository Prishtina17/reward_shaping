#!/usr/bin/env bash

set -euo pipefail

MAP_NAME="${1:-3s_vs_3z}"
ALG_CONFIG="qmix"

ENV_CONFIGS=(
  melee_range_control_ab
  melee_range_control_ap
  melee_range_control_asp
  melee_range_control_pb
  melee_range_control_sb
  melee_range_control_sp
  sc2
)

for ENV_CFG in "${ENV_CONFIGS[@]}"; do
  echo "=== Running ${ENV_CFG} on map ${MAP_NAME} ==="
  python src/main.py --config="${ALG_CONFIG}" --env-config="${ENV_CFG}" with env_args.map_name="${MAP_NAME}"
done
