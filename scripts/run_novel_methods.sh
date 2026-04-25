#!/usr/bin/env bash
# Run novel-method experiments (NOVEL_METHODS_IMPLEMENTATION.md), 4-layer LoRA base.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -f "${CONDA_PREFIX:-}"/etc/profile.d/conda.sh ]]; then
  # shellcheck source=/dev/null
  source "${CONDA_PREFIX}"/etc/profile.d/conda.sh
fi
if command -v conda &>/dev/null; then
  # shellcheck source=/dev/null
  source "$(conda info --base 2>/dev/null)"/etc/profile.d/conda.sh 2>/dev/null || true
  conda activate fedlora 2>/dev/null || true
fi

SEED="${SEED:-42}"
PY=python3

echo "=== Two-Phase (K in {5,8,10,12}) ==="
for k in 5 8 10 12; do
  $PY scripts/run_experiment.py --config "config/exp_two_phase_k${k}.yaml" --method two_phase --seed "$SEED"
done

echo "=== Reverse adaptive ==="
$PY scripts/run_experiment.py --config config/exp_reverse_adaptive.yaml --method reverse_adaptive --seed "$SEED"

echo "=== Budget adaptive (sweep) ==="
for b in 800 1200 1600 2000; do
  $PY scripts/run_experiment.py --config "config/exp_budget_${b}.yaml" --method budget_adaptive --seed "$SEED"
done

echo "=== Baselines (4-layer) ==="
$PY scripts/run_experiment.py --config config/base_config_4layers.yaml --method flora --seed "$SEED"
$PY scripts/run_experiment.py --config config/base_config_4layers.yaml --method ffa_lora --seed "$SEED"
$PY scripts/run_experiment.py --config config/base_config_4layers.yaml --method fedit --seed "$SEED"

echo "Done. (Curriculum rank: add model-side dynamic rank, then implement aggregator.)"
