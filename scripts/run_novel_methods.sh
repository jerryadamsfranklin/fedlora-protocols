#!/usr/bin/env bash
# Run novel-method experiments (NOVEL_METHODS_IMPLEMENTATION.md), 4-layer LoRA base.
# Each job is numbered RUN N/12 in the log, on disk (run_NN_of_12/), and in run_meta.json.
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
TOTAL=12
N=0

# Usage: run_batch "short description" --config ... --method ...
run_batch() {
  local desc="$1"
  shift
  N=$((N+1))
  echo ""
  echo "################################################################"
  echo "#  RUN ${N}/${TOTAL} — ${desc}"
  echo "################################################################"
  $PY scripts/run_experiment.py "$@" --seed "$SEED" --run-index "$N" --run-total "$TOTAL"
}

echo "Full batch: ${TOTAL} runs (update TOTAL in this script if you add or remove steps)."

echo "=== Two-Phase (K in {5,8,10,12}) ==="
for k in 5 8 10 12; do
  run_batch "Two-phase, K=${k}" --config "config/exp_two_phase_k${k}.yaml" --method two_phase
done

echo "=== Reverse adaptive ==="
run_batch "Reverse adaptive" --config config/exp_reverse_adaptive.yaml --method reverse_adaptive

echo "=== Budget adaptive (sweep) ==="
for b in 800 1200 1600 2000; do
  run_batch "Budget adaptive, ${b} MB" --config "config/exp_budget_${b}.yaml" --method budget_adaptive
done

echo "=== Baselines (4-layer) ==="
run_batch "Baseline FLoRA" --config config/base_config_4layers.yaml --method flora
run_batch "Baseline FFA-LoRA" --config config/base_config_4layers.yaml --method ffa_lora
run_batch "Baseline FedIT" --config config/base_config_4layers.yaml --method fedit

echo "Done. (Curriculum rank: add model-side dynamic rank, then implement aggregator.)"
