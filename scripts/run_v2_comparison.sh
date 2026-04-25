#!/usr/bin/env bash
# Compare FedLoRA-Adaptive v2 against baselines (sequential).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-fedlora}"
LOG_DIR="${LOG_DIR:-logs}"
CONFIG="${CONFIG:-config/exp_adaptive_v2.yaml}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR"

echo "========================================"
echo "FedLoRA-Adaptive v2 — main comparison"
echo "Started: $(date)"
echo "Repo: $ROOT"
echo "Config: $CONFIG"
echo "Seed: $SEED"
echo "========================================"

if command -v conda >/dev/null 2>&1; then
  # shellcheck source=/dev/null
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
fi

run() {
  local label=$1
  shift
  echo ""
  echo ">>> $label"
  "$@" 2>&1 | tee "$LOG_DIR/$(echo "$label" | tr ' /' '__').log"
}

for method in fedit ffa_lora flora flexlora fedlora_adaptive fedlora_adaptive_v2; do
  run "v2_compare_${method}" python scripts/run_experiment.py \
    --config "$CONFIG" --method "$method" --seed "$SEED"
done

echo ""
echo "========================================"
echo "Comparison finished: $(date)"
echo "Logs: $LOG_DIR"
echo "========================================"

