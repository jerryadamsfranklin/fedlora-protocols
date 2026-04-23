#!/usr/bin/env bash
# Compare soft vs hard transition for FedLoRA-Adaptive v2.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-fedlora}"
LOG_DIR="${LOG_DIR:-logs}"
SEED="${SEED:-42}"

mkdir -p "$LOG_DIR"

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

# Soft transition (default)
run "v2_soft_transition" python scripts/run_experiment.py \
  --config config/exp_adaptive_v2.yaml --method fedlora_adaptive_v2 --seed "$SEED"

# Hard transition
run "v2_hard_transition" python scripts/run_experiment.py \
  --config config/ablation_hard_transition.yaml --method fedlora_adaptive_v2 --seed "$SEED"

