#!/usr/bin/env bash
# Run revised-protocol experiments (sequential). Logs under logs/.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONDA_ENV="${CONDA_ENV:-fedlora}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Federated LoRA — revised protocol"
echo "Started: $(date)"
echo "Repo: $ROOT"
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

# EXP1 — IID (4 methods)
for method in fedit ffa_lora flora flexlora; do
  run "exp1_${method}" python scripts/run_experiment.py \
    --config config/revised_exp1_iid.yaml --method "$method"
done

# EXP2 — label skew
for method in fedit ffa_lora flora flexlora; do
  run "exp2_${method}" python scripts/run_experiment.py \
    --config config/revised_exp2_noniid_label.yaml --method "$method"
done

# EXP3 — quantity skew
for method in fedit ffa_lora flora flexlora; do
  run "exp3_${method}" python scripts/run_experiment.py \
    --config config/revised_exp3_noniid_qty.yaml --method "$method"
done

# EXP4 — rank sweep (2 × 3)
for method in fedit ffa_lora; do
  for rank in 8 16 32; do
    run "exp4_${method}_r${rank}" python scripts/run_experiment.py \
      --config config/revised_exp4_rank.yaml --method "$method" --lora_r "$rank"
  done
done

# EXP5 — client scaling (2 × 2)
for method in fedit flora; do
  for clients in 5 10; do
    run "exp5_${method}_c${clients}" python scripts/run_experiment.py \
      --config config/revised_exp5_scaling.yaml --method "$method" \
      --num_clients "$clients"
  done
done

echo ""
echo "========================================"
echo "All jobs finished: $(date)"
echo "========================================"
