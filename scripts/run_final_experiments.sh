#!/usr/bin/env bash
# Final experiments for Two-Phase paper (IID multi-seed + non-IID Dirichlet alpha=0.5).
# See CURSOR_FINAL_EXPERIMENTS.md / README.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PYTHON:-python3}"

SEEDS="42 123 456"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${ROOT}/logs/final_experiments_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "FINAL EXPERIMENTS FOR TWO-PHASE PAPER"
echo "Started: $(date)"
echo "Log directory: $LOG_DIR"
echo "=============================================="

TOTAL_RUNS=18
CURRENT_RUN=0

run_experiment() {
  local config="$1"
  local seed="$2"
  local run_idx="$3"

  echo ""
  echo "[$run_idx/$TOTAL_RUNS] Running: ${config}.yaml seed=${seed}"
  echo "Time: $(date)"

  "${PY}" scripts/run_experiment.py \
    --config "config/${config}.yaml" \
    --seed "${seed}" \
    --run-index "${run_idx}" \
    --run-total "${TOTAL_RUNS}" \
    2>&1 | tee "${LOG_DIR}/${config}_seed${seed}.log"

  echo "Completed: ${config} seed=${seed}"
  echo ""
}

echo ""
echo "=== SECTION A: IID Multi-Seed Replication ==="

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_two_phase_k8" "${seed}" "${CURRENT_RUN}"
done

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_two_phase_k10" "${seed}" "${CURRENT_RUN}"
done

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_flora_iid" "${seed}" "${CURRENT_RUN}"
done

echo ""
echo "=== SECTION B: Non-IID Robustness (Dirichlet alpha=0.5, label_skew proxy) ==="

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_two_phase_k8_noniid_alpha05" "${seed}" "${CURRENT_RUN}"
done

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_two_phase_k10_noniid_alpha05" "${seed}" "${CURRENT_RUN}"
done

for seed in ${SEEDS}; do
  CURRENT_RUN=$((CURRENT_RUN + 1))
  run_experiment "exp_flora_noniid_alpha05" "${seed}" "${CURRENT_RUN}"
done

echo ""
echo "=============================================="
echo "ALL EXPERIMENTS COMPLETED"
echo "Finished: $(date)"
echo "Total runs: ${TOTAL_RUNS}"
echo "Logs saved to: $LOG_DIR"
echo "=============================================="

echo ""
echo "Generating results summary..."
"${PY}" scripts/analyze_final_results.py --results-dir "${ROOT}/results/raw" \
  || echo "analyze_final_results.py failed or produced no rows (check results/)."
