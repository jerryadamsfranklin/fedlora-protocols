#!/usr/bin/env bash
# Phase 2 Part 3.3(b): TinyLlama Non-IID α=0.1 baselines (FLoRA + Two-Phase K=8).
# Train 6 runs on CUDA, then full downstream eval (MMLU, ARC-Easy, BoolQ, HellaSwag).
#
# Usage (on RTX 4090, from repo root, venv active):
#   bash scripts/run_part33b_alpha01_baselines.sh train
#   bash scripts/run_part33b_alpha01_baselines.sh eval
#   bash scripts/run_part33b_alpha01_baselines.sh all

set -euo pipefail

TAG="${TAG:-phase1b_alpha01_baselines}"
DEVICE="${DEVICE:-cuda}"
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
BENCH=(mmlu arc_easy boolq hellaswag)

SEEDS=(42 123 456)
declare -a CONFIGS=(
  "config/exp_flora_noniid_alpha01.yaml"
  "config/exp_two_phase_k8_noniid_alpha01.yaml"
)

train_one() {
  local cfg="$1" seed="$2"
  echo "============================================================"
  echo "TRAIN ${cfg} seed=${seed} device=${DEVICE} tag=${TAG}"
  echo "============================================================"
  python3 scripts/run_experiment.py \
    --config "${cfg}" \
    --seed "${seed}" \
    --device "${DEVICE}" \
    --tag "${TAG}" \
    --save-every 10
}

eval_latest() {
  local exp="$1" method="$2" seed="$3"
  local raw_base="results/raw/${exp}/${method}/seed_${seed}/${TAG}"
  local ckpt
  ckpt="$(ls -td "${raw_base}"/*/final_adapter_state.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "[WARN] missing checkpoint under ${raw_base}"
    return 1
  fi
  local run_dir
  run_dir="$(dirname "${ckpt}")"
  local stamp
  stamp="$(basename "${run_dir}")"
  local out_dir="results/downstream/${exp}/${method}/seed_${seed}/${TAG}"
  mkdir -p "${out_dir}"
  local out_file="${out_dir}/downstream_results.json"
  if [[ -f "${out_file}" ]]; then
    echo "[skip eval] ${out_file}"
    return 0
  fi
  echo "============================================================"
  echo "EVAL ${ckpt} -> ${out_file}"
  echo "============================================================"
  python3 scripts/evaluate_checkpoint.py \
    --checkpoint "${ckpt}" \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --benchmarks "${BENCH[@]}" \
    --num-examples "${NUM_EXAMPLES}" \
    --seed 42 \
    --device "${DEVICE}" \
    --output "${out_file}"
  echo "Training run stamp was ${stamp}"
}

do_train() {
  for cfg in "${CONFIGS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      train_one "${cfg}" "${seed}"
    done
  done
}

do_eval() {
  for seed in "${SEEDS[@]}"; do
    eval_latest "exp_flora_noniid_alpha01" "flora" "${seed}"
    eval_latest "exp_two_phase_k8_noniid_alpha01" "two_phase" "${seed}"
  done
}

summarize() {
  python3 - <<'PY'
import json
from pathlib import Path

rows = []
root = Path("results/raw")
tag = "phase1b_alpha01_baselines"
for exp, method in [
    ("exp_flora_noniid_alpha01", "flora"),
    ("exp_two_phase_k8_noniid_alpha01", "two_phase"),
]:
    for seed in (42, 123, 456):
        runs = sorted((root / exp / method / f"seed_{seed}" / tag).glob("*/results.json"))
        if not runs:
            print(f"MISSING train {exp} seed={seed}")
            continue
        r = json.loads(runs[-1].read_text())
        rounds = r if isinstance(r, list) else r["rounds"]
        last = rounds[-1]
        sw = (last.get("aggregator_stats") or {}).get("switch_round")
        first_b_only = next((r["round"] for r in rounds if r.get("broadcast_b_only")), None)
        ds_path = Path("results/downstream") / exp / method / f"seed_{seed}" / tag / "downstream_results.json"
        ds = {}
        if ds_path.is_file():
            bms = json.loads(ds_path.read_text()).get("benchmarks", {})
            ds = {k: bms[k]["accuracy"] for k in ("mmlu", "arc_easy", "boolq", "hellaswag") if k in bms}
        rows.append((exp, method, seed, last["communication_mb"], last["avg_loss"], sw, first_b_only, ds, runs[-1]))

print(f"{'exp':40} {'seed':4} {'comm_mb':>10} {'loss':>8} {'sw':>4} {'bstart':>6}  mmlu  arc  boolq  hella")
for exp, method, seed, comm, loss, sw, bstart, ds, path in rows:
    print(
        f"{exp:40} {seed:4d} {comm:10.3f} {loss:8.4f} {str(sw):>4} {str(bstart):>6}  "
        f"{ds.get('mmlu', float('nan')):5.3f} {ds.get('arc_easy', float('nan')):5.3f} "
        f"{ds.get('boolq', float('nan')):5.3f} {ds.get('hellaswag', float('nan')):5.3f}"
    )
    print(f"  raw={path}")
PY
}

cmd="${1:-all}"
case "${cmd}" in
  train) do_train ;;
  eval) do_eval ;;
  summarize) summarize ;;
  all)
    do_train
    do_eval
    summarize
    python3 scripts/build_results_table.py
    ;;
  *)
    echo "Usage: $0 {train|eval|summarize|all}"
    exit 2
    ;;
esac
