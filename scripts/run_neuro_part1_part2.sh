#!/usr/bin/env bash
# Neurocomputing Final Plan - Part 1 + Part 2 run orchestrator.
# Part 1: Dolly-15k (3k subset) for FLoRA / Two-Phase K=8 / ReverseAdaptive.
# Part 2: Alpaca-3k published baselines FFA-LoRA and FedIT.
#
# Usage:
#   bash scripts/run_neuro_part1_part2.sh train
#   bash scripts/run_neuro_part1_part2.sh holdout
#   bash scripts/run_neuro_part1_part2.sh summarize
#   RUN_MC=1 bash scripts/run_neuro_part1_part2.sh mc_eval
#   RUN_MC=1 bash scripts/run_neuro_part1_part2.sh all

set -euo pipefail

MANIFEST="${MANIFEST:-docs/neuro_part12_run_manifest.csv}"
TAG="${TAG:-neuro_part1_part2_4090}"
DEVICE="${DEVICE:-cuda}"
NUM_EXAMPLES="${NUM_EXAMPLES:-500}"
BENCHMARKS=("${BENCHMARKS[@]:-mmlu arc_easy boolq hellaswag}")

PART1_EXPS=(
  "exp_dolly_flora_iid"
  "exp_dolly_two_phase_k8_iid"
  "exp_dolly_reverse_adaptive_iid"
  "exp_dolly_reverse_adaptive_noniid_alpha05"
)

PART2_EXPS=(
  "exp_ffa_lora_iid"
  "exp_ffa_lora_noniid_alpha05"
  "exp_fedit_iid"
  "exp_fedit_noniid_alpha05"
)

run_train_manifest() {
  python3 - <<'PY'
import csv
import os
import subprocess
import sys

manifest = os.environ["MANIFEST"]
tag = os.environ["TAG"]
device = os.environ["DEVICE"]

with open(manifest, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: int(r["exec_order"]))
total = len(rows)
for i, row in enumerate(rows, start=1):
    cfg = row["config"]
    seed = row["seed"]
    cmd = [
        sys.executable,
        "scripts/run_experiment.py",
        "--config", cfg,
        "--seed", str(seed),
        "--device", device,
        "--tag", tag,
        "--run-index", str(i),
        "--run-total", str(total),
        "--save-every", "10",
    ]
    print("=" * 76)
    print(f"[train {i}/{total}] exec_order={row['exec_order']} exp={row['experiment']} seed={seed}")
    print(" ".join(cmd))
    subprocess.check_call(cmd)
PY
}

eval_latest_checkpoint() {
  local exp="$1"
  local method="$2"
  local seed="$3"
  local raw_base="results/raw/${exp}/${method}/seed_${seed}/${TAG}"
  local ckpt
  ckpt="$(ls -td "${raw_base}"/*/final_adapter_state.pt 2>/dev/null | head -1 || true)"
  if [[ -z "${ckpt}" ]]; then
    echo "[WARN] no checkpoint: ${raw_base}"
    return 1
  fi
  local out_dir="results/downstream/${exp}/${method}/seed_${seed}/${TAG}"
  local out_file="${out_dir}/downstream_results.json"
  mkdir -p "${out_dir}"
  if [[ -f "${out_file}" ]]; then
    echo "[skip] ${out_file}"
    return 0
  fi
  echo "[mc_eval] ${exp} ${method} seed=${seed}"
  python3 scripts/evaluate_checkpoint.py \
    --checkpoint "${ckpt}" \
    --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --benchmarks "${BENCHMARKS[@]}" \
    --num-examples "${NUM_EXAMPLES}" \
    --seed 42 \
    --device "${DEVICE}" \
    --output "${out_file}"
}

run_mc_eval_manifest() {
  python3 - <<'PY' > /tmp/neuro_part12_eval_rows.tsv
import csv
import os

manifest = os.environ["MANIFEST"]
with open(manifest, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
rows.sort(key=lambda r: int(r["exec_order"]))
for r in rows:
    print("\t".join([r["experiment"], r["method"], r["seed"]]))
PY

  while IFS=$'\t' read -r exp method seed; do
    eval_latest_checkpoint "${exp}" "${method}" "${seed}"
  done < /tmp/neuro_part12_eval_rows.tsv
}

run_holdout() {
  python3 scripts/evaluate_instruction_holdout.py \
    --discover-root results/raw \
    --include-exp "${PART1_EXPS[0]}" \
    --include-exp "${PART1_EXPS[1]}" \
    --include-exp "${PART1_EXPS[2]}" \
    --include-exp "${PART1_EXPS[3]}" \
    --device "${DEVICE}" \
    --start-index 3000 \
    --num-examples 500 \
    --eval-batch-size 4 \
    --summary-csv analysis/neuro_part1_dolly_holdout.csv \
    --skip-existing

  python3 scripts/evaluate_instruction_holdout.py \
    --discover-root results/raw \
    --include-exp "${PART2_EXPS[0]}" \
    --include-exp "${PART2_EXPS[1]}" \
    --include-exp "${PART2_EXPS[2]}" \
    --include-exp "${PART2_EXPS[3]}" \
    --device "${DEVICE}" \
    --start-index 3000 \
    --num-examples 500 \
    --eval-batch-size 4 \
    --summary-csv analysis/neuro_part2_baselines_holdout.csv \
    --skip-existing
}

summarize() {
  python3 - <<'PY'
import csv
import json
import os
from pathlib import Path

manifest = Path(os.environ.get("MANIFEST", "docs/neuro_part12_run_manifest.csv"))
tag = os.environ.get("TAG", "neuro_part1_part2_4090")
rows = list(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
rows.sort(key=lambda r: int(r["exec_order"]))

print(f"{'exec':>4} {'experiment':38} {'seed':>4} {'comm_mb':>10} {'loss':>8} {'switch':>6} {'b_only':>6}")
for r in rows:
    exp = r["experiment"]
    method = r["method"]
    seed = r["seed"]
    base = Path("results/raw") / exp / method / f"seed_{seed}" / tag
    result_files = sorted(base.glob("*/results.json"))
    if not result_files:
        print(f"{int(r['exec_order']):4d} {exp:38} {seed:>4} {'MISSING':>10} {'-':>8} {'-':>6} {'-':>6}")
        continue
    data = json.loads(result_files[-1].read_text())
    rounds = data if isinstance(data, list) else data.get("rounds", [])
    if not rounds:
        print(f"{int(r['exec_order']):4d} {exp:38} {seed:>4} {'EMPTY':>10} {'-':>8} {'-':>6} {'-':>6}")
        continue
    last = rounds[-1]
    switch = (last.get("aggregator_stats") or {}).get("switch_round")
    b_only = next((x["round"] for x in rounds if x.get("broadcast_b_only")), None)
    print(
        f"{int(r['exec_order']):4d} {exp:38} {seed:>4} "
        f"{last.get('communication_mb', float('nan')):10.3f} "
        f"{last.get('avg_loss', float('nan')):8.4f} "
        f"{str(switch):>6} {str(b_only):>6}"
    )

print("\nHoldout CSVs:")
for p in [
    Path("analysis/neuro_part1_dolly_holdout.csv"),
    Path("analysis/neuro_part2_baselines_holdout.csv"),
]:
    print(f" - {p} {'(ok)' if p.is_file() else '(missing)'}")
PY
}

cmd="${1:-all}"
case "${cmd}" in
  train)
    run_train_manifest
    ;;
  holdout)
    run_holdout
    ;;
  mc_eval)
    run_mc_eval_manifest
    ;;
  summarize)
    summarize
    ;;
  all)
    run_train_manifest
    run_holdout
    if [[ "${RUN_MC:-0}" == "1" ]]; then
      run_mc_eval_manifest
    fi
    summarize
    ;;
  *)
    echo "Usage: $0 {train|holdout|mc_eval|summarize|all}"
    exit 2
    ;;
esac
