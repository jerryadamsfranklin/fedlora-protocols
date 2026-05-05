#!/bin/bash
# Run downstream evaluation on all completed checkpoints.
#
# Skips runs where final_adapter_state.pt is missing (those need re-running
# under Stage 5's full matrix).

set -u
ROOT="results/raw"
OUT_ROOT="results/downstream"
mkdir -p "${OUT_ROOT}"

# Find all final_adapter_state.pt files.
find "${ROOT}" -name "final_adapter_state.pt" | while read ckpt; do
    rel=$(echo "${ckpt}" | sed "s|${ROOT}/||" | sed "s|/final_adapter_state.pt||")
    out_dir="${OUT_ROOT}/${rel}"
    mkdir -p "${out_dir}"
    out_file="${out_dir}/downstream_results.json"

    if [[ -f "${out_file}" ]]; then
        echo "[skip] ${rel} (already evaluated)"
        continue
    fi

    echo "[eval] ${rel}"
    python scripts/evaluate_checkpoint.py \
        --checkpoint "${ckpt}" \
        --output "${out_file}" \
        --num-examples 500 \
        --seed 42
done
echo "All evaluations complete."

