"""
Compute paired statistical tests for headline IID loss comparisons.

Output: analysis/statistical_tests.csv
"""

from __future__ import annotations

import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

_SEED_RE = re.compile(r"seed_(\d+)")


def _latest_path_per_seed(pattern: str) -> Dict[int, str]:
    """Select newest timestamp folder per seed for overlapping globs."""
    by_seed: Dict[int, List[Tuple[str, str]]] = {}
    for path in sorted(glob.glob(pattern)):
        m = _SEED_RE.search(path)
        if not m:
            continue
        seed = int(m.group(1))
        ts = Path(path).parent.name
        by_seed.setdefault(seed, []).append((ts, path))
    out: Dict[int, str] = {}
    for seed, items in by_seed.items():
        best_ts, best_path = max(items, key=lambda x: x[0])
        out[seed] = best_path
    return out


def collect_losses_aligned(
    pattern: str, common_seeds: List[int], key: str = "avg_loss"
) -> List[float]:
    latest = _latest_path_per_seed(pattern)
    values: List[float] = []
    for seed in common_seeds:
        path = latest.get(seed)
        if path is None:
            raise KeyError(f"missing seed {seed} for pattern {pattern}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not data:
            raise ValueError(f"empty results: {path}")
        v = data[-1].get(key)
        if v is None:
            raise ValueError(f"no {key} in {path}")
        values.append(float(v))
    return values


def paired_test(a: List[float], b: List[float], label: str) -> Dict[str, object]:
    if len(a) != len(b) or len(a) < 2:
        return {
            "comparison": label,
            "n": len(a),
            "mean_a": "",
            "mean_b": "",
            "diff": "",
            "p_value": "",
            "cohens_d": "",
            "note": "insufficient seeds",
        }
    arr_a = np.array(a, dtype=float)
    arr_b = np.array(b, dtype=float)
    _, p = stats.ttest_rel(arr_a, arr_b)
    diff = arr_a - arr_b
    sd = float(np.std(diff, ddof=1))
    d = float(np.mean(diff) / sd) if sd > 0 else 0.0
    return {
        "comparison": label,
        "n": len(a),
        "mean_a": float(np.mean(arr_a)),
        "mean_b": float(np.mean(arr_b)),
        "diff": float(np.mean(diff)),
        "p_value": float(p),
        "cohens_d": d,
        "note": "",
    }


def main() -> None:
    flora_pat = "results/raw/exp_flora_iid/flora/seed_*/run_*/*/results.json"
    tp_k8_pat = "results/raw/exp_two_phase_k8/two_phase/seed_*/stage1_bidirectional/*/results.json"
    ra_pat = "results/raw/exp_reverse_adaptive_iid/reverse_adaptive/seed_*/stage2_adaptive/*/results.json"

    flora_seeds = set(_latest_path_per_seed(flora_pat).keys())
    tp_seeds = set(_latest_path_per_seed(tp_k8_pat).keys())
    ra_seeds = set(_latest_path_per_seed(ra_pat).keys())
    common = sorted(flora_seeds & tp_seeds & ra_seeds)

    print(f"FLoRA IID seeds: {sorted(flora_seeds)}")
    print(f"Two-Phase K=8 IID seeds: {sorted(tp_seeds)}")
    print(f"ReverseAdaptive IID seeds: {sorted(ra_seeds)}")
    print(f"Paired seeds (intersection): {common}")

    flora = collect_losses_aligned(flora_pat, common)
    tp_k8 = collect_losses_aligned(tp_k8_pat, common)
    ra = collect_losses_aligned(ra_pat, common)

    rows = []
    if len(common) >= 2:
        rows.append(paired_test(tp_k8, flora, "Two-Phase K=8 vs FLoRA (loss)"))
        rows.append(paired_test(ra, flora, "ReverseAdaptive vs FLoRA (loss)"))
        rows.append(paired_test(ra, tp_k8, "ReverseAdaptive vs Two-Phase K=8 (loss)"))

    llama_flora_pat = "results/raw/exp_llama3_flora_iid/flora/seed_*/stage5_llama3*/*/results.json"
    llama_tp8_pat = "results/raw/exp_llama3_two_phase_k8_iid/two_phase/seed_*/stage5_llama3*/*/results.json"
    llama_ra_pat = "results/raw/exp_llama3_reverse_adaptive_iid/reverse_adaptive/seed_*/stage5_llama3*/*/results.json"

    lf_s = set(_latest_path_per_seed(llama_flora_pat).keys())
    lt_s = set(_latest_path_per_seed(llama_tp8_pat).keys())
    lr_s = set(_latest_path_per_seed(llama_ra_pat).keys())
    llama_common = sorted(lf_s & lt_s & lr_s)

    print(f"\n[LLaMA-3.2-3B IID]")
    print(f"FLoRA seeds: {sorted(lf_s)}")
    print(f"Two-Phase K=8 seeds: {sorted(lt_s)}")
    print(f"ReverseAdaptive seeds: {sorted(lr_s)}")
    print(f"Paired seeds (intersection): {llama_common}")

    if len(llama_common) >= 2:
        llama_flora = collect_losses_aligned(llama_flora_pat, llama_common)
        llama_tp8 = collect_losses_aligned(llama_tp8_pat, llama_common)
        llama_ra = collect_losses_aligned(llama_ra_pat, llama_common)
        rows.append(paired_test(llama_tp8, llama_flora, "[LLaMA-3B] Two-Phase K=8 vs FLoRA (loss)"))
        rows.append(paired_test(llama_ra, llama_flora, "[LLaMA-3B] ReverseAdaptive vs FLoRA (loss)"))
        rows.append(paired_test(llama_ra, llama_tp8, "[LLaMA-3B] ReverseAdaptive vs Two-Phase K=8 (loss)"))

    repo_root = Path(__file__).resolve().parents[1]
    out_path = repo_root / "analysis" / "statistical_tests.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
    print(f"Wrote {len(rows)} statistical tests to {out_path}")


if __name__ == "__main__":
    main()
