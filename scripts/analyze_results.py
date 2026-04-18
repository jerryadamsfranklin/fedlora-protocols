#!/usr/bin/env python3
"""
Analyze experiment results and generate comparison tables.

Loads every results/*/results.json and writes CSV summaries under analysis/.
"""

from __future__ import annotations

import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


RESULTS_DIR = "results"

# .../{exp_slug}_{method}[_r{rank}][_c{n}]_{YYYYMMDD}_{HHMMSS}
_RUN_TAIL_RE = re.compile(
    r"_(fedit|ffa_lora|flora|flexlora)((?:_r\d+)?)((?:_c\d+)?)_(\d{8}_\d{6})$"
)


def find_convergence_round(data: List[Dict[str, Any]], rel_threshold: float = 0.002) -> int:
    """
    First round index (1-based) where relative improvement from previous round
    falls below rel_threshold. Falls back to len(data) if never triggered.
    """
    for i in range(1, len(data)):
        prev_loss = data[i - 1]["avg_loss"]
        cur_loss = data[i]["avg_loss"]
        if prev_loss <= 0:
            continue
        rel_improve = (prev_loss - cur_loss) / prev_loss
        if rel_improve < rel_threshold:
            return i + 1
    return len(data)


def classify_experiment(slug: str) -> str:
    s = slug.lower()
    if "exp5" in s or "scaling" in s:
        return "Scaling"
    if "exp4" in s or "rank" in s:
        return "Rank"
    if "exp3" in s or "qty" in s or "quantity" in s:
        return "NonIID-Qty"
    if "exp2" in s or "noniid_label" in s or "label" in s:
        return "NonIID-Label"
    if "exp1" in s or "_iid" in s:
        return "IID"
    return "Unknown"


def parse_run_directory(dirname: str) -> Optional[Dict[str, Any]]:
    m = _RUN_TAIL_RE.search(dirname)
    if not m:
        return None
    method = m.group(1)
    r_part, c_part = m.group(2), m.group(3)
    slug = dirname[: m.start()].rstrip("_")

    rank = None
    clients = None
    rm = re.search(r"_r(\d+)", r_part + c_part)
    if rm:
        rank = int(rm.group(1))
    cm = re.search(r"_c(\d+)", r_part + c_part)
    if cm:
        clients = int(cm.group(1))
    return {
        "slug": slug,
        "method": method,
        "lora_r": rank,
        "num_clients_override": clients,
        "experiment_type": classify_experiment(slug),
    }


def load_all_results(results_dir: str = RESULTS_DIR) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    for exp_dir in sorted(glob(os.path.join(results_dir, "*"))):
        if not os.path.isdir(exp_dir):
            continue
        results_file = os.path.join(exp_dir, "results.json")
        if not os.path.exists(results_file):
            continue
        dirname = os.path.basename(exp_dir)
        meta = parse_run_directory(dirname)
        if meta is None:
            continue

        with open(results_file) as f:
            data = json.load(f)
        if not data:
            continue

        initial = data[0]
        final = data[-1]
        total_sec = sum(float(r["round_time"]) for r in data)

        row: Dict[str, Any] = {
            "run_dir": dirname,
            "experiment_type": meta["experiment_type"],
            "slug": meta["slug"],
            "method": meta["method"],
            "lora_r": meta["lora_r"],
            "num_clients_override": meta["num_clients_override"],
            "initial_loss": initial["avg_loss"],
            "final_loss": final["avg_loss"],
            "loss_reduction": initial["avg_loss"] - final["avg_loss"],
            "total_time_hours": total_sec / 3600.0,
            "communication_mb": final["communication_mb"],
            "num_rounds": len(data),
            "convergence_round": find_convergence_round(data),
        }
        records.append(row)

    return pd.DataFrame(records)


def generate_comparison_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    sub = df[df["experiment_type"] != "Unknown"]
    if sub.empty:
        return sub
    return pd.pivot_table(
        sub,
        index="experiment_type",
        columns="method",
        values=["final_loss", "communication_mb"],
        aggfunc="mean",
    )


def compute_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Pairwise t-tests when at least two independent runs exist per cell."""
    results: List[Dict[str, Any]] = []
    for exp in df["experiment_type"].unique():
        exp_data = df[df["experiment_type"] == exp]
        methods = sorted(exp_data["method"].unique())
        for i, m1 in enumerate(methods):
            for m2 in methods[i + 1 :]:
                loss1 = exp_data[exp_data["method"] == m1]["final_loss"].values
                loss2 = exp_data[exp_data["method"] == m2]["final_loss"].values
                if len(loss1) > 1 and len(loss2) > 1:
                    stat, pval = stats.ttest_ind(loss1, loss2, equal_var=False)
                else:
                    stat, pval = np.nan, np.nan
                results.append(
                    {
                        "experiment": exp,
                        "comparison": f"{m1} vs {m2}",
                        "t_statistic": stat,
                        "p_value": pval,
                        "significant": bool(pval < 0.05)
                        if not np.isnan(pval)
                        else None,
                        "note": None
                        if len(loss1) > 1 and len(loss2) > 1
                        else "need multiple seeds/runs per cell",
                    }
                )
    return pd.DataFrame(results)


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print("Loading results...")
    df = load_all_results()
    if df.empty:
        print("No results found under results/*/results.json")
        sys.exit(1)

    print(f"\nFound {len(df)} experiment runs\n")
    print("=" * 60)

    print("\n=== SUMMARY TABLE ===\n")
    cols = [
        "experiment_type",
        "method",
        "slug",
        "final_loss",
        "total_time_hours",
        "communication_mb",
        "num_rounds",
    ]
    print(df[cols].to_string(index=False))

    print("\n=== COMPARISON BY EXPERIMENT (mean over repeated runs) ===\n")
    comparison = generate_comparison_table(df)
    print(comparison.to_string())

    print("\n=== STATISTICAL SIGNIFICANCE (multi-run only) ===\n")
    stats_df = compute_statistical_tests(df)
    print(stats_df.to_string(index=False))

    os.makedirs("analysis", exist_ok=True)
    df.to_csv("analysis/all_results.csv", index=False)
    comparison.to_csv("analysis/comparison_table.csv")
    stats_df.to_csv("analysis/statistical_tests.csv", index=False)
    print("\n\nSaved: analysis/all_results.csv, comparison_table.csv, statistical_tests.csv")


if __name__ == "__main__":
    main()
