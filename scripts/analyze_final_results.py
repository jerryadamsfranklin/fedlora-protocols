#!/usr/bin/env python3
"""
Analyze final experiment results for the Two-Phase paper batch.
Computes mean/std across seeds grouped by experiment folder under results/raw/.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np


def find_results_files(results_dir: str) -> dict[str, list[str]]:
    """Group results.json paths by experiment name (directory under results/raw/)."""
    grouped: dict[str, list[str]] = defaultdict(list)
    root_path = Path(results_dir)
    if not root_path.is_dir():
        return {}

    for results_json in root_path.rglob("results.json"):
        path = results_json.parent
        parts = path.parts
        try:
            raw_idx = parts.index("raw")
            exp_name = parts[raw_idx + 1]
        except (ValueError, IndexError):
            continue
        grouped[exp_name].append(str(results_json))

    return dict(grouped)


def load_results(filepath: str) -> dict | None:
    """Load key metrics from the last round of results.json."""
    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        return None

    final_round = data[-1]

    upload = final_round.get("upload_mb")
    if upload is None:
        upload = final_round.get("communication_mb", 0.0)

    metrics = {
        "final_loss": final_round.get("avg_loss", float("nan")),
        "total_comm_mb": final_round.get("communication_mb", 0.0),
        "upload_mb": upload,
    }

    agg_stats = final_round.get("aggregator_stats") or {}
    if "comm_savings_vs_flora" in agg_stats:
        metrics["savings"] = agg_stats["comm_savings_vs_flora"]
    if "total_comm_mb" in agg_stats:
        metrics["agg_total_comm"] = agg_stats["total_comm_mb"]

    return metrics


def analyze_experiments(results_dir: str) -> None:
    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT ANALYSIS")
    print("=" * 70 + "\n")

    grouped = find_results_files(results_dir)

    if not grouped:
        print("No results found under", results_dir)
        return

    summary = []

    for exp_name in sorted(grouped.keys()):
        files = grouped[exp_name]
        losses = []
        uploads = []
        savings_list = []

        for fpath in files:
            m = load_results(fpath)
            if not m:
                continue
            losses.append(m["final_loss"])
            uploads.append(m["upload_mb"])
            if "savings" in m:
                savings_list.append(m["savings"])

        if not losses:
            continue

        row = {
            "experiment": exp_name,
            "n_seeds": len(losses),
            "loss_mean": float(np.mean(losses)),
            "loss_std": float(np.std(losses)),
            "upload_mean": float(np.mean(uploads)),
            "upload_std": float(np.std(uploads)),
        }

        if savings_list:
            row["savings_mean"] = float(np.mean(savings_list))
            row["savings_std"] = float(np.std(savings_list))

        summary.append(row)

    print(
        f"{'Experiment':<42} {'Seeds':>6} {'Loss (mean±std)':>22} "
        f"{'Upload MB (mean±std)':>24} {'Savings':>10}"
    )
    print("-" * 110)

    for row in summary:
        loss_str = f"{row['loss_mean']:.4f} ± {row['loss_std']:.4f}"
        upload_str = f"{row['upload_mean']:.1f} ± {row['upload_std']:.1f}"

        savings_str = ""
        if "savings_mean" in row:
            savings_str = f"{row['savings_mean'] * 100:.1f}%"

        print(
            f"{row['experiment']:<42} {row['n_seeds']:>6} "
            f"{loss_str:>22} {upload_str:>24} {savings_str:>10}"
        )

    print("\n" + "=" * 70)
    print("Note: Report mean ± std across seeds for the paper.")
    print("Upload uses final-round cumulative upload_mb when present; else communication_mb.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        default=os.path.join("results", "raw"),
        help="Directory containing experiment subfolders",
    )
    parser.add_argument(
        "--log-dir",
        default=None,
        help="Ignored; accepted for compatibility with batch script.",
    )
    args = parser.parse_args()

    analyze_experiments(os.path.abspath(args.results_dir))


if __name__ == "__main__":
    main()
