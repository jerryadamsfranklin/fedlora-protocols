#!/usr/bin/env python3
"""
Generate figures from results/*.json for quick paper drafts.
"""

from __future__ import annotations

import json
import os
import re
import sys
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    }
)

METHOD_COLORS = {
    "fedit": "#1f77b4",
    "ffa_lora": "#ff7f0e",
    "flora": "#2ca02c",
    "flexlora": "#d62728",
}

METHOD_LABELS = {
    "fedit": "FedIT",
    "ffa_lora": "FFA-LoRA",
    "flora": "FLoRA",
    "flexlora": "FlexLoRA",
}

_RUN_TAIL_RE = re.compile(
    r"_(fedit|ffa_lora|flora|flexlora)((?:_r\d+)?)((?:_c\d+)?)_(\d{8}_\d{6})$"
)


def load_experiment_data(results_dir: str = RESULTS_DIR) -> dict:
    out = {}
    for exp_dir in glob(os.path.join(results_dir, "*")):
        if not os.path.isdir(exp_dir):
            continue
        rf = os.path.join(exp_dir, "results.json")
        if not os.path.exists(rf):
            continue
        with open(rf) as f:
            out[os.path.basename(exp_dir)] = json.load(f)
    return out


def _method_from_name(name: str) -> str | None:
    m = _RUN_TAIL_RE.search(name)
    return m.group(1) if m else None


def plot_convergence_curves(experiments: dict, exp_tag: str, output_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    plotted = set()
    for name, data in sorted(experiments.items()):
        if exp_tag not in name:
            continue
        method = _method_from_name(name)
        if method is None:
            continue
        label = METHOD_LABELS[method]
        if label in plotted:
            label = "_nolegend_"
        else:
            plotted.add(label)
        rounds = [d["round"] for d in data]
        losses = [d["avg_loss"] for d in data]
        ax.plot(
            rounds,
            losses,
            color=METHOD_COLORS[method],
            label=label if label != "_nolegend_" else None,
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.85,
        )

    ax.set_xlabel("Communication round")
    ax.set_ylabel("Training loss (avg over clients)")
    ax.set_title(f"Convergence ({exp_tag})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {output_path}")


def plot_iid_vs_noniid_comparison(experiments: dict, output_path: str) -> None:
    rows = []
    for name, results in experiments.items():
        if "exp1" not in name and "revised_exp1" not in name:
            continue
        if "iid" not in name.lower():
            continue
        method = _method_from_name(name)
        if method is None:
            continue
        rows.append(
            {
                "setting": "IID",
                "method": method,
                "loss": results[-1]["avg_loss"],
            }
        )

    for name, results in experiments.items():
        if "exp2" not in name and "revised_exp2" not in name:
            continue
        method = _method_from_name(name)
        if method is None:
            continue
        rows.append(
            {
                "setting": "Non-IID",
                "method": method,
                "loss": results[-1]["avg_loss"],
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for IID vs Non-IID comparison")
        return

    df["label"] = df["method"].map(METHOD_LABELS)
    pivot = df.pivot_table(
        index="label", columns="setting", values="loss", aggfunc="mean"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot.index))
    w = 0.35
    if "IID" in pivot.columns:
        ax.bar(x - w / 2, pivot["IID"], w, label="IID", color="#2ecc71")
    if "Non-IID" in pivot.columns:
        ax.bar(x + w / 2, pivot["Non-IID"], w, label="Non-IID", color="#e74c3c")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Final training loss")
    ax.set_title("IID vs non-IID (mean if multiple runs)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {output_path}")


def plot_communication_efficiency(experiments: dict, output_path: str) -> None:
    rows = []
    for name, results in experiments.items():
        if "exp1" not in name and "revised_exp1" not in name:
            continue
        method = _method_from_name(name)
        if method is None:
            continue
        rows.append(
            {
                "method": method,
                "loss": results[-1]["avg_loss"],
                "comm": results[-1]["communication_mb"],
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        print("No data for communication efficiency plot")
        return

    df = df.groupby("method", as_index=False).mean()

    fig, ax = plt.subplots(figsize=(8, 6))
    for _, row in df.iterrows():
        m = row["method"]
        ax.scatter(
            row["comm"],
            row["loss"],
            c=METHOD_COLORS[m],
            s=200,
            label=METHOD_LABELS[m],
            edgecolors="black",
            linewidth=1,
        )
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Final training loss")
    ax.set_title("Communication vs loss (IID runs)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.savefig(output_path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(FIGURES_DIR, exist_ok=True)

    experiments = load_experiment_data()
    if not experiments:
        print("No results found.")
        sys.exit(1)

    print(f"Found {len(experiments)} runs with results.json\n")

    for tag, fname in [
        ("exp1", "convergence_exp1.pdf"),
        ("exp2", "convergence_exp2.pdf"),
        ("exp3", "convergence_exp3.pdf"),
    ]:
        plot_convergence_curves(
            experiments,
            tag,
            os.path.join(FIGURES_DIR, fname),
        )

    plot_iid_vs_noniid_comparison(
        experiments,
        os.path.join(FIGURES_DIR, "iid_vs_noniid.pdf"),
    )
    plot_communication_efficiency(
        experiments,
        os.path.join(FIGURES_DIR, "communication_efficiency.pdf"),
    )
    print(f"\nFigures written to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
