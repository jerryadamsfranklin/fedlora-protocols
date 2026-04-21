#!/usr/bin/env python3
"""
Generate publication figures aligned with Federated_LoRA_Complete_Guide.md (RQ1–RQ5).

Writes under figures/:
  fig1_convergence_comparison.pdf — three settings: IID, label skew, quantity skew
  fig2_iid_vs_noniid.pdf — IID vs label-skew final loss by method (RQ2)
  fig3_communication_tradeoff.pdf — cumulative communication vs final loss on IID (RQ3)
  fig4_rank_sensitivity.pdf — LoRA rank vs final loss (RQ4)
  fig5_client_scaling.pdf — number of clients vs final loss (RQ5)

Legacy aliases: convergence_exp*.pdf, iid_vs_noniid.pdf, communication_efficiency.pdf
"""

from __future__ import annotations

import json
import os
import re
import sys
from glob import glob
from typing import Any, Dict, List, Tuple

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


def load_experiment_data(results_dir: str = RESULTS_DIR) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
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


def extract_timestamp(name: str) -> str:
    m = re.search(r"_(\d{8}_\d{6})$", name)
    return m.group(1) if m else ""


def dedupe_latest_per_method(
    experiments: Dict[str, List[Dict[str, Any]]], prefix: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Keep newest run per method among directories whose name contains prefix."""
    candidates = {
        k: v
        for k, v in experiments.items()
        if prefix in k and k.startswith("revised_")
    }
    if not candidates:
        candidates = {k: v for k, v in experiments.items() if prefix in k}
    best: Dict[str, Tuple[str, str, List[Dict[str, Any]]]] = {}
    for name, data in candidates.items():
        meth = _method_from_name(name)
        if meth is None:
            continue
        ts = extract_timestamp(name)
        cur = best.get(meth)
        if cur is None or ts > cur[0]:
            best[meth] = (ts, name, data)
    return {v[1]: v[2] for v in best.values()}


def _save_fig(path: str) -> None:
    plt.savefig(path)
    plt.savefig(path.replace(".pdf", ".png"))
    plt.close()
    print(f"Saved: {path}")


def plot_convergence_one_ax(
    ax: Any,
    experiments: Dict[str, List[Dict[str, Any]]],
    prefix: str,
    title: str,
) -> None:
    sub = dedupe_latest_per_method(experiments, prefix)
    plotted_labels: set[str] = set()
    for name in sorted(sub.keys()):
        data = sub[name]
        method = _method_from_name(name)
        if method is None:
            continue
        label = METHOD_LABELS[method]
        use_label = label if label not in plotted_labels else "_nolegend_"
        if use_label != "_nolegend_":
            plotted_labels.add(label)
        rounds = [d["round"] for d in data]
        losses = [d["avg_loss"] for d in data]
        ax.plot(
            rounds,
            losses,
            color=METHOD_COLORS[method],
            label=label if use_label != "_nolegend_" else None,
            linewidth=2,
            marker="o",
            markersize=4,
            alpha=0.9,
        )
    ax.set_xlabel("Communication round")
    ax.set_ylabel("Avg. training loss")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_fig1_convergence_comparison(
    experiments: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    specs = [
        ("revised_exp1_iid", "IID (Alpaca)"),
        ("revised_exp2_noniid_label", "Label skew (CommonsenseQA)"),
        ("revised_exp3_noniid_qty", "Quantity skew (Alpaca)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    for ax, (prefix, title) in zip(axes, specs):
        plot_convergence_one_ax(ax, experiments, prefix, title)
    fig.suptitle(
        "Convergence across federated LoRA aggregation methods",
        fontsize=13,
        y=1.02,
    )
    plt.tight_layout()
    _save_fig(output_path)


def plot_fig2_iid_vs_noniid(
    experiments: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    rows: List[Dict[str, Any]] = []
    for name, data in dedupe_latest_per_method(
        experiments, "revised_exp1_iid"
    ).items():
        m = _method_from_name(name)
        if m:
            rows.append(
                {"setting": "IID", "method": m, "loss": data[-1]["avg_loss"]}
            )
    for name, data in dedupe_latest_per_method(
        experiments, "revised_exp2_noniid_label"
    ).items():
        m = _method_from_name(name)
        if m:
            rows.append(
                {
                    "setting": "Label skew",
                    "method": m,
                    "loss": data[-1]["avg_loss"],
                }
            )
    df = pd.DataFrame(rows)
    if df.empty:
        print("fig2: no revised IID / label-skew data")
        return
    df["label"] = df["method"].map(METHOD_LABELS)
    pivot = df.pivot_table(
        index="label", columns="setting", values="loss", aggfunc="first"
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(pivot.index))
    w = 0.35
    if "IID" in pivot.columns:
        ax.bar(x - w / 2, pivot["IID"], w, label="IID (Alpaca)", color="#2ecc71")
    if "Label skew" in pivot.columns:
        ax.bar(
            x + w / 2,
            pivot["Label skew"],
            w,
            label="Label skew (CommonsenseQA)",
            color="#e74c3c",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index)
    ax.set_ylabel("Final training loss")
    ax.set_title("IID vs label-skew non-IID (final loss)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    _save_fig(output_path)


def plot_fig3_communication_tradeoff(
    experiments: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    sub = dedupe_latest_per_method(experiments, "revised_exp1_iid")
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, data in sorted(sub.items()):
        m = _method_from_name(name)
        if m is None:
            continue
        ax.scatter(
            data[-1]["communication_mb"],
            data[-1]["avg_loss"],
            c=METHOD_COLORS[m],
            s=220,
            label=METHOD_LABELS[m],
            edgecolors="black",
            linewidth=1,
            zorder=3,
        )
    ax.set_xlabel("Cumulative communication (MB)")
    ax.set_ylabel("Final training loss")
    ax.set_title("Communication vs loss (IID baseline, RQ3)")
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def _dedupe_rank_scaling_rows(
    rows: List[Dict[str, Any]], key_fields: Tuple[str, ...]
) -> List[Dict[str, Any]]:
    """Keep row with latest timestamp per key_fields."""
    tmp: Dict[Tuple[Any, ...], Tuple[str, Dict[str, Any]]] = {}
    for row in rows:
        name = row["_name"]
        ts = extract_timestamp(name)
        key = tuple(row[k] for k in key_fields)
        cur = tmp.get(key)
        if cur is None or ts > cur[0]:
            tmp[key] = (ts, row)
    out = [v[1] for v in tmp.values()]
    for r in out:
        del r["_name"]
    return out


def plot_fig4_rank_sensitivity(
    experiments: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    rows: List[Dict[str, Any]] = []
    for name, data in experiments.items():
        if "revised_exp4_rank" not in name:
            continue
        rm = re.search(r"_r(\d+)_", name)
        if not rm:
            continue
        m = _method_from_name(name)
        if m is None:
            continue
        rows.append(
            {
                "_name": name,
                "rank": int(rm.group(1)),
                "method": m,
                "loss": data[-1]["avg_loss"],
            }
        )
    rows = _dedupe_rank_scaling_rows(rows, ("rank", "method"))
    if not rows:
        print("fig4: no revised_exp4_rank data")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m].sort_values("rank")
        ax.plot(
            sub["rank"],
            sub["loss"],
            marker="o",
            linewidth=2,
            label=METHOD_LABELS[m],
            color=METHOD_COLORS[m],
        )
    ax.set_xlabel("LoRA rank $r$")
    ax.set_ylabel("Final training loss")
    ax.set_title("Rank sensitivity (IID Alpaca subset, RQ4)")
    ax.set_xticks(sorted(df["rank"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def plot_fig5_client_scaling(
    experiments: Dict[str, List[Dict[str, Any]]], output_path: str
) -> None:
    rows: List[Dict[str, Any]] = []
    for name, data in experiments.items():
        if "revised_exp5_scaling" not in name:
            continue
        cm = re.search(r"_c(\d+)_", name)
        if not cm:
            continue
        m = _method_from_name(name)
        if m is None:
            continue
        rows.append(
            {
                "_name": name,
                "clients": int(cm.group(1)),
                "method": m,
                "loss": data[-1]["avg_loss"],
            }
        )
    rows = _dedupe_rank_scaling_rows(rows, ("clients", "method"))
    if not rows:
        print("fig5: no revised_exp5_scaling data")
        return
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in sorted(df["method"].unique()):
        sub = df[df["method"] == m].sort_values("clients")
        ax.plot(
            sub["clients"],
            sub["loss"],
            marker="s",
            linewidth=2,
            markersize=8,
            label=METHOD_LABELS[m],
            color=METHOD_COLORS[m],
        )
    ax.set_xlabel("Number of clients")
    ax.set_ylabel("Final training loss")
    ax.set_title("Client scaling (IID Alpaca, RQ5)")
    ax.set_xticks(sorted(df["clients"].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save_fig(output_path)


def main() -> None:
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    os.makedirs(FIGURES_DIR, exist_ok=True)

    experiments = load_experiment_data()
    if not experiments:
        print("No results found.")
        sys.exit(1)

    print(f"Found {len(experiments)} runs with results.json\n")

    plot_fig1_convergence_comparison(
        experiments, os.path.join(FIGURES_DIR, "fig1_convergence_comparison.pdf")
    )
    plot_fig2_iid_vs_noniid(
        experiments, os.path.join(FIGURES_DIR, "fig2_iid_vs_noniid.pdf")
    )
    plot_fig3_communication_tradeoff(
        experiments, os.path.join(FIGURES_DIR, "fig3_communication_tradeoff.pdf")
    )
    plot_fig4_rank_sensitivity(
        experiments, os.path.join(FIGURES_DIR, "fig4_rank_sensitivity.pdf")
    )
    plot_fig5_client_scaling(
        experiments, os.path.join(FIGURES_DIR, "fig5_client_scaling.pdf")
    )

    # Legacy filenames (single-tag convergence)
    for prefix, fname in [
        ("revised_exp1_iid", "convergence_exp1.pdf"),
        ("revised_exp2_noniid_label", "convergence_exp2.pdf"),
        ("revised_exp3_noniid_qty", "convergence_exp3.pdf"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_convergence_one_ax(ax, experiments, prefix, prefix.replace("_", " "))
        plt.tight_layout()
        _save_fig(os.path.join(FIGURES_DIR, fname))

    print(f"\nPrimary figures: fig1–fig5 in {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
