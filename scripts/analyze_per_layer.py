#!/usr/bin/env python3
"""
Analyze per-layer switching patterns from FedLoRA-Adaptive v2 results.

Reads `results.json` files and prints:
- final per-layer phase
- per-layer switch round (if present)

Optionally writes a timeline heatmap if matplotlib is available.
"""

from __future__ import annotations

import argparse
import json
import os
from glob import glob
from typing import Any, Dict, List, Optional


def _load_results(path: str) -> List[Dict[str, Any]]:
    with open(path) as f:
        return json.load(f)


def _find_results_files(results_root: str, method: str) -> List[str]:
    pattern = os.path.join(results_root, "**", method, "**", "results.json")
    return sorted(glob(pattern, recursive=True))


def _extract_layer_analysis(rounds: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for row in reversed(rounds):
        stats = row.get("aggregator_stats") or {}
        layer_analysis = stats.get("layer_analysis")
        if layer_analysis:
            return layer_analysis
    return None


def _print_summary(layer_analysis: Dict[str, Any]) -> None:
    layer_phases = layer_analysis.get("layer_phases", {})
    switch_rounds = layer_analysis.get("layer_switch_rounds", {})

    print("\n" + "=" * 60)
    print("PER-LAYER SWITCHING ANALYSIS")
    print("=" * 60)
    print(f"\n{'Layer':<20} {'Phase':<15} {'Switch round':<12}")
    print("-" * 50)
    for layer in sorted(layer_phases.keys()):
        phase = layer_phases.get(layer, "unknown")
        sw = switch_rounds.get(layer)
        sw_str = str(sw) if sw is not None else "Never"
        print(f"{layer:<20} {phase:<15} {sw_str:<12}")

    still = layer_analysis.get("layers_still_ffa_lora", [])
    print("\nLayers still in FFA-LoRA:", still)
    print("Earliest switch:", layer_analysis.get("earliest_switch"))
    print("Latest switch:", layer_analysis.get("latest_switch"))


def _plot_timeline(rounds: List[Dict[str, Any]], out_path: str) -> None:
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as e:
        print(f"Skipping plot (matplotlib missing): {e}")
        return

    phase_map = {"FFA-LoRA": 0, "TRANSITION": 1, "FLoRA": 2}

    per_round = []
    for row in rounds:
        stats = row.get("aggregator_stats") or {}
        la = (stats.get("layer_analysis") or {}).get("layer_phases")
        if la:
            per_round.append((row.get("round"), la))

    if not per_round:
        print("No per-round layer phases found; skipping plot.")
        return

    layer_names = sorted(per_round[0][1].keys())
    matrix = np.zeros((len(layer_names), len(per_round)))
    for j, (_r, phases) in enumerate(per_round):
        for i, layer in enumerate(layer_names):
            matrix[i, j] = phase_map.get(phases.get(layer, "FFA-LoRA"), 0)

    cmap = plt.cm.colors.ListedColormap(["#3498db", "#f39c12", "#2ecc71"])
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0, vmax=2)
    ax.set_yticks(range(len(layer_names)))
    ax.set_yticklabels(layer_names)
    ax.set_xlabel("Round index")
    ax.set_ylabel("Layer")
    ax.set_title("Per-layer phase timeline (v2)")
    cbar = plt.colorbar(im, ax=ax, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["FFA-LoRA", "TRANSITION", "FLoRA"])
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved plot: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_root", default="results", help="Results directory root")
    ap.add_argument("--method", default="fedlora_adaptive_v2")
    ap.add_argument("--pick", default="latest", choices=["latest", "first"])
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--plot_out", default="figures/per_layer_timeline.png")
    args = ap.parse_args()

    files = _find_results_files(args.results_root, args.method)
    if not files:
        raise SystemExit(f"No results.json found for method={args.method} under {args.results_root}")

    path = files[-1] if args.pick == "latest" else files[0]
    print(f"Using: {path}")
    rounds = _load_results(path)
    la = _extract_layer_analysis(rounds)
    if not la:
        raise SystemExit("No layer_analysis found in results (missing aggregator_stats?)")

    _print_summary(la)
    if args.plot:
        _plot_timeline(rounds, args.plot_out)


if __name__ == "__main__":
    main()

