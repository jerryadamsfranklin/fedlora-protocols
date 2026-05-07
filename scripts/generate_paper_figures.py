"""Generate Stage 5 paper figures into figures/. Requires analysis/final_results_table.csv."""

from __future__ import annotations

import csv
import glob
import json
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

REPO = Path(__file__).resolve().parents[1]
OUTDIR = REPO / "figures"
TABLE_PATH = REPO / "analysis" / "final_results_table.csv"

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
    }
)

_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def load_table(path: Path) -> List[Dict[str, str]]:
    if not path.is_file():
        warnings.warn(f"Missing {path}; figures may skip or be empty.")
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _f(x: str) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def _latest_per_seed(pattern: str) -> Dict[int, str]:
    by_seed: Dict[int, List[Tuple[str, str]]] = {}
    for path in sorted(glob.glob(str(REPO / pattern))):
        m = re.search(r"seed_(\d+)", path)
        if not m:
            continue
        seed = int(m.group(1))
        ts = Path(path).parent.name
        if not _TS_RE.match(ts):
            continue
        by_seed.setdefault(seed, []).append((ts, path))
    out: Dict[int, str] = {}
    for seed, items in by_seed.items():
        out[seed] = max(items, key=lambda x: x[0])[1]
    return out


def _load_rounds(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _mean_std(vals: Sequence[float]) -> Tuple[float, float]:
    a = np.array(vals, dtype=float)
    return float(np.mean(a)), float(np.std(a, ddof=1)) if len(a) > 1 else 0.0


def _merge_downstream_jsons(dir_path: Path) -> Dict[str, float]:
    acc: Dict[str, float] = {}
    if not dir_path.is_dir():
        return acc
    for fp in sorted(dir_path.rglob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for name, block in (data.get("benchmarks") or {}).items():
            if isinstance(block, dict) and "accuracy" in block:
                acc[name] = float(block["accuracy"])
    return acc


def _save(fig: Figure, name: str) -> None:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    pdf = OUTDIR / f"{name}.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {pdf}")


def figure1_pareto_iid(rows: List[Dict[str, str]]) -> None:
    """Pareto with FLoRA, K10, K8 means plus four ReverseAdaptive threshold points (seed 42)."""
    tiny = [
        r
        for r in rows
        if r.get("setting") == "iid"
        and r.get("scale") == "tinyllama_1b"
        and _f(r.get("total_mb", "") or "") is not None
        and _f(r.get("final_loss", "") or "") is not None
    ]

    def collect(exp: str, tag_filter: Optional[str] = None) -> Tuple[float, float, float, float]:
        rs = [r for r in tiny if r["exp_name"] == exp]
        if tag_filter:
            rs = [r for r in rs if tag_filter in r["tag"]]
        losses = [_f(r["final_loss"]) for r in rs]
        mbs = [_f(r["total_mb"]) for r in rs]
        losses = [x for x in losses if x is not None]
        mbs = [x for x in mbs if x is not None]
        if not losses:
            return float("nan"), float("nan"), float("nan"), float("nan")
        lm, ls = _mean_std(losses)
        mm, ms = _mean_std(mbs)
        return mm, ms, lm, ls

    mb_f, eb_f, loss_f, el_f = collect("exp_flora_iid", "run_")
    mb_10, eb_10, loss_10, el_10 = collect("exp_two_phase_k10", "stage1_bidirectional")
    mb_8, eb_8, loss_8, el_8 = collect("exp_two_phase_k8", "stage1_bidirectional")

    ra_thresholds: List[Tuple[str, str, str]] = [
        ("0.001", "stage2_threshold_extended_0.001", "τ=0.001"),
        ("0.002", "stage2_threshold_extended_0.002", "τ=0.002"),
        ("0.005", "stage2_threshold_0.005", "τ=0.005"),
        ("0.010", "stage2_threshold_0.01", "τ=0.01"),
    ]
    ra_points: List[Tuple[float, float, str]] = []
    for _tid, folder, label in ra_thresholds:
        pat = str(
            REPO
            / "results/raw/exp_reverse_adaptive_threshold_ablation/reverse_adaptive/seed_42"
            / folder
            / "*"
            / "results.json"
        )
        matches = sorted(glob.glob(pat))
        if not matches:
            warnings.warn(f"Figure 1: missing RA threshold run for {folder}")
            continue
        path = matches[-1]
        data = _load_rounds(path)
        if not data:
            continue
        final = data[-1]
        mb = float(final["communication_mb"])
        loss = float(final["avg_loss"])
        ra_points.append((mb, loss, label))

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.errorbar(
        [mb_f],
        [loss_f],
        xerr=[eb_f],
        yerr=[el_f],
        fmt="o",
        color=_COLORS[0],
        capsize=3,
        label="FLoRA",
    )
    ax.errorbar(
        [mb_10],
        [loss_10],
        xerr=[eb_10],
        yerr=[el_10],
        fmt="s",
        color=_COLORS[1],
        capsize=3,
        label="Two-Phase K=10",
    )
    ax.errorbar(
        [mb_8],
        [loss_8],
        xerr=[eb_8],
        yerr=[el_8],
        fmt="^",
        color=_COLORS[2],
        capsize=3,
        label="Two-Phase K=8",
    )

    if len(ra_points) >= 2:
        rax = [p[0] for p in ra_points]
        ray = [p[1] for p in ra_points]
        ax.plot(rax, ray, "k--", alpha=0.55, linewidth=1.2, label="ReverseAdaptive (τ sweep)")
        for mb, loss, lab in ra_points:
            ax.scatter([mb], [loss], color=_COLORS[3], s=38, zorder=5, edgecolors="k", linewidths=0.4)
            ax.annotate(lab, (mb, loss), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Total round-trip communication (MB)")
    ax.set_ylabel("Final training loss")
    ax.set_xlim(1200, 2700)
    ax.set_ylim(1.255, 1.295)
    ax.set_title("Communication-quality tradeoff (TinyLlama-1.1B, IID)")
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig1_pareto_iid")


def figure2_convergence() -> None:
    flora = _latest_per_seed("results/raw/exp_flora_iid/flora/seed_*/run_*/*/results.json")
    tp = _latest_per_seed("results/raw/exp_two_phase_k8/two_phase/seed_*/stage1_bidirectional/*/results.json")
    ra = _latest_per_seed("results/raw/exp_reverse_adaptive_iid/reverse_adaptive/seed_*/stage2_adaptive/*/results.json")
    seeds = sorted(set(flora.keys()) & set(tp.keys()) & set(ra.keys()))
    if len(seeds) < 2:
        warnings.warn("Figure 2: insufficient seeds")
        return

    def curve(paths: Dict[int, str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mats = []
        for s in seeds:
            rds = _load_rounds(paths[s])
            mats.append([float(x["avg_loss"]) for x in rds])
        arr = np.array(mats, dtype=float)
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        return np.arange(1, len(mean) + 1), mean, std

    r1, m1, s1 = curve(flora)
    r2, m2, s2 = curve(tp)
    r3, m3, s3 = curve(ra)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(r1, m1, label="FLoRA", color=_COLORS[0])
    ax.fill_between(r1, m1 - s1, m1 + s1, color=_COLORS[0], alpha=0.2)
    ax.plot(r2, m2, label="Two-Phase K=8", color=_COLORS[1])
    ax.fill_between(r2, m2 - s2, m2 + s2, color=_COLORS[1], alpha=0.2)
    ax.plot(r3, m3, label="ReverseAdaptive", color=_COLORS[2])
    ax.fill_between(r3, m3 - s3, m3 + s3, color=_COLORS[2], alpha=0.2)

    ax.axvline(8, color="gray", linestyle=":", linewidth=1.0, label="Two-Phase switch (round 8)")
    ax.axvline(6, color="black", linestyle="--", linewidth=1.0, label="ReverseAdaptive switch (round 6)")

    ax.set_xlabel("Round")
    ax.set_ylabel("Training loss")
    ax.set_xlim(1, 15)
    ax.set_title("Convergence trajectories (mean ± 1 std over seeds)")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.92)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig2_convergence")


def figure3_cumulative_comm() -> None:
    seed = 42
    paths = {
        "FLoRA": _latest_per_seed(
            f"results/raw/exp_flora_iid/flora/seed_{seed}/run_*/*/results.json"
        ).get(seed),
        "Two-Phase K=8": _latest_per_seed(
            f"results/raw/exp_two_phase_k8/two_phase/seed_{seed}/stage1_bidirectional/*/results.json"
        ).get(seed),
        "ReverseAdaptive": _latest_per_seed(
            f"results/raw/exp_reverse_adaptive_iid/reverse_adaptive/seed_{seed}/stage2_adaptive/*/results.json"
        ).get(seed),
    }
    if any(v is None for v in paths.values()):
        warnings.warn("Figure 3: missing seed-42 run for a method")
        return

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    for label, p in paths.items():
        assert p is not None
        rd = _load_rounds(p)
        xs = np.arange(1, len(rd) + 1)
        ys = [float(x["communication_mb"]) for x in rd]
        ax.plot(xs, ys, label=label, linewidth=1.8)

    ax.set_xlabel("Round")
    ax.set_ylabel("Cumulative communication (MB)")
    ax.set_xlim(1, 15)
    ax.set_ylim(0, 2600)
    ax.set_title("Cumulative communication (seed 42)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig3_cumulative_comm")


def figure4_downstream() -> None:
    tiny_base: Dict[str, float] = {}
    tb_path = REPO / "results/downstream/baseline_no_adapter_500.json"
    if tb_path.is_file():
        with open(tb_path, encoding="utf-8") as f:
            d0 = json.load(f)
        for k, v in (d0.get("benchmarks") or {}).items():
            tiny_base[k] = float(v["accuracy"])
    with open(REPO / "results/downstream/baseline_no_adapter_hellaswag.json", encoding="utf-8") as f:
        d1 = json.load(f)
    for k, v in (d1.get("benchmarks") or {}).items():
        tiny_base[k] = float(v["accuracy"])

    llama_base: Dict[str, float] = {}
    for fn in ("baseline_no_adapter_llama3_3b.json", "baseline_no_adapter_llama3_3b_hellaswag.json"):
        fp = REPO / "results/downstream" / fn
        if fp.is_file():
            with open(fp, encoding="utf-8") as f:
                d = json.load(f)
            for k, v in (d.get("benchmarks") or {}).items():
                llama_base[k] = float(v["accuracy"])

    tiny_dirs = {
        "FLoRA": REPO / "results/downstream/exp_flora_iid/flora/seed_42/stage3_checkpoint",
        "Two-Phase K=8": REPO / "results/downstream/exp_two_phase_k8/two_phase/seed_42/stage3_checkpoint",
        "ReverseAdaptive": REPO / "results/downstream/exp_reverse_adaptive_iid/reverse_adaptive/seed_42/stage3_checkpoint",
    }
    llama_dirs = {
        "FLoRA": REPO / "results/downstream/exp_llama3_flora_iid/flora/seed_42/stage5_llama3_full",
        "Two-Phase K=8": REPO / "results/downstream/exp_llama3_two_phase_k8_iid/two_phase/seed_42/stage5_llama3_full",
        "ReverseAdaptive": REPO / "results/downstream/exp_llama3_reverse_adaptive_iid/reverse_adaptive/seed_42/stage5_llama3_full",
    }

    tiny_bench = ["arc_easy", "boolq", "hellaswag"]
    llama_bench = ["mmlu", "arc_easy", "boolq", "hellaswag"]

    def series_for(
        dirs: Dict[str, Path], keys: List[str], base: Dict[str, float]
    ) -> Dict[str, List[float]]:
        out: Dict[str, List[float]] = {k: [] for k in keys}
        for label in ("Base", "FLoRA", "Two-Phase K=8", "ReverseAdaptive"):
            if label == "Base":
                vals = [base.get(k, float("nan")) for k in keys]
            else:
                m = _merge_downstream_jsons(dirs[label])
                vals = [m.get(k, float("nan")) for k in keys]
            for k, v in zip(keys, vals):
                out[k].append(v)
        return out

    tiny_data = series_for(tiny_dirs, tiny_bench, tiny_base)
    llama_data = series_for(llama_dirs, llama_bench, llama_base)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(tiny_bench))
    w = 0.2
    labels_row = ["Base", "FLoRA", "Two-Phase K=8", "ReverseAdaptive"]
    for i, lab in enumerate(labels_row):
        ys = [tiny_data[k][i] for k in tiny_bench]
        ax0.bar(x + (i - 1.5) * w, ys, width=w, label=lab)

    ax0.set_xticks(x)
    ax0.set_xticklabels(["ARC-Easy", "BoolQ", "HellaSwag"])
    ax0.set_ylabel("Accuracy")
    ax0.set_ylim(0, 0.75)
    ax0.set_title("TinyLlama-1.1B (500 examples each)")
    ax0.grid(True, axis="y", alpha=0.25)
    ax0.legend(fontsize=7, ncol=2)

    x2 = np.arange(len(llama_bench))
    for i, lab in enumerate(labels_row):
        ys = [llama_data[k][i] for k in llama_bench]
        ax1.bar(x2 + (i - 1.5) * w, ys, width=w, label=lab)

    ax1.set_xticks(x2)
    ax1.set_xticklabels(["MMLU", "ARC-Easy", "BoolQ", "HellaSwag"])
    ax1.set_ylim(0, 1.0)
    ax1.set_title("LLaMA-3.2-3B (500 examples each)")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(fontsize=7, ncol=2)

    fig.suptitle("Downstream zero-shot accuracy")
    fig.tight_layout()
    _save(fig, "fig4_downstream_accuracy")


def figure5_threshold_ablation() -> None:
    root = REPO / "results/raw/exp_reverse_adaptive_threshold_ablation/reverse_adaptive/seed_42"
    paths = sorted(glob.glob(str(root / "*" / "*" / "results.json")))
    rows = []
    for p in paths:
        data = _load_rounds(p)
        if not data:
            continue
        fin = data[-1]
        agg = fin.get("aggregator_stats") or {}
        thr = agg.get("switch_threshold")
        sw = agg.get("switch_round")
        if thr is None:
            continue
        rows.append((float(thr), float(sw), float(fin["avg_loss"])))
    if len(rows) < 3:
        warnings.warn("Figure 5: insufficient threshold runs")
        return
    rows.sort(key=lambda t: t[0])
    xs = np.array([r[0] for r in rows])
    sw = np.array([r[1] for r in rows])
    loss = np.array([r[2] for r in rows])

    fig, ax0 = plt.subplots(figsize=(7.0, 4.0))
    ax1 = ax0.twinx()
    ax0.plot(xs, sw, "o-", color=_COLORS[0], label="Switch round")
    ax1.plot(xs, loss, "s--", color=_COLORS[3], label="Final loss")

    ax0.set_xscale("log")
    ax0.set_xlabel("switch_threshold τ (log scale)")
    ax0.set_ylabel("Switch round", color=_COLORS[0])
    ax1.set_ylabel("Final loss", color=_COLORS[3])
    ax0.set_title("ReverseAdaptive threshold ablation (seed 42)")
    ax0.grid(True, alpha=0.25)

    h0, l0 = ax0.get_legend_handles_labels()
    h1, l1 = ax1.get_legend_handles_labels()
    ax0.legend(h0 + h1, l0 + l1, loc="center right", fontsize=8)
    fig.tight_layout()
    _save(fig, "fig5_threshold_ablation")


def figure6_scale_validation(rows: List[Dict[str, str]]) -> None:
    tiny = [
        r
        for r in rows
        if r.get("setting") == "iid"
        and r.get("scale") == "tinyllama_1b"
        and _f(r.get("total_mb", "") or "") is not None
        and _f(r.get("final_loss", "") or "") is not None
    ]

    def agg(exp: str, tag_filter: Optional[str]) -> Tuple[float, float, float, float]:
        rs = [r for r in tiny if r["exp_name"] == exp]
        if tag_filter:
            rs = [r for r in rs if tag_filter in r["tag"]]
        losses = [_f(r["final_loss"]) for r in rs]
        mbs = [_f(r["total_mb"]) for r in rs]
        losses = [x for x in losses if x is not None]
        mbs = [x for x in mbs if x is not None]
        if not losses:
            return (float("nan"),) * 4
        lm, ls = _mean_std(losses)
        mm, ms = _mean_std(mbs)
        return mm, ms, lm, ls

    mb_f, eb_f, lf, elf = agg("exp_flora_iid", "run_")
    mb_8, eb_8, l8, el8 = agg("exp_two_phase_k8", "stage1_bidirectional")
    mb_ra, eb_ra, l_ra, el_ra = agg("exp_reverse_adaptive_iid", "stage2_adaptive")

    def pt(exp: str) -> Tuple[float, float]:
        rs = [
            r
            for r in rows
            if r.get("exp_name") == exp
            and r.get("setting") == "iid"
            and r.get("scale") == "llama3_3b"
            and r.get("tag") == "stage5_llama3_full"
            and _f(r.get("total_mb", "") or "") is not None
        ]
        if not rs:
            return float("nan"), float("nan")
        r = max(rs, key=lambda x: x.get("timestamp", ""))
        return float(r["total_mb"]), float(r["final_loss"])

    lf_l, los_l = pt("exp_llama3_flora_iid")
    l8_mb, l8_ls = pt("exp_llama3_two_phase_k8_iid")
    lr_mb, lr_ls = pt("exp_llama3_reverse_adaptive_iid")

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.5, 4.0))

    ax0.errorbar([mb_f], [lf], xerr=[eb_f], yerr=[elf], fmt="o", color=_COLORS[0], capsize=3, label="FLoRA")
    ax0.errorbar([mb_8], [l8], xerr=[eb_8], yerr=[el8], fmt="^", color=_COLORS[2], capsize=3, label="Two-Phase K=8")
    ax0.errorbar([mb_ra], [l_ra], xerr=[eb_ra], yerr=[el_ra], fmt="D", color=_COLORS[3], capsize=3, label="ReverseAdaptive")
    ax0.plot([mb_f, mb_8, mb_ra], [lf, l8, l_ra], color="gray", linestyle=":", alpha=0.7)
    ax0.set_title("TinyLlama-1.1B (mean ± 1 std, 3 seeds)")
    ax0.set_xlabel("Total round-trip MB")
    ax0.set_ylabel("Final loss")
    ax0.grid(True, alpha=0.25)
    ax0.legend(fontsize=8)

    if not (np.isnan(lf_l) or np.isnan(l8_mb)):
        ax1.scatter([lf_l], [los_l], color=_COLORS[0], s=55, label="FLoRA")
        ax1.scatter([l8_mb], [l8_ls], color=_COLORS[2], s=55, marker="^", label="Two-Phase K=8")
        ax1.scatter([lr_mb], [lr_ls], color=_COLORS[3], s=55, marker="D", label="ReverseAdaptive")
        ax1.plot([lf_l, l8_mb, lr_mb], [los_l, l8_ls, lr_ls], color="gray", linestyle=":", alpha=0.6)

    ax1.set_title("LLaMA-3.2-3B (single seed)")
    ax1.set_xlabel("Total round-trip MB")
    ax1.set_ylabel("Final loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=8)

    fig.suptitle("Scale validation: communication versus final loss")
    fig.tight_layout()
    _save(fig, "fig6_scale_validation")


def main() -> None:
    os.chdir(REPO)
    rows = load_table(TABLE_PATH)
    figure1_pareto_iid(rows)
    figure2_convergence()
    figure3_cumulative_comm()
    figure4_downstream()
    figure5_threshold_ablation()
    figure6_scale_validation(rows)
    print(f"Figures directory: {OUTDIR}")


if __name__ == "__main__":
    main()
