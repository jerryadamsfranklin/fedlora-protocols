#!/usr/bin/env python3
"""
Neurocomputing Phase 1 — enumerate the canonical 34 MPS training runs.

Walks results/raw/, resolves the paper-suite canonical results.json for each
(config × seed), and writes a CSV checklist for CUDA reruns.

Default mode (--suite paper) emits exactly the 34 runs used by the paper /
figure pipelines (FLoRA, Two-Phase K=8, ReverseAdaptive; TinyLlama + LLaMA-3.2-3B).
Rows are ordered for the Phase 1 early-gate protocol:
  1) cheapest Non-IID gate
  2) FLoRA IID seed 42 gate
  3) remaining 32

Usage:
  python3 scripts/list_existing_runs.py \\
      --results-dir results/raw/ \\
      --output docs/phase1_run_manifest.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


REPO = Path(__file__).resolve().parents[1]


@dataclass
class SuiteSpec:
    experiment: str
    method: str
    model: str  # TinyLlama-1.1B | LLaMA-3.2-3B
    data_setting: str
    seed: int
    config: str
    tag_glob: str  # relative to seed dir: e.g. "run_*" or "stage2_adaptive"
    switch_threshold: Optional[float] = None
    notes: str = ""


# Canonical Phase 1 / paper suite (exactly 34). Tag globs match
# scripts/generate_paper_figures.py and scripts/statistical_analysis.py.
PAPER_SUITE: List[SuiteSpec] = []

def _add(
    experiment: str,
    method: str,
    model: str,
    data_setting: str,
    seeds: Sequence[int],
    config: str,
    tag_glob: str,
    switch_threshold: Optional[float] = None,
    notes: str = "",
) -> None:
    for seed in seeds:
        PAPER_SUITE.append(
            SuiteSpec(
                experiment=experiment,
                method=method,
                model=model,
                data_setting=data_setting,
                seed=int(seed),
                config=config,
                tag_glob=tag_glob,
                switch_threshold=switch_threshold,
                notes=notes,
            )
        )


SEEDS_MAIN = (42, 123, 456)

_add("exp_flora_iid", "flora", "TinyLlama-1.1B", "iid", SEEDS_MAIN,
     "config/exp_flora_iid.yaml", "run_*")
_add("exp_flora_noniid_alpha05", "flora", "TinyLlama-1.1B", "noniid_alpha0.5", SEEDS_MAIN,
     "config/exp_flora_noniid_alpha05.yaml", "run_*")
_add("exp_two_phase_k8", "two_phase", "TinyLlama-1.1B", "iid", SEEDS_MAIN,
     "config/exp_two_phase_k8.yaml", "stage1_bidirectional")
_add("exp_two_phase_k8_noniid_alpha05", "two_phase", "TinyLlama-1.1B", "noniid_alpha0.5", SEEDS_MAIN,
     "config/exp_two_phase_k8_noniid_alpha05.yaml", "run_*")
_add("exp_reverse_adaptive_iid", "reverse_adaptive", "TinyLlama-1.1B", "iid", SEEDS_MAIN,
     "config/exp_reverse_adaptive_iid.yaml", "stage2_adaptive")
_add("exp_reverse_adaptive_noniid_alpha05", "reverse_adaptive", "TinyLlama-1.1B", "noniid_alpha0.5", SEEDS_MAIN,
     "config/exp_reverse_adaptive_noniid_alpha05.yaml", "stage2_adaptive")
_add("exp_reverse_adaptive_noniid_alpha01", "reverse_adaptive", "TinyLlama-1.1B", "noniid_alpha0.1", SEEDS_MAIN,
     "config/exp_reverse_adaptive_noniid_alpha01.yaml", "stage2_adaptive")

# Fig 1 / Fig 5 threshold points (4 runs, seed 42) used by generate_paper_figures.py
for tau, tag in [
    (0.001, "stage2_threshold_extended_0.001"),
    (0.002, "stage2_threshold_extended_0.002"),
    (0.005, "stage2_threshold_0.005"),
    (0.01, "stage2_threshold_0.01"),
]:
    _add(
        "exp_reverse_adaptive_threshold_ablation",
        "reverse_adaptive",
        "TinyLlama-1.1B",
        f"iid_threshold_tau={tau}",
        (42,),
        "config/exp_reverse_adaptive_threshold_ablation.yaml",
        tag,
        switch_threshold=tau,
        notes="threshold ablation; pass --switch_threshold or --override on CUDA rerun",
    )

_add("exp_llama3_flora_iid", "flora", "LLaMA-3.2-3B", "iid", SEEDS_MAIN,
     "config/exp_llama3_flora_iid.yaml", "stage5_llama3*")
_add("exp_llama3_two_phase_k8_iid", "two_phase", "LLaMA-3.2-3B", "iid", SEEDS_MAIN,
     "config/exp_llama3_two_phase_k8_iid.yaml", "stage5_llama3*")
_add("exp_llama3_reverse_adaptive_iid", "reverse_adaptive", "LLaMA-3.2-3B", "iid", SEEDS_MAIN,
     "config/exp_llama3_reverse_adaptive_iid.yaml", "stage5_llama3*")

assert len(PAPER_SUITE) == 34, f"PAPER_SUITE must have 34 specs, got {len(PAPER_SUITE)}"


@dataclass
class ManifestRow:
    exec_order: int
    gate_role: str  # gate1_noniid | gate2_flora_or_twophase | remaining
    model: str
    method: str
    data_setting: str
    seed: int
    experiment: str
    config: str
    switch_threshold: str
    mps_results_json: str
    mps_device: str
    final_avg_loss: str
    final_communication_mb: str
    notes: str


def _latest_results(seed_dir: Path, tag_glob: str) -> Optional[Path]:
    matches = list(seed_dir.glob(f"{tag_glob}/*/results.json"))
    if not matches:
        return None
    return max(matches, key=lambda p: p.parent.name)


def _read_meta_device(run_dir: Path) -> str:
    meta = run_dir / "run_meta.json"
    if not meta.is_file():
        return ""
    try:
        return str(json.loads(meta.read_text()).get("device", "") or "")
    except Exception:
        return ""


def _final_metrics(results_path: Path) -> Tuple[str, str]:
    try:
        data = json.loads(results_path.read_text())
    except Exception:
        return "", ""
    rounds = data if isinstance(data, list) else data.get("rounds") or []
    if not rounds:
        return "", ""
    last = rounds[-1]
    loss = last.get("avg_loss", "")
    comm = last.get("communication_mb", "")
    return ("" if loss == "" else f"{float(loss):.6f}",
            "" if comm == "" else f"{float(comm):.4f}")


def resolve_suite(results_dir: Path) -> List[Tuple[SuiteSpec, Path]]:
    resolved: List[Tuple[SuiteSpec, Path]] = []
    missing: List[SuiteSpec] = []
    for spec in PAPER_SUITE:
        seed_dir = results_dir / spec.experiment / spec.method / f"seed_{spec.seed}"
        path = _latest_results(seed_dir, spec.tag_glob)
        if path is None:
            missing.append(spec)
        else:
            resolved.append((spec, path))
    if missing:
        lines = [
            f"  - {m.experiment} seed={m.seed} tag={m.tag_glob!r} ({m.data_setting})"
            for m in missing
        ]
        raise SystemExit(
            "ERROR: could not resolve MPS results.json for "
            f"{len(missing)} suite member(s):\n" + "\n".join(lines)
        )
    return resolved


def gate_order(resolved: List[Tuple[SuiteSpec, Path]]) -> List[Tuple[str, SuiteSpec, Path]]:
    """
    Phase 1 early-gate ordering:
      gate1: cheapest TinyLlama non-IID (prefer ReverseAdaptive α=0.5 seed 42)
      gate2: FLoRA IID TinyLlama seed 42 (else Two-Phase K=8 IID seed 42)
      remaining: everything else
    """
    remaining = list(resolved)

    def take(pred) -> Optional[Tuple[SuiteSpec, Path]]:
        for i, item in enumerate(remaining):
            if pred(item[0]):
                return remaining.pop(i)
        return None

    gate1 = take(
        lambda s: (
            s.model == "TinyLlama-1.1B"
            and "noniid" in s.data_setting
            and s.method == "reverse_adaptive"
            and s.seed == 42
            and "threshold" not in s.data_setting
        )
    )
    if gate1 is None:
        gate1 = take(
            lambda s: s.model == "TinyLlama-1.1B" and "noniid" in s.data_setting and s.seed == 42
        )
    if gate1 is None:
        gate1 = take(lambda s: "noniid" in s.data_setting)

    gate2 = take(
        lambda s: s.model == "TinyLlama-1.1B" and s.method == "flora" and s.data_setting == "iid" and s.seed == 42
    )
    if gate2 is None:
        gate2 = take(
            lambda s: (
                s.model == "TinyLlama-1.1B"
                and s.method == "two_phase"
                and s.data_setting == "iid"
                and s.seed == 42
            )
        )

    if gate1 is None or gate2 is None:
        raise SystemExit("ERROR: could not select gate1 (non-IID) and/or gate2 (FLoRA/Two-Phase) runs")

    ordered: List[Tuple[str, SuiteSpec, Path]] = [
        ("gate1_noniid", gate1[0], gate1[1]),
        ("gate2_flora_or_twophase", gate2[0], gate2[1]),
    ]
    # Prefer TinyLlama remaining then LLaMA-3.2-3B for rental batching
    rem_sorted = sorted(
        remaining,
        key=lambda sp: (
            0 if sp[0].model.startswith("TinyLlama") else 1,
            sp[0].method,
            sp[0].data_setting,
            sp[0].seed,
        ),
    )
    for spec, path in rem_sorted:
        ordered.append(("remaining", spec, path))
    return ordered


def build_rows(
    ordered: List[Tuple[str, SuiteSpec, Path]],
    repo_root: Path,
) -> List[ManifestRow]:
    rows: List[ManifestRow] = []
    for i, (role, spec, path) in enumerate(ordered, start=1):
        loss, comm = _final_metrics(path)
        try:
            rel = str(path.resolve().relative_to(repo_root.resolve()).as_posix())
        except ValueError:
            rel = str(path.as_posix())
        rows.append(
            ManifestRow(
                exec_order=i,
                gate_role=role,
                model=spec.model,
                method=spec.method,
                data_setting=spec.data_setting,
                seed=spec.seed,
                experiment=spec.experiment,
                config=spec.config,
                switch_threshold="" if spec.switch_threshold is None else str(spec.switch_threshold),
                mps_results_json=rel,
                mps_device=_read_meta_device(path.parent),
                final_avg_loss=loss,
                final_communication_mb=comm,
                notes=spec.notes,
            )
        )
    return rows


def write_csv(rows: Sequence[ManifestRow], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise SystemExit("ERROR: no rows to write")
    fieldnames = list(asdict(rows[0]).keys())
    with output.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO / "results" / "raw",
        help="Root results directory (default: results/raw)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO / "docs" / "phase1_run_manifest.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--suite",
        choices=["paper"],
        default="paper",
        help="Which suite definition to emit (only 'paper' = 34 canonical runs)",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=34,
        help="Fail if resolved row count != this (Phase 1 guide requires 34)",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    if not results_dir.is_absolute():
        results_dir = (Path.cwd() / results_dir).resolve()
    if not results_dir.is_dir():
        raise SystemExit(f"ERROR: results dir not found: {results_dir}")

    if args.suite != "paper":
        raise SystemExit("Only --suite paper is implemented")

    resolved = resolve_suite(results_dir)
    if len(resolved) != args.expected_count:
        raise SystemExit(
            f"ERROR: expected {args.expected_count} runs, resolved {len(resolved)}. "
            "Stop and reconcile with the maintainer before renting GPUs."
        )

    ordered = gate_order(resolved)
    rows = build_rows(ordered, REPO)
    out = args.output if args.output.is_absolute() else Path.cwd() / args.output
    write_csv(rows, out)

    print(f"Wrote {len(rows)} runs -> {out}")
    print()
    print("Gate order:")
    for r in rows[:2]:
        print(
            f"  [{r.gate_role}] #{r.exec_order} {r.model} {r.method} "
            f"{r.data_setting} seed={r.seed}"
        )
        print(f"      MPS: {r.mps_results_json}")
    print(f"  ... plus {len(rows) - 2} remaining runs")
    print()
    print("Breakdown:")
    from collections import Counter
    c = Counter((r.model, r.method) for r in rows)
    for (model, method), n in sorted(c.items()):
        print(f"  {model:16} {method:18} {n}")


if __name__ == "__main__":
    main()
