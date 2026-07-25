#!/usr/bin/env python3
"""
Build / run Phase 1 CUDA rerun commands from docs/phase1_run_manifest.csv.

Critical: threshold-ablation rows share one YAML; this script ALWAYS passes
`--switch_threshold` when the CSV column is non-empty so CUDA reruns cannot
silently reuse the config default (0.01).

Usage (dry-run is default — prints commands only):
  python3 scripts/run_phase1_from_manifest.py --rows 1,2
  python3 scripts/run_phase1_from_manifest.py --gate-only
  python3 scripts/run_phase1_from_manifest.py --rows 11-14   # threshold set

To execute on a CUDA host:
  python3 scripts/run_phase1_from_manifest.py --rows 1 --execute
"""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence


REPO = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPO / "docs" / "phase1_run_manifest.csv"


def parse_rows_spec(spec: str, n: int) -> List[int]:
    """Parse '1,2,5-7' into 1-based exec_order list."""
    out: List[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            lo, hi = int(a), int(b)
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if i < 1 or i > n]
    if bad:
        raise SystemExit(f"ERROR: row indices out of range 1..{n}: {bad}")
    return out


def load_manifest(path: Path) -> List[dict]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"ERROR: empty manifest {path}")
    return rows


def build_command(row: dict, tag: str, device: str) -> List[str]:
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "run_experiment.py"),
        "--config",
        row["config"],
        "--seed",
        str(row["seed"]),
        "--device",
        device,
        "--tag",
        tag,
    ]
    thr = (row.get("switch_threshold") or "").strip()
    if thr:
        cmd.extend(["--switch_threshold", thr])
    return cmd


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--rows", default=None, help="1-based exec_order list, e.g. 1,2 or 11-14")
    p.add_argument("--gate-only", action="store_true", help="Run/print gate rows only")
    p.add_argument("--tag", default="phase1_cuda_rerun")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run commands (default: dry-run / print only)",
    )
    args = p.parse_args()

    rows = load_manifest(args.manifest if args.manifest.is_absolute() else Path.cwd() / args.manifest)
    by_order = {int(r["exec_order"]): r for r in rows}

    if args.gate_only:
        selected = [
            by_order[int(r["exec_order"])]
            for r in rows
            if r["gate_role"] in ("gate1_noniid", "gate2_flora_or_twophase")
        ]
        selected.sort(key=lambda r: int(r["exec_order"]))
    elif args.rows:
        idxs = parse_rows_spec(args.rows, len(rows))
        selected = [by_order[i] for i in idxs]
    else:
        raise SystemExit("Specify --gate-only or --rows")

    for row in selected:
        cmd = build_command(row, args.tag, args.device)
        thr = (row.get("switch_threshold") or "").strip()
        print(
            f"# exec_order={row['exec_order']} gate={row['gate_role']} "
            f"{row['experiment']} seed={row['seed']} "
            f"switch_threshold={thr or '(config default)'}"
        )
        print(f"# MPS: {row['mps_results_json']}")
        print(shlex.join(cmd))
        if thr:
            print(f"# OK: will pass --switch_threshold {thr}")
        elif "threshold" in row["data_setting"]:
            raise SystemExit(
                f"ERROR: threshold row {row['exec_order']} missing switch_threshold in CSV"
            )
        print()
        if args.execute:
            print(f">>> executing exec_order={row['exec_order']} ...")
            subprocess.check_call(cmd, cwd=str(REPO))


if __name__ == "__main__":
    main()
