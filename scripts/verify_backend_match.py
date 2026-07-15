#!/usr/bin/env python3
"""
Neurocomputing Phase 0 (A.3 / Part C.7): compare MPS vs CUDA experiment outputs.

Compares final (and optionally per-round) training metrics from two results.json
files under an absolute tolerance of 0.01 (rtol=0).

Usage:
  python3 scripts/verify_backend_match.py \\
      --mps path/to/mps/results.json \\
      --cuda path/to/cuda/results.json

  # Or directories containing results.json:
  python3 scripts/verify_backend_match.py --mps <mps_run_dir> --cuda <cuda_run_dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

# Neurocomputing Phase 0 acceptance: absolute 0.01, no relative slack.
ATOL = 0.01
RTOL = 0.0


def _resolve_results(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = path / "results.json"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"No results.json at {path}")


def _load(path: Path) -> Any:
    with open(_resolve_results(path), encoding="utf-8") as f:
        return json.load(f)


def _rounds(data: Any) -> List[Dict[str, Any]]:
    # Older dumps are a bare list of round dicts
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("rounds"), list):
        return data["rounds"]
    return []


def _final_loss(data: Any) -> Optional[float]:
    rounds = _rounds(data)
    if rounds:
        last = rounds[-1]
        for key in ("avg_loss", "loss", "train_loss"):
            if key in last and last[key] is not None:
                return float(last[key])
    if isinstance(data, dict):
        for key in ("final_loss", "avg_loss"):
            if key in data and data[key] is not None:
                return float(data[key])
    return None


def _metric_pairs(mps: Any, cuda: Any) -> List[Tuple[str, float, float]]:
    pairs: List[Tuple[str, float, float]] = []
    m_loss = _final_loss(mps)
    c_loss = _final_loss(cuda)
    if m_loss is None or c_loss is None:
        raise ValueError("Could not extract final avg_loss from one or both results.json")
    pairs.append(("final_avg_loss", m_loss, c_loss))

    # Optional: communication should be protocol-deterministic (exact), but
    # still check under the same absolute band for uniformity.
    m_rounds = _rounds(mps)
    c_rounds = _rounds(cuda)
    if m_rounds and c_rounds:
        m_comm = m_rounds[-1].get("communication_mb")
        c_comm = c_rounds[-1].get("communication_mb")
        if m_comm is not None and c_comm is not None:
            pairs.append(("final_communication_mb", float(m_comm), float(c_comm)))

        n = min(len(m_rounds), len(c_rounds))
        for i in range(n):
            ml = m_rounds[i].get("avg_loss")
            cl = c_rounds[i].get("avg_loss")
            if ml is not None and cl is not None:
                pairs.append((f"round_{i+1}_avg_loss", float(ml), float(cl)))
    return pairs


def compare(
    mps_path: Path,
    cuda_path: Path,
    atol: float = ATOL,
    rtol: float = RTOL,
    rounds_only_final: bool = False,
) -> int:
    mps = _load(mps_path)
    cuda = _load(cuda_path)
    pairs = _metric_pairs(mps, cuda)
    if rounds_only_final:
        pairs = [p for p in pairs if p[0].startswith("final_")]

    print("Neurocomputing Phase 0 — backend match (MPS vs CUDA)")
    print(f"  MPS:  {_resolve_results(mps_path)}")
    print(f"  CUDA: {_resolve_results(cuda_path)}")
    print(f"  atol={atol}  rtol={rtol}")
    print()

    failures = 0
    for name, m_val, c_val in pairs:
        ok = bool(np.isclose(m_val, c_val, atol=atol, rtol=rtol))
        # Explicit form matching guide: torch/np allclose(atol=0.01, rtol=0)
        delta = abs(m_val - c_val)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  [{status}] {name}: mps={m_val:.6f}  cuda={c_val:.6f}  "
            f"|Δ|={delta:.6f}"
        )

    print()
    if failures:
        print(f"Result: {failures} failure(s) — do not proceed to Neurocomputing Phase 1.")
        return 1
    print("Result: all checks passed at atol=0.01.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare MPS vs CUDA federated LoRA results (atol=0.01)."
    )
    parser.add_argument("--mps", type=Path, required=True, help="MPS results.json or run dir")
    parser.add_argument("--cuda", type=Path, required=True, help="CUDA results.json or run dir")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only compare final_avg_loss / final_communication_mb",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=ATOL,
        help=f"Absolute tolerance (default {ATOL} per Neurocomputing Phase 0)",
    )
    args = parser.parse_args(argv)
    return compare(args.mps, args.cuda, atol=args.atol, rounds_only_final=args.final_only)


if __name__ == "__main__":
    sys.exit(main())
