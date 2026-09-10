#!/usr/bin/env python3
"""
Compare MPS vs CUDA experiment outputs.

Setting-aware absolute tolerances (rtol=0):
  - IID:     per-round and final loss atol=0.01 (Phase 0)
  - Non-IID: per-round and final loss atol=0.025, plus:
      * B-only / phase-switch round must match exactly (hard)
      * at most 2 consecutive rounds may have |Δloss| > 0.01
        (even when within the relaxed 0.025 band)

Communication is always checked at atol=0.01 (protocol should be exact).

Usage:
  python3 scripts/verify_backend_match.py \\
      --mps path/to/mps/results.json \\
      --cuda path/to/cuda/results.json \\
      --setting auto|iid|noniid
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ATOL_IID = 0.01
ATOL_NONIID = 0.025
ATOL_COMM = 0.01
ATOL_STRICT_STREAK = 0.01  # used for Non-IID consecutive-spike rule
MAX_CONSEC_OVER_STRICT = 2
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


def _switch_round(data: Any) -> Optional[int]:
    """
    Round when B-only / FFA switch activates (1-based).

    Prefer aggregator_stats.switch_round once present; else first round with
    broadcast_b_only=True. Returns None if never switches (e.g. pure FLoRA).
    """
    rounds = _rounds(data)
    for r in rounds:
        stats = r.get("aggregator_stats") or {}
        sr = stats.get("switch_round")
        if sr is not None:
            return int(sr)
        for ev in stats.get("events") or []:
            if isinstance(ev, dict) and ev.get("event") in (
                "switch_to_ffa",
                "switch",
            ):
                if ev.get("round") is not None:
                    return int(ev["round"])
    for r in rounds:
        if r.get("broadcast_b_only"):
            return int(r["round"])
    return None


def resolve_setting(setting: str, mps_path: Path, cuda_path: Path) -> str:
    s = (setting or "auto").strip().lower()
    if s in ("iid", "noniid"):
        return s
    if s != "auto":
        raise SystemExit(f"ERROR: unknown --setting {setting!r} (use auto|iid|noniid)")
    blob = f"{mps_path} {cuda_path}".lower()
    if "noniid" in blob or "non_iid" in blob or "alpha0" in blob:
        return "noniid"
    return "iid"


def _metric_pairs(mps: Any, cuda: Any) -> List[Tuple[str, float, float]]:
    pairs: List[Tuple[str, float, float]] = []
    m_loss = _final_loss(mps)
    c_loss = _final_loss(cuda)
    if m_loss is None or c_loss is None:
        raise ValueError("Could not extract final avg_loss from one or both results.json")
    pairs.append(("final_avg_loss", m_loss, c_loss))

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


def _check_consec_strict_spikes(
    round_deltas: List[Tuple[int, float]],
) -> Tuple[bool, str]:
    """Fail if any streak of rounds with |Δ| > 0.01 is longer than 2."""
    streak = 0
    max_streak = 0
    worst_end = 0
    for rnd, delta in round_deltas:
        if delta > ATOL_STRICT_STREAK:
            streak += 1
            if streak > max_streak:
                max_streak = streak
                worst_end = rnd
        else:
            streak = 0
    if max_streak > MAX_CONSEC_OVER_STRICT:
        start = worst_end - max_streak + 1
        return (
            False,
            f"longest streak with |Δ|>{ATOL_STRICT_STREAK} is {max_streak} "
            f"(rounds {start}-{worst_end}); max allowed consecutive is "
            f"{MAX_CONSEC_OVER_STRICT}",
        )
    return True, f"max consecutive |Δ|>{ATOL_STRICT_STREAK} streak={max_streak}"


def compare(
    mps_path: Path,
    cuda_path: Path,
    setting: str = "auto",
    rounds_only_final: bool = False,
) -> int:
    resolved = resolve_setting(setting, mps_path, cuda_path)
    loss_atol = ATOL_IID if resolved == "iid" else ATOL_NONIID

    mps = _load(mps_path)
    cuda = _load(cuda_path)
    pairs = _metric_pairs(mps, cuda)
    if rounds_only_final:
        pairs = [p for p in pairs if p[0].startswith("final_")]

    print("Backend match (MPS vs CUDA)")
    print(f"  MPS:  {_resolve_results(mps_path)}")
    print(f"  CUDA: {_resolve_results(cuda_path)}")
    print(f"  setting={resolved}  loss_atol={loss_atol}  comm_atol={ATOL_COMM}  rtol={RTOL}")
    if resolved == "noniid":
        print(
            f"  non-IID extras: switch_round exact match; "
            f"≤{MAX_CONSEC_OVER_STRICT} consecutive rounds with |Δloss|>{ATOL_STRICT_STREAK}"
        )
    print()

    failures = 0
    round_deltas: List[Tuple[int, float]] = []

    for name, m_val, c_val in pairs:
        delta = abs(m_val - c_val)
        if name == "final_communication_mb":
            atol = ATOL_COMM
        else:
            atol = loss_atol
        ok = bool(np.isclose(m_val, c_val, atol=atol, rtol=RTOL))
        if name.startswith("round_") and name.endswith("_avg_loss"):
            rnd = int(name.split("_")[1])
            round_deltas.append((rnd, delta))
        status = "PASS" if ok else "FAIL"
        if not ok:
            failures += 1
        print(
            f"  [{status}] {name}: mps={m_val:.6f}  cuda={c_val:.6f}  "
            f"|Δ|={delta:.6f}  (atol={atol})"
        )

    # Non-IID hard extras (even with --final-only we still check switch if rounds exist)
    if resolved == "noniid" and not rounds_only_final:
        m_sw = _switch_round(mps)
        c_sw = _switch_round(cuda)
        sw_ok = m_sw == c_sw
        status = "PASS" if sw_ok else "FAIL"
        if not sw_ok:
            failures += 1
        print(
            f"  [{status}] switch_round_exact: mps={m_sw}  cuda={c_sw}  "
            f"(hard match, no tolerance)"
        )

        streak_ok, streak_msg = _check_consec_strict_spikes(round_deltas)
        status = "PASS" if streak_ok else "FAIL"
        if not streak_ok:
            failures += 1
        print(f"  [{status}] noniid_consec_strict_spikes: {streak_msg}")

    print()
    if failures:
        print(
            f"Result: {failures} failure(s)."
        )
        return 1
    print(f"Result: all checks passed (setting={resolved}, loss_atol={loss_atol}).")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare MPS vs CUDA federated LoRA results (setting-aware atol)."
    )
    parser.add_argument("--mps", type=Path, required=True, help="MPS results.json or run dir")
    parser.add_argument("--cuda", type=Path, required=True, help="CUDA results.json or run dir")
    parser.add_argument(
        "--setting",
        default="auto",
        choices=["auto", "iid", "noniid"],
        help="Tolerance profile (auto detects 'noniid' in paths)",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Only compare final_avg_loss / final_communication_mb (skips Non-IID extras)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Deprecated: ignored. Use --setting iid|noniid instead.",
    )
    args = parser.parse_args(argv)
    if args.atol is not None:
        print(
            f"Warning: --atol={args.atol} is deprecated and ignored; "
            "use --setting iid|noniid.",
            file=sys.stderr,
        )
    return compare(
        args.mps,
        args.cuda,
        setting=args.setting,
        rounds_only_final=args.final_only,
    )


if __name__ == "__main__":
    sys.exit(main())
