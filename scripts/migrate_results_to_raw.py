#!/usr/bin/env python3
"""
Migrate legacy results layout into the organized results/raw/... hierarchy.

This script COPIES (does not delete) existing runs under:
  results/<run_dir>/results.json
into:
  results/raw/<experiment>/<method>/seed_unknown[_rX][_cY]/<timestamp>/results.json

It also writes a best-effort run_meta.json in the destination folder so future
analysis can attribute runs without relying on directory-name parsing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from glob import glob
from typing import Any, Dict, Optional, Tuple


_RUN_TAIL_RE = re.compile(
    r"_(fedit|ffa_lora|flora|flexlora|fedlora_adaptive)((?:_r\d+)?)((?:_c\d+)?)_(\d{8}_\d{6})$"
)


def _parse_legacy_dirname(dirname: str) -> Optional[Dict[str, Any]]:
    """
    Parse legacy run dir name:
      {slug}_{method}[_r{rank}][_c{clients}]_{YYYYMMDD}_{HHMMSS}
    """
    m = _RUN_TAIL_RE.search(dirname)
    if not m:
        return None
    method = m.group(1)
    r_part, c_part = m.group(2), m.group(3)
    timestamp = m.group(4)
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
        "experiment": slug,
        "method": method,
        "timestamp": timestamp,
        "lora_r": rank,
        "num_clients_override": clients,
    }


def _dest_seed_dir(parsed: Dict[str, Any]) -> str:
    parts = ["seed_unknown"]
    if parsed.get("lora_r") is not None:
        parts.append(f"r{parsed['lora_r']}")
    if parsed.get("num_clients_override") is not None:
        parts.append(f"c{parsed['num_clients_override']}")
    return "_".join(parts)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def migrate_one(
    results_root: str,
    legacy_run_dir: str,
    dry_run: bool = False,
    overwrite: bool = False,
) -> Tuple[bool, str]:
    dirname = os.path.basename(legacy_run_dir.rstrip(os.sep))
    parsed = _parse_legacy_dirname(dirname)
    if parsed is None:
        return False, f"skip (unparseable): {dirname}"

    src_results = os.path.join(legacy_run_dir, "results.json")
    if not os.path.exists(src_results):
        return False, f"skip (missing results.json): {dirname}"

    dest_dir = os.path.join(
        results_root,
        "raw",
        str(parsed["experiment"]),
        str(parsed["method"]),
        _dest_seed_dir(parsed),
        str(parsed["timestamp"]),
    )
    dest_results = os.path.join(dest_dir, "results.json")
    dest_meta = os.path.join(dest_dir, "run_meta.json")

    if os.path.exists(dest_results) and not overwrite:
        return False, f"skip (already migrated): {dirname}"

    if dry_run:
        return True, f"would migrate: {dirname} -> {os.path.relpath(dest_dir, results_root)}"

    _ensure_dir(dest_dir)
    shutil.copy2(src_results, dest_results)

    meta: Dict[str, Any] = {
        "experiment": parsed["experiment"],
        "method": parsed["method"],
        "seed": None,
        "timestamp": parsed["timestamp"],
        "device": None,
        "output_dir": os.path.relpath(dest_dir, os.getcwd()),
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
        "overrides": {
            "lora_r": parsed.get("lora_r"),
            "num_clients": parsed.get("num_clients_override"),
        },
        "migrated_from": os.path.relpath(legacy_run_dir, os.getcwd()),
    }
    with open(dest_meta, "w") as f:
        json.dump(meta, f, indent=2)

    return True, f"migrated: {dirname} -> {os.path.relpath(dest_dir, results_root)}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_root", default="results")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    results_root = os.path.abspath(args.results_root)

    # Legacy runs are immediate children of results/ (exclude new raw/)
    candidates = sorted(glob(os.path.join(results_root, "*")))
    legacy_dirs = [
        p
        for p in candidates
        if os.path.isdir(p) and os.path.basename(p) != "raw"
    ]

    migrated = 0
    skipped = 0
    for d in legacy_dirs:
        ok, msg = migrate_one(
            results_root=results_root,
            legacy_run_dir=d,
            dry_run=args.dry_run,
            overwrite=args.overwrite,
        )
        print(msg)
        if ok:
            migrated += 1
        else:
            skipped += 1

    print(f"\nDone. migrated={migrated}, skipped={skipped}")


if __name__ == "__main__":
    main()

