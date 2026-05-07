"""
Build the master results CSV by walking results/raw/ and results/downstream/.

Output: analysis/final_results_table.csv with one row per results.json under raw/.
"""

from __future__ import annotations

import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TIMESTAMP_RE = re.compile(r"^\d{8}_\d{6}$")


def parse_path(results_json_path: str) -> Optional[Dict[str, str]]:
    """Extract exp_name, method, seed, tag, timestamp from any nested results.json path."""
    path = Path(results_json_path).resolve()
    parts = path.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    if parts[-1] != "results.json":
        return None
    timestamp = parts[-2]
    if not TIMESTAMP_RE.match(timestamp):
        return None
    mid = parts[idx + 1 : -2]
    if len(mid) < 3:
        return None
    exp_name = mid[0]
    method = mid[1]
    seed_part = mid[2]
    if not seed_part.startswith("seed_"):
        return None
    seed = seed_part.replace("seed_", "")
    tag_parts = mid[3:]
    tag = "/".join(tag_parts)
    return {
        "exp_name": exp_name,
        "method": method,
        "seed": seed,
        "tag": tag,
        "timestamp": timestamp,
    }


def get_setting(exp_name: str) -> Tuple[str, str]:
    """Map exp_name to (setting, scale)."""
    scale = "llama3_3b" if "llama3" in exp_name else "tinyllama_1b"
    if "noniid_alpha01" in exp_name:
        setting = "noniid_alpha01"
    elif "noniid_alpha05" in exp_name:
        setting = "noniid_alpha05"
    else:
        setting = "iid"
    return setting, scale


def load_final_metrics(results_json_path: str) -> Dict[str, Any]:
    with open(results_json_path, encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        return {}
    final = data[-1]
    agg = final.get("aggregator_stats") or {}
    switch_round = agg.get("switch_round") if agg else None
    return {
        "final_loss": final.get("avg_loss"),
        "total_mb": final.get("communication_mb"),
        "upload_mb": final.get("upload_mb"),
        "download_mb": final.get("download_mb"),
        "switch_round": switch_round,
    }


def find_downstream(parsed: Dict[str, str], results_root: str = "results") -> Dict[str, str]:
    """Merge accuracy fields from any JSON under downstream/<exp>/<method>/seed_<seed>/<tag>/."""
    base_path = Path(results_root) / "downstream" / parsed["exp_name"] / parsed["method"]
    base_path = base_path / f"seed_{parsed['seed']}" / parsed["tag"]
    out = {
        "mmlu_acc": "",
        "arc_easy_acc": "",
        "boolq_acc": "",
        "hellaswag_acc": "",
    }
    if not base_path.is_dir():
        return out

    for fp in sorted(base_path.rglob("*.json")):
        try:
            with open(fp, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        bms = data.get("benchmarks", {})
        for key in ("mmlu", "arc_easy", "boolq", "hellaswag"):
            col = f"{key}_acc"
            if col in out and not out[col] and key in bms:
                acc = bms[key].get("accuracy")
                if acc is not None:
                    out[col] = acc
    return out


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    raw_root = repo_root / "results" / "raw"
    rows: List[Dict[str, Any]] = []

    paths = sorted(raw_root.rglob("results.json"))
    print(f"Found {len(paths)} results.json files under {raw_root}")

    for path in paths:
        sp = str(path)
        parsed = parse_path(sp)
        if parsed is None:
            print(f"  skip (unparseable): {path.relative_to(repo_root)}")
            continue
        setting, scale = get_setting(parsed["exp_name"])
        metrics = load_final_metrics(sp)
        downstream = find_downstream(parsed, str(repo_root / "results"))
        rows.append(
            {
                "exp_name": parsed["exp_name"],
                "method": parsed["method"],
                "setting": setting,
                "scale": scale,
                "seed": parsed["seed"],
                "tag": parsed["tag"],
                "timestamp": parsed["timestamp"],
                "final_loss": metrics.get("final_loss", ""),
                "total_mb": metrics.get("total_mb", ""),
                "upload_mb": metrics.get("upload_mb", ""),
                "download_mb": metrics.get("download_mb", ""),
                "switch_round": metrics.get("switch_round", "") or "",
                **downstream,
            }
        )

    out_dir = repo_root / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "final_results_table.csv"
    if not rows:
        print("No rows to write.")
        return

    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Wrote {len(rows)} rows to {out_path}")


if __name__ == "__main__":
    main()
