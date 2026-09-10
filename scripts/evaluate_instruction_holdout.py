#!/usr/bin/env python3
"""
Evaluate held-out instruction-following loss/perplexity for LoRA checkpoints.

Supports:
1) Single checkpoint evaluation
2) Sweep mode via discovered checkpoints under results/raw
3) Sweep mode via manifest CSV rows (e.g., results/run_checklist.csv)

Primary use:
- Held-out metric on training distribution (default Alpaca train[3000:3500]).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.lora_model import FederatedLoRAModel


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = REPO_ROOT / "results" / "raw"
DOWNSTREAM_INSTR_ROOT = REPO_ROOT / "results" / "downstream_instruction"
DEFAULT_BASE = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DEFAULT_TARGET_MODULES = ["q_proj", "v_proj"]


@dataclass
class CheckpointSpec:
    checkpoint: Path
    exp_name: str
    method: str
    seed: str
    tag: str
    timestamp: str


def resolve_device(choice: str) -> str:
    choice = (choice or "auto").strip().lower()
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: --device cuda requested but CUDA is unavailable.")
        return "cuda"
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("ERROR: --device mps requested but MPS is unavailable.")
        return "mps"
    if choice == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def infer_target_modules_from_state(state: Dict[str, torch.Tensor]) -> List[str]:
    known = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    return [name for name in known if any(f".{name}." in k for k in state)]


def parse_checkpoint_path(path: Path) -> Optional[CheckpointSpec]:
    """Parse results/raw/.../final_adapter_state.pt into metadata."""
    p = path.resolve()
    if p.name != "final_adapter_state.pt":
        return None
    parts = p.parts
    try:
        idx = parts.index("raw")
    except ValueError:
        return None
    rel = parts[idx + 1 :]
    # exp/method/seed_*/.../timestamp/final_adapter_state.pt
    if len(rel) < 6:
        return None
    exp_name, method, seed_part = rel[0], rel[1], rel[2]
    if not seed_part.startswith("seed_"):
        return None
    timestamp = rel[-2]
    tag_parts = rel[3:-2]
    return CheckpointSpec(
        checkpoint=p,
        exp_name=exp_name,
        method=method,
        seed=seed_part.replace("seed_", ""),
        tag="/".join(tag_parts),
        timestamp=timestamp,
    )


def discover_checkpoints(root: Path) -> List[CheckpointSpec]:
    out: List[CheckpointSpec] = []
    for ckpt in sorted(root.rglob("final_adapter_state.pt")):
        parsed = parse_checkpoint_path(ckpt)
        if parsed is not None:
            out.append(parsed)
    return out


def load_manifest_checkpoints(manifest_csv: Path) -> List[CheckpointSpec]:
    """
    Load checkpoints from manifest rows with mps_results_json or results_json path.
    Converts each .../results.json to .../final_adapter_state.pt.
    """
    rows: List[CheckpointSpec] = []
    with manifest_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_path = row.get("mps_results_json") or row.get("results_json") or ""
            if not raw_path:
                continue
            rp = Path(raw_path)
            if not rp.is_absolute():
                rp = (REPO_ROOT / rp).resolve()
            if rp.name != "results.json":
                continue
            ckpt = rp.parent / "final_adapter_state.pt"
            parsed = parse_checkpoint_path(ckpt)
            if parsed is not None:
                rows.append(parsed)
    # Deduplicate by absolute checkpoint path
    seen = set()
    uniq: List[CheckpointSpec] = []
    for r in rows:
        key = str(r.checkpoint)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    return uniq


def get_checkpoint_config(ckpt: Path) -> Dict[str, Any]:
    cfg_path = ckpt.parent / "config_merged.yaml"
    if not cfg_path.is_file():
        return {}
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def format_instruction_texts(examples: Dict[str, List[Any]]) -> List[str]:
    cols = set(examples.keys())
    if {"instruction", "output"}.issubset(cols):
        return [
            f"### Instruction:\n{inst}\n\n### Response:\n{out}"
            for inst, out in zip(examples["instruction"], examples["output"])
        ]
    if {"instruction", "response"}.issubset(cols):
        return [
            f"### Instruction:\n{inst}\n\n### Response:\n{resp}"
            for inst, resp in zip(examples["instruction"], examples["response"])
        ]
    if {"prompt", "response"}.issubset(cols):
        return [
            f"### Instruction:\n{prompt}\n\n### Response:\n{resp}"
            for prompt, resp in zip(examples["prompt"], examples["response"])
        ]
    if "text" in cols:
        return [str(t) for t in examples["text"]]
    if {"question", "answer"}.issubset(cols):
        return [
            f"### Instruction:\n{q}\n\n### Response:\n{a}"
            for q, a in zip(examples["question"], examples["answer"])
        ]
    # Fallback: stringify first field
    first = next(iter(examples.keys()))
    return [str(x) for x in examples[first]]


def build_eval_loader(
    tokenizer,
    dataset_name: str,
    split: str,
    start_index: int,
    num_examples: int,
    max_seq_length: int,
    eval_batch_size: int,
) -> DataLoader:
    ds = load_dataset(dataset_name, split=split)
    end = min(start_index + num_examples, len(ds))
    if start_index >= len(ds):
        raise SystemExit(
            f"ERROR: start_index={start_index} out of range for {dataset_name}/{split} "
            f"(len={len(ds)})."
        )
    ds = ds.select(range(start_index, end))

    def _to_text(examples: Dict[str, List[Any]]) -> Dict[str, List[str]]:
        return {"text": format_instruction_texts(examples)}

    ds = ds.map(_to_text, batched=True)
    ds = ds.map(
        lambda x: tokenizer(
            x["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        ),
        batched=True,
        remove_columns=ds.column_names,
    )
    ds.set_format("torch")
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    return DataLoader(
        ds,
        batch_size=eval_batch_size,
        shuffle=False,
        collate_fn=collator,
    )


@torch.no_grad()
def evaluate_loss(model, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        labels = batch.get("labels")
        if labels is None:
            continue
        n = int((labels != -100).sum().item())
        total_loss += float(out.loss.item()) * n
        total_tokens += n
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
    ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return {"loss": avg_loss, "perplexity": ppl, "tokens": float(total_tokens)}


def out_path_for(spec: CheckpointSpec) -> Path:
    base = DOWNSTREAM_INSTR_ROOT / spec.exp_name / spec.method / f"seed_{spec.seed}"
    if spec.tag:
        base = base / spec.tag
    return base / spec.timestamp / "instruction_holdout.json"


def should_include(
    spec: CheckpointSpec,
    include_exps: Optional[set[str]],
    include_methods: Optional[set[str]],
) -> bool:
    if include_exps and spec.exp_name not in include_exps:
        return False
    if include_methods and spec.method not in include_methods:
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Single checkpoint path.")
    parser.add_argument(
        "--manifest-csv",
        default=None,
        help="Use checkpoints from manifest CSV (mps_results_json/results_json).",
    )
    parser.add_argument(
        "--discover-root",
        default=str(RAW_ROOT),
        help="Root for checkpoint discovery in sweep mode.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-exp", action="append", default=None)
    parser.add_argument("--include-method", action="append", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--dataset", default="tatsu-lab/alpaca")
    parser.add_argument("--split", default="train")
    parser.add_argument("--start-index", type=int, default=3000)
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=4)
    parser.add_argument("--base-model", default=None, help="Override base model for all evals.")
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=None,
        help="Override target modules for all evals.",
    )
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument(
        "--summary-csv",
        default="analysis/instruction_holdout_table.csv",
        help="Consolidated output CSV path (repo-relative or absolute).",
    )
    args = parser.parse_args()

    device = resolve_device(args.device)
    include_exps = set(args.include_exp or [])
    include_methods = set(args.include_method or [])

    # Resolve checkpoints
    specs: List[CheckpointSpec]
    if args.checkpoint:
        p = Path(args.checkpoint)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        parsed = parse_checkpoint_path(p)
        if parsed is None:
            raise SystemExit(f"ERROR: could not parse checkpoint path: {p}")
        specs = [parsed]
    elif args.manifest_csv:
        manifest = Path(args.manifest_csv)
        if not manifest.is_absolute():
            manifest = (Path.cwd() / manifest).resolve()
        specs = load_manifest_checkpoints(manifest)
    else:
        root = Path(args.discover_root)
        if not root.is_absolute():
            root = (Path.cwd() / root).resolve()
        specs = discover_checkpoints(root)

    specs = [s for s in specs if should_include(s, include_exps, include_methods)]
    if args.limit is not None:
        specs = specs[: args.limit]
    if not specs:
        raise SystemExit("No checkpoints selected.")

    print(f"Selected checkpoints: {len(specs)}")
    print(f"Device: {device}")
    print(
        f"Held-out: {args.dataset} {args.split}[{args.start_index}:{args.start_index + args.num_examples}]"
    )

    rows: List[Dict[str, Any]] = []
    # Cache base loss by eval configuration to avoid recomputing.
    base_cache: Dict[Tuple[str, str, str, int, int, int, int], Dict[str, float]] = {}

    for i, spec in enumerate(specs, start=1):
        if not spec.checkpoint.is_file():
            print(f"[{i}/{len(specs)}] missing checkpoint: {spec.checkpoint}")
            continue
        out_path = out_path_for(spec)
        if args.skip_existing and out_path.is_file():
            print(f"[{i}/{len(specs)}] skip existing: {out_path}")
            with out_path.open("r", encoding="utf-8") as f:
                existing = json.load(f)
            rows.append(existing.get("row", {}))
            continue

        cfg = get_checkpoint_config(spec.checkpoint)
        model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
        lora_cfg = cfg.get("lora", {}) if isinstance(cfg, dict) else {}
        train_cfg = cfg.get("training", {}) if isinstance(cfg, dict) else {}
        data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}

        base_model = args.base_model or model_cfg.get("name") or DEFAULT_BASE
        lora_r = args.lora_r if args.lora_r is not None else int(lora_cfg.get("r", 16))
        lora_alpha = (
            args.lora_alpha
            if args.lora_alpha is not None
            else int(lora_cfg.get("lora_alpha", 32))
        )
        dataset_name = str(data_cfg.get("dataset_name") or args.dataset)
        split = str(data_cfg.get("dataset_split") or args.split)
        max_seq_length = int(train_cfg.get("max_seq_length", args.max_seq_length))

        state = torch.load(spec.checkpoint, map_location="cpu")
        target_modules = args.target_modules
        if target_modules is None:
            inferred = infer_target_modules_from_state(state)
            target_modules = inferred if inferred else list(DEFAULT_TARGET_MODULES)

        print(
            f"[{i}/{len(specs)}] {spec.exp_name} {spec.method} seed={spec.seed} "
            f"tag={spec.tag or '-'}"
        )

        model = FederatedLoRAModel(
            model_name=base_model,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=list(target_modules),
            device=device,
        )
        model.load_model()
        loader = build_eval_loader(
            tokenizer=model.tokenizer,
            dataset_name=dataset_name,
            split=split,
            start_index=args.start_index,
            num_examples=args.num_examples,
            max_seq_length=max_seq_length,
            eval_batch_size=args.eval_batch_size,
        )

        # Tuned (adapter)
        model.set_lora_state_dict(state)
        tuned = evaluate_loss(model.model, loader, device=device)

        # Base (no adapter) computed once per evaluation config key
        base_key = (
            base_model,
            dataset_name,
            split,
            args.start_index,
            args.num_examples,
            max_seq_length,
            args.eval_batch_size,
        )
        if base_key not in base_cache:
            base_model_obj = FederatedLoRAModel(
                model_name=base_model,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=list(target_modules),
                device=device,
            )
            base_model_obj.load_model()
            base_loader = build_eval_loader(
                tokenizer=base_model_obj.tokenizer,
                dataset_name=dataset_name,
                split=split,
                start_index=args.start_index,
                num_examples=args.num_examples,
                max_seq_length=max_seq_length,
                eval_batch_size=args.eval_batch_size,
            )
            base_cache[base_key] = evaluate_loss(base_model_obj.model, base_loader, device=device)
        base = base_cache[base_key]

        row = {
            "exp_name": spec.exp_name,
            "method": spec.method,
            "seed": spec.seed,
            "tag": spec.tag,
            "timestamp": spec.timestamp,
            "checkpoint": str(spec.checkpoint),
            "dataset": dataset_name,
            "split": split,
            "heldout_start": args.start_index,
            "heldout_examples": args.num_examples,
            "max_seq_length": max_seq_length,
            "base_model": base_model,
            "target_modules": ",".join(target_modules),
            "lora_r": lora_r,
            "lora_alpha": lora_alpha,
            "device": device,
            "base_loss": base["loss"],
            "base_perplexity": base["perplexity"],
            "tuned_loss": tuned["loss"],
            "tuned_perplexity": tuned["perplexity"],
            "delta_loss_tuned_minus_base": tuned["loss"] - base["loss"],
            "delta_ppl_tuned_minus_base": tuned["perplexity"] - base["perplexity"],
        }
        rows.append(row)

        out_payload = {
            "row": row,
            "base_metrics": base,
            "tuned_metrics": tuned,
            "meta": {
                "checkpoint": str(spec.checkpoint),
                "dataset": dataset_name,
                "split": split,
                "heldout_start": args.start_index,
                "heldout_examples": args.num_examples,
                "max_seq_length": max_seq_length,
            },
        }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(out_payload, f, indent=2)
        print(
            f"  tuned_loss={tuned['loss']:.4f}, base_loss={base['loss']:.4f}, "
            f"delta={tuned['loss'] - base['loss']:.4f}"
        )

    if not rows:
        raise SystemExit("No rows evaluated/written.")

    summary_csv = Path(args.summary_csv)
    if not summary_csv.is_absolute():
        summary_csv = (REPO_ROOT / summary_csv).resolve()
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"\nWrote summary CSV: {summary_csv}")
    print(f"Wrote per-run JSONs under: {DOWNSTREAM_INSTR_ROOT}")


if __name__ == "__main__":
    main()
