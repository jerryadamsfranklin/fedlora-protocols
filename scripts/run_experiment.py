"""
Run a single federated LoRA experiment.

Usage:
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit
    python scripts/run_experiment.py --config config/exp_reverse_adaptive_iid.yaml \\
        --seed 42 --device cuda --tag phase0_cuda_validation

"""

import argparse
import json
import os
import platform
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
import math
import re
from typing import Any, Dict

# Silence HF tokenizers fork-parallelism warning by default.
# This only affects tokenization throughput; it does not change model numerics.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_partitioner import DataPartitioner
from src.federation.checkpoint import resolve_checkpoint_path, run_dir_from_checkpoint
from src.federation.client import FederatedClient
from src.federation.server import FederatedServer
from src.models.lora_model import FederatedLoRAModel


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge override into base (override wins)."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = deepcopy(v)
    return result


def load_config(path: str) -> Dict[str, Any]:
    """
    Load YAML with recursive `_inherit` (e.g. revised_exp1 → revised_base → base_config).
    Each child file overrides its parent via deep merge.
    """
    abs_path = os.path.abspath(path)
    with open(abs_path) as f:
        config = yaml.safe_load(f)

    if not config:
        return {}

    inherit = config.pop("_inherit", None)
    if inherit:
        config_dir = os.path.dirname(abs_path)
        base_path = os.path.join(config_dir, inherit)
        base = load_config(base_path)
        config = _deep_merge(base, config)

    return config


def resolve_device(requested: str | None) -> str:
    """
    Resolve training device.

    --device cuda|mps|cpu selects explicitly (validated).
    --device omit/"auto" keeps the historical default: MPS if available else CPU
    (does not auto-pick CUDA, so Mac behavior stays unchanged).
    """
    choice = (requested or "auto").strip().lower()
    if choice in ("", "auto"):
        return "mps" if torch.backends.mps.is_available() else "cpu"
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "ERROR: --device cuda requested but torch.cuda.is_available() is False. "
                "Check the GPU instance / PyTorch CUDA build before training."
            )
        return "cuda"
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit(
                "ERROR: --device mps requested but torch.backends.mps.is_available() is False."
            )
        return "mps"
    if choice == "cpu":
        return "cpu"
    raise SystemExit(
        f"ERROR: unknown --device '{requested}'. Use one of: auto, cuda, mps, cpu."
    )


def apply_overrides(config: Dict[str, Any], overrides: list[str] | None) -> Dict[str, Any]:
    """
    Apply dotted-key overrides (KEY=VALUE) to a nested config dict.

    Values are parsed with YAML for type inference (e.g. "2" -> int, "true" -> bool).
    """
    if not overrides:
        return config

    _NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

    def _coerce_scalar(value: Any) -> Any:
        """
        yaml.safe_load is great for '2', '0.01', 'true', '[1,2]'.
        But it treats scientific notation like '5e-05' as a string.
        Coerce numeric-looking strings (incl. sci-notation) into numbers.
        """
        if not isinstance(value, str):
            return value
        s = value.strip()
        if not s:
            return value
        if _NUM_RE.match(s):
            # Prefer int when it is clearly an int literal (no dot, no exponent).
            if ("." not in s) and ("e" not in s.lower()):
                try:
                    return int(s)
                except Exception:
                    pass
            try:
                return float(s)
            except Exception:
                return value
        return value

    def _set_dot_path(d: Dict[str, Any], path: str, value: Any) -> None:
        parts = [p for p in path.split(".") if p]
        if not parts:
            return
        cur: Dict[str, Any] = d
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must be KEY=VALUE, got: {item!r}")
        k, v = item.split("=", 1)
        try:
            parsed = yaml.safe_load(v)
        except Exception:
            parsed = v
        _set_dot_path(config, k.strip(), _coerce_scalar(parsed))
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--method", default=None, help="Override aggregation method"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Legacy single seed: used for both data partitioning and training RNG "
            "unless --data-seed / --run-seed are set."
        ),
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Seed for data partitioning only (Dirichlet / IID splits). "
            "Defaults to --seed. Keep fixed across diagnostics that vary --run-seed."
        ),
    )
    parser.add_argument(
        "--run-seed",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Seed for training / init RNG (torch, numpy, python random). "
            "Defaults to --seed. Vary this while holding --data-seed fixed "
            "to test switch-round stability (Phase 1 #17 diagnostic)."
        ),
    )
    parser.add_argument(
        "--tag",
        default=None,
        help="Optional run tag (adds a folder in the output path)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Override a config value using dot paths (repeatable), e.g. federated.num_rounds=1",
    )
    parser.add_argument(
        "--lora_r",
        type=int,
        default=None,
        help="Override LoRA rank (lora.r); for rank sweeps (EXP4)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Override federated.num_clients (and clients_per_round); for scaling (EXP5)",
    )
    parser.add_argument(
        "--switch_threshold",
        type=float,
        default=None,
        help="Override reverse_adaptive.switch_threshold (tau)",
    )
    parser.add_argument(
        "--run-index",
        type=int,
        default=None,
        metavar="N",
        help="Position in a batch of runs (1-based); use with --run-total",
    )
    parser.add_argument(
        "--run-total",
        type=int,
        default=None,
        metavar="M",
        help="Total runs in the batch; adds run_N_of_M to the output path",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help=(
            "Compute device. 'auto' = MPS if available else CPU (legacy default). "
            "Use 'cuda' on NVIDIA hosts."
        ),
    )
    parser.add_argument(
        "--resume",
        default=None,
        metavar="PATH",
        help=(
            "Resume from a checkpoint .pt, a checkpoints/ dir, or a run output dir "
            "(uses checkpoints/latest.pt). Continues writing into that run directory."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        metavar="N",
        help="Override checkpointing.save_every (0 disables mid-run checkpoints).",
    )
    args = parser.parse_args()
    if (args.run_index is None) ^ (args.run_total is None):
        parser.error("--run-index and --run-total must be used together")
    if args.run_index is not None:
        if args.run_total < 1 or args.run_index < 1 or args.run_index > args.run_total:
            parser.error(
                "--run-index must be 1..--run-total and --run-total must be >= 1"
            )

    # Load config (recursive _inherit merge)
    config = load_config(args.config)
    try:
        config = apply_overrides(config, args.override)
    except ValueError as e:
        parser.error(str(e))

    if args.lora_r is not None:
        config.setdefault("lora", {})["r"] = args.lora_r
    if args.num_clients is not None:
        nc = args.num_clients
        config.setdefault("federated", {})["num_clients"] = nc
        config.setdefault("federated", {})["clients_per_round"] = nc
    if args.switch_threshold is not None:
        config.setdefault("reverse_adaptive", {})["switch_threshold"] = float(
            args.switch_threshold
        )

    # Override method if specified; else from config or methods[0]
    method = args.method
    if method is None:
        method = config.get("federated", {}).get(
            "aggregation_method",
            (config.get("methods") or ["fedit"])[0],
        )

    # Separate data-partition seed from training RNG (Phase 1 #17 / Phase 2 Part 0).
    data_seed = int(args.data_seed if args.data_seed is not None else args.seed)
    run_seed = int(args.run_seed if args.run_seed is not None else args.seed)
    import random as _random
    import numpy as _np

    # Seed RNGs for partitioning / any pre-train randomness with data_seed first.
    _random.seed(data_seed)
    _np.random.seed(data_seed)
    torch.manual_seed(data_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(data_seed)

    # Output directory (suffix for sweeps so names stay readable)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment", {}).get("name", "exp")
    resume_ckpt_path = None
    if args.resume:
        resume_ckpt_path = resolve_checkpoint_path(args.resume)
        output_dir = str(run_dir_from_checkpoint(resume_ckpt_path))
        print(f"Resume: {resume_ckpt_path}")
    else:
        suffix = ""
        if args.lora_r is not None:
            suffix += f"_r{args.lora_r}"
        if args.num_clients is not None:
            suffix += f"_c{args.num_clients}"
        if args.run_seed is not None and run_seed != data_seed:
            suffix += f"_run{run_seed}"
        # Organize under seed_<data_seed>/… so partition identity stays browsable.
        seed_dir = f"seed_{data_seed}{suffix}" if suffix else f"seed_{data_seed}"
        run_folder = None
        if args.run_index is not None:
            run_folder = f"run_{args.run_index:02d}_of_{args.run_total:02d}"
        tag_folder = (
            args.tag.strip() if isinstance(args.tag, str) and args.tag.strip() else None
        )
        output_dir = os.path.join(
            "results",
            "raw",
            exp_name,
            method,
            seed_dir,
            *([run_folder] if run_folder else []),
            *([tag_folder] if tag_folder else []),
            timestamp,
        )
    os.makedirs(output_dir, exist_ok=True)

    if args.run_index is not None:
        print(
            f"\n{'='*60}\n"
            f"  BATCH RUN  {args.run_index}/{args.run_total}\n"
            f"{'='*60}"
        )
    print(f"\nExperiment: {exp_name}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")

    device = resolve_device(args.device)
    config.setdefault("model", {})["device"] = device
    print(f"Device: {device}  (requested={args.device})")

    train_cfg = config.get("training", {})

    # Load model
    print("\n[1/4] Loading model...")
    model_cfg = config.get("model", {})
    torch_dtype = model_cfg.get("torch_dtype", "float32")
    model = FederatedLoRAModel(
        model_name=model_cfg.get(
            "name", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        ),
        lora_r=config.get("lora", {}).get("r", 16),
        lora_alpha=config.get("lora", {}).get("lora_alpha", 32),
        target_modules=config.get("lora", {}).get("target_modules"),
        device=device,
        torch_dtype=torch_dtype,
        lora_param_dtype=train_cfg.get("lora_param_dtype"),
    )
    model.load_model()

    # Load data
    print("\n[2/4] Loading data...")
    data_cfg = config.get("data", {})
    eval_cfg = config.get("evaluation", {})
    dataset = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg.get("dataset_split", "train"),
    )
    if "max_samples" in data_cfg:
        dataset = dataset.select(
            range(min(data_cfg["max_samples"], len(dataset)))
        )

    def _format_commonsenseqa_items(examples):
        questions = examples["question"]
        choices = examples["choices"]
        answer_keys = examples.get("answerKey")

        formatted = []
        for i, q in enumerate(questions):
            # `choices` can appear as either:
            # - a dict-of-lists (batched): {"label": [[...]], "text": [[...]]}
            # - a list-of-dicts: [{"label": [...], "text": [...]}, ...]
            if isinstance(choices, dict):
                labels = choices["label"][i]
                texts = choices["text"][i]
            else:
                labels = choices[i]["label"]
                texts = choices[i]["text"]

            choices_lines = "\n".join(
                f"{lab}) {txt}"
                for lab, txt in zip(labels, texts)
            )
            ans = answer_keys[i] if answer_keys is not None else ""
            formatted.append(
                f"Question: {q}\n\nChoices:\n{choices_lines}\n\nAnswer: {ans}"
            )
        return {"text": formatted}

    def _to_text_field(ds):
        # Ensure we have a "text" field for eval tokenization, without
        # disturbing partition labels used earlier.
        cols = set(ds.column_names)
        if "text" in cols:
            return ds
        if "instruction" in cols and "output" in cols:
            def _fmt_alpaca(examples):
                texts = [
                    f"### Instruction:\n{inst}\n\n### Response:\n{out}"
                    for inst, out in zip(examples["instruction"], examples["output"])
                ]
                return {"text": texts}
            return ds.map(_fmt_alpaca, batched=True)
        if "question" in cols and "choices" in cols and "answerKey" in cols:
            return ds.map(_format_commonsenseqa_items, batched=True)
        if "question" in cols:
            return ds.map(lambda x: {"text": x["question"]})
        return ds

    def _tokenize_text(examples, max_seq_length: int):
        return model.tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    # Optional held-out eval dataset (Phase 1c)
    eval_fn = None
    eval_split = data_cfg.get("eval_split")
    eval_samples = data_cfg.get("eval_samples")
    if eval_split:
        try:
            eval_ds = load_dataset(data_cfg["dataset_name"], split=eval_split)
            if eval_samples:
                eval_ds = eval_ds.select(range(min(int(eval_samples), len(eval_ds))))
            eval_ds = _to_text_field(eval_ds)
            eval_ds = eval_ds.map(
                lambda x: _tokenize_text(x, train_cfg.get("max_seq_length", 512)),
                batched=True,
                remove_columns=eval_ds.column_names,
            )
            eval_ds.set_format("torch")

            collator = DataCollatorForLanguageModeling(
                tokenizer=model.tokenizer,
                mlm=False,
            )
            eval_loader = DataLoader(
                eval_ds,
                batch_size=eval_cfg.get("eval_batch_size", 4),
                shuffle=False,
                collate_fn=collator,
            )

            def _eval_fn(_state_dict):
                model.set_lora_state_dict(_state_dict)
                model.model.eval()
                total_loss = 0.0
                total_tokens = 0
                with torch.no_grad():
                    for batch in eval_loader:
                        batch = {k: v.to(model.device) for k, v in batch.items()}
                        outputs = model.model(**batch)
                        loss = outputs.loss
                        labels = batch.get("labels")
                        if labels is None:
                            continue
                        num_tokens = (labels != -100).sum().item()
                        total_loss += float(loss.item()) * num_tokens
                        total_tokens += num_tokens
                model.model.train()
                avg_loss = (
                    total_loss / total_tokens if total_tokens > 0 else float("inf")
                )
                ppl = math.exp(avg_loss) if avg_loss < 100 else float("inf")
                return {
                    "val_loss": avg_loss,
                    "val_perplexity": ppl,
                    "val_tokens": total_tokens,
                }

            eval_fn = _eval_fn
            print(
                f"Eval: enabled (split={eval_split}, samples={len(eval_ds)})"
            )
        except Exception as e:
            print(f"Eval: disabled (failed to load eval split '{eval_split}'): {e}")

    # Partition
    num_clients = config.get("federated", {}).get("num_clients", 10)
    partition_method = data_cfg.get("partition_method", "iid")
    label_column = data_cfg.get("label_column", "label")
    partition_alpha = float(
        data_cfg.get(
            "partition_alpha",
            data_cfg.get("dirichlet_alpha", 0.5),
        )
    )

    dataset_for_partition = dataset
    if partition_method == "label_skew" and label_column not in dataset.column_names:
        # Alpaca has no class labels; bucket by instruction length as a task-diversity proxy.
        def _proxy_labels_from_length(examples):
            if "instruction" in examples:
                texts = examples["instruction"]
            elif "text" in examples:
                texts = examples["text"]
            else:
                n = len(next(iter(examples.values())))
                texts = [""] * n
            return {
                label_column: [min(len(t or "") // 50, 9) for t in texts]
            }

        dataset_for_partition = dataset.map(
            _proxy_labels_from_length,
            batched=True,
            desc="Proxy labels (length buckets) for label_skew",
        )
        print(
            f"Partition: added '{label_column}' via instruction-length buckets "
            "(non-IID label_skew on Alpaca)."
        )

    partitioner = DataPartitioner(
        dataset_for_partition, num_clients=num_clients, seed=data_seed
    )

    if partition_method == "iid":
        client_datasets = partitioner.iid_partition()
    elif partition_method == "label_skew":
        client_datasets = partitioner.label_skew_partition(
            label_column=label_column,
            alpha=partition_alpha,
        )
    elif partition_method == "quantity_skew":
        q_alpha = float(data_cfg.get("quantity_alpha", partition_alpha))
        client_datasets = partitioner.quantity_skew_partition(
            alpha=q_alpha,
            min_samples=data_cfg.get("min_samples_per_client", 10),
        )
    else:
        raise ValueError(
            f"Unknown data.partition_method: {partition_method!r} "
            "(expected 'iid', 'label_skew', or 'quantity_skew')"
        )

    print(f"Partition stats: {partitioner.get_stats(client_datasets)}")
    print(f"Seeds: data_seed={data_seed}  run_seed={run_seed}")

    # Re-seed for training / client init so --run-seed can vary independently.
    _random.seed(run_seed)
    _np.random.seed(run_seed)
    torch.manual_seed(run_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(run_seed)

    # Create clients
    print("\n[3/4] Creating clients...")
    clients = []
    for i, ds in enumerate(client_datasets):
        if len(ds) == 0:
            continue
        client = FederatedClient(
            client_id=i,
            model=model,
            dataset=ds,
            batch_size=train_cfg.get("batch_size", 4),
            local_epochs=train_cfg.get("local_epochs", 2),
            learning_rate=train_cfg.get("learning_rate", 2e-4),
            max_seq_length=train_cfg.get("max_seq_length", 512),
            gradient_accumulation_steps=train_cfg.get(
                "gradient_accumulation_steps", 4
            ),
        )
        clients.append(client)

    if not clients:
        print("No clients with data. Exiting.")
        return

    # Create server
    fed_cfg = config.get("federated", {})
    two_phase_cfg = config.get("two_phase", {}) or {}
    reverse_adaptive_cfg = config.get("reverse_adaptive", {}) or {}
    ckpt_cfg = config.get("checkpointing", {}) or {}
    save_every = (
        args.save_every
        if args.save_every is not None
        else int(ckpt_cfg.get("save_every", 0) or 0)
    )
    keep_last_n = int(ckpt_cfg.get("keep_last_n", 3) or 3)

    server = FederatedServer(
        aggregation_method=method,
        num_rounds=fed_cfg.get("num_rounds", 30),
        eval_every=eval_cfg.get("eval_every", 5),
        output_dir=output_dir,
        lora_r=config.get("lora", {}).get("r", 16),
        switch_threshold=reverse_adaptive_cfg.get("switch_threshold", 0.01),
        warmup_rounds=reverse_adaptive_cfg.get("warmup_rounds", 3),
        transition_rounds=reverse_adaptive_cfg.get("transition_rounds", 3),
        stability_threshold=reverse_adaptive_cfg.get("stability_threshold", 1.1),
        two_phase=two_phase_cfg,
        reverse_adaptive=reverse_adaptive_cfg,
        save_every=save_every,
        keep_last_n=keep_last_n,
        seed=run_seed,
    )
    server.set_clients(clients)

    # Run training
    print("\n[4/4] Training...")
    if save_every > 0:
        print(f"Checkpointing: every {save_every} round(s), keep_last_n={keep_last_n}")
    if resume_ckpt_path is not None:
        server.load_checkpoint(str(resume_ckpt_path))
    results = server.train(eval_fn=eval_fn)

    # Persist merged config + run metadata for reproducibility
    try:
        with open(os.path.join(output_dir, "config_merged.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
    except Exception as e:
        print(f"Warning: failed to write config_merged.yaml: {e}")

    def _git(cmd: list[str]) -> str:
        try:
            return subprocess.check_output(cmd, text=True).strip()
        except Exception:
            return ""

    run_meta = {
        "experiment": exp_name,
        "method": method,
        "seed": data_seed,  # partition identity (legacy field)
        "data_seed": data_seed,
        "run_seed": run_seed,
        "timestamp": timestamp,
        "device": device,
        "output_dir": output_dir,
        "git_commit": _git(["git", "rev-parse", "HEAD"]),
        "git_branch": _git(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "git_dirty": bool(_git(["git", "status", "--porcelain"])),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "overrides": {
            "lora_r": args.lora_r,
            "num_clients": args.num_clients,
            "switch_threshold": args.switch_threshold,
            "data_seed": data_seed,
            "run_seed": run_seed,
            "device": args.device,
            "save_every": save_every,
            "resume": str(resume_ckpt_path) if resume_ckpt_path else None,
        },
    }
    if args.run_index is not None:
        run_meta["run_index"] = args.run_index
        run_meta["run_total"] = args.run_total
        run_meta["run_label"] = f"{args.run_index}/{args.run_total}"
    try:
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write run_meta.json: {e}")

    print(f"\n{'='*60}")
    print("Complete!")
    if args.run_index is not None:
        print(f"Batch run: {args.run_index}/{args.run_total}")
    print(f"Communication: {results['total_communication_mb']:.2f} MB")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
