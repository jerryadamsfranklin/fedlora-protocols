"""
Run a single federated LoRA experiment.

Usage:
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit

CURSOR AI: Implement this script as specified.
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
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorForLanguageModeling

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.data_partitioner import DataPartitioner
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config YAML path")
    parser.add_argument(
        "--method", default=None, help="Override aggregation method"
    )
    parser.add_argument("--seed", type=int, default=42)
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
        help="Override switch threshold for fedlora_adaptive (tau)",
    )
    args = parser.parse_args()

    # Load config (recursive _inherit merge)
    config = load_config(args.config)

    if args.lora_r is not None:
        config.setdefault("lora", {})["r"] = args.lora_r
    if args.num_clients is not None:
        nc = args.num_clients
        config.setdefault("federated", {})["num_clients"] = nc
        config.setdefault("federated", {})["clients_per_round"] = nc
    if args.switch_threshold is not None:
        config.setdefault("fedlora_adaptive", {})["switch_threshold"] = float(
            args.switch_threshold
        )

    # Override method if specified; else from config or methods[0]
    method = args.method
    if method is None:
        method = config.get("federated", {}).get(
            "aggregation_method",
            (config.get("methods") or ["fedit"])[0],
        )

    # Set seed
    torch.manual_seed(args.seed)

    # Output directory (suffix for sweeps so names stay readable)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get("experiment", {}).get("name", "exp")
    suffix = ""
    if args.lora_r is not None:
        suffix += f"_r{args.lora_r}"
    if args.num_clients is not None:
        suffix += f"_c{args.num_clients}"
    # Organize runs under results/raw/<exp>/<method>/seed_<seed>/<timestamp>/
    # Keep sweep context in the seed directory name for easy browsing.
    seed_dir = f"seed_{args.seed}{suffix}" if suffix else f"seed_{args.seed}"
    output_dir = os.path.join(
        "results",
        "raw",
        exp_name,
        method,
        seed_dir,
        timestamp,
    )
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExperiment: {exp_name}")
    print(f"Method: {method}")
    print(f"Output: {output_dir}")

    # Device: MPS on Mac, else CPU
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

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
        device=device,
        torch_dtype=torch_dtype,
    )
    model.load_model()

    # Load data
    print("\n[2/4] Loading data...")
    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
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
    partitioner = DataPartitioner(dataset, num_clients=num_clients, seed=args.seed)
    partition_method = data_cfg.get("partition_method", "iid")

    if partition_method == "iid":
        client_datasets = partitioner.iid_partition()
    elif partition_method == "label_skew":
        client_datasets = partitioner.label_skew_partition(
            label_column=data_cfg.get("label_column", "label"),
            alpha=data_cfg.get("dirichlet_alpha", 0.5),
        )
    elif partition_method == "quantity_skew":
        client_datasets = partitioner.quantity_skew_partition(
            alpha=data_cfg.get("quantity_alpha", 0.5),
            min_samples=data_cfg.get("min_samples_per_client", 10),
        )
    else:
        client_datasets = partitioner.iid_partition()

    print(f"Partition stats: {partitioner.get_stats(client_datasets)}")

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
    adaptive_cfg = config.get("fedlora_adaptive", {})
    server = FederatedServer(
        aggregation_method=method,
        num_rounds=fed_cfg.get("num_rounds", 30),
        eval_every=eval_cfg.get("eval_every", 5),
        output_dir=output_dir,
        lora_r=config.get("lora", {}).get("r", 16),
        switch_threshold=adaptive_cfg.get("switch_threshold", 0.01),
        warmup_rounds=adaptive_cfg.get("warmup_rounds", 3),
        fixed_switch_round=adaptive_cfg.get("fixed_switch_round", None),
    )
    server.set_clients(clients)

    # Run training
    print("\n[4/4] Training...")
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
        "seed": args.seed,
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
        },
    }
    try:
        with open(os.path.join(output_dir, "run_meta.json"), "w") as f:
            json.dump(run_meta, f, indent=2)
    except Exception as e:
        print(f"Warning: failed to write run_meta.json: {e}")

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Communication: {results['total_communication_mb']:.2f} MB")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
