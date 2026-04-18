"""
Run a single federated LoRA experiment.

Usage:
    python scripts/run_experiment.py --config config/exp1_iid.yaml --method fedit

CURSOR AI: Implement this script as specified.
"""

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import torch
import yaml
from datasets import load_dataset

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
    args = parser.parse_args()

    # Load config (recursive _inherit merge)
    config = load_config(args.config)

    if args.lora_r is not None:
        config.setdefault("lora", {})["r"] = args.lora_r
    if args.num_clients is not None:
        nc = args.num_clients
        config.setdefault("federated", {})["num_clients"] = nc
        config.setdefault("federated", {})["clients_per_round"] = nc

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
    output_dir = os.path.join(
        "results", f"{exp_name}_{method}{suffix}_{timestamp}"
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
    dataset = load_dataset(
        data_cfg["dataset_name"],
        split=data_cfg.get("dataset_split", "train"),
    )
    if "max_samples" in data_cfg:
        dataset = dataset.select(
            range(min(data_cfg["max_samples"], len(dataset)))
        )

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
    train_cfg = config.get("training", {})
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
    eval_cfg = config.get("evaluation", {})
    server = FederatedServer(
        aggregation_method=method,
        num_rounds=fed_cfg.get("num_rounds", 30),
        eval_every=eval_cfg.get("eval_every", 5),
        output_dir=output_dir,
    )
    server.set_clients(clients)

    # Run training
    print("\n[4/4] Training...")
    results = server.train()

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"Communication: {results['total_communication_mb']:.2f} MB")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
