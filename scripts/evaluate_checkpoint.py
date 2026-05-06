"""
Evaluate a trained federated LoRA checkpoint on downstream benchmarks.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint results/raw/exp_two_phase_k8/two_phase/seed_42/<run_id>/final_adapter_state.pt \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --benchmarks mmlu arc_easy boolq \
        --num-examples 500 \
        --output downstream_results.json
"""

import argparse
import json
import os
import sys

import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import evaluate_downstream
from src.models.lora_model import FederatedLoRAModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        help="Path to final_adapter_state.pt (omit when using --no-adapter)",
    )
    parser.add_argument(
        "--no-adapter",
        action="store_true",
        help="Evaluate the base model without applying any adapter (baseline).",
    )
    parser.add_argument(
        "--base-model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    )
    parser.add_argument(
        "--benchmarks", nargs="+", default=["mmlu", "arc_easy", "boolq"]
    )
    parser.add_argument("--num-examples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if bool(args.no_adapter) == bool(args.checkpoint):
        raise SystemExit(
            "Specify exactly one of --checkpoint or --no-adapter."
        )

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading base model: {args.base_model}")

    fed_model = FederatedLoRAModel(
        model_name=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        device=device,
    )
    fed_model.load_model()

    if args.no_adapter:
        print("No adapter: evaluating base model only.")
        checkpoint_label = None
    else:
        print(f"Loading adapter state: {args.checkpoint}")
        state = torch.load(args.checkpoint, map_location="cpu")
        fed_model.set_lora_state_dict(state)
        checkpoint_label = args.checkpoint

    print(f"Running benchmarks: {args.benchmarks}")
    results = evaluate_downstream(
        model=fed_model.model,
        tokenizer=fed_model.tokenizer,
        benchmarks=args.benchmarks,
        device=device,
        num_examples=args.num_examples,
        seed=args.seed,
    )

    out = {
        "checkpoint": checkpoint_label,
        "no_adapter": bool(args.no_adapter),
        "base_model": args.base_model,
        "num_examples": args.num_examples,
        "benchmarks": results,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults written to {args.output}")
    for name, r in results.items():
        print(
            f"  {name}: accuracy = {r['accuracy']:.4f} ({r['num_correct']}/{r['num_examples']})"
        )


if __name__ == "__main__":
    main()

