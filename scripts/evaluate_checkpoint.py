"""
Evaluate a trained federated LoRA checkpoint on downstream benchmarks.

Usage:
    python scripts/evaluate_checkpoint.py \
        --checkpoint results/raw/exp_two_phase_k8/two_phase/seed_42/<run_id>/final_adapter_state.pt \
        --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --benchmarks mmlu arc_easy boolq hellaswag \
        --num-examples 500 \
        --device cuda \
        --output downstream_results.json
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import torch

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.metrics import evaluate_downstream
from src.models.lora_model import FederatedLoRAModel


def infer_target_modules_from_state(state: Dict[str, torch.Tensor]) -> List[str]:
    """Infer LoRA target modules from adapter state_dict keys."""
    known = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    found = [name for name in known if any(f".{name}." in k for k in state)]
    if not found:
        raise SystemExit(
            "Could not infer --target-modules from checkpoint keys. "
            "Pass --target-modules explicitly."
        )
    return found


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
        default=None,
        help=(
            "LoRA target modules. If omitted with --checkpoint, inferred from "
            "adapter state keys."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Inference device. 'auto' prefers CUDA, then MPS, else CPU.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if bool(args.no_adapter) == bool(args.checkpoint):
        raise SystemExit(
            "Specify exactly one of --checkpoint or --no-adapter."
        )

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("ERROR: --device cuda requested but CUDA is unavailable.")
        device = "cuda"
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            raise SystemExit("ERROR: --device mps requested but MPS is unavailable.")
        device = "mps"
    elif args.device == "cpu":
        device = "cpu"
    else:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Device: {device}")
    print(f"Loading base model: {args.base_model}")

    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")

    target_modules = args.target_modules
    if target_modules is None:
        if state is not None:
            target_modules = infer_target_modules_from_state(state)
            print(
                "Inferred target modules from checkpoint:",
                ", ".join(target_modules),
            )
        else:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
            print(
                "No adapter checkpoint provided; defaulting target modules to:",
                ", ".join(target_modules),
            )

    fed_model = FederatedLoRAModel(
        model_name=args.base_model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        device=device,
    )
    fed_model.load_model()

    if args.no_adapter:
        print("No adapter: evaluating base model only.")
        checkpoint_label = None
    else:
        print(f"Loading adapter state: {args.checkpoint}")
        # Verify we are actually loading adapter tensors, not silently skipping.
        model_state = fed_model.model.state_dict()
        matched = [k for k in state.keys() if k in model_state]
        missing = [k for k in state.keys() if k not in model_state]
        if not matched:
            raise SystemExit(
                "Adapter checkpoint has zero keys matching current model state. "
                "Check --target-modules and base model."
            )
        pre = {k: model_state[k].detach().cpu().clone() for k in matched}
        fed_model.set_lora_state_dict(state)
        changed = 0
        for k in matched:
            cur = fed_model.model.state_dict()[k].detach().cpu()
            if not torch.equal(pre[k], cur):
                changed += 1
        print(
            "Adapter load stats:",
            f"state_keys={len(state)}",
            f"matched={len(matched)}",
            f"missing={len(missing)}",
            f"changed={changed}",
        )
        if missing:
            print("Warning: unmatched adapter keys (first 5):", missing[:5])
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

