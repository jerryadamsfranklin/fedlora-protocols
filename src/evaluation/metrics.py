"""
Public evaluation API.

Used by the federated server (in-loop training-loss evaluation) and by
post-hoc downstream evaluation scripts.
"""

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from .benchmarks import BENCHMARKS
from .scorers import evaluate_multiple_choice


@torch.no_grad()
def evaluate_loss(model, dataloader: DataLoader, device: str) -> Dict[str, float]:
    """In-training validation: return mean cross-entropy and perplexity."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        if not torch.isfinite(out.loss):
            continue
        total_loss += float(out.loss.item())
        n_batches += 1
    avg_loss = total_loss / n_batches if n_batches > 0 else float("inf")
    return {
        "val_loss": avg_loss,
        "val_perplexity": float(torch.exp(torch.tensor(avg_loss)).item()),
    }


def evaluate_downstream(
    model,
    tokenizer,
    benchmarks: List[str],
    device: str,
    num_examples: int = 500,
    seed: int = 42,
) -> Dict[str, Dict]:
    """
    Run all named benchmarks and return per-benchmark accuracy.

    benchmarks: subset of BENCHMARKS keys, e.g. ["mmlu", "arc_easy", "boolq"].
    """
    results: Dict[str, Dict] = {}
    for name in benchmarks:
        if name not in BENCHMARKS:
            raise ValueError(
                f"Unknown benchmark: {name}. Known: {list(BENCHMARKS)}"
            )
        items = BENCHMARKS[name](num_examples=num_examples, seed=seed)
        results[name] = evaluate_multiple_choice(
            model, tokenizer, items, device, desc=name
        )
    return results

