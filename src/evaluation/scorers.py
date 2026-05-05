"""
Multiple-choice scoring via log-likelihood of choice continuations.

For each (prompt, choices) tuple:
  - For each choice, compute log P(choice | prompt) using teacher forcing.
  - Predict argmax_c log P(c | prompt).
  - Accuracy = fraction of correct predictions.

This is the standard "lm-eval" style multiple choice protocol.
"""

from typing import Dict, List

import torch
from torch.nn import functional as F
from tqdm import tqdm


@torch.no_grad()
def score_choice_loglikelihood(
    model,
    tokenizer,
    prompt: str,
    choice: str,
    device: str,
    max_seq_length: int = 1024,
) -> float:
    """
    Compute log P(choice | prompt) under the model.

    We tokenize prompt and choice separately so we know which tokens to score.
    Then we feed (prompt + choice) through the model and sum log-probs of the
    choice tokens.
    """
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].to(device)
    choice_ids = tokenizer(
        choice, return_tensors="pt", add_special_tokens=False
    ).input_ids[0].to(device)

    full_ids = torch.cat([prompt_ids, choice_ids]).unsqueeze(0)
    if full_ids.shape[1] > max_seq_length:
        # Truncate the prompt from the left if too long.
        excess = full_ids.shape[1] - max_seq_length
        full_ids = full_ids[:, excess:]

    outputs = model(full_ids)
    logits = outputs.logits[0]  # (seq_len, vocab)

    # We score choice tokens. The token at position i is predicted by logits at i-1.
    n_choice = choice_ids.shape[0]
    target_positions = range(full_ids.shape[1] - n_choice, full_ids.shape[1])
    log_probs = F.log_softmax(logits, dim=-1)
    score = 0.0
    for pos, target in zip(target_positions, choice_ids):
        if pos == 0:
            continue
        score += log_probs[pos - 1, target].item()
    return score


@torch.no_grad()
def evaluate_multiple_choice(
    model,
    tokenizer,
    items: List[Dict],
    device: str,
    desc: str = "eval",
) -> Dict:
    """
    Evaluate accuracy on multiple-choice items.

    Returns:
        {"accuracy": float, "num_examples": int, "num_correct": int}
    """
    model.eval()
    correct = 0
    for item in tqdm(items, desc=desc):
        scores = [
            score_choice_loglikelihood(
                model, tokenizer, item["prompt"], choice, device
            )
            for choice in item["choices"]
        ]
        pred = int(scores.index(max(scores)))
        if pred == item["answer"]:
            correct += 1
    n = len(items)
    return {
        "accuracy": correct / n if n > 0 else 0.0,
        "num_examples": n,
        "num_correct": correct,
    }

