"""
Benchmark dataset loaders for downstream LLM evaluation.

Each loader returns a List[Dict] with a uniform schema:
    {"prompt": str, "choices": List[str], "answer": int}

Where "answer" is the integer index of the correct choice in "choices".
"""

from typing import Dict, List

from datasets import load_dataset


def load_mmlu_subset(num_examples: int = 500, seed: int = 42) -> List[Dict]:
    """
    Load a deterministic subset of MMLU (test split).

    Full MMLU is large; this samples `num_examples` after shuffling with `seed`.
    """
    ds = load_dataset("cais/mmlu", "all", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))
    items: List[Dict] = []
    for row in ds:
        choices = list(row["choices"])
        prompt = (
            f"Question: {row['question']}\n"
            f"A) {choices[0]}\n"
            f"B) {choices[1]}\n"
            f"C) {choices[2]}\n"
            f"D) {choices[3]}\n"
            f"Answer:"
        )
        items.append(
            {
                "prompt": prompt,
                "choices": [" A", " B", " C", " D"],
                "answer": int(row["answer"]),
            }
        )
    return items


def load_arc_easy(num_examples: int = 500, seed: int = 42) -> List[Dict]:
    ds = load_dataset("ai2_arc", "ARC-Easy", split="test")
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))
    items: List[Dict] = []
    for row in ds:
        labels = row["choices"]["label"]
        texts = row["choices"]["text"]
        ans_label = row["answerKey"]
        if ans_label not in labels:
            continue
        ans_idx = labels.index(ans_label)
        choices_str = "\n".join(f"{lab}) {txt}" for lab, txt in zip(labels, texts))
        prompt = f"Question: {row['question']}\n{choices_str}\nAnswer:"
        items.append(
            {
                "prompt": prompt,
                "choices": [f" {lab}" for lab in labels],
                "answer": ans_idx,
            }
        )
    return items


def load_boolq(num_examples: int = 500, seed: int = 42) -> List[Dict]:
    ds = load_dataset("google/boolq", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))
    items: List[Dict] = []
    for row in ds:
        prompt = (
            f"Passage: {row['passage']}\n"
            f"Question: {row['question']}\n"
            f"Answer:"
        )
        ans = 1 if row["answer"] else 0  # 1 = yes, 0 = no
        items.append(
            {
                "prompt": prompt,
                "choices": [" no", " yes"],
                "answer": ans,
            }
        )
    return items


def load_hellaswag(num_examples: int = 500, seed: int = 42) -> List[Dict]:
    """
    Load a deterministic subset of HellaSwag (validation split).

    HellaSwag is 4-way sentence-completion. Each item has a context
    and 4 candidate endings; the task is to pick the most plausible.
    """
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=seed).select(range(min(num_examples, len(ds))))
    items: List[Dict] = []
    for row in ds:
        ctx = row.get("ctx", "") or row.get("ctx_a", "")
        endings = row["endings"]
        if len(endings) != 4:
            continue
        ans_str = row["label"]
        try:
            ans_idx = int(ans_str)
        except (TypeError, ValueError):
            continue
        prompt = f"{ctx}"
        items.append(
            {
                "prompt": prompt,
                "choices": [f" {e}" for e in endings],
                "answer": ans_idx,
            }
        )
    return items


BENCHMARKS = {
    "mmlu": load_mmlu_subset,
    "arc_easy": load_arc_easy,
    "boolq": load_boolq,
    "hellaswag": load_hellaswag,
}

