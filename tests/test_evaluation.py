"""
Smoke tests for evaluation pipeline. Uses a tiny synthetic dataset; does
not download any benchmarks.
"""

from unittest.mock import MagicMock

from src.evaluation.scorers import evaluate_multiple_choice


def test_evaluate_multiple_choice_perfect_oracle():
    """If the model always scores the correct choice highest, accuracy = 1."""
    items = [
        {"prompt": "Q1", "choices": [" A", " B"], "answer": 0},
        {"prompt": "Q2", "choices": [" A", " B"], "answer": 1},
    ]

    # Mock model and tokenizer that returns scores aligned with answers.
    def fake_score(model, tokenizer, prompt, choice, device, max_seq_length=1024):
        # Return higher score for the correct choice based on the prompt.
        if prompt == "Q1":
            return 1.0 if choice == " A" else -1.0
        return 1.0 if choice == " B" else -1.0

    # Patch the score function for this test.
    import src.evaluation.scorers as scorers

    original = scorers.score_choice_loglikelihood
    scorers.score_choice_loglikelihood = fake_score
    try:
        result = evaluate_multiple_choice(MagicMock(), MagicMock(), items, "cpu")
    finally:
        scorers.score_choice_loglikelihood = original

    assert result["accuracy"] == 1.0
    assert result["num_correct"] == 2
    assert result["num_examples"] == 2


def test_evaluate_multiple_choice_random_baseline():
    """If the scorer is constant, accuracy is 0.5 for alternating answers."""
    items = [
        {"prompt": f"Q{i}", "choices": [" A", " B"], "answer": i % 2}
        for i in range(20)
    ]

    def fake_score(model, tokenizer, prompt, choice, device, max_seq_length=1024):
        return 0.0  # always tie -> argmax picks first index

    import src.evaluation.scorers as scorers

    original = scorers.score_choice_loglikelihood
    scorers.score_choice_loglikelihood = fake_score
    try:
        result = evaluate_multiple_choice(MagicMock(), MagicMock(), items, "cpu")
    finally:
        scorers.score_choice_loglikelihood = original

    # Always picks index 0; correct on items where answer=0 (half).
    assert result["accuracy"] == 0.5

