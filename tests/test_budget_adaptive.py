"""Tests for BudgetAdaptiveAggregator."""

import torch

from src.federation.aggregators.budget_adaptive import BudgetAdaptiveAggregator


def _mock_states(n: int = 3):
    states = []
    for _ in range(n):
        states.append(
            {
                "m.q_proj.lora_A.default": torch.randn(16, 8),
                "m.q_proj.lora_B.default": torch.randn(8, 16),
            }
        )
    return states


def test_starts_flora_with_generous_budget():
    agg = BudgetAdaptiveAggregator(
        total_budget_mb=1_000_000.0, num_rounds=5, max_rank=16
    )
    states = _mock_states()
    agg.aggregate(states, round_num=1)
    assert agg.round_methods[-1] == "FLoRA"
    assert agg.get_freeze_a() is False
