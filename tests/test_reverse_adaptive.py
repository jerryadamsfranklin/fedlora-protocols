"""Tests for ReverseAdaptiveAggregator."""

import torch

from src.federation.aggregators.flora import FLoRAAggregator
from src.federation.aggregators.reverse_adaptive import ReverseAdaptiveAggregator


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


def test_warmup_blocks_switch():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=3, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    for i, loss in enumerate([1.5, 1.49, 1.489]):
        agg.aggregate(states, current_loss=loss, round_num=i + 1)
    assert agg.current_mode == "FLoRA"


def test_switches_after_warmup_when_flat():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=3, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    losses = [1.5, 1.4, 1.3, 1.25, 1.22, 1.219]
    for i, loss in enumerate(losses):
        agg.aggregate(states, current_loss=loss, round_num=i + 1)
    assert agg.current_mode == "FFA-LoRA"


def test_no_switch_baseline_matches_flora_keys():
    """
    With switch_threshold=-1 and warmup_rounds=999, aggregator never switches.
    Aggregated output keys must match a plain FLoRA aggregation's keys.
    """
    flora = FLoRAAggregator(max_rank=16)
    rev = ReverseAdaptiveAggregator(
        switch_threshold=-1.0, warmup_rounds=999, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    for i, loss in enumerate([1.5, 1.4, 1.3, 1.2, 1.1]):
        rev.aggregate(states, current_loss=loss, round_num=i + 1)

    assert rev.current_mode == "FLoRA"
    out_rev = rev.aggregate(states, current_loss=1.05, round_num=6)
    out_flora = flora.aggregate(states)
    assert set(out_rev.keys()) == set(out_flora.keys())


def test_events_logged_on_switch():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=2, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    for i, loss in enumerate([1.5, 1.4, 1.39, 1.389]):
        agg.aggregate(states, current_loss=loss, round_num=i + 1)
    stats = agg.get_stats()
    assert "events" in stats
    assert any(e["event"] == "switch_to_ffa" for e in stats["events"])
