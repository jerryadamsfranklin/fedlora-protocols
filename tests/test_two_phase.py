"""Tests for TwoPhaseAggregator."""

import torch

from src.federation.aggregators.two_phase import TwoPhaseAggregator


def _mock_states(n: int = 3):
    states = []
    for _ in range(n):
        states.append(
            {
                "base.model.layers.0.self_attn.q_proj.lora_A.default": torch.randn(
                    16, 8
                ),
                "base.model.layers.0.self_attn.q_proj.lora_B.default": torch.randn(
                    8, 16
                ),
            }
        )
    return states


def test_phase1_flora_freeze_false():
    agg = TwoPhaseAggregator(phase_boundary=8, max_rank=16)
    states = _mock_states()
    for r in range(1, 8):
        agg.aggregate(states, round_num=r)
        assert agg.current_phase == 1
        assert agg.get_freeze_a() is False
    # After round-8 aggregate, the *next* training round (9) is FFA-prep: freeze A
    agg.aggregate(states, round_num=8)
    assert agg.current_phase == 1
    assert agg.get_freeze_a() is True


def test_phase2_ffa_freeze_true():
    agg = TwoPhaseAggregator(phase_boundary=3, max_rank=16)
    states = _mock_states()
    for r in range(1, 4):
        agg.aggregate(states, round_num=r)
    agg.aggregate(states, round_num=4)
    assert agg.current_phase == 2
    assert agg.get_freeze_a() is True


def test_comm_savings_positive_after_mixed_phases():
    agg = TwoPhaseAggregator(phase_boundary=5, max_rank=16)
    states = _mock_states()
    for r in range(1, 11):
        agg.aggregate(states, round_num=r)
    s = agg.get_stats()
    assert s["comm_savings_vs_flora"] > 0
