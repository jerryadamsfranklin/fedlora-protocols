"""Tests for CurriculumRankAggregator padding and schedule."""

import torch

from src.federation.aggregators.curriculum_rank import CurriculumRankAggregator


def _mock_states(n: int = 3, r: int = 16, din: int = 8, dout: int = 8):
    states = []
    for _ in range(n):
        states.append(
            {
                "m.q_proj.lora_A.default": torch.randn(r, din),
                "m.q_proj.lora_B.default": torch.randn(dout, r),
            }
        )
    return states


def test_schedule_selects_rank():
    agg = CurriculumRankAggregator(rank_schedule={5: 4, 10: 8, 15: 16}, full_rank=16)
    assert agg._rank_for_round(1) == 4
    assert agg._rank_for_round(5) == 4
    assert agg._rank_for_round(6) == 8
    assert agg._rank_for_round(10) == 8
    assert agg._rank_for_round(11) == 16


def test_padding_keeps_full_rank_shapes():
    states = _mock_states()
    agg = CurriculumRankAggregator(rank_schedule={1: 4, 2: 8}, full_rank=16)
    out = agg.aggregate(states, round_num=1)
    assert out["m.q_proj.lora_A.default"].shape[0] == 16
    assert out["m.q_proj.lora_B.default"].shape[1] == 16
    out2 = agg.aggregate(states, round_num=2)
    assert out2["m.q_proj.lora_A.default"].shape[0] == 16
    assert out2["m.q_proj.lora_B.default"].shape[1] == 16

