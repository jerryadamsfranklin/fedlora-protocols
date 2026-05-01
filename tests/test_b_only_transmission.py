import torch

from src.federation.aggregators.ffa_lora import FFALoRAAggregator
from src.federation.aggregators.two_phase import TwoPhaseAggregator


def _full_state():
    return {
        "layer.q_proj.lora_A.default": torch.randn(16, 8),
        "layer.q_proj.lora_B.default": torch.randn(8, 16),
        "layer.v_proj.lora_A.default": torch.randn(16, 8),
        "layer.v_proj.lora_B.default": torch.randn(8, 16),
    }


def _filter_b_only(state):
    # Keep this local to avoid importing client.py (which depends on transformers).
    return {k: v for k, v in state.items() if ("lora_B" in k or "lora_b" in k)}


def test_ffa_get_frozen_a_and_reconstruction_roundtrip():
    agg = FFALoRAAggregator()
    full_states = [_full_state() for _ in range(3)]
    _ = agg.aggregate(full_states)
    frozen_a = agg.get_frozen_a()
    assert frozen_a
    assert all(("lora_a" in k.lower()) for k in frozen_a.keys())

    b_only_states = [_filter_b_only(s) for s in full_states]
    assert all(not any("lora_a" in k.lower() for k in st.keys()) for st in b_only_states)

    # server-side reconstruction
    for st in b_only_states:
        for a_key, a_tensor in frozen_a.items():
            if a_key not in st:
                st[a_key] = a_tensor.clone()

    assert all(any("lora_a" in k.lower() for k in st.keys()) for st in b_only_states)


def test_two_phase_should_upload_b_only_after_first_ffa_round():
    agg = TwoPhaseAggregator(phase_boundary=1, max_rank=16)
    states = [_full_state() for _ in range(2)]

    # Before any rounds: round_count=0, next training is round 1 (phase 1)
    assert agg.get_freeze_a() is False
    assert agg.should_upload_b_only() is False

    # After round 1 aggregate, next training is round 2 (first FFA round)
    agg.aggregate(states, round_num=1)
    assert agg.get_freeze_a() is True
    assert agg.should_upload_b_only() is False

    # After round 2 aggregate (first FFA round completed), next training (round 3)
    # can upload B-only.
    agg.aggregate(states, round_num=2)
    assert agg.get_freeze_a() is True
    assert agg.should_upload_b_only() is True

