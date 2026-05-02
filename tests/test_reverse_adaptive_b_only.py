import torch

from src.federation.aggregators.reverse_adaptive import ReverseAdaptiveAggregator


def _mock_states(n=3):
    return [
        {
            "m.q_proj.lora_A.default": torch.randn(16, 8),
            "m.q_proj.lora_B.default": torch.randn(8, 16),
        }
        for _ in range(n)
    ]


def test_reverse_adaptive_b_only_methods_exist():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=2, transition_rounds=0, max_rank=16
    )
    assert hasattr(agg, "get_frozen_a")
    assert hasattr(agg, "should_upload_b_only")
    assert hasattr(agg, "should_broadcast_b_only")


def test_reverse_adaptive_no_b_only_in_flora_phase():
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=5, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    # FLoRA phase: no B-only.
    for i, loss in enumerate([1.5, 1.49, 1.488]):
        agg.aggregate(states, current_loss=loss, round_num=i + 1)
        assert agg.should_upload_b_only() is False
        assert agg.should_broadcast_b_only() is False


def test_reverse_adaptive_b_only_after_switch_and_init():
    """
    Once ReverseAdaptive switches to FFA-LoRA AND the FFA-LoRA aggregator has
    initialized frozen A, B-only must be enabled.
    """
    agg = ReverseAdaptiveAggregator(
        switch_threshold=0.01, warmup_rounds=2, transition_rounds=0, max_rank=16
    )
    states = _mock_states()
    losses = [1.5, 1.4, 1.3, 1.25, 1.22, 1.219, 1.2185, 1.2184]
    for i, loss in enumerate(losses):
        agg.aggregate(states, current_loss=loss, round_num=i + 1)

    assert agg.current_mode == "FFA-LoRA"
    assert agg.should_upload_b_only() is True
    assert agg.should_broadcast_b_only() is True
    frozen_a = agg.get_frozen_a()
    assert frozen_a, "frozen_a must be populated after FFA-LoRA initialization"

