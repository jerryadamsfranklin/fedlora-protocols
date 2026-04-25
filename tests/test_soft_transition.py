"""Test soft transition functionality."""

from src.federation.aggregators.mixins.soft_transition import SoftTransitionMixin


class TestSoftTransition:
    def setup_method(self):
        # Create a test class that uses the mixin
        class TestAggregator(SoftTransitionMixin):
            pass

        self.aggregator = TestAggregator()
        self.aggregator.init_soft_transition(transition_rounds=3)

    def test_ffa_lora_phase_fully_frozen(self):
        """FFA-LoRA phase should return freeze_ratio=1.0"""
        ratio = self.aggregator.get_freeze_ratio(round_num=2, phase="FFA-LoRA")
        assert ratio == 1.0

    def test_flora_phase_fully_trainable(self):
        """FLoRA phase should return freeze_ratio=0.0"""
        ratio = self.aggregator.get_freeze_ratio(round_num=10, phase="FLoRA")
        assert ratio == 0.0

    def test_transition_gradual_decrease(self):
        """TRANSITION phase should gradually decrease freeze_ratio"""
        self.aggregator.start_transition(round_num=4)

        # Round 4 (start): should be ~1.0 or close
        r4 = self.aggregator.get_freeze_ratio(round_num=4, phase="TRANSITION")
        assert r4 >= 0.9

        # Round 5: should be ~0.67
        r5 = self.aggregator.get_freeze_ratio(round_num=5, phase="TRANSITION")
        assert 0.5 < r5 < 0.8

        # Round 6: should be ~0.33
        r6 = self.aggregator.get_freeze_ratio(round_num=6, phase="TRANSITION")
        assert 0.2 < r6 < 0.5

        # Round 7: should be 0.0 (transition complete)
        r7 = self.aggregator.get_freeze_ratio(round_num=7, phase="TRANSITION")
        assert r7 <= 0.1

    def test_transition_complete_detection(self):
        """Should detect when transition is complete"""
        self.aggregator.start_transition(round_num=4)

        assert not self.aggregator.is_transition_complete(round_num=5)
        assert not self.aggregator.is_transition_complete(round_num=6)
        assert self.aggregator.is_transition_complete(round_num=7)


def _run_without_pytest():
    t = TestSoftTransition()
    t.setup_method()
    t.test_ffa_lora_phase_fully_frozen()
    t.setup_method()
    t.test_flora_phase_fully_trainable()
    t.setup_method()
    t.test_transition_gradual_decrease()
    t.setup_method()
    t.test_transition_complete_detection()


if __name__ == "__main__":
    _run_without_pytest()
    print("OK")

