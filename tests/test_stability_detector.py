"""Test stability detection functionality."""

from src.federation.aggregators.mixins.stability_detector import StabilityDetectorMixin


class TestStabilityDetector:
    def setup_method(self):
        class TestAggregator(StabilityDetectorMixin):
            pass

        self.aggregator = TestAggregator()
        self.aggregator.init_stability_detector(spike_threshold=1.1, sustained_rounds=2)

    def test_no_instability_on_decreasing_loss(self):
        losses = [1.5, 1.4, 1.3, 1.2]
        for i, loss in enumerate(losses, start=1):
            self.aggregator.record_loss(loss, round_num=i)
        assert not self.aggregator.detect_instability(1.1)

    def test_spike_detection(self):
        self.aggregator.loss_history = [1.5, 1.4, 1.3]
        assert self.aggregator.detect_instability(1.3 * 1.15)  # 15% spike

    def test_no_spike_on_small_increase(self):
        self.aggregator.loss_history = [1.5, 1.4, 1.3]
        assert not self.aggregator.detect_instability(1.3 * 1.05)  # 5% increase

    def test_sustained_increase_detection(self):
        self.aggregator.loss_history = [1.2, 1.25, 1.30]
        assert self.aggregator.detect_instability(1.31)

    def test_detailed_detection_returns_reason(self):
        self.aggregator.loss_history = [1.5, 1.4, 1.3]
        result = self.aggregator.detect_instability_detailed(1.3 * 1.15)
        assert result["unstable"] is True
        assert "spike" in result["reason"].lower()


def _run_without_pytest():
    t = TestStabilityDetector()
    t.setup_method()
    t.test_no_instability_on_decreasing_loss()
    t.setup_method()
    t.test_spike_detection()
    t.setup_method()
    t.test_no_spike_on_small_increase()
    t.setup_method()
    t.test_sustained_increase_detection()
    t.setup_method()
    t.test_detailed_detection_returns_reason()


if __name__ == "__main__":
    _run_without_pytest()
    print("OK")

