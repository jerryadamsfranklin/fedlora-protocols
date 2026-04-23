"""Test per-layer tracking functionality."""

from src.federation.aggregators.mixins.per_layer_tracker import PerLayerTrackerMixin


class TestPerLayerTracker:
    def setup_method(self):
        class TestAggregator(PerLayerTrackerMixin):
            pass

        self.aggregator = TestAggregator()
        self.layer_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.aggregator.init_per_layer_tracker(
            layer_names=self.layer_names,
            switch_threshold=0.01,
            warmup_rounds=3,
            transition_rounds=3,
        )

    def test_initial_state_all_ffa_lora(self):
        for layer in self.layer_names:
            assert self.aggregator.layer_phases[layer] == "FFA-LoRA"

    def test_layer_freeze_ratio_in_ffa_lora(self):
        ratios = self.aggregator.get_layer_freeze_ratios(round_num=1)
        for layer in self.layer_names:
            assert ratios[layer] == 1.0

    def test_layer_analysis_output(self):
        self.aggregator.layer_phases["q_proj"] = "FLoRA"
        self.aggregator.layer_switch_rounds["q_proj"] = 4
        analysis = self.aggregator.get_layer_analysis()
        assert "layer_phases" in analysis
        assert "layers_still_ffa_lora" in analysis
        assert "layers_in_flora" in analysis
        assert "q_proj" in analysis["layers_in_flora"]

    def test_revert_layer(self):
        self.aggregator.layer_phases["q_proj"] = "FLoRA"
        self.aggregator.layer_switch_rounds["q_proj"] = 4
        self.aggregator.revert_layer("q_proj", round_num=6)
        assert self.aggregator.layer_phases["q_proj"] == "FFA-LoRA"
        assert self.aggregator.layer_switch_rounds["q_proj"] is None


def _run_without_pytest():
    t = TestPerLayerTracker()
    t.setup_method()
    t.test_initial_state_all_ffa_lora()
    t.setup_method()
    t.test_layer_freeze_ratio_in_ffa_lora()
    t.setup_method()
    t.test_layer_analysis_output()
    t.setup_method()
    t.test_revert_layer()


if __name__ == "__main__":
    _run_without_pytest()
    print("OK")

