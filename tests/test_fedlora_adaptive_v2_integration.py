"""Integration test for FedLoRA-Adaptive v2 (lightweight).

Note: this test imports torch and will only run in the project environment
where torch is installed and functioning (e.g. your local `fedlora` conda env).
"""

try:
    import torch
except Exception as e:  # pragma: no cover
    raise SystemExit(f"SKIP: torch import failed: {e}")

from src.federation.aggregators.fedlora_adaptive_v2 import (  # noqa: E402
    FedLoRAAdaptiveV2Aggregator,
)


class TestFedLoRAAdaptiveV2Integration:
    def setup_method(self):
        self.layer_names = ["q_proj", "k_proj", "v_proj", "o_proj"]
        self.aggregator = FedLoRAAdaptiveV2Aggregator(
            layer_names=self.layer_names,
            max_rank=4,
            switch_threshold=0.01,
            warmup_rounds=3,
            transition_rounds=3,
            stability_threshold=1.1,
            per_layer_enabled=True,
        )

    def _mock_client_states(self, num_clients: int = 3):
        # Minimal shapes consistent with LoRA math.
        # A: (r, d_in), B: (d_out, r)
        r = 4
        d_in = 8
        d_out = 8
        states = []
        for _ in range(num_clients):
            s = {}
            for layer in self.layer_names:
                s[f"base.{layer}.lora_A.weight"] = torch.randn(r, d_in)
                s[f"base.{layer}.lora_B.weight"] = torch.randn(d_out, r)
            states.append(s)
        return states

    def test_aggregate_returns_all_keys(self):
        states = self._mock_client_states()
        out = self.aggregator.aggregate(
            client_states=states,
            weights=[1 / len(states)] * len(states),
            current_loss=1.5,
            layer_metrics={k: 1.0 for k in self.layer_names},
            round_num=1,
        )
        # Expect all keys returned
        for layer in self.layer_names:
            assert f"base.{layer}.lora_A.weight" in out
            assert f"base.{layer}.lora_B.weight" in out

    def test_get_stats_has_layer_analysis(self):
        stats = self.aggregator.get_stats()
        assert "layer_analysis" in stats
        assert "layer_phases" in stats["layer_analysis"]


def _run_without_pytest():
    t = TestFedLoRAAdaptiveV2Integration()
    t.setup_method()
    t.test_aggregate_returns_all_keys()
    t.setup_method()
    t.test_get_stats_has_layer_analysis()


if __name__ == "__main__":
    _run_without_pytest()
    print("OK")

