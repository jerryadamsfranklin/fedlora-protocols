"""
Per-Layer Tracking Mixin for FedLoRA-Adaptive v2.

Tracks convergence metrics per-layer and enables independent switching.
"""

from __future__ import annotations

from typing import Dict, List, Optional


class PerLayerTrackerMixin:
    """
    Mixin that provides per-layer tracking and switching functionality.

    Each tracked layer is assigned a phase:
    - "FFA-LoRA": keep A frozen for that layer
    - "TRANSITION": gradually unfreeze A
    - "FLoRA": fully trainable
    """

    def init_per_layer_tracker(
        self,
        layer_names: List[str],
        switch_threshold: float = 0.01,
        warmup_rounds: int = 3,
        transition_rounds: int = 3,
    ) -> None:
        self.layer_names = list(layer_names)
        self.layer_switch_threshold = float(switch_threshold)
        self.layer_warmup_rounds = int(warmup_rounds)
        self.layer_transition_rounds = int(transition_rounds)

        self.layer_phases: Dict[str, str] = {name: "FFA-LoRA" for name in layer_names}
        self.layer_switch_rounds: Dict[str, Optional[int]] = {
            name: None for name in layer_names
        }
        self.layer_metric_history: Dict[str, List[float]] = {
            name: [] for name in layer_names
        }

    def record_layer_metrics(
        self, layer_metrics: Dict[str, float], round_num: Optional[int] = None
    ) -> None:
        _ = round_num
        for layer_name, metric in layer_metrics.items():
            if layer_name in self.layer_metric_history:
                self.layer_metric_history[layer_name].append(float(metric))

    def check_layer_switches(self, round_num: int) -> Dict[str, bool]:
        switched: Dict[str, bool] = {}
        for layer_name in self.layer_names:
            switched[layer_name] = False
            if self.layer_phases.get(layer_name, "FFA-LoRA") != "FFA-LoRA":
                continue
            if self._should_layer_switch(layer_name, round_num):
                self.layer_phases[layer_name] = "TRANSITION"
                self.layer_switch_rounds[layer_name] = int(round_num)
                switched[layer_name] = True
        return switched

    def update_layer_phases(self, round_num: int) -> None:
        for layer_name in self.layer_names:
            if self.layer_phases.get(layer_name) != "TRANSITION":
                continue
            switch_round = self.layer_switch_rounds.get(layer_name)
            if switch_round is None:
                continue
            if int(round_num) >= switch_round + max(1, self.layer_transition_rounds):
                self.layer_phases[layer_name] = "FLoRA"

    def _should_layer_switch(self, layer_name: str, round_num: int) -> bool:
        if int(round_num) <= self.layer_warmup_rounds:
            return False
        history = self.layer_metric_history.get(layer_name, [])
        if len(history) < 2:
            return False
        prev_metric = history[-2]
        curr_metric = history[-1]
        if prev_metric == 0:
            return False
        improvement_rate = (prev_metric - curr_metric) / prev_metric
        return improvement_rate < self.layer_switch_threshold

    def get_layer_freeze_ratio(self, layer_name: str, round_num: int) -> float:
        phase = self.layer_phases.get(layer_name, "FFA-LoRA")
        if phase == "FFA-LoRA":
            return 1.0
        if phase == "FLoRA":
            return 0.0
        if phase == "TRANSITION":
            switch_round = self.layer_switch_rounds.get(layer_name)
            if switch_round is None:
                return 0.5
            rounds_since_switch = int(round_num) - int(switch_round)
            denom = max(1, self.layer_transition_rounds)
            progress = min(1.0, rounds_since_switch / denom)
            return 1.0 - progress
        return 1.0

    def get_layer_freeze_ratios(self, round_num: int) -> Dict[str, float]:
        return {
            layer_name: self.get_layer_freeze_ratio(layer_name, round_num)
            for layer_name in self.layer_names
        }

    def revert_layer(self, layer_name: str, round_num: Optional[int] = None) -> None:
        _ = round_num
        if self.layer_phases.get(layer_name, "FFA-LoRA") != "FFA-LoRA":
            self.layer_phases[layer_name] = "FFA-LoRA"
            self.layer_switch_rounds[layer_name] = None

    def revert_all_layers(self, round_num: Optional[int] = None) -> None:
        _ = round_num
        for layer_name in self.layer_names:
            self.revert_layer(layer_name, round_num=round_num)

    def get_layer_analysis(self) -> Dict:
        switch_rounds = [r for r in self.layer_switch_rounds.values() if r is not None]
        return {
            "layer_phases": dict(self.layer_phases),
            "layer_switch_rounds": dict(self.layer_switch_rounds),
            "layers_still_ffa_lora": [
                name
                for name, phase in self.layer_phases.items()
                if phase == "FFA-LoRA"
            ],
            "layers_in_flora": [
                name for name, phase in self.layer_phases.items() if phase == "FLoRA"
            ],
            "earliest_switch": min(switch_rounds) if switch_rounds else None,
            "latest_switch": max(switch_rounds) if switch_rounds else None,
        }

