"""
FedLoRA-Adaptive v2: Enhanced Adaptive Aggregation with:
1. Soft Transition (gradual A unfreezing)
2. Per-Layer Switching (independent layer decisions)
3. Stability Detection (bidirectional switching)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator
from .mixins.per_layer_tracker import PerLayerTrackerMixin
from .mixins.soft_transition import SoftTransitionMixin
from .mixins.stability_detector import StabilityDetectorMixin


class FedLoRAAdaptiveV2Aggregator(
    SoftTransitionMixin, StabilityDetectorMixin, PerLayerTrackerMixin
):
    """
    FedLoRA-Adaptive v2 aggregator.

    Uses:
    - per-layer switching signals driven by per-layer metrics (e.g. grad norms)
    - soft transition via freeze ratios
    - stability detection that can revert layers back to FFA-LoRA
    """

    def __init__(
        self,
        layer_names: List[str],
        max_rank: int = 64,
        switch_threshold: float = 0.01,
        warmup_rounds: int = 3,
        transition_rounds: int = 3,
        stability_threshold: float = 1.1,
        per_layer_enabled: bool = True,
    ):
        self.name = "FedLoRA-Adaptive-v2"

        self.max_rank = max_rank
        self.per_layer_enabled = bool(per_layer_enabled)

        # Mixins
        self.init_soft_transition(transition_rounds=transition_rounds)
        self.init_stability_detector(spike_threshold=stability_threshold, sustained_rounds=2)
        self.init_per_layer_tracker(
            layer_names=layer_names,
            switch_threshold=switch_threshold,
            warmup_rounds=warmup_rounds,
            transition_rounds=transition_rounds,
        )

        # Component aggregators
        self.ffa_lora = FFALoRAAggregator()
        self.flora = FLoRAAggregator(max_rank=max_rank)

        # Global fallback state for non-per-layer mode
        self.round_count = 0
        self.global_phase = "FFA-LoRA"
        self.switch_round: Optional[int] = None

    @staticmethod
    def _extract_layer_id_from_key(key: str) -> str:
        parts = key.split(".")
        for i, part in enumerate(parts):
            if "lora" in part.lower():
                if i > 0:
                    return parts[i - 1]
        return "unknown"

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        current_loss: Optional[float] = None,
        layer_metrics: Optional[Dict[str, float]] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states to aggregate")

        # Round bookkeeping
        if round_num is None:
            self.round_count += 1
            round_num = self.round_count
        else:
            self.round_count = int(round_num)

        # Stability tracking (uses current_loss vs prior recorded)
        if current_loss is not None:
            if self.detect_instability(float(current_loss)):
                self.record_revert(round_num)
                if self.per_layer_enabled:
                    self.revert_all_layers(round_num)
                else:
                    self.global_phase = "FFA-LoRA"
                    self.switch_round = None
            self.record_loss(float(current_loss), round_num=round_num)

        # Per-layer switching updates
        if self.per_layer_enabled and layer_metrics is not None:
            self.record_layer_metrics(layer_metrics, round_num=round_num)
            self.check_layer_switches(round_num)
            self.update_layer_phases(round_num)

        # Global (non-per-layer) switching: mimic v1 with soft transition phases.
        if not self.per_layer_enabled and current_loss is not None:
            if self.global_phase == "FFA-LoRA":
                if (
                    round_num > self.layer_warmup_rounds
                    and len(self.loss_history) >= 2
                ):
                    prev_loss = self.loss_history[-2]
                    curr_loss = self.loss_history[-1]
                    if prev_loss > 0:
                        improvement_rate = (prev_loss - curr_loss) / prev_loss
                        if improvement_rate < self.layer_switch_threshold:
                            self.global_phase = "TRANSITION"
                            self.switch_round = round_num
                            self.start_transition(round_num)
            elif self.global_phase == "TRANSITION":
                if self.is_transition_complete(round_num):
                    self.global_phase = "FLoRA"

        if not self.per_layer_enabled:
            freeze_ratio = self.get_freeze_ratio(round_num, self.global_phase)
            agg = self.ffa_lora if freeze_ratio > 0.5 else self.flora
            return agg.aggregate(client_states, weights=weights)

        # True per-layer dispatch: each layer chooses its own aggregator.
        # Group all keys by layer id.
        all_keys = set()
        for state in client_states:
            all_keys.update(state.keys())

        layer_to_keys: Dict[str, List[str]] = {}
        for key in all_keys:
            layer_id = self._extract_layer_id_from_key(key)
            layer_to_keys.setdefault(layer_id, []).append(key)

        layer_freeze = self.get_layer_freeze_ratios(round_num)

        aggregated: Dict[str, torch.Tensor] = {}
        for layer_id, keys in layer_to_keys.items():
            ratio = layer_freeze.get(layer_id, 1.0)
            agg = self.ffa_lora if ratio > 0.5 else self.flora

            layer_client_states = [{k: s[k] for k in keys if k in s} for s in client_states]
            layer_agg = agg.aggregate(layer_client_states, weights=weights)
            aggregated.update(layer_agg)

        return aggregated

    def get_stats(self) -> Dict:
        stats = {
            "name": self.name,
            "round_count": self.round_count,
            "per_layer_enabled": self.per_layer_enabled,
        }
        stats.update(self.get_stability_stats())
        if self.per_layer_enabled:
            stats["layer_analysis"] = self.get_layer_analysis()
        else:
            stats["global_phase"] = self.global_phase
            stats["switch_round"] = self.switch_round
        return stats

    def reset(self) -> None:
        self.round_count = 0
        self.global_phase = "FFA-LoRA"
        self.switch_round = None
        self.loss_history = []
        self.revert_count = 0
        for layer in self.layer_names:
            self.layer_phases[layer] = "FFA-LoRA"
            self.layer_switch_rounds[layer] = None
            self.layer_metric_history[layer] = []

