"""
FedLoRA-Adaptive: Adaptive switching between FFA-LoRA and FLoRA.

This is the main novel method described in CURSOR_INSTRUCTIONS_COMPLETE.md.

High-level behavior:
- Start in FFA-LoRA mode (freeze LoRA A; aggregate B only).
- Monitor convergence each round using the server-provided current_loss.
- Switch to FLoRA when improvement_rate < switch_threshold (after warmup).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch

from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator


class FedLoRAAdaptiveAggregator:
    """
    FedLoRA-Adaptive aggregator.

    Parameters
    ----------
    max_rank:
        Rank used for FLoRA compression (should match LoRA rank r for PEFT shapes).
    switch_threshold:
        Switch from FFA-LoRA -> FLoRA when improvement_rate < threshold.
        improvement_rate = (prev_loss - curr_loss) / prev_loss
    warmup_rounds:
        Minimum number of rounds before considering switching.
    fixed_switch_round:
        If set, forces switch at (or after) this round (for ablations).
    """

    def __init__(
        self,
        max_rank: int = 64,
        switch_threshold: float = 0.01,
        warmup_rounds: int = 3,
        fixed_switch_round: Optional[int] = None,
    ):
        self.name = "FedLoRA-Adaptive"

        self.max_rank = max_rank
        self.switch_threshold = switch_threshold
        self.warmup_rounds = warmup_rounds
        self.fixed_switch_round = fixed_switch_round

        self.ffa_lora = FFALoRAAggregator()
        self.flora = FLoRAAggregator(max_rank=max_rank)

        self.current_mode = "FFA-LoRA"
        self.round_count = 0
        self.loss_history: List[float] = []
        self.switch_round: Optional[int] = None

        self.ffa_lora_rounds = 0
        self.flora_rounds = 0

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        current_loss: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client states using current mode.

        IMPORTANT: current_loss must be passed by the server for automatic switching.
        """
        self.round_count += 1

        if current_loss is not None:
            self.loss_history.append(float(current_loss))

        if self.current_mode == "FFA-LoRA" and self._should_switch():
            self._switch_to_flora()

        if self.current_mode == "FFA-LoRA":
            self.ffa_lora_rounds += 1
            return self.ffa_lora.aggregate(client_states, weights=weights)

        self.flora_rounds += 1
        return self.flora.aggregate(client_states, weights=weights)

    def _should_switch(self) -> bool:
        if self.fixed_switch_round is not None:
            return self.round_count >= self.fixed_switch_round

        if self.round_count <= self.warmup_rounds:
            return False

        if len(self.loss_history) < 2:
            return False

        prev_loss = self.loss_history[-2]
        curr_loss = self.loss_history[-1]
        if prev_loss <= 0:
            return False

        improvement_rate = (prev_loss - curr_loss) / prev_loss
        return improvement_rate < self.switch_threshold

    def _switch_to_flora(self) -> None:
        self.current_mode = "FLoRA"
        self.switch_round = self.round_count
        print(
            f"\n[FedLoRA-Adaptive] Switching to FLoRA at round {self.switch_round} "
            f"(threshold={self.switch_threshold}, warmup={self.warmup_rounds})\n"
        )

    def get_mode(self) -> str:
        return self.current_mode

    def get_switch_round(self) -> Optional[int]:
        return self.switch_round

    def get_stats(self) -> Dict:
        total = max(1, self.round_count)
        # Approx: FFA-LoRA saves ~50% communication during its rounds.
        comm_savings = 0.5 * (self.ffa_lora_rounds / total)
        return {
            "name": self.name,
            "current_mode": self.current_mode,
            "switch_round": self.switch_round,
            "total_rounds": self.round_count,
            "ffa_lora_rounds": self.ffa_lora_rounds,
            "flora_rounds": self.flora_rounds,
            "switch_threshold": self.switch_threshold,
            "warmup_rounds": self.warmup_rounds,
            "communication_savings_est": comm_savings,
        }

    def reset(self) -> None:
        self.current_mode = "FFA-LoRA"
        self.round_count = 0
        self.loss_history = []
        self.switch_round = None
        self.ffa_lora_rounds = 0
        self.flora_rounds = 0
        if hasattr(self.ffa_lora, "reset"):
            self.ffa_lora.reset()

