"""
Two-Phase: FLoRA for early rounds, then FFA-LoRA (fixed boundary).
"""

from typing import Any, Dict, List, Optional

import torch

from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator


class TwoPhaseAggregator:
    """FLoRA through `phase_boundary`, then FFA-LoRA."""

    def __init__(self, phase_boundary: int = 8, max_rank: int = 16) -> None:
        self.name = "Two-Phase"
        self.phase_boundary = int(phase_boundary)
        self.max_rank = int(max_rank)
        self.flora = FLoRAAggregator(max_rank=self.max_rank)
        self.ffa_lora = FFALoRAAggregator()
        self.round_count = 0
        self.current_phase = 1
        self.phase1_rounds = 0
        self.phase2_rounds = 0
        self.flora_comm_per_round: Optional[float] = None
        self.ffa_comm_per_round: Optional[float] = None
        self.total_comm_mb = 0.0

    def _track_comm(self, is_flora: bool, client_states: List[Dict[str, torch.Tensor]]) -> None:
        if not client_states:
            return
        sample = client_states[0]
        n = len(client_states)
        if is_flora:
            bpc = sum(
                p.numel() * p.element_size() for p in sample.values()
            ) / (1024 * 1024)
            if self.flora_comm_per_round is None:
                self.flora_comm_per_round = bpc * n
            self.total_comm_mb += bpc * n
        else:
            bpc = sum(
                p.numel() * p.element_size()
                for n_, p in sample.items()
                if "lora_B" in n_ or "lora_b" in n_
            ) / (1024 * 1024)
            if self.ffa_comm_per_round is None:
                self.ffa_comm_per_round = bpc * n
            self.total_comm_mb += bpc * n

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states to aggregate")
        if round_num is None:
            self.round_count += 1
            round_num = self.round_count
        else:
            self.round_count = int(round_num)

        if self.round_count <= self.phase_boundary:
            self.current_phase = 1
            self.phase1_rounds += 1
            out = self.flora.aggregate(client_states, weights)
            self._track_comm(True, client_states)
        else:
            self.current_phase = 2
            self.phase2_rounds += 1
            out = self.ffa_lora.aggregate(client_states, weights)
            self._track_comm(False, client_states)
        return out

    def get_freeze_a(self) -> bool:
        # Server calls this *before* aggregate; `round_count` is rounds already finished.
        # This round's training (1-based) is round_count + 1, which must match aggregate().
        nxt = self.round_count + 1
        return nxt > self.phase_boundary

    def get_stats(self) -> Dict[str, Any]:
        savings = 0.0
        if self.flora_comm_per_round and self.round_count > 0:
            flora_total = self.flora_comm_per_round * self.round_count
            if flora_total > 0:
                savings = 1.0 - (self.total_comm_mb / flora_total)
        return {
            "name": self.name,
            "phase_boundary": self.phase_boundary,
            "current_phase": self.current_phase,
            "round_count": self.round_count,
            "phase1_rounds": self.phase1_rounds,
            "phase2_rounds": self.phase2_rounds,
            "flora_comm_per_round_mb": self.flora_comm_per_round,
            "ffa_comm_per_round_mb": self.ffa_comm_per_round,
            "total_comm_mb": self.total_comm_mb,
            "comm_savings_vs_flora": savings,
        }

    def get_frozen_a(self) -> Dict[str, torch.Tensor]:
        """
        Return cached frozen A matrices from the internal FFA-LoRA aggregator.

        Used by the server to reconstruct full client states from B-only uploads.
        Returns empty dict if FFA-LoRA has not been initialized yet.
        """
        return self.ffa_lora.get_frozen_a()

    def should_upload_b_only(self) -> bool:
        """
        Whether clients should upload B-only tensors for this round.

        The first FFA round (round phase_boundary+1) must upload full A+B so the
        server-side FFA aggregator can initialize frozen A. Therefore B-only
        begins starting at round phase_boundary+2.
        """
        # Server queries this before training the next round. At the start of
        # round (phase_boundary+2), self.round_count == phase_boundary+1.
        return self.round_count > self.phase_boundary

    def should_broadcast_b_only(self) -> bool:
        """
        Whether the server should broadcast B-only state for the upcoming round.

        Symmetric to should_upload_b_only: True only after the first FFA round
        has completed (frozen A is cached and clients have it locally).
        """
        return self.round_count > self.phase_boundary

    def reset(self) -> None:
        self.ffa_lora.reset()
        self.round_count = 0
        self.current_phase = 1
        self.phase1_rounds = 0
        self.phase2_rounds = 0
        self.flora_comm_per_round = None
        self.ffa_comm_per_round = None
        self.total_comm_mb = 0.0

    def state_dict(self) -> Dict[str, Any]:
        return {
            "round_count": self.round_count,
            "current_phase": self.current_phase,
            "phase1_rounds": self.phase1_rounds,
            "phase2_rounds": self.phase2_rounds,
            "flora_comm_per_round": self.flora_comm_per_round,
            "ffa_comm_per_round": self.ffa_comm_per_round,
            "total_comm_mb": self.total_comm_mb,
            "phase_boundary": self.phase_boundary,
            "max_rank": self.max_rank,
            "ffa_lora": self.ffa_lora.state_dict(),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.round_count = int(state.get("round_count", 0))
        self.current_phase = int(state.get("current_phase", 1))
        self.phase1_rounds = int(state.get("phase1_rounds", 0))
        self.phase2_rounds = int(state.get("phase2_rounds", 0))
        self.flora_comm_per_round = state.get("flora_comm_per_round")
        self.ffa_comm_per_round = state.get("ffa_comm_per_round")
        self.total_comm_mb = float(state.get("total_comm_mb", 0.0))
        if "ffa_lora" in state:
            self.ffa_lora.load_state_dict(state["ffa_lora"])
