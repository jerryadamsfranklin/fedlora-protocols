"""
Reverse adaptive: FLoRA first, then FFA-LoRA when per-round loss improvement is small.
"""

from typing import Any, Dict, List, Optional

import torch

from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator


class ReverseAdaptiveAggregator:
    def __init__(
        self,
        switch_threshold: float = 0.01,
        warmup_rounds: int = 5,
        transition_rounds: int = 2,
        stability_threshold: float = 1.1,
        max_rank: int = 16,
    ) -> None:
        self.name = "Reverse-Adaptive"
        self.switch_threshold = float(switch_threshold)
        self.warmup_rounds = int(warmup_rounds)
        self.transition_rounds = int(transition_rounds)
        self.stability_threshold = float(stability_threshold)
        self.max_rank = int(max_rank)
        self.flora = FLoRAAggregator(max_rank=self.max_rank)
        self.ffa_lora = FFALoRAAggregator()
        self.round_count = 0
        self.current_mode = "FLoRA"
        self.switch_round: Optional[int] = None
        self.loss_history: List[float] = []
        self.revert_count = 0
        self.flora_rounds = 0
        self.ffa_lora_rounds = 0
        self.total_comm_mb = 0.0
        self.flora_comm_per_round: Optional[float] = None
        self.ffa_comm_per_round: Optional[float] = None
        self._last_fr = 0.0  # after last aggregate; <0.5 means FLoRA-like
        self.events: List[Dict[str, Any]] = []

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

    def _get_freeze_ratio(self, round_num: int) -> float:
        if self.current_mode == "FLoRA":
            return 0.0
        if self.current_mode == "FFA-LoRA":
            return 1.0
        if self.switch_round is None or self.transition_rounds <= 0:
            return 0.0
        progress = (round_num - self.switch_round) / self.transition_rounds
        progress = min(1.0, max(0.0, float(progress)))
        return progress

    def _should_switch(self, round_num: int) -> bool:
        if self.current_mode != "FLoRA":
            return False
        if round_num <= self.warmup_rounds:
            return False
        if len(self.loss_history) < 2:
            return False
        prev_loss = self.loss_history[-2]
        curr_loss = self.loss_history[-1]
        if prev_loss <= 0:
            return False
        improvement = (prev_loss - curr_loss) / prev_loss
        return bool(improvement < self.switch_threshold)

    def _loss_history_tail(self) -> List[float]:
        if len(self.loss_history) >= 3:
            return self.loss_history[-3:]
        return list(self.loss_history)

    def _start_transition(self, round_num: int) -> None:
        self.switch_round = round_num
        if self.transition_rounds > 0:
            self.current_mode = "TRANSITION"
        else:
            self.ffa_lora.reset()
            self.current_mode = "FFA-LoRA"
        self.events.append(
            {
                "round": self.round_count,
                "event": "switch_to_ffa",
                "loss_history_tail": self._loss_history_tail(),
            }
        )

    def _revert_to_flora(self) -> None:
        self.ffa_lora.reset()
        self.current_mode = "FLoRA"
        self.switch_round = None
        self.revert_count += 1
        self.events.append(
            {
                "round": self.round_count,
                "event": "revert_to_flora",
                "loss_history_tail": self._loss_history_tail(),
            }
        )

    def _detect_instability(self, current_loss: float) -> bool:
        if self.current_mode == "FLoRA" or len(self.loss_history) < 2:
            return False
        prev_loss = self.loss_history[-2]
        if prev_loss <= 0:
            return False
        return bool(current_loss > prev_loss * self.stability_threshold)

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        current_loss: Optional[float] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states to aggregate")
        if round_num is None:
            self.round_count += 1
            round_num = self.round_count
        else:
            self.round_count = int(round_num)

        if current_loss is not None:
            self.loss_history.append(float(current_loss))
            if self._detect_instability(float(current_loss)):
                self._revert_to_flora()
            elif (
                self.current_mode == "FLoRA"
                and self._should_switch(self.round_count)
            ):
                self._start_transition(self.round_count)
        if self.current_mode == "TRANSITION" and self.switch_round is not None:
            if self.round_count >= self.switch_round + self.transition_rounds:
                # Keep FFA's frozen A from the end of the transition; only flip mode
                self.current_mode = "FFA-LoRA"

        fr = self._get_freeze_ratio(self.round_count)
        if fr < 0.5:
            self.flora_rounds += 1
            out = self.flora.aggregate(client_states, weights)
            self._track_comm(True, client_states)
        else:
            if self._last_fr < 0.5:
                self.ffa_lora.reset()
            self.ffa_lora_rounds += 1
            out = self.ffa_lora.aggregate(client_states, weights)
            self._track_comm(False, client_states)
        self._last_fr = fr
        return out

    def get_freeze_a(self) -> bool:
        # Align with the upcoming round: `round_count` = rounds already aggregated.
        nxt = self.round_count + 1
        return self._get_freeze_ratio(nxt) >= 0.5

    def get_stats(self) -> Dict[str, Any]:
        savings = 0.0
        if self.flora_comm_per_round and self.round_count > 0:
            den = self.flora_comm_per_round * self.round_count
            if den > 0:
                savings = 1.0 - (self.total_comm_mb / den)
        return {
            "name": self.name,
            "current_mode": self.current_mode,
            "switch_round": self.switch_round,
            "round_count": self.round_count,
            "flora_rounds": self.flora_rounds,
            "ffa_lora_rounds": self.ffa_lora_rounds,
            "revert_count": self.revert_count,
            "switch_threshold": self.switch_threshold,
            "warmup_rounds": self.warmup_rounds,
            "total_comm_mb": self.total_comm_mb,
            "comm_savings_vs_flora": savings,
            "events": list(self.events),
        }

    def get_frozen_a(self) -> Dict[str, torch.Tensor]:
        """
        Return the frozen A matrices currently held by the internal FFA-LoRA
        aggregator. Returns an empty dict if FFA-LoRA has not been initialized
        (i.e., we are still in FLoRA phase).
        """
        return self.ffa_lora.get_frozen_a()

    def should_upload_b_only(self) -> bool:
        """
        Clients should upload B-only when the next round will be in freeze-A
        mode AND the FFA-LoRA aggregator has already cached frozen A.
        """
        nxt = self.round_count + 1
        next_freeze = self._get_freeze_ratio(nxt) >= 0.5
        return next_freeze and bool(self.ffa_lora.initialized)

    def should_broadcast_b_only(self) -> bool:
        """
        Server broadcasts B-only under the same condition as upload.
        """
        return self.should_upload_b_only()

    def reset(self) -> None:
        self.ffa_lora.reset()
        self.round_count = 0
        self.current_mode = "FLoRA"
        self.switch_round = None
        self.loss_history.clear()
        self.revert_count = 0
        self.flora_rounds = 0
        self.ffa_lora_rounds = 0
        self.total_comm_mb = 0.0
        self.flora_comm_per_round = None
        self.ffa_comm_per_round = None
        self._last_fr = 0.0
        self.events.clear()

    def state_dict(self) -> Dict[str, Any]:
        return {
            "round_count": self.round_count,
            "current_mode": self.current_mode,
            "switch_round": self.switch_round,
            "loss_history": list(self.loss_history),
            "revert_count": self.revert_count,
            "flora_rounds": self.flora_rounds,
            "ffa_lora_rounds": self.ffa_lora_rounds,
            "total_comm_mb": self.total_comm_mb,
            "flora_comm_per_round": self.flora_comm_per_round,
            "ffa_comm_per_round": self.ffa_comm_per_round,
            "_last_fr": self._last_fr,
            "events": list(self.events),
            "ffa_lora": self.ffa_lora.state_dict(),
            # Hyperparameters kept for sanity checks on resume
            "switch_threshold": self.switch_threshold,
            "warmup_rounds": self.warmup_rounds,
            "transition_rounds": self.transition_rounds,
            "stability_threshold": self.stability_threshold,
            "max_rank": self.max_rank,
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.round_count = int(state.get("round_count", 0))
        self.current_mode = state.get("current_mode", "FLoRA")
        self.switch_round = state.get("switch_round")
        self.loss_history = list(state.get("loss_history") or [])
        self.revert_count = int(state.get("revert_count", 0))
        self.flora_rounds = int(state.get("flora_rounds", 0))
        self.ffa_lora_rounds = int(state.get("ffa_lora_rounds", 0))
        self.total_comm_mb = float(state.get("total_comm_mb", 0.0))
        self.flora_comm_per_round = state.get("flora_comm_per_round")
        self.ffa_comm_per_round = state.get("ffa_comm_per_round")
        self._last_fr = float(state.get("_last_fr", 0.0))
        self.events = list(state.get("events") or [])
        if "ffa_lora" in state:
            self.ffa_lora.load_state_dict(state["ffa_lora"])
