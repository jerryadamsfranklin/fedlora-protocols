"""
Budget-adaptive: choose FLoRA vs FFA-LoRA each round to stay under a total comm budget.
"""

from typing import Any, Dict, List, Optional

import torch

from .ffa_lora import FFALoRAAggregator
from .flora import FLoRAAggregator


class BudgetAdaptiveAggregator:
    def __init__(
        self,
        total_budget_mb: float,
        num_rounds: int,
        priority: str = "quality",
        max_rank: int = 16,
    ) -> None:
        if priority not in ("quality", "efficiency"):
            raise ValueError("priority must be 'quality' or 'efficiency'")
        self.name = "Budget-Adaptive"
        self.total_budget_mb = float(total_budget_mb)
        self.num_rounds = int(num_rounds)
        self.priority = priority
        self.max_rank = int(max_rank)
        self.flora = FLoRAAggregator(max_rank=self.max_rank)
        self.ffa_lora = FFALoRAAggregator()
        self.round_count = 0
        self.budget_remaining = self.total_budget_mb
        self.budget_spent = 0.0
        self.flora_comm_per_round: Optional[float] = None
        self.ffa_comm_per_round: Optional[float] = None
        self.flora_rounds = 0
        self.ffa_lora_rounds = 0
        self.round_methods: List[str] = []
        self._last_fr = 0.0  # 0 = FLoRA path, 1 = FFA path (for reset)
        # Chosen in plan_round() before client training; aggregate() must match it.
        self._plan_use_flora: Optional[bool] = None

    def _measure_costs(self, client_states: List[Dict[str, torch.Tensor]]) -> None:
        if not client_states:
            self.flora_comm_per_round = 172.0
            self.ffa_comm_per_round = 86.0
            return
        s0 = client_states[0]
        n = len(client_states)
        fb = sum(p.numel() * p.element_size() for p in s0.values()) * n
        self.flora_comm_per_round = fb / (1024 * 1024)
        ffa_b = sum(
            p.numel() * p.element_size()
            for n_, p in s0.items()
            if "lora_B" in n_ or "lora_b" in n_
        ) * n
        self.ffa_comm_per_round = ffa_b / (1024 * 1024)

    def _should_use_flora(self, round_num: int) -> bool:
        if self.flora_comm_per_round is None or self.ffa_comm_per_round is None:
            return True
        r_rem = self.num_rounds - round_num + 1
        if r_rem <= 0:
            return False
        if self.priority == "quality":
            min_future = self.ffa_comm_per_round * (r_rem - 1)
            can = (self.flora_comm_per_round + min_future) <= self.budget_remaining
            return bool(can)
        avg = self.budget_remaining / r_rem
        return bool(avg >= self.flora_comm_per_round * 1.1)

    def plan_round(self, round_num: int) -> None:
        """
        Call once at the start of each federated round (1-based) before get_freeze_a.
        Picks FLoRA vs FFA for this round so client training matches aggregation.
        """
        r = int(round_num)
        if self.flora_comm_per_round is None or self.ffa_comm_per_round is None:
            # No measured costs yet: first policy decision defaults to FLoRA.
            # Costs are filled after the first aggregate() from real client state.
            self._plan_use_flora = True
        else:
            self._plan_use_flora = self._should_use_flora(r)
        return None

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
        if self.flora_comm_per_round is None:
            self._measure_costs(client_states)
        assert self.flora_comm_per_round is not None
        assert self.ffa_comm_per_round is not None
        if self._plan_use_flora is not None:
            use_flora = self._plan_use_flora
            self._plan_use_flora = None
        else:
            use_flora = self._should_use_flora(self.round_count)
        if use_flora:
            cost = self.flora_comm_per_round
            out = self.flora.aggregate(client_states, weights)
            self.flora_rounds += 1
            self.round_methods.append("FLoRA")
        else:
            if self._last_fr < 0.5:
                self.ffa_lora.reset()
            out = self.ffa_lora.aggregate(client_states, weights)
            cost = self.ffa_comm_per_round
            self.ffa_lora_rounds += 1
            self.round_methods.append("FFA-LoRA")
        self.budget_spent += cost
        self.budget_remaining -= cost
        self._last_fr = 0.0 if use_flora else 1.0
        return out

    def get_freeze_a(self) -> bool:
        if self._plan_use_flora is not None:
            return not self._plan_use_flora
        if not self.round_methods:
            return False
        return self.round_methods[-1] == "FFA-LoRA"

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "total_budget_mb": self.total_budget_mb,
            "budget_spent_mb": self.budget_spent,
            "budget_remaining_mb": self.budget_remaining,
            "budget_utilization": self.budget_spent / self.total_budget_mb
            if self.total_budget_mb > 0
            else 0.0,
            "round_count": self.round_count,
            "flora_rounds": self.flora_rounds,
            "ffa_lora_rounds": self.ffa_lora_rounds,
            "priority": self.priority,
            "round_methods": list(self.round_methods),
        }

    def reset(self) -> None:
        self.ffa_lora.reset()
        self.round_count = 0
        self.budget_remaining = self.total_budget_mb
        self.budget_spent = 0.0
        self.flora_comm_per_round = None
        self.ffa_comm_per_round = None
        self.flora_rounds = 0
        self.ffa_lora_rounds = 0
        self.round_methods.clear()
        self._last_fr = 0.0
        self._plan_use_flora = None
