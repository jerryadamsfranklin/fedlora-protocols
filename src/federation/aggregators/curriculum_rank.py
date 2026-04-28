"""
Curriculum Rank Aggregator (client-side rank control, server-side schedule).

In the *fixed* curriculum-rank protocol, clients actually *train and send* LoRA
updates at the current rank r_t. This yields real communication savings because
clients upload smaller tensors.

Implementation notes for this repo:
- The server provides `current_rank` each round via `get_current_rank(round_num)`.
- Clients keep a full-rank adapter in memory but freeze/slice so they only train
  and *send* the leading r_t rows/cols of LoRA A/B.
- This aggregator must therefore handle variable-shaped A/B tensors by padding
  them back to full rank before doing weighted averaging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch


class CurriculumRankAggregator:
    """
    Curriculum rank: progressively increase effective rank over rounds.

    Args:
        rank_schedule: mapping of inclusive round threshold -> effective rank.
            Example: {5: 4, 10: 8, 15: 16} means:
              rounds 1-5: r=4, rounds 6-10: r=8, rounds 11-15: r=16
        final_rank: fallback rank if schedule is empty
        full_rank: adapter rank expected by clients (pad back to this).
            If None, inferred from incoming client tensors each round.
    """

    def __init__(
        self,
        rank_schedule: Optional[Dict[int, int]] = None,
        final_rank: int = 16,
        full_rank: Optional[int] = None,
    ) -> None:
        self.name = "Curriculum-Rank"
        self.rank_schedule = rank_schedule or {5: 4, 10: 8, 15: int(final_rank)}
        self.final_rank = int(final_rank)
        self.full_rank = int(full_rank) if full_rank is not None else None

        self.round_count = 0
        self.current_rank = self._rank_for_round(1)
        self.rank_history: List[int] = []

    def _rank_for_round(self, round_num: int) -> int:
        if not self.rank_schedule:
            return self.final_rank
        r = min(self.rank_schedule.values())
        for thr, rr in sorted(self.rank_schedule.items()):
            if round_num <= int(thr):
                return int(rr)
            r = int(rr)
        return int(r)

    @staticmethod
    def _pad_to_full_rank(
        a: torch.Tensor,
        b: torch.Tensor,
        full_rank: int,
        a_dtype: torch.dtype,
        b_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # A: [r, d_in], B: [d_out, r]
        r_eff = a.shape[0]
        if r_eff == full_rank:
            return a.to(a_dtype), b.to(b_dtype)
        a_full = torch.zeros((full_rank, a.shape[1]), device=a.device, dtype=a_dtype)
        b_full = torch.zeros((b.shape[0], full_rank), device=b.device, dtype=b_dtype)
        a_full[:r_eff, :] = a.to(a_dtype)
        b_full[:, :r_eff] = b.to(b_dtype)
        return a_full, b_full

    def get_current_rank(self, round_num: int) -> int:
        """Server calls this to tell clients what rank to use this round."""
        return self._rank_for_round(int(round_num))

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
        round_num: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        if not client_states:
            raise ValueError("No client states")

        if round_num is None:
            self.round_count += 1
            round_num = self.round_count
        else:
            self.round_count = int(round_num)

        self.current_rank = self._rank_for_round(self.round_count)
        self.rank_history.append(self.current_rank)

        # Default weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total = float(sum(weights))
        weights = [float(w) / total for w in weights]

        # Find all A/B pairs from the union of keys.
        all_keys = set()
        for st in client_states:
            all_keys.update(st.keys())
        a_keys = [k for k in all_keys if "lora_A" in k or "lora_a" in k]

        aggregated: Dict[str, torch.Tensor] = {}
        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B").replace("lora_a", "lora_b")
            # Gather present tensors across clients.
            a_list = [(w, st[a_key]) for w, st in zip(weights, client_states) if a_key in st]
            b_list = [(w, st[b_key]) for w, st in zip(weights, client_states) if b_key in st]
            if not a_list or not b_list:
                continue

            # Infer full rank from config or max seen across clients this round.
            full_rank = self.full_rank
            if full_rank is None:
                full_rank = max(int(t.shape[0]) for _, t in a_list)

            a0 = a_list[0][1]
            b0 = b_list[0][1]

            # Weighted average in full-rank space (pad per-client tensors).
            A_acc = torch.zeros((full_rank, a0.shape[1]), dtype=torch.float32, device=a0.device)
            B_acc = torch.zeros((b0.shape[0], full_rank), dtype=torch.float32, device=b0.device)

            for w, a in a_list:
                r_eff = int(a.shape[0])
                A_acc[:r_eff, :] += float(w) * a.float()
            for w, b in b_list:
                r_eff = int(b.shape[1])
                B_acc[:, :r_eff] += float(w) * b.float()

            aggregated[a_key] = A_acc.to(a0.dtype).detach()
            aggregated[b_key] = B_acc.to(b0.dtype).detach()

        return aggregated

    def get_freeze_a(self) -> bool:
        # Curriculum rank does not require freezing A.
        return False

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "round_count": self.round_count,
            "current_rank": self.current_rank,
            "rank_schedule": dict(self.rank_schedule),
            "rank_history": list(self.rank_history),
            "full_rank": self.full_rank,
        }

    def reset(self) -> None:
        self.round_count = 0
        self.current_rank = self._rank_for_round(1)
        self.rank_history.clear()

