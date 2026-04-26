"""
Curriculum Rank Aggregator (server-side effective rank schedule).

Clients always train with full adapter rank r_full (e.g. 16). The server aggregates
client updates by computing ΔW = Σ w_k (B_k @ A_k), then taking an SVD and keeping
only the top-r components for the current round's scheduled rank.

To avoid any client/model changes, the aggregated LoRA tensors are padded back to
the original adapter rank so `FederatedLoRAModel.set_lora_state_dict()` can copy
them into existing PEFT parameters without shape mismatches.
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

        # Find A/B pairs
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k or "lora_a" in k]
        aggregated: Dict[str, torch.Tensor] = {}

        for a_key in a_keys:
            # Support both lora_A and lora_a casing if present (PEFT uses lora_A)
            b_key = a_key.replace("lora_A", "lora_B").replace("lora_a", "lora_b")
            if b_key not in client_states[0]:
                continue

            a0 = client_states[0][a_key]
            b0 = client_states[0][b_key]
            full_rank = self.full_rank or int(a0.shape[0])

            # Compute weighted ΔW
            accumulated = None
            for w, st in zip(weights, client_states):
                a = st[a_key].float()
                b = st[b_key].float()
                upd = b @ a  # [d_out, d_in]
                accumulated = (w * upd) if accumulated is None else (accumulated + w * upd)

            # SVD and truncate to effective rank (capped by full rank)
            U, S, Vh = torch.linalg.svd(accumulated, full_matrices=False)
            r_eff = int(min(self.current_rank, full_rank, S.shape[0]))
            U_r = U[:, :r_eff]
            S_r = S[:r_eff]
            Vh_r = Vh[:r_eff, :]

            sqrt_s = torch.sqrt(S_r)
            new_b = U_r * sqrt_s.unsqueeze(0)          # [d_out, r_eff]
            new_a = sqrt_s.unsqueeze(1) * Vh_r         # [r_eff, d_in]

            # Pad back to full adapter rank so client can load without shape mismatch
            a_full, b_full = self._pad_to_full_rank(
                new_a, new_b, full_rank=full_rank, a_dtype=a0.dtype, b_dtype=b0.dtype
            )

            aggregated[a_key] = a_full.detach()
            aggregated[b_key] = b_full.detach()

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

