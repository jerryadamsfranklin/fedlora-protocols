"""
FLoRA Aggregator: Stacking-based Aggregation

From Wang et al., NeurIPS 2024.
Instead of averaging, stack LoRA modules then compress with SVD.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FLoRAAggregator:
    """
    FLoRA: Stack client LoRAs, compress with SVD.

    - Stack A matrices: [A_1; A_2; ... A_K]
    - Concat B matrices: [B_1, B_2, ..., B_K]
    - Compress using SVD to limit size
    """

    def __init__(self, max_rank: int = 64):
        self.name = "FLoRA"
        self.max_rank = max_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Stack and compress client LoRAs.
        """
        if not client_states:
            raise ValueError("No client states")

        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        aggregated = {}

        # Find A/B pairs (PEFT uses lora_A / lora_B)
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            # Weighted average of low-rank updates in full space:
            #   ΔW_k = B_k @ A_k  →  ΔW = Σ_k w_k ΔW_k
            # (Stacking B and concat A gives stacked_B @ stacked_A = Σ ΔW_k without
            # weights — K times too large when clients are equally weighted.)
            ba = None
            for w, state in zip(weights, client_states):
                a = state[a_key].float()
                b = state[b_key].float()
                term = w * (b @ a)
                ba = term if ba is None else ba + term

            U, S, Vh = torch.linalg.svd(ba, full_matrices=False)

            r = min(self.max_rank, S.shape[0])
            U = U[:, :r]
            S = S[:r]
            Vh = Vh[:r, :]

            sqrt_s = torch.sqrt(S)
            new_b = (U * sqrt_s.unsqueeze(0)).to(client_states[0][b_key].dtype)
            new_a = (sqrt_s.unsqueeze(1) * Vh).to(client_states[0][a_key].dtype)

            aggregated[a_key] = new_a
            aggregated[b_key] = new_b

        return aggregated

    def get_communication_cost(
        self,
        state: Dict[str, torch.Tensor],
        num_clients: int = 1,
    ) -> int:
        """Bytes transmitted."""
        params = sum(p.numel() for p in state.values())
        return params * 2 * 2  # Approximate
