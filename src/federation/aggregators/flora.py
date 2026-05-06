"""
FLoRA Aggregator: Stacking-based Aggregation

From Wang et al., NeurIPS 2024 "FLoRA: Federated Fine-Tuning Large Language
Models with Heterogeneous Low-Rank Adaptations".

Key insight: Instead of averaging LoRA matrices (which introduces noise),
STACK them to preserve each client's contribution exactly, then compress.

This is DIFFERENT from FlexLoRA which computes weighted Σ(BA).
FLoRA stacks without weighting, preserving client contributions.
"""

from typing import Dict, List, Optional

import torch


class FLoRAAggregator:
    """
    FLoRA: Stack client LoRAs, then compress with SVD.

    Algorithm:
    1. Stack A matrices vertically: [A_1; A_2; ...; A_K] → shape (K*r, d_in)
    2. Concat B matrices horizontally: [B_1, B_2, ..., B_K] → shape (d_out, K*r)
    3. Product B_stacked @ A_stacked = Σ(B_k @ A_k) exactly
    4. Apply SVD and truncate to max_rank
    5. Reconstruct A_global, B_global

    Key difference from FlexLoRA:
    - FLoRA: Unweighted stacking (preserves each client equally)
    - FlexLoRA: Weighted sum (clients weighted by data size)
    """

    def __init__(self, max_rank: int = 64):
        self.name = "FLoRA"
        self.max_rank = max_rank

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,  # IGNORED in true FLoRA
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client LoRA states using stacking (not averaging).

        Note: weights parameter is accepted for API compatibility but
        IGNORED because true FLoRA does not weight clients - it stacks
        them equally and lets SVD determine importance.
        """
        if not client_states:
            raise ValueError("No client states to aggregate")

        num_clients = len(client_states)
        aggregated = {}

        # Find all LoRA A/B pairs (PEFT uses lora_A / lora_B)
        a_keys = [k for k in client_states[0].keys() if "lora_A" in k]

        for a_key in a_keys:
            b_key = a_key.replace("lora_A", "lora_B")
            if b_key not in client_states[0]:
                continue

            original_dtype = client_states[0][a_key].dtype

            # STEP 1: Stack A vertically: (K*r, d_in)
            a_matrices = [state[a_key].float() for state in client_states]
            stacked_a = torch.cat(a_matrices, dim=0)

            # STEP 2: Concat B horizontally: (d_out, K*r)
            b_matrices = [state[b_key].float() for state in client_states]
            stacked_b = torch.cat(b_matrices, dim=1)

            # Determine ranks
            original_rank = a_matrices[0].shape[0]

            # STEP 3: Full product (exact Σ(B_k @ A_k), unweighted)
            ba_product = stacked_b @ stacked_a
            # Normalize to keep update scale comparable across client counts.
            ba_product = ba_product / max(1, num_clients)

            # STEP 4: SVD compression
            if not torch.isfinite(ba_product).all():
                n_bad = int((~torch.isfinite(ba_product)).sum().item())
                total = int(ba_product.numel())
                print(
                    f"[warn] FLoRA non-finite BA product for {a_key}: "
                    f"{n_bad}/{total} entries non-finite. "
                    "Sanitizing to enable SVD; consider lowering learning_rate for stability."
                )
                ba_product = torch.nan_to_num(
                    ba_product,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                )
            U, S, Vh = torch.linalg.svd(ba_product, full_matrices=False)

            # Keep adapter rank compatible with PEFT (default: original_rank),
            # capped by max_rank.
            target_rank = min(original_rank, self.max_rank, S.shape[0])
            U = U[:, :target_rank]
            S = S[:target_rank]
            Vh = Vh[:target_rank, :]

            # STEP 5: Reconstruct A and B using sqrt(S) split
            sqrt_s = torch.sqrt(S)
            new_b = (U * sqrt_s.unsqueeze(0)).to(original_dtype)
            new_a = (sqrt_s.unsqueeze(1) * Vh).to(original_dtype)

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
