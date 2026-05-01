"""
FFA-LoRA Aggregator: Freeze A, Aggregate Only B

From Sun et al., ICLR 2024.
Key insight: Freezing A reduces gradient coupling and halves communication.

CURSOR AI: Implement this file as specified.
"""

from typing import Dict, List, Optional

import torch


class FFALoRAAggregator:
    """
    FFA-LoRA: Freeze A matrices, only aggregate B.

    - A matrices frozen from initialization
    - Only B matrices are trained and aggregated
    - 50% communication reduction
    """

    def __init__(self):
        self.name = "FFA-LoRA"
        self.frozen_a = None
        self.initialized = False

    def aggregate(
        self,
        client_states: List[Dict[str, torch.Tensor]],
        weights: Optional[List[float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate: freeze A, average B.
        """
        if not client_states:
            raise ValueError("No client states")

        # Initialize frozen A on first call
        if not self.initialized:
            self.frozen_a = {}
            for name, param in client_states[0].items():
                if "lora_A" in name or "lora_a" in name:
                    self.frozen_a[name] = param.clone()
            self.initialized = True
        elif self.frozen_a is None:
            # Defensive: should not happen, but keep behavior sane.
            self.frozen_a = {}

        # Default weights
        if weights is None:
            weights = [1.0 / len(client_states)] * len(client_states)
        total = sum(weights)
        weights = [w / total for w in weights]

        # Aggregate
        aggregated = {}
        for name in client_states[0].keys():
            if "lora_A" in name or "lora_a" in name:
                # Use frozen A. If we haven't seen this A key yet (e.g. when
                # aggregating layer subsets), cache it lazily.
                if name not in self.frozen_a:
                    self.frozen_a[name] = client_states[0][name].clone()
                aggregated[name] = self.frozen_a[name].clone()
            else:
                # Average B (and any other params)
                weighted_sum = sum(
                    w * state[name].float()
                    for w, state in zip(weights, client_states)
                )
                aggregated[name] = weighted_sum.to(client_states[0][name].dtype)

        return aggregated

    def get_communication_cost(self, state: Dict[str, torch.Tensor]) -> int:
        """Bytes for one client (only B matrices after init)."""
        b_params = sum(
            p.numel()
            for name, p in state.items()
            if "lora_B" in name or "lora_b" in name
        )
        return b_params * 2 * 2  # float16 * 2 (upload + download)

    def reset(self):
        """Reset for new experiment."""
        self.frozen_a = None
        self.initialized = False

    def get_frozen_a(self) -> Dict[str, torch.Tensor]:
        """
        Return cached frozen A matrices.

        Used by the server to reconstruct full client states from B-only uploads.
        Returns empty dict if not yet initialized.
        """
        if self.frozen_a is None:
            return {}
        return {k: v.clone() for k, v in self.frozen_a.items()}

    def should_upload_b_only(self) -> bool:
        """
        Whether clients should upload B-only tensors for this round.

        We can only do B-only uploads after at least one aggregate() call has
        initialized frozen A on the server side.
        """
        return bool(self.initialized)

    def should_broadcast_b_only(self) -> bool:
        """
        Whether the server should broadcast B-only state to clients for the
        upcoming round.

        Mirrors should_upload_b_only(). We can broadcast B-only only after
        frozen A has been initialized on the server (i.e., after at least one
        aggregate() call). Before that, clients have no A to retain.
        """
        return bool(self.initialized)
