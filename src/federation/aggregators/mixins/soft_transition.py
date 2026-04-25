"""
Soft Transition Mixin for FedLoRA-Adaptive v2.

Provides gradual unfreezing of A matrices instead of a hard switch.
"""

from __future__ import annotations


class SoftTransitionMixin:
    """
    Mixin that provides soft transition functionality.

    Instead of binary switch:
        Before: freeze_ratio = 1.0 if mode == "FFA-LoRA" else 0.0

    We use gradual transition:
        During TRANSITION phase: freeze_ratio decreases from 1.0 to 0.0
    """

    def init_soft_transition(self, transition_rounds: int = 3):
        """Initialize soft transition state."""
        self.transition_rounds = int(transition_rounds)
        self.transition_start_round = None

    def get_freeze_ratio(self, round_num: int, phase: str) -> float:
        """
        Get the fraction of A matrices to freeze.

        Args:
            round_num: Current round number
            phase: Current phase ("FFA-LoRA", "TRANSITION", or "FLoRA")

        Returns:
            Float in [0, 1] where:
            - 1.0 = fully frozen (pure FFA-LoRA)
            - 0.0 = fully trainable (pure FLoRA)
            - 0.5 = half frozen (mid-transition)
        """
        if phase == "FFA-LoRA":
            return 1.0
        if phase == "FLoRA":
            return 0.0
        if phase == "TRANSITION":
            if self.transition_start_round is None:
                return 0.5  # Fallback
            rounds_since_switch = round_num - self.transition_start_round
            denom = max(1, self.transition_rounds)
            progress = min(1.0, rounds_since_switch / denom)
            # Linear interpolation: 1.0 -> 0.0
            return 1.0 - progress
        raise ValueError(f"Unknown phase: {phase}")

    def start_transition(self, round_num: int):
        """Mark the start of transition phase."""
        self.transition_start_round = int(round_num)

    def is_transition_complete(self, round_num: int) -> bool:
        """Check if transition is complete."""
        if self.transition_start_round is None:
            return False
        return int(round_num) >= self.transition_start_round + max(
            1, self.transition_rounds
        )

